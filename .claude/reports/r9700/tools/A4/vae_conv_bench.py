"""Per-conv microbenchmark of the Krea 2 (Wan 2.1), Qwen-Image 2.1 and MiniMax H3 VAE convolutions.

The conv shapes are collected by walking each VAE on the meta device (no checkpoint needed), so they
are exactly what ComfyUI runs at the chosen resolution. For every unique shape it times:

  torch        F.conv3d / F.conv2d as ComfyUI calls it today (MIOpen off -> slow_conv_dilated3d / slow_conv2d)
  nobias_add   same conv without bias, bias added in one op (isolates the per-channel bias fill of slow_conv_dilated3d)
  conv2d       single-frame conv3d (T == 1, kT == 1) run as conv2d (proposal A4-1)
  kitchen      comfy-kitchen HIP fp16_conv3d on a pre-padded NDHWC input (fp16 only; prep time reported apart)

Run once with MIOpen off (default) and once with --miopen to compare MIOpen (first call = Find cost).
MIOPEN_FIND_MODE / MIOPEN_USER_DB_PATH must be set in the environment before starting.

Usage (from anywhere, production venv, GPU 1 so a running ComfyUI on GPU 0 is not disturbed):
  .venv-rocm-100\\Scripts\\python vae_conv_bench.py --comfy C:\\path\\to\\ComfyUI --gpu 1 --res 1024
  set COMFYUI_ENABLE_MIOPEN=1 & ... vae_conv_bench.py --comfy ... --gpu 1 --miopen
Options: --models krea2,qwen21,minimax  --dtypes bf16,fp16  --top 15  --profile  --out conv.jsonl
WARNING: MIOpen Find benchmarks naive kernels on large shapes; on Windows that can hit the 2 s TDR.
Start with --res 512 when testing --miopen for the first time.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--comfy", required=True)
p.add_argument("--gpu", default=None, help="device index to expose (sets HIP_VISIBLE_DEVICES)")
p.add_argument("--res", type=int, default=1024)
p.add_argument("--models", default="krea2,qwen21,minimax")
p.add_argument("--dtypes", default="bf16,fp16")
p.add_argument("--top", type=int, default=15, help="shapes per model, by FLOPs x count")
p.add_argument("--iters", type=int, default=5)
p.add_argument("--miopen", action="store_true", help="also time with torch.backends.cudnn.enabled=True (needs COMFYUI_ENABLE_MIOPEN=1 or is forced here)")
p.add_argument("--miopen-immediate", action="store_true", help="torch.backends.miopen.immediate=True for the MIOpen runs")
p.add_argument("--profile", action="store_true", help="count GPU kernels per call for the largest shape of each model")
p.add_argument("--out", default=None)
p.add_argument("--comfy-args", default="", help="extra ComfyUI args, e.g. \"--cpu\" for a dry run")
a = p.parse_args()

if a.gpu is not None:
    os.environ["HIP_VISIBLE_DEVICES"] = a.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
sys.argv = [sys.argv[0]] + a.comfy_args.split()
sys.path.insert(0, a.comfy)
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.quant_ops  # noqa: E402
import comfy.ldm.wan.vae as wan21  # noqa: E402
import comfy.ldm.wan.vae2_2 as wan22  # noqa: E402
import comfy.ldm.minimax.vae as mmvae  # noqa: E402

dev = mm.get_torch_device()
ck = getattr(comfy.quant_ops, "ck", None)
cudnn_default = torch.backends.cudnn.enabled


def env_report():
    r = {"torch": torch.__version__, "hip": torch.version.hip, "device": str(dev),
         "cudnn_enabled_after_comfy_import": cudnn_default}
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        r.update(arch=props.gcnArchName if torch.version.hip else props.name,
                 amd_min_version_rdna3=mm.amd_min_version(dev, min_rdna_version=3),
                 miopen_version=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    try:
        import comfy_kitchen
        r["comfy_kitchen"] = getattr(comfy_kitchen, "__version__", "?")
        r["kitchen_backends"] = {k: str(v) for k, v in comfy_kitchen.list_backends().items()}
    except Exception as e:
        r["comfy_kitchen"] = f"unavailable: {e}"
    for k in ("MIOPEN_FIND_MODE", "MIOPEN_USER_DB_PATH", "COMFYUI_ENABLE_MIOPEN", "PYTORCH_MIOPEN_SUGGEST_NHWC"):
        r[k] = os.environ.get(k)
    return r


# ---- collect conv shapes on the meta device ----
calls = []
_c2, _c3 = F.conv2d, F.conv3d


def _tup(v, n):
    return tuple(v) if isinstance(v, (list, tuple)) else (v,) * n


def rec2(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    calls.append(("2d", tuple(x.shape), tuple(w.shape), _tup(stride, 2), _tup(padding, 2), b is not None))
    return _c2(x, w, b, stride, padding, dilation, groups)


def rec3(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    calls.append(("3d", tuple(x.shape), tuple(w.shape), _tup(stride, 3), _tup(padding, 3), b is not None))
    return _c3(x, w, b, stride, padding, dilation, groups)


def collect(model, res):
    F.conv2d, F.conv3d = rec2, rec3
    calls.clear()
    try:
        with torch.device("meta"):
            if model == "krea2":
                v = wan21.WanVAE(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                                 temperal_downsample=[False, True, True], image_channels=3, conv_out_channels=3)
                for m in v.modules():
                    if hasattr(m, "optimized_attention"):
                        m.optimized_attention = lambda q, k, v: q
                v.decode(torch.empty(1, 16, 1, res // 8, res // 8))
            elif model == "qwen21":
                v = wan22.WanVAE(dim=96, dec_dim=144, z_dim=64, dim_mult=[1, 2, 4, 8, 8], num_res_blocks=2, attn_scales=[],
                                 temperal_downsample=[False, True, True, True], image_channels=4, patch_size=1, temporal_kernel=1)
                for m in v.modules():
                    if hasattr(m, "optimized_attention"):
                        m.optimized_attention = lambda q, k, v: q
                v.decode(torch.empty(1, 64, res // 16, res // 16))
            elif model == "minimax":
                # single-frame encode of one 256 px tile, the shapes that miss the kitchen conv today
                e = mmvae.EncoderFCN3D(ch=128, ch_mult=[1, 2, 2, 4, 4, 8], space_down=[2, 2, 2, 2, 1, 1],
                                       time_down=[1, 2, 2, 1, 1, 1], num_res_blocks=2, in_channels=3, z_channels=24)
                e(torch.empty(1, 3, 1, 256, 256))
    finally:
        F.conv2d, F.conv3d = _c2, _c3
    counts = {}
    for c in calls:
        counts[c] = counts.get(c, 0) + 1
    return counts


def flops(c):
    kind, xs, ws, st, pd, _ = c
    out = [(xs[2 + i] + 2 * pd[i] - ws[2 + i]) // st[i] + 1 for i in range(len(ws) - 2)]
    return 2 * xs[0] * ws[0] * math.prod(ws[1:]) * math.prod(out)


# ---- timing ----
def timed(fn, iters):
    if dev.type != "cuda":
        t = time.perf_counter(); fn(); first = time.perf_counter() - t
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        return first * 1e3, statistics.median(ts) * 1e3, 0
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); fn(); torch.cuda.synchronize(); first = (time.perf_counter() - t) * 1e3
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    return first, statistics.median(ts), (torch.cuda.max_memory_allocated() - base) / 2**20


def variants(c, dtype):
    kind, xs, ws, st, pd, has_bias = c
    x = torch.randn(xs, device=dev, dtype=dtype)
    w = torch.randn(ws, device=dev, dtype=dtype) * (1.0 / math.sqrt(math.prod(ws[1:])))
    b = torch.randn(ws[0], device=dev, dtype=dtype) if has_bias else None
    conv = F.conv3d if kind == "3d" else F.conv2d
    v = {"torch": lambda: conv(x, w, b, st, pd)}
    if b is not None:
        def nobias():
            o = conv(x, w, None, st, pd)
            return o.add_(b.view((1, -1) + (1,) * (o.ndim - 2)))
        v["nobias_add"] = nobias
    if kind == "3d" and xs[2] == 1 and ws[2] == 1 and pd[0] == 0:
        v["conv2d"] = lambda: F.conv2d(x[:, :, 0], w[:, :, 0], b, st[1:], pd[1:]).unsqueeze(2)
    if dtype == torch.float16 and ck is not None and hasattr(ck, "fp16_conv3d"):
        x5 = x if kind == "3d" else x.unsqueeze(2)
        w5 = w if kind == "3d" else w.unsqueeze(2)
        st3 = st if kind == "3d" else (1,) + st
        pd3 = pd if kind == "3d" else (0,) + pd

        def prep():
            xp = F.pad(x5, (pd3[2], pd3[2], pd3[1], pd3[1], pd3[0], pd3[0])) if any(pd3) else x5
            return xp.contiguous(memory_format=torch.channels_last_3d)
        xp = prep()
        wcl = w5.contiguous(memory_format=torch.channels_last_3d)
        v["kitchen"] = lambda: ck.fp16_conv3d(xp, wcl, b, None, list(st3))
        v["kitchen_prep"] = prep
    return v


def kernel_count(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    names = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            names[e.name] = names.get(e.name, 0) + 1
    return sum(names.values()), sorted(names.items(), key=lambda kv: -kv[1])[:4]


def main():
    out = open(a.out, "a") if a.out else None
    rep = env_report()
    print(json.dumps(rep, indent=1))
    if out:
        out.write(json.dumps({"env": rep}) + "\n")
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    modes = [False] + ([True] if a.miopen else [])
    for model in a.models.split(","):
        counts = collect(model, a.res)
        shapes = sorted(counts, key=lambda c: -flops(c) * counts[c])[:a.top]
        shapes.sort(key=flops)  # small first: a TDR on a large MIOpen Find shows up last
        print(f"\n== {model} @ {a.res}px: {len(counts)} unique conv shapes, {sum(counts.values())} calls; timing top {len(shapes)}")
        totals = {}
        for dname in a.dtypes.split(","):
            dtype = dtypes[dname]
            for miopen in modes:
                torch.backends.cudnn.enabled = miopen
                if a.miopen_immediate and miopen:
                    torch.backends.miopen.immediate = True
                for c in shapes:
                    for vname, fn in variants(c, dtype).items():
                        if miopen and vname.startswith("kitchen"):
                            continue
                        try:
                            first, med, peak = timed(fn, a.iters)
                        except Exception as e:
                            print(f"   {c} {vname}: FAILED {type(e).__name__}: {e}")
                            continue
                        tag = f"{dname}/{'miopen' if miopen else 'nomiopen'}/{vname}"
                        totals[tag] = totals.get(tag, 0.0) + med * counts[c]
                        row = {"model": model, "res": a.res, "dtype": dname, "miopen": miopen, "variant": vname,
                               "kind": c[0], "x": c[1], "w": c[2], "stride": c[3], "pad": c[4], "count": counts[c],
                               "first_ms": round(first, 3), "median_ms": round(med, 3), "peak_mib": round(peak, 1)}
                        print(f"   {dname:4} {'MIOpen' if miopen else 'noMIOp':6} {vname:12} {c[0]} x={c[1]} w={c[2]} s={c[3]} x{counts[c]:<3}"
                              f" first {first:8.2f} ms  median {med:8.3f} ms  peak {peak:8.1f} MiB")
                        if out:
                            out.write(json.dumps(row) + "\n")
                if a.profile and dev.type == "cuda" and not miopen:
                    c = shapes[-1]
                    for vname, fn in variants(c, dtype).items():
                        n, top = kernel_count(fn)
                        print(f"   kernels per call [{vname}] {c[0]} x={c[1]} w={c[2]}: {n}  {top}")
        torch.backends.cudnn.enabled = cudnn_default
        print(f"-- {model}: sum(median x count) over the timed shapes, ms")
        for k, v in sorted(totals.items()):
            print(f"   {k:32} {v:9.1f}")
    if out:
        out.close()


if __name__ == "__main__":
    main()
