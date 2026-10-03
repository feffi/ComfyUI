"""Attention microbenchmark for Krea 2, Qwen-Image 2.1 and MiniMax H3 shapes on ROCm.

Standalone: needs torch, optionally comfy_kitchen and a SolAttn_triton checkout.
Does not import ComfyUI, so it can run while a ComfyUI instance is up.

  python attn_bench.py --device 1                      # all default cases
  python attn_bench.py --device 1 --explain            # only print which SDPA backend each case can use, and why not
  python attn_bench.py --device 1 --cases krea2_1024,h3_21760 --long
  python attn_bench.py --device 1 --solattn custom_nodes/ComfyUI-SolAttn_triton
  python attn_bench.py --device 1 --explain --profile --cases krea2_1024 --dtypes bf16,fp32   # kernels SDPA really runs
  python attn_bench.py --smoke                          # tiny shapes, any device incl. CPU (syntax/flow check)

Per backend it reports median ms per call, peak extra memory over the inputs
(torch.cuda.max_memory_allocated) and the relative L2 error against an fp32
reference computed in query chunks.
"""
import argparse
import gc
import json
import math
import os
import statistics
import sys
import types
import warnings

# ComfyUI's main.py sets this before torch is imported; without it AOTriton keeps
# experimental arches (gfx1201 on some AOTriton builds) off.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

PRIORITY = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

# name: (B, Hq, Hkv, Lq, Lk, D, mask, layout, note)
# layout "bhsd" = contiguous (B, H, L, D); "h3" = q/k/v are strided views of one (L, 3*H*D) qkv buffer like MiniMax H3.
CASES = {
    "krea2_1024":        (2, 48, 12, 4352, 4352, 128, None, "bhsd", "Krea 2 DiT block, 1024^2 (4096 img + 256 txt), CFG batch 2, 28 calls/step"),
    "krea2_2048":        (2, 48, 12, 16640, 16640, 128, None, "bhsd", "Krea 2 DiT block, 2048^2, CFG batch 2"),
    "krea2_txtfusion":   (24, 20, 20, 256, 256, 128, None, "bhsd", "Krea 2 text fusion layerwise block (B*12 layers)"),
    "qi21_1024_target":  (1, 32, 32, 4096, 4352, 128, None, "bhsd", "Qwen-Image 2.1 target rows over prefix+target, 1024^2"),
    "qi21_1024_text":    (1, 32, 32, 256, 256, 128, "causal", "bhsd", "Qwen-Image 2.1 causal text segment (bool mask)"),
    "qi21_2048_target":  (1, 32, 32, 16384, 16640, 128, None, "bhsd", "Qwen-Image 2.1 target rows, 2048^2"),
    "h3_8501":           (1, 56, 56, 8501, 8501, 128, None, "h3", "MiniMax H3 block, 8.5k tokens, 50 calls per forward"),
    "h3_21760":          (1, 56, 56, 21760, 21760, 128, None, "h3", "MiniMax H3 block, 1024x576x90f"),
    "vae_wan21_1024":    (1, 1, 1, 16384, 16384, 384, None, "bhsd", "Wan 2.1 VAE mid attention (Krea 2), 1024^2 decode"),
    "vae_qi21_1024_c512": (1, 1, 1, 4096, 4096, 512, None, "bhsd", "Qwen-Image 2.1 VAE mid attention if C=512"),
    "vae_qi21_1024_c1024": (1, 1, 1, 4096, 4096, 1024, None, "bhsd", "Qwen-Image 2.1 VAE mid attention if C=1024"),
}
LONG_CASES = {
    "h3_43520": (1, 56, 56, 43520, 43520, 128, None, "h3", "MiniMax H3 block, 2x the 90f sequence"),
}
SMOKE_CASES = {
    "smoke_gqa": (1, 8, 2, 256, 256, 128, None, "bhsd", "tiny GQA"),
    "smoke_h3": (1, 4, 4, 300, 300, 128, None, "h3", "tiny strided"),
    "smoke_mask": (1, 4, 4, 128, 128, 64, "causal", "bhsd", "tiny causal mask"),
    "smoke_vae": (1, 1, 1, 256, 256, 384, None, "bhsd", "tiny vae"),
}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def make_inputs(case, dtype, dev, seed=0):
    B, Hq, Hkv, Lq, Lk, D, mask_kind, layout, _ = case
    g = torch.Generator(device=dev).manual_seed(seed)
    if layout == "h3":
        qkv = torch.randn(Lq, 3 * Hq * D, device=dev, dtype=dtype, generator=g)
        q, k, v = qkv.split(Hq * D, dim=-1)
        q, k, v = (t.view(Lq, Hq, D).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    else:
        q = torch.randn(B, Hq, Lq, D, device=dev, dtype=dtype, generator=g)
        k = torch.randn(B, Hkv, Lk, D, device=dev, dtype=dtype, generator=g)
        v = torch.randn(B, Hkv, Lk, D, device=dev, dtype=dtype, generator=g)
    mask = None
    if mask_kind == "causal":
        mask = torch.ones(Lq, Lk, dtype=torch.bool, device=dev).tril(Lk - Lq).view(1, 1, Lq, Lk)
    return q, k, v, mask


def expand_kv(k, v, hq):
    rep = hq // k.shape[1]
    if rep == 1:
        return k, v
    return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)


def reference(q, k, v, mask, budget=1 << 30):
    """fp32 attention in query chunks so long sequences fit."""
    k, v = expand_kv(k, v, q.shape[1])
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = q.shape[-1] ** -0.5
    B, H, Lq, _ = q.shape
    Lk = k.shape[2]
    chunk = max(1, min(Lq, budget // max(1, B * H * Lk * 4)))
    out = torch.empty(B, H, Lq, v.shape[-1], dtype=torch.float32, device=q.device)
    for i in range(0, Lq, chunk):
        s = torch.matmul(qf[:, :, i:i + chunk], kf.transpose(-1, -2)) * scale
        if mask is not None:
            s.masked_fill_(~mask[..., i:i + chunk, :], float("-inf"))
        out[:, :, i:i + chunk] = torch.matmul(s.softmax(-1), vf)
        del s
    return out


def rel_err(out, ref):
    out = out.float()
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def sdpa_params(q, k, v, mask, gqa):
    return torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, gqa)


def eligibility(q, k, v, mask, gqa, debug=False):
    p = sdpa_params(q, k, v, mask, gqa)
    res = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, check in (("flash", torch.backends.cuda.can_use_flash_attention),
                            ("cudnn", torch.backends.cuda.can_use_cudnn_attention),
                            ("efficient", torch.backends.cuda.can_use_efficient_attention)):
            try:
                res[name] = check(p, debug)
            except RuntimeError as e:
                res[name] = False
                caught.append(warnings.WarningMessage(f"{name}: {e}", RuntimeWarning, __file__, 0))
    # torch may print the debug reasons straight to stderr instead of raising Python warnings
    reasons = [str(w.message).replace("\n", " ") for w in caught] if debug else []
    chosen = next((n for n in ("flash", "cudnn", "efficient") if res[n]), "math")
    return chosen, res, reasons


# --- backends: fn(q, k, v, mask) -> (B, Hq, Lq, D) ---------------------------------

def be_sdpa_native(q, k, v, mask):
    gqa = q.shape[1] != k.shape[1]
    kw = {"enable_gqa": True} if gqa else {}
    with sdpa_kernel(PRIORITY, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kw)


def be_sdpa_expanded(q, k, v, mask):
    k, v = expand_kv(k, v, q.shape[1])
    with sdpa_kernel(PRIORITY, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def forced(backend, expand):
    def fn(q, k, v, mask):
        kw = {}
        if q.shape[1] != k.shape[1]:
            if expand:
                k, v = expand_kv(k, v, q.shape[1])
            else:
                kw["enable_gqa"] = True
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kw)
    return fn


def be_slice_attention(q, k, v, mask):
    """ComfyUI's VAE fallback (normal_attention / slice_attention) for one head, B=1."""
    assert q.shape[1] == 1 and mask is None
    qq = q[:, 0]                      # b, hw, c
    kk = k[:, 0].transpose(1, 2)      # b, c, hw
    vv = v[:, 0].transpose(1, 2)      # b, c, hw
    r1 = torch.zeros_like(kk)
    scale = q.shape[-1] ** -0.5
    free = torch.cuda.mem_get_info(q.device)[0] if q.device.type == "cuda" else 8 << 30
    need = qq.shape[0] * qq.shape[1] * kk.shape[2] * qq.element_size() * (3 if qq.element_size() == 2 else 2.5)
    steps = 1 if need <= free else 2 ** math.ceil(math.log(need / free, 2))
    size = qq.shape[1] // steps if qq.shape[1] % steps == 0 else qq.shape[1]
    for i in range(0, qq.shape[1], size):
        s1 = torch.bmm(qq[:, i:i + size], kk) * scale
        s2 = torch.softmax(s1, dim=2).permute(0, 2, 1)
        del s1
        r1[:, :, i:i + size] = torch.bmm(vv, s2)
        del s2
    return r1.transpose(1, 2).unsqueeze(1)


def kitchen_version():
    try:
        from importlib.metadata import version
        return version("comfy-kitchen")
    except Exception:  # noqa: BLE001
        return None


def load_kitchen():
    try:
        import comfy_kitchen as ck
    except Exception as e:  # noqa: BLE001 - optional package
        print(f"comfy_kitchen not importable: {e}")
        return None
    return ck


def kitchen_backends(ck, dev):
    out = {}
    if ck is None:
        return out
    if ck.int8_attention_is_available(dev):
        out["ck_int8"] = lambda q, k, v, mask: ck.int8_attention(q, k, v, attn_mask=mask)
    if ck.sol_attn_is_available(dev):
        def sol(tau, token_aug):
            def fn(q, k, v, mask):
                if mask is not None or q.shape[2] != k.shape[2]:
                    raise NotImplementedError("sol_attn: self-attention without mask only")
                k, v = expand_kv(k, v, q.shape[1])
                qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
                return ck.sol_attn(qs, ks, vs, tau=tau, token_aug=token_aug).transpose(1, 2)
            return fn
        out["ck_sol_tau1.3"] = sol(1.3, 0)
        out["ck_sol_tau1.3_aug256"] = sol(1.3, 256)
    return out


def solattn_backends(path):
    if not path:
        return {}
    path = os.path.abspath(path)
    pkg = types.ModuleType("solattn_pkg")
    pkg.__path__ = [path]
    sys.modules["solattn_pkg"] = pkg
    out = {}
    try:
        import importlib
        tri = importlib.import_module("solattn_pkg._tri_fwd")

        def bf16(q, k, v, mask):
            if mask is not None or q.shape != k.shape:
                raise NotImplementedError("SolAttn_triton declines masks and GQA")
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
            return tri.sol_attn(qs, ks, vs, scale=None, tau=1.3, sink_blocks=(0, 0), sink_q=(0, 0), use_tma=False).transpose(1, 2)
        out["solattn_triton_bf16"] = bf16
    except Exception as e:  # noqa: BLE001
        print(f"SolAttn_triton bf16 kernel not importable: {e!r}")
    try:
        import importlib
        i8 = importlib.import_module("solattn_pkg._int8_fwd")

        def int8(q, k, v, mask):
            if mask is not None or q.shape != k.shape:
                raise NotImplementedError("SolAttn_triton declines masks and GQA")
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
            return i8.sol_attn_int8(qs, ks, vs, scale=None, tau=1.3, use_tma=False, int8_pv=True).transpose(1, 2)
        out["solattn_triton_int8"] = int8
    except Exception as e:  # noqa: BLE001
        print(f"SolAttn_triton int8 kernel not importable: {e!r}")
    return out


def run_one(fn, q, k, v, mask, dev, warmup, iters):
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    out = None
    for _ in range(warmup):
        out = fn(q, k, v, mask)
        del out
    sync(dev)
    times = []
    if dev.type == "cuda":
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(q, k, v, mask)
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
            if _ != iters - 1:
                del out
        peak = (torch.cuda.max_memory_allocated(dev) - base) / 2**20
    else:
        import time
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn(q, k, v, mask)
            times.append((time.perf_counter() - t0) * 1000)
        peak = float("nan")
    return statistics.median(times), peak, out


def math_bytes(case, dtype):
    B, Hq, _, Lq, Lk, _, _, _, _ = case
    return B * Hq * Lq * Lk * max(4, torch.finfo(dtype).bits // 8) * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--cases", default="")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--dtypes", default="bf16", help="comma list of bf16,fp16,fp32")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--solattn", default="", help="path to a ComfyUI-SolAttn_triton checkout")
    ap.add_argument("--math-limit-gb", type=float, default=8.0, help="skip forced MATH above this score-matrix size")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", default="", help="also write results to this file")
    ap.add_argument("--profile", action="store_true", help="print the top GPU kernels of sdpa_native per case (shows AOTriton vs math)")
    args = ap.parse_args()

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    except AttributeError:
        pass

    if torch.cuda.is_available():
        dev = torch.device("cuda", args.device)
        torch.cuda.set_device(dev)
        props = torch.cuda.get_device_properties(dev)
        arch = getattr(props, "gcnArchName", f"sm{props.major}{props.minor}")
    else:
        dev = torch.device("cpu")
        arch = "cpu"
    ck = load_kitchen()
    print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip, "device": str(dev), "arch": arch,
                      "comfy_kitchen": kitchen_version(),
                      "AOTRITON_EXPERIMENTAL": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
                      "flash_built": torch.backends.cuda.is_flash_attention_available() if dev.type == "cuda" else None}))

    cases = dict(SMOKE_CASES) if args.smoke else dict(CASES)
    if args.long:
        cases.update(LONG_CASES)
    if args.cases:
        allc = {**CASES, **LONG_CASES, **SMOKE_CASES}
        cases = {n: allc[n] for n in args.cases.split(",")}
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dlist = [dtypes[d] for d in args.dtypes.split(",")]
    if args.smoke and dev.type == "cpu":
        dlist = [torch.float32]

    extra = {}
    if dev.type == "cuda":
        extra.update(kitchen_backends(ck, dev))
        extra.update(solattn_backends(args.solattn))

    results = []
    for name, case in cases.items():
        for dtype in dlist:
            q, k, v, mask = make_inputs(case, dtype, dev)
            gqa = q.shape[1] != k.shape[1]
            print(f"\n== {name} {str(dtype)[6:]} q{tuple(q.shape)} k{tuple(k.shape)} mask={case[6]} : {case[8]}")
            if dev.type == "cuda":
                chosen, res, reasons = eligibility(q, k, v, mask, gqa, debug=args.explain)
                print(f"   SDPA (comfy priority) native{' GQA' if gqa else ''}: {chosen}  {res}")
                for r in reasons:
                    print(f"     - {r}")
                if gqa:
                    ke, ve = expand_kv(k, v, q.shape[1])
                    chosen_e, res_e, _ = eligibility(q, ke, ve, mask, False)
                    print(f"   SDPA with K/V expanded: {chosen_e}  {res_e}")
                    del ke, ve
            if args.profile and dev.type == "cuda":
                be_sdpa_native(q, k, v, mask)
                sync(dev)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    be_sdpa_native(q, k, v, mask)
                    sync(dev)
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=6, max_name_column_width=90))
            if args.explain:
                continue
            ref = reference(q, k, v, mask)
            backends = {"sdpa_native": be_sdpa_native}
            if gqa:
                backends["sdpa_expanded"] = be_sdpa_expanded
            if dev.type == "cuda":
                backends["force_flash"] = forced(SDPBackend.FLASH_ATTENTION, False)
                backends["force_efficient_expanded"] = forced(SDPBackend.EFFICIENT_ATTENTION, True)
                if math_bytes(case, dtype) <= args.math_limit_gb * 2**30:
                    backends["force_math"] = forced(SDPBackend.MATH, False)
                backends.update(extra)
            if case[1] == 1 and mask is None:
                backends["comfy_vae_slice"] = be_slice_attention
            for bname, fn in backends.items():
                try:
                    ms, peak, out = run_one(fn, q, k, v, mask, dev, args.warmup, args.iters)
                    err = rel_err(out, ref)
                    del out
                    print(f"   {bname:26s} {ms:9.3f} ms  peak +{peak:9.1f} MiB  relL2 {err:.2e}")
                    results.append({"case": name, "dtype": str(dtype)[6:], "backend": bname, "ms": ms, "peak_mib": peak, "rel_l2": err})
                except Exception as e:  # noqa: BLE001 - report and continue
                    msg = str(e).splitlines()[0][:140] if str(e) else type(e).__name__
                    print(f"   {bname:26s} unavailable: {msg}")
                    results.append({"case": name, "dtype": str(dtype)[6:], "backend": bname, "error": msg})
                gc.collect()
            del q, k, v, mask, ref
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
