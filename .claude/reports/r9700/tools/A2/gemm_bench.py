"""Linear-layer (GEMM) microbenchmark for the ComfyUI paths on ROCm, at a model's real shapes.

Run from the ComfyUI folder with the production venv's python, on an idle GPU (GPU 1 if GPU 0 is busy):

  python gemm_bench.py --preset krea2 --tokens 4608 --device 1
  python gemm_bench.py --model models\\diffusion_models\\<file>.safetensors --tokens 4608 --device 1 --csv krea2_gemm.csv

--model reads only the safetensors header: every 2-D ".weight" with both dims >= 256 becomes an (N, K)
shape, counted per occurrence. --tokens is rows per linear call (batch * sequence; double it for CFG
batch 2). The summary line "per model call" is the sum over all linears of ms * count.

TunableOp A/B (torch GEMMs only, not the kitchen kernels): run once with
  set PYTORCH_TUNABLEOP_ENABLED=1
  set PYTORCH_TUNABLEOP_FILENAME=%CD%\\tunableop_bench.csv
then again with the same variables (it reads the file, no retuning), and compare the bf16 rows.

Variants (each one that cannot run on this stack prints "n/a" with the reason):
  bf16_hipblaslt  F.linear bf16, torch.backends.cuda.preferred_blas_library("hipblaslt")  (torch default on gfx1201)
  bf16_rocblas    F.linear bf16, preferred_blas_library("cublas") = rocBLAS
  fp16_torch      F.linear fp16 (default BLAS)
  fp16_kitchen    comfy_kitchen.fp16_linear (HIP WMMA, fp32 accumulate) - what --fast fp16_accumulation reaches
  fp8_comfy       ComfyUI fp8 path: QuantizedTensor.from_float(x) + F.linear -> kitchen scaled_mm_v2 -> HIP WMMA
  fp8_hip_mm      comfy_kitchen.backends.hip.scaled_mm_fp8 on pre-quantized input (GEMM only)
  fp8_quantize    the per-call activation quantize alone (ck.quantize_per_tensor_fp8)
  fp8_torch_tw    torch._scaled_mm tensor-wise scales (hipBLASLt)
  fp8_torch_rw    torch._scaled_mm row-wise scales (hipBLASLt)
  int8_kitchen    comfy_kitchen.int8_linear W8A8 (dynamic per-row activation, per-channel weight)
"""
import argparse
import csv
import math
import os
import sys

import torch
import torch.nn.functional as F

PRESETS = {
    # (N, K, count) from the model __init__ defaults in ComfyUI; prefer --model for the real checkpoint
    "krea2": [(6144, 6144, 3 * 28), (1536, 6144, 2 * 28), (16384, 6144, 2 * 28), (6144, 16384, 28)],
    "qwen21": [(4096, 4096, 4 * 32), (24576, 4096, 32), (4096, 12288, 32)],
    "minimax": [(21504, 5376, 50), (5376, 7168, 50), (28672, 5376, 50), (5376, 14336, 50)],
}


def shapes_from_model(path):
    from safetensors import safe_open
    counts = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if not k.endswith(".weight"):
                continue
            shape = tuple(f.get_slice(k).get_shape())
            if len(shape) == 2 and min(shape) >= 256:
                counts[shape] = counts.get(shape, 0) + 1
    return [(n, k, c) for (n, k), c in sorted(counts.items(), key=lambda i: -i[1] * i[0][0] * i[0][1])]


def bench(fn, rotate, min_ms=200.0, warmup=3):
    for i in range(warmup):
        fn(i % rotate)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn(0)
    end.record()
    torch.cuda.synchronize()
    iters = max(5, min(200, int(min_ms / max(start.elapsed_time(end), 1e-3))))
    start.record()
    for i in range(iters):
        fn(i % rotate)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def set_blas(name):
    try:
        torch.backends.cuda.preferred_blas_library(name)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def build_variants(ck, quant_ops):
    fp8 = torch.float8_e4m3fn
    one = lambda dev: torch.ones((), device=dev, dtype=torch.float32)

    def prep_fp8(w):
        s = (w.abs().amax().float() / 448.0).clamp(min=1e-12)
        return (w.float() / s).clamp(-448, 448).to(fp8), s

    def prep_int8(w):
        s = (w.abs().amax(dim=1).float() / 127.0).clamp(min=1e-12)
        return (w.float() / s[:, None]).round().clamp(-127, 127).to(torch.int8), s

    v = {}

    def bf16(blas):
        def setup(x, ws):
            err = set_blas(blas)
            if err:
                raise RuntimeError(err)
            xb, wb = x.bfloat16(), [w.bfloat16() for w in ws]
            return lambda i: F.linear(xb, wb[i])
        return setup
    v["bf16_hipblaslt"] = bf16("hipblaslt")
    v["bf16_rocblas"] = bf16("cublas")

    def fp16_torch(x, ws):
        set_blas("hipblaslt")
        xh, wh = x.half(), [w.half() for w in ws]
        return lambda i: F.linear(xh, wh[i])
    v["fp16_torch"] = fp16_torch

    def fp16_kitchen(x, ws):
        xh, wh = x.half(), [w.half() for w in ws]
        return lambda i: ck.fp16_linear(xh, wh[i])
    v["fp16_kitchen"] = fp16_kitchen

    def fp8_comfy(x, ws):
        QT = quant_ops.QuantizedTensor
        layout = "TensorCoreFP8E4M3Layout" if quant_ops.__name__ == "comfy.quant_ops" else "TensorCoreFP8Layout"
        qws = []
        for w in ws:
            q, s = prep_fp8(w)
            params = quant_ops.get_layout_class(layout).Params(scale=s.reshape(()), orig_dtype=torch.bfloat16, orig_shape=tuple(q.shape))
            qws.append(QT(q, layout, params))
        xb = x.bfloat16()

        def run(i):
            qx = QT.from_float(xb, layout, scale=None) if quant_ops.__name__ == "comfy.quant_ops" else QT.from_float(xb, layout, scale=one(xb.device))
            return F.linear(qx, qws[i])
        return run
    v["fp8_comfy"] = fp8_comfy

    def fp8_hip_mm(x, ws):
        from comfy_kitchen.backends import hip
        if not hip.has_wmma():
            raise RuntimeError("kitchen HIP backend without WMMA on this device set")
        qws = [prep_fp8(w) for w in ws]
        qx = x.bfloat16().clamp(-448, 448).to(fp8)
        sa = one(x.device)
        return lambda i: hip.scaled_mm_fp8(qx, qws[i][0].t(), sa, qws[i][1], None, torch.bfloat16)
    v["fp8_hip_mm"] = fp8_hip_mm

    def fp8_quantize(x, ws):
        xb, sa = x.bfloat16(), one(x.device)
        return lambda i: ck.quantize_per_tensor_fp8(xb, sa, fp8)
    v["fp8_quantize"] = fp8_quantize

    def fp8_torch_tw(x, ws):
        set_blas("hipblaslt")
        qws = [prep_fp8(w) for w in ws]
        qx = x.bfloat16().clamp(-448, 448).to(fp8)
        sa = one(x.device)
        return lambda i: torch._scaled_mm(qx, qws[i][0].t(), scale_a=sa, scale_b=qws[i][1].reshape(()), out_dtype=torch.bfloat16)
    v["fp8_torch_tw"] = fp8_torch_tw

    def fp8_torch_rw(x, ws):
        set_blas("hipblaslt")
        qws = []
        for w in ws:
            s = (w.abs().amax(dim=1).float() / 448.0).clamp(min=1e-12)
            qws.append(((w.float() / s[:, None]).clamp(-448, 448).to(fp8), s.reshape(1, -1).contiguous()))
        sx = (x.abs().amax(dim=1, keepdim=True).float() / 448.0).clamp(min=1e-12)
        qx = (x.float() / sx).to(fp8)
        return lambda i: torch._scaled_mm(qx, qws[i][0].t(), scale_a=sx, scale_b=qws[i][1], out_dtype=torch.bfloat16)
    v["fp8_torch_rw"] = fp8_torch_rw

    def int8_kitchen(x, ws):
        qws = [prep_int8(w) for w in ws]
        xb = x.bfloat16()
        return lambda i: ck.int8_linear(xb, qws[i][0], qws[i][1], None, torch.bfloat16)
    v["int8_kitchen"] = int8_kitchen
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--model", help="safetensors file; shapes are read from its header")
    ap.add_argument("--tokens", type=int, default=4608)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--max-shapes", type=int, default=12)
    ap.add_argument("--csv")
    ap.add_argument("--comfy-root", default=os.getcwd())
    a = ap.parse_args()

    torch.cuda.set_device(a.device)
    dev = torch.device("cuda", a.device)
    sys.path.insert(0, a.comfy_root)
    try:
        import comfy.quant_ops as quant_ops  # ComfyUI's layouts and backend selection, as in a real run
    except Exception as e:
        print(f"comfy.quant_ops not importable ({e}); using comfy_kitchen.tensor directly")
        import comfy_kitchen.tensor as quant_ops
    import comfy_kitchen as ck

    props = torch.cuda.get_device_properties(dev)
    print(f"torch {torch.__version__} hip {torch.version.hip} device {a.device} {props.name} {getattr(props, 'gcnArchName', '')}")
    print(f"comfy_kitchen {getattr(ck, '__version__', '?')} backends {ck.list_backends()}")
    print(f"default blas {torch.backends.cuda.preferred_blas_library()}  TunableOp {os.environ.get('PYTORCH_TUNABLEOP_ENABLED', '0')}")

    shapes = shapes_from_model(a.model) if a.model else PRESETS[a.preset or "krea2"]
    shapes = shapes[:a.max_shapes]
    variants = build_variants(ck, quant_ops)
    names = list(variants) if a.variants == "all" else a.variants.split(",")
    m = a.tokens
    rows = []
    totals = {n: 0.0 for n in names}
    for n_out, k_in, count in shapes:
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn(m, k_in, device=dev, generator=g) * 2.0
        rotate = max(1, min(8, math.ceil(256e6 / (n_out * k_in))))  # weights alternate past the 64 MB Infinity Cache
        ws = [torch.randn(n_out, k_in, device=dev, generator=g) * 0.02 for _ in range(rotate)]
        ref = x @ ws[0].t()
        flops = 2.0 * m * n_out * k_in
        print(f"\nM={m} N={n_out} K={k_in} x{count}  (weights rotated over {rotate} copies)")
        for name in names:
            try:
                fn = variants[name](x, ws)
                out = fn(0)
                err = rel_err(out, ref) if out.shape == ref.shape else float("nan")
                ms = bench(fn, rotate)
                tflops = flops / ms / 1e9 if name != "fp8_quantize" else float("nan")
                totals[name] += ms * count
                print(f"  {name:15} {ms:8.3f} ms  {tflops:7.1f} TFLOPS  rel_err {err:.2e}")
                rows.append([m, n_out, k_in, count, name, f"{ms:.4f}", f"{tflops:.2f}", f"{err:.3e}"])
            except Exception as e:
                print(f"  {name:15} n/a: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
                rows.append([m, n_out, k_in, count, name, "", "", f"n/a {type(e).__name__}"])
                totals[name] = float("nan")
            finally:
                set_blas("hipblaslt")
        del x, ws, ref
        torch.cuda.empty_cache()
    print("\nper model call (sum of ms * count over the listed shapes):")
    for name in names:
        print(f"  {name:15} {totals[name]:9.1f} ms")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["M", "N", "K", "count", "variant", "ms", "tflops", "rel_err"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
