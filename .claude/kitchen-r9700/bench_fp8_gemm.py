# ruff: noqa: T201
"""fp8 GEMM microbenchmark: comfy-kitchen's HIP WMMA kernel vs hipBLASLt (torch._scaled_mm).

Plain torch, under torch.inference_mode(), one GPU. Shapes are the Krea 2 block at
1280x2048 (M = image + text tokens), Qwen-Image 2.1 at 1024x1024 and the small-M
text/modulation GEMMs.

Compare two kitchen builds in two runs of the same interpreter, since one process can
import only one comfy_kitchen:

    python bench_fp8_gemm.py --device 1 --save-ref ref_0236.pt          # installed kitchen
    set PYTHONPATH=C:\\kt\\kitchen-patched
    python bench_fp8_gemm.py --device 1 --ref ref_0236.pt               # patched kitchen

The reference stores a SHA-256 of every kitchen output plus every 64th row, so the
second run reports bit identity and max |diff| against the installed kernel.
--sweep-tiles times each fp8 tile of the patched build separately (it sets
COMFY_KITCHEN_HIP_FP8_TILE per subprocess), to check what the tuner picks.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

import torch

FP8 = torch.float8_e4m3fn
NUM_TILES = 11  # kFp8Tiles in the patched gemm_fp8.hip


def shapes():
    # (group, M, K, N, uses per Krea 2 block)
    out = []
    for m in (10264, 10347):
        out += [
            (f"krea2 M={m}", m, 6144, 6144, 3),    # wq, gate, wo
            (f"krea2 M={m}", m, 6144, 1536, 2),    # wk, wv
            (f"krea2 M={m}", m, 6144, 16384, 2),   # mlp gate, up
            (f"krea2 M={m}", m, 16384, 6144, 1),   # mlp down
        ]
    out += [
        ("qwen21", 4096, 4096, 24576, 1),
        ("qwen21", 4096, 12288, 4096, 1),
        ("qwen21", 4096, 4096, 4096, 1),
    ]
    out += [("small-M", m, 4096, 16384, 1) for m in (1, 2, 8, 16, 77, 335)]
    out += [("small-M", 2, 6144, 36864, 1)]
    return out


def operands(m, k, n, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(m, k, device=device, generator=g).mul_(4).clamp_(-448, 448).to(FP8)
    w = torch.randn(n, k, device=device, generator=g).mul_(4).clamp_(-448, 448).to(FP8)
    sa = torch.tensor(0.01, device=device)
    sb = torch.tensor(0.02, device=device)
    return a, w, sa, sb


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times), min(times), max(times)


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def run(args):
    import comfy_kitchen
    from comfy_kitchen.backends import hip as ckhip

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    info = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "kitchen": getattr(comfy_kitchen, "__version__", "?"),
        "kitchen_path": os.path.dirname(comfy_kitchen.__file__),
        "gpu": f"{props.name} {getattr(props, 'gcnArchName', '')}",
        "tile": os.environ.get("COMFY_KITCHEN_HIP_FP8_TILE", "tuned"),
    }
    ref = torch.load(args.ref) if args.ref else None
    save = {}
    rows = []
    for i, (group, m, k, n, uses) in enumerate(shapes()):
        if args.only and args.only not in group:
            continue
        a, w, sa, sb = operands(m, k, n, device, seed=1000 + i)
        def kitchen(a=a, w=w, sa=sa, sb=sb):
            return ckhip.scaled_mm_fp8(a, w.t(), sa, sb, None, torch.bfloat16)

        iters = args.iters if m >= 1024 else args.iters * 5
        k_ms = time_ms(kitchen, args.warmup, iters)
        out = kitchen()
        sample = out[::64].float().cpu()
        key = f"{m}x{k}x{n}"
        row = {"group": group, "M": m, "K": k, "N": n, "uses": uses, "kitchen_ms": k_ms[0],
               "kitchen_minmax": k_ms[1:], "tflops": 2 * m * k * n / k_ms[0] / 1e9}
        if not args.no_hipblaslt:
            def blas(a=a, w=w, sa=sa, sb=sb):
                return torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

            row["hipblaslt_ms"] = time_ms(blas, args.warmup, iters)[0]
            row["vs_hipblaslt_maxdiff"] = (blas()[::64].float().cpu() - sample).abs().max().item()
        if args.save_ref or ref is not None:
            row_digest = digest(out)
            save[key] = {"sha256": row_digest, "sample": sample}
            if ref is not None and key in ref:
                row["identical"] = ref[key]["sha256"] == row_digest
                row["maxdiff_vs_ref"] = (ref[key]["sample"] - sample).abs().max().item()
        rows.append(row)
        a = w = out = None
        torch.cuda.empty_cache()
    if args.save_ref:
        torch.save(save, args.save_ref)
    return info, rows


def print_table(info, rows):
    print(json.dumps(info))
    hdr = f"{'group':<16}{'M':>7}{'K':>7}{'N':>7}{'kitchen ms':>12}{'TFLOP/s':>9}{'hipBLASLt':>11}{'k/blas':>8}{'identical':>11}{'maxdiff':>10}"
    print(hdr)
    for r in rows:
        blas = r.get("hipblaslt_ms")
        blas_s = "" if blas is None else f"{blas:.3f}"
        ratio = "" if blas is None else f"{r['kitchen_ms'] / blas:.2f}"
        print(f"{r['group']:<16}{r['M']:>7}{r['K']:>7}{r['N']:>7}{r['kitchen_ms']:>12.3f}{r['tflops']:>9.1f}"
              f"{blas_s:>11}{ratio:>8}{str(r.get('identical', '')):>11}{str(r.get('maxdiff_vs_ref', '')):>10}")
    for m in sorted({r["M"] for r in rows if r["group"].startswith("krea2")}):
        block = [r for r in rows if r["group"] == f"krea2 M={m}"]
        k_total = sum(r["kitchen_ms"] * r["uses"] for r in block)
        line = f"Krea 2 block GEMMs, M={m}: kitchen {k_total:.2f} ms"
        if all("hipblaslt_ms" in r for r in block):
            line += f", hipBLASLt {sum(r['hipblaslt_ms'] * r['uses'] for r in block):.2f} ms"
        print(line)


def sweep(args):
    results = {}
    for tile in [None] + list(range(NUM_TILES)):
        env = dict(os.environ)
        env.pop("COMFY_KITCHEN_HIP_FP8_TILE", None)
        if tile is not None:
            env["COMFY_KITCHEN_HIP_FP8_TILE"] = str(tile)
        cmd = [sys.executable, __file__, "--device", str(args.device), "--json", "--no-hipblaslt",
               "--iters", str(args.iters), "--warmup", str(args.warmup)]
        if args.only:
            cmd += ["--only", args.only]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
        results["tuned" if tile is None else str(tile)] = json.loads(out.strip().splitlines()[-1])["rows"]
    cols = list(results)
    print(f"{'M':>7}{'K':>7}{'N':>7}" + "".join(f"{c:>9}" for c in cols) + "   best")
    for i, r in enumerate(results["tuned"]):
        ms = [results[c][i]["kitchen_ms"] for c in cols]
        best = min(range(1, len(cols)), key=lambda j: ms[j])
        print(f"{r['M']:>7}{r['K']:>7}{r['N']:>7}" + "".join(f"{v:>9.3f}" for v in ms) + f"   tile {cols[best]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--only", help="substring of the shape group, e.g. krea2 or small-M")
    p.add_argument("--no-hipblaslt", action="store_true")
    p.add_argument("--save-ref", help="write output hashes and samples of this kitchen build")
    p.add_argument("--ref", help="compare against a file written by --save-ref")
    p.add_argument("--sweep-tiles", action="store_true", help="time every forced tile (patched build)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    if args.sweep_tiles:
        sweep(args)
        return
    with torch.inference_mode():
        info, rows = run(args)
    if args.json:
        for r in rows:
            r["kitchen_minmax"] = list(r["kitchen_minmax"])
        print(json.dumps({"info": info, "rows": rows}))
    else:
        print_table(info, rows)


if __name__ == "__main__":
    main()
