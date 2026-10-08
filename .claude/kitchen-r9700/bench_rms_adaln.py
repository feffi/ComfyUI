# ruff: noqa: T201
"""rms_adaln / adaln microbenchmark for comfy-kitchen's HIP kernels (patch 0005).

Shapes: Krea 2 at 1280x2048 as the model calls it (rms_adaln on the (1, T, 6144) bf16
residual stream with one (1, 1, 6144) scale and shift, twice per block), then partial,
odd, LayerNorm and fallback shapes.

    python bench_rms_adaln.py --device 1 --save-ref C:\\kt\\adaln_0236.pt   # installed kitchen
    set PYTHONPATH=C:\\kt\\kitchen-patched
    python bench_rms_adaln.py --device 1 --ref C:\\kt\\adaln_0236.pt        # patched kitchen

GB/s counts what a launch has to move once (x read, out written, the distinct scale and
shift rows read) against the ~600 GB/s practical peak. "fast" mirrors the launcher's
test for 0005's one-wave-per-row path; "-" runs adaln_kernel. --ref reports bit
identity against the reference build; "err64" is the max |error| of the first 64 rows
against an fp64 evaluation, for both builds.
"""

import argparse
import json
import os

import torch

from bench_fp8_gemm import digest, time_ms

PEAK_GBS = 600.0
BF16, FP16, FP32 = torch.bfloat16, torch.float16, torch.float32


def cases():
    # (name, batch, rows, D, x dtype, scale/shift dtype, per-token modulation, LayerNorm)
    out = [(f"krea2 T={t}", 1, t, 6144, BF16, BF16, False, False) for t in (10264, 10347)]
    out += [
        ("krea2 T=77", 1, 77, 6144, BF16, BF16, False, False),
        ("krea2 T=1", 1, 1, 6144, BF16, BF16, False, False),
        ("B=2 T=4103 D=3072 per-token", 2, 4103, 3072, BF16, BF16, True, False),
        ("B=2 T=4103 D=3072 LayerNorm", 2, 4103, 3072, BF16, BF16, False, True),
        ("T=1031 D=1536 fp16", 1, 1031, 1536, FP16, FP16, False, False),
        ("T=1031 D=256", 1, 1031, 256, BF16, BF16, True, False),
        ("fallback: D=1000", 1, 1031, 1000, BF16, BF16, False, False),
        ("fallback: fp32 scale/shift", 1, 4103, 6144, BF16, FP32, False, False),
        ("fallback: fp32 x", 1, 1031, 6144, FP32, FP32, False, False),
    ]
    return out


def fast_path(x, scale, shift, d):
    """The launcher's test for the one-wave-per-row path."""
    return (x.dtype in (FP16, BF16) and scale.dtype == x.dtype and shift.dtype == x.dtype
            and d % 256 == 0 and all(t.data_ptr() % 16 == 0 for t in (x, scale, shift)))


def ref64(x, scale, shift, eps, layernorm):
    x = x.double()
    if layernorm:
        x = x - x.mean(-1, keepdim=True)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + scale.double()) + shift.double()


def run(args):
    import comfy_kitchen
    from comfy_kitchen.backends import hip as ck

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    info = {"torch": torch.__version__, "hip": torch.version.hip,
            "kitchen": getattr(comfy_kitchen, "__version__", "?"),
            "kitchen_path": os.path.dirname(comfy_kitchen.__file__),
            "gpu": f"{props.name} {getattr(props, 'gcnArchName', '')}"}
    ref = torch.load(args.ref) if args.ref else None
    save, rows = {}, []
    eps = 1e-5
    for i, (name, b, t, d, xd, sd, per_token, layernorm) in enumerate(cases()):
        if args.only and args.only not in name:
            continue
        g = torch.Generator(device=device).manual_seed(3000 + i)
        x = torch.randn(b, t, d, device=device, generator=g).mul_(3).add_(0.5).to(xd)
        # Krea 2 slices scale and shift out of one modulation projection.
        mod_rows = t if per_token else 1
        mod = torch.randn(b, mod_rows, 6 * d, device=device, generator=g).mul_(0.3).to(sd)
        scale, shift = mod[..., d:2 * d], mod[..., 3 * d:4 * d]
        fn = ck.adaln if layernorm else ck.rms_adaln

        want = ref64(x[:, :64], scale[:, :64], shift[:, :64], eps, layernorm)
        out = fn(x, scale, shift, eps)
        err = (out[:, :64].double() - want).abs().max().item()
        row_digest = digest(out)
        sample = out.flatten()[::97].float().cpu()

        moved = 2 * x.numel() * x.element_size() + 2 * b * mod_rows * d * mod.element_size()
        ms = time_ms(lambda x=x, scale=scale, shift=shift, fn=fn: fn(x, scale, shift, eps), args.warmup, args.iters)
        row = {"case": name, "ms": ms[0], "minmax": list(ms[1:]), "gbs": moved / ms[0] / 1e6,
               "fast": "y" if fast_path(x, scale, shift, d) else "-", "err64": err}
        save[name] = {"sha256": row_digest, "sample": sample}
        if ref is not None and name in ref:
            row["identical"] = ref[name]["sha256"] == row_digest
            row["maxdiff_vs_ref"] = (ref[name]["sample"] - sample).abs().max().item()
        rows.append(row)
        x = mod = out = None
        torch.cuda.empty_cache()
    if args.save_ref:
        torch.save(save, args.save_ref)
    return info, rows


def print_table(info, rows):
    print(json.dumps(info))
    print(f"{'case':<32}{'ms':>9}{'GB/s':>8}{'%peak':>7}{'fast':>6}{'err64':>10}{'identical':>11}{'maxdiff':>9}")
    for r in rows:
        print(f"{r['case']:<32}{r['ms']:>9.3f}{r['gbs']:>8.0f}{100 * r['gbs'] / PEAK_GBS:>7.0f}{r['fast']:>6}"
              f"{r['err64']:>10.3g}{str(r.get('identical', '')):>11}{str(r.get('maxdiff_vs_ref', '')):>9}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--only", help="substring of the case name, e.g. krea2")
    p.add_argument("--save-ref", help="write output hashes and samples of this kitchen build")
    p.add_argument("--ref", help="compare against a file written by --save-ref")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    with torch.inference_mode():
        info, rows = run(args)
    if args.json:
        print(json.dumps({"info": info, "rows": rows}))
    else:
        print_table(info, rows)


if __name__ == "__main__":
    main()
