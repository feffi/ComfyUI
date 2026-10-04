# ruff: noqa: T201
"""Exhaustive check of comfy-kitchen's per-tensor fp8 quantize (patch 0004).

Patch 0004 swaps the software e4m3 encoder for the gfx12 hardware convert. Its
equivalence could not be tested without a GPU, so run this before using the patch:

    python check_fp8_quantize.py --device 1 --save ref_0236.pt     # installed kitchen
    set PYTHONPATH=C:\\kt\\kitchen-patched
    python check_fp8_quantize.py --device 1 --ref ref_0236.pt      # patched kitchen

Inputs are every bf16 and every fp16 bit pattern (NaN, inf, subnormals, both zeros)
at three scales, plus float32 values on and next to every e4m3 rounding midpoint.
The patched build must match the installed one byte for byte; the eager reference
(clamp to +-448, then torch's cast) is printed for information.
"""

import argparse

import torch

FP8 = torch.float8_e4m3fn


def inputs(device):
    out = {}
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device=device).to(torch.int16)
    out["bf16 all"] = bits.view(torch.bfloat16)
    out["fp16 all"] = bits.view(torch.float16)
    # Every e4m3 magnitude, the midpoints between neighbours and one float32 ulp
    # either side of each midpoint, both signs, plus values past 448.
    codes = torch.arange(0, 127, dtype=torch.uint8, device=device).view(FP8).float()
    mids = (codes[:-1] + codes[1:]) / 2
    near = torch.cat([mids, torch.nextafter(mids, mids + 1), torch.nextafter(mids, mids - 1), codes,
                      torch.tensor([448.0, 449.0, 464.0, 465.0, 1e4, 3e38], device=device)])
    out["fp32 midpoints"] = torch.cat([near, -near])
    return out


def run(args):
    from comfy_kitchen.backends import hip as ckhip

    device = torch.device("cuda", args.device)
    results = {}
    for name, x in inputs(device).items():
        for scale_value in (1.0, 0.0625, 3.0):
            scale = torch.tensor(scale_value, device=device)
            q = ckhip.quantize_per_tensor_fp8(x, scale, FP8).view(torch.uint8).cpu()
            ref = (x.float() / scale).clamp(-448, 448).to(FP8).view(torch.uint8).cpu()
            key = f"{name} scale={scale_value}"
            results[key] = q
            nan = torch.isnan(x.float().cpu())
            eager = (q[~nan] != ref[~nan]).sum().item()
            print(f"{key:<32} n={x.numel():>6}  differs from eager cast (non-NaN): {eager}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--save")
    p.add_argument("--ref")
    args = p.parse_args()
    with torch.inference_mode():
        results = run(args)
    if args.save:
        torch.save(results, args.save)
    if args.ref:
        ref = torch.load(args.ref)
        bad = 0
        for key, q in results.items():
            diff = (q != ref[key]).nonzero().flatten()
            bad += diff.numel()
            if diff.numel():
                i = diff[0].item()
                print(f"MISMATCH {key}: {diff.numel()} bytes, first at {i}: patched {q[i].item():#04x}, "
                      f"installed {ref[key][i].item():#04x}")
        print("identical to the installed build" if bad == 0 else f"{bad} bytes differ: do not use patch 0004")


if __name__ == "__main__":
    main()
