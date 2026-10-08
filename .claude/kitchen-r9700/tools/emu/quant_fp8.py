# ruff: noqa: T201
"""Compare kitchen's HIP per-tensor fp8 encoder (pack_fp8, compiled for the host) with
torch's clamp-then-cast over every bf16 and fp16 bit pattern and every positive float32
in [2^-10, 2^9], with log2f nudged by up to +-16 ulps. Below that range both give a
signed zero, above it both saturate.
usage: quant_fp8.py <kitchen>/comfy_kitchen/backends/hip
Expected: 0 mismatches outside NaN. Kitchen keeps a NaN's sign and torch's vectorized
CPU clamp sets it for bf16, so the 127 positive bf16 NaNs differ in the sign bit only.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ULPS = (-16, -4, -1, 0, 1, 4, 16)
FP32_FIRST, FP32_END, CHUNK = 0x3A800000, 0x44000000, 1 << 24


def run(exe, out, type_, ulps, first, count):
    subprocess.run([exe, type_, str(ulps), hex(first), str(count), out], check=True)
    return torch.from_numpy(np.fromfile(out, dtype=np.uint8))


def torch_ref(x):
    return torch.clamp(x, min=-448, max=448).to(torch.float8_e4m3fn).view(torch.uint8)


def main():
    hip = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        exe, out = os.path.join(tmp, "quant_fp8"), os.path.join(tmp, "out.bin")
        subprocess.run(["clang++-20", "-std=c++20", "-O2", "-ffp-contract=off", "-D__gfx1201__",
                        f"-I{HERE}/include", f"-I{hip}", f"-I{HERE}/../gen",
                        "-Wno-unknown-attributes", "-Wno-ignored-attributes",
                        f"{HERE}/quant_fp8.cpp", "-o", exe], check=True)
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        for ulps in ULPS:
            for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
                x = bits.view(dtype)
                got, want = run(exe, out, name, ulps, 0, 65536), torch_ref(x)
                bad, nan = got != want, x.isnan()
                print(f"ulps {ulps:+3d} {name}: {int((bad & ~nan).sum())} mismatches of 65536, "
                      f"{int((bad & nan & ((got ^ want) == 0x80)).sum())} NaNs differing in the sign bit only, "
                      f"{int((bad & nan & ((got ^ want) != 0x80)).sum())} otherwise")
            bad = 0
            for first in range(FP32_FIRST, FP32_END, CHUNK):
                n = min(CHUNK, FP32_END - first)
                x = torch.arange(first, first + n, dtype=torch.int64).to(torch.int32).view(torch.float32)
                bad += int((run(exe, out, "fp32", ulps, first, n) != torch_ref(x)).sum())
            print(f"ulps {ulps:+3d} fp32: {bad} mismatches of {FP32_END - FP32_FIRST} in [2^-10, 2^9]")


if __name__ == "__main__":
    main()
