# ruff: noqa: T201
"""Kitchen's HIP fp8 quantize against torch's clamp-then-cast, on the CPU.

ComfyUI's fp8_linear quantizes with QuantizedTensor.from_float (c028067), which on
the R9700 runs kitchen's quantize_per_tensor_fp8 kernel. Before, it clamped to
+-448 and cast with torch. This compiles that kernel's encoder for the host and
compares the two over every bf16 and fp16 bit pattern and every positive float32 in
[2^-10, 2^9] (below, both give a signed zero; above, both saturate), with the
log2f result nudged by up to +-16 ulps.

    python quant_fp8.py <kitchen>/comfy_kitchen/backends/hip

Expected: 0 mismatches outside NaN. Kitchen keeps a NaN's sign; torch's CPU clamp
may not, so some bf16 NaNs differ in the sign bit only (NaN either way).
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
                bad = run(exe, out, name, ulps, 0, 65536) != torch_ref(x)
                print(f"ulps {ulps:+3d} {name}: {int((bad & ~x.isnan()).sum())} mismatches of 65536, "
                      f"{int((bad & x.isnan()).sum())} NaNs differing in the sign bit")
            bad = 0
            for first in range(FP32_FIRST, FP32_END, CHUNK):
                n = min(CHUNK, FP32_END - first)
                x = torch.arange(first, first + n, dtype=torch.int64).to(torch.int32).view(torch.float32)
                bad += int((run(exe, out, "fp32", ulps, first, n) != torch_ref(x)).sum())
            print(f"ulps {ulps:+3d} fp32: {bad} mismatches of {FP32_END - FP32_FIRST} in [2^-10, 2^9]")


if __name__ == "__main__":
    main()
