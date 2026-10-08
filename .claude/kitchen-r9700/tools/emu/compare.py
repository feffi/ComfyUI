# ruff: noqa: T201
"""Compare two dump directories of test_norm: output bytes (*.bin) and the fp32 value of
every element before its narrowing store (*.f32, 0xffffffff where nothing was stored).
usage: compare.py <dir a> <dir b>"""
import glob
import os
import struct
import sys

a_dir, b_dir = sys.argv[1:3]
total = {"bin": [0, 0], "f32": [0, 0]}
for ext, width, fmt in (("bin", 2, "H"), ("f32", 4, "I")):
    for path in sorted(glob.glob(os.path.join(a_dir, f"*.{ext}"))):
        name = os.path.basename(path)
        x = open(path, "rb").read()
        y = open(os.path.join(b_dir, name), "rb").read()
        xs = struct.unpack(f"{len(x) // width}{fmt}", x[: len(x) // width * width])
        ys = struct.unpack(f"{len(y) // width}{fmt}", y[: len(y) // width * width])
        n = sum(1 for v in xs if not (ext == "f32" and v == 0xFFFFFFFF))
        d = sum(1 for u, v in zip(xs, ys) if u != v) + abs(len(xs) - len(ys))
        total[ext][0] += d
        total[ext][1] += n
        if d:
            print(f"DIFF {name}: {d}")
print(f"output elements: {total['bin'][0]} differ of {total['bin'][1]}; "
      f"fp32 pre-store values: {total['f32'][0]} differ of {total['f32'][1]}")
sys.exit(1 if total["bin"][0] or total["f32"][0] else 0)
