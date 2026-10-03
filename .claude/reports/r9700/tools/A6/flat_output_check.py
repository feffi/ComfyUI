r"""Flag flat-colour (NaN/inf-derived) outputs, the gfx1201 failure mode of ComfyUI issue #16313.

    python flat_output_check.py output\ [--since-hours 24] [--threshold 3.5]

Images (png/jpg/webp): per-image luminance stddev. Videos (mp4/webm/mov, needs PyAV): stddev of
up to 8 sampled frames. stddev below --threshold (the issue reports 0.0-3.4 for broken runs,
45-70 for real renders) is reported as FLAT. Use it after repeated fixed-seed runs per model and
per driver (26.7.1 vs 26.8.1): any FLAT result, or differing hashes for the same seed, is a finding.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
from PIL import Image

IMG = (".png", ".jpg", ".jpeg", ".webp")
VID = (".mp4", ".webm", ".mov", ".mkv")


def luma_std(arr):
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(arr.std())


def video_stds(path, n=8):
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        total = s.frames or 0
        step = max(1, total // n) if total else 1
        out = []
        for i, f in enumerate(c.decode(s)):
            if i % step == 0:
                out.append(luma_std(f.to_ndarray(format="rgb24")))
            if len(out) >= n:
                break
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--since-hours", type=float, default=24)
    ap.add_argument("--threshold", type=float, default=3.5)
    a = ap.parse_args()
    cutoff = time.time() - a.since_hours * 3600
    flat = 0
    for root, _, files in os.walk(a.folder):
        for name in sorted(files):
            p = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            if os.path.getmtime(p) < cutoff or ext not in IMG + VID:
                continue
            try:
                if ext in IMG:
                    stds = [luma_std(np.asarray(Image.open(p)))]
                else:
                    stds = video_stds(p)
            except Exception as e:
                print(f"SKIP  {p}: {e}")
                continue
            digest = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
            worst = min(stds) if stds else float("nan")
            is_flat = worst < a.threshold
            flat += is_flat
            print(f"{'FLAT ' if is_flat else 'ok   '} min_std={worst:6.2f} sha={digest} {p}")
    print(f"\n{flat} flat output(s)")
    sys.exit(1 if flat else 0)


if __name__ == "__main__":
    main()
