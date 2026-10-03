"""Reproduce the Windows RDNA4 idle VRAM page-out (ROCm/TheRock#7221, llama.cpp discussion #23443).

Adrenalin 26.5.1 .. 26.8.1 page live HIP allocations out of VRAM after ~10 s idle; PRO 26.9.2
(32.0.32015.2008) is reported fixed. Run with ComfyUI stopped, using the production venv python:
    python idle_pageout_check.py --gib 8 --idle 20 --devices 0,1

Per device: allocate --gib GiB, touch it (warm), idle, touch again. A post-idle touch that is
seconds instead of milliseconds, together with a jump in process commit, means the allocation was paged out.
Keep --gib * devices well below free system RAM: on affected drivers the page-out lands in system RAM.
"""
import argparse
import ctypes
import os
import time

import torch


def commit_gib():
    if os.name != "nt":
        return float("nan")

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                           "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PrivateUsage / 2**30


def touch(t):
    torch.cuda.synchronize(t.device)
    s = time.perf_counter()
    t.add_(1)
    torch.cuda.synchronize(t.device)
    return (time.perf_counter() - s) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=8)
    ap.add_argument("--idle", type=float, default=20)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    print("torch", torch.__version__, "hip", torch.version.hip)
    bufs = []
    for d in [int(x) for x in a.devices.split(",") if x != ""]:
        if d >= torch.cuda.device_count():
            print(f"cuda:{d} not visible, skipped (check HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES)")
            continue
        t = torch.empty(int(a.gib * 2**30) // 2, dtype=torch.float16, device=f"cuda:{d}")
        t.zero_()
        bufs.append(t)
        print(f"cuda:{d} {torch.cuda.get_device_name(d)}: allocated {a.gib} GiB, warm touch {touch(t):.1f} ms")
    for r in range(a.rounds):
        c0 = commit_gib()
        print(f"round {r}: idle {a.idle:.0f} s (process private commit {c0:.2f} GiB)")
        time.sleep(a.idle)
        c1 = commit_gib()
        for t in bufs:
            ms = touch(t)
            verdict = "PAGED OUT" if ms > 250 else "resident"
            print(f"  {t.device}: touch after idle {ms:.1f} ms -> {verdict}")
        print(f"  private commit before touch {c1:.2f} GiB, after {commit_gib():.2f} GiB")


if __name__ == "__main__":
    main()
