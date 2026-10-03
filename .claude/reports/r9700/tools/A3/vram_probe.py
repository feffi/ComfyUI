"""A3 memory probe for 2x R9700 on Windows ROCm. Run in .venv-rocm-100 with ComfyUI stopped.

  python vram_probe.py                      # devices, P2P, copy bandwidths, cross-process free-memory visibility
  python vram_probe.py --pin-probe 60       # also pin host RAM in 2 GiB steps up to 60% of RAM (can make the desktop sluggish)
  python vram_probe.py --aimdo-log C:\\path\\ComfyUI   # start ComfyUI once with DEBUG logs and print the comfy-aimdo WDDM/device lines

Answers: does hipMemGetInfo (ComfyUI get_free_memory) see other processes' VRAM on Windows;
is GPU0<->GPU1 peer access available and how fast are cross-GPU and host copies; how much host
memory can actually be pinned versus ComfyUI's Windows cap of 40% of RAM (model_management.py:1640);
does comfy-aimdo match a distinct WDDM adapter (LUID) to each of the two identical cards.
"""
import argparse
import subprocess
import sys
import time

import psutil
import torch

GiB = 1024 ** 3


def bw(fn, nbytes, reps=5):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return nbytes / ts[len(ts) // 2] / 1e9


def child_alloc(dev, gib):
    x = torch.empty(int(gib * GiB), dtype=torch.uint8, device=f"cuda:{dev}")
    x.fill_(1)
    torch.cuda.synchronize()
    print("ALLOCATED", flush=True)
    sys.stdin.readline()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pin-probe", type=float, default=0, help="pin up to this percent of RAM")
    p.add_argument("--aimdo-log", default=None, help="ComfyUI dir; runs main.py --quick-test-for-ci --verbose DEBUG")
    p.add_argument("--child", nargs=2, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        return child_alloc(int(a.child[0]), float(a.child[1]))

    n = torch.cuda.device_count()
    ram = psutil.virtual_memory().total
    print(f"torch {torch.__version__} hip {torch.version.hip} devices {n}")
    print(f"RAM {ram / GiB:.1f} GiB; ComfyUI Windows pinned cap (40%) = {0.4 * ram / GiB:.1f} GiB")
    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"cuda:{i} {pr.name} {pr.gcnArchName} total {total / GiB:.2f} GiB free {free / GiB:.2f} GiB  pci {getattr(pr, 'pci_bus_id', '?')}")

    if n >= 2:
        print(f"peer access 0->1 {torch.cuda.can_device_access_peer(0, 1)}  1->0 {torch.cuda.can_device_access_peer(1, 0)}")
        a0 = torch.empty(GiB, dtype=torch.uint8, device="cuda:0")
        a1 = torch.empty(GiB, dtype=torch.uint8, device="cuda:1")
        print(f"copy cuda:0->cuda:1 {bw(lambda: a1.copy_(a0, non_blocking=True), GiB):.1f} GB/s, cuda:1->cuda:0 {bw(lambda: a0.copy_(a1, non_blocking=True), GiB):.1f} GB/s")
        del a0, a1
    host = torch.empty(GiB, dtype=torch.uint8)
    pinned = torch.empty(GiB, dtype=torch.uint8, pin_memory=True)
    for i in range(n):
        d = torch.empty(GiB, dtype=torch.uint8, device=f"cuda:{i}")
        print(f"H2D cuda:{i} pageable {bw(lambda: d.copy_(host), GiB):.1f} GB/s, pinned {bw(lambda: d.copy_(pinned, non_blocking=True), GiB):.1f} GB/s, "
              f"D2H pinned {bw(lambda: pinned.copy_(d, non_blocking=True), GiB):.1f} GB/s")
        del d
    del host, pinned
    torch.cuda.empty_cache()

    for i in range(n):
        before = torch.cuda.mem_get_info(i)[0]
        child = subprocess.Popen([sys.executable, __file__, "--child", str(i), "4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        child.stdout.readline()
        after = torch.cuda.mem_get_info(i)[0]
        child.stdin.write("\n")
        child.stdin.flush()
        child.wait()
        print(f"cuda:{i}: another process allocated 4.00 GiB -> this process's hipMemGetInfo free dropped by {(before - after) / GiB:.2f} GiB "
              f"({'visible' if before - after > 3.5 * GiB else 'NOT visible: ComfyUI get_free_memory ignores other processes here'})")

    if a.pin_probe > 0:
        limit = ram * a.pin_probe / 100
        chunks, total = [], 0
        try:
            while total + 2 * GiB <= limit:
                t = time.perf_counter()
                chunks.append(torch.empty(2 * GiB, dtype=torch.uint8, pin_memory=True))
                total += 2 * GiB
                print(f"  pinned {total / GiB:.0f} GiB ({100 * total / ram:.0f}% of RAM) in {time.perf_counter() - t:.2f} s", flush=True)
        except RuntimeError as e:
            print(f"  pinning failed at {total / GiB:.0f} GiB ({100 * total / ram:.0f}% of RAM): {e}")
        del chunks

    if a.aimdo_log:
        out = subprocess.run([sys.executable, "main.py", "--quick-test-for-ci", "--verbose", "DEBUG", "--port", "8199", "--database-url", "sqlite:///:memory:"],
                             cwd=a.aimdo_log, capture_output=True, text=True, timeout=900)
        for line in (out.stdout + out.stderr).splitlines():
            low = line.lower()
            if "aimdo" in low or "wddm" in low or "dynamicvram" in low or "pinned memory" in low or "async weight offloading" in low:
                print("  ", line[:240])


if __name__ == "__main__":
    main()
