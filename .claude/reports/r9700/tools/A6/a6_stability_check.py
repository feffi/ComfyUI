r"""A6 stability check for ComfyUI on 2x R9700 (gfx1201), Windows, ROCm wheels. Read-only.

Run with the production venv python from the ComfyUI folder:
    .venv-rocm-100\Scripts\python.exe a6_stability_check.py --comfy . [--log comfyui.log | --url http://127.0.0.1:8188] [--free-mem-test 16]

Prints: torch/HIP versions, visible devices (order, gcnArchName, PCI bus), visibility env vars,
the AOTriton flash-attention probe ComfyUI runs at import (with torch's reason when it fails,
with and without TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL), installed vs required comfy packages,
onnxruntime providers, and what ComfyUI resolved at startup (from its log).
--free-mem-test N: a child process holds N GiB on GPU 1 while this process reads mem_get_info(1),
to show whether free memory on Windows accounts for other processes.
Nothing is changed, installed or sent over the network (except the optional --url to your local ComfyUI).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.request

PROBE = textwrap.dedent(r"""
    import os, warnings, json, torch
    out = {"env": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")}
    dev = torch.device("cuda", 0)
    out["flash_available"] = torch.backends.cuda.is_flash_attention_available()
    q = torch.zeros((1, 1, 8, 64), dtype=torch.float16, device=dev)
    params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, False, False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out["can_use_flash"] = torch.backends.cuda.can_use_flash_attention(params, True)
        out["can_use_efficient"] = torch.backends.cuda.can_use_efficient_attention(params, True)
    out["reasons"] = sorted({str(x.message) for x in w})
    if out["can_use_flash"]:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            torch.zeros(1, device=dev).add_(1).item()
            out["launch"] = "ok"
        except Exception as e:
            out["launch"] = repr(e)
    print("PROBE_JSON " + json.dumps(out))
""")

HOLD = textwrap.dedent(r"""
    import sys, time, torch
    gib = float(sys.argv[1])
    t = torch.empty(int(gib * (1 << 30)), dtype=torch.uint8, device="cuda:1")
    t.fill_(1); torch.cuda.synchronize()
    print("HELD", flush=True)
    time.sleep(float(sys.argv[2]))
""")

LOG_PATTERNS = [
    ("attention", r"Using (pytorch|sub quadratic|split|sage|xformers|Flash|Comfy Kitchen) .*attention|Using .*optimization for attention"),
    ("amd_arch", r"AMD arch: .*"),
    ("rocm_version", r"ROCm version: .*"),
    ("flash_probe_failed", r"Could not run flash attention.*"),
    ("miopen", r"cudnn.enabled = False.*"),
    ("fp16_accum", r"Enabled fp16 accumulation.*"),
    ("single_gpu_forced", r"forcing single GPU mode.*"),
    ("cuda_device", r"Set cuda devices? to.*"),
    ("device", r"Device: .*"),
    ("vram_state", r"Set vram state to: .*"),
    ("dynamic_vram", r".*DynamicVRAM.*|.*aimdo.*"),
    ("kitchen_backend", r"Found comfy_kitchen backend .*"),
    ("pinned", r"Enabled pinned memory.*"),
    ("async_offload", r"Using async weight offloading.*"),
    ("outdated_pkgs", r"Installed .* is lower than the recommended version.*"),
    ("sol_attn", r"\[sol_attn\].*"),
    ("nan_or_oom", r".*(out of memory|OOM|NaN|nan).*"),
]


def section(title):
    print("\n== " + title)


def run_probe(env_value):
    env = dict(os.environ)
    if env_value is None:
        env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)
    else:
        env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = env_value
    p = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=300)
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    return {"error": (p.stderr or p.stdout)[-800:]}


def required_versions(comfy_dir):
    req = {}
    path = os.path.join(comfy_dir, "requirements.txt")
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            m = re.match(r"^(comfy[a-z0-9_\-]*)==([^\s;#]+)", line.strip())
            if m:
                req[m.group(1)] = m.group(2)
    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=".", help="ComfyUI folder (for requirements.txt)")
    ap.add_argument("--log", help="saved ComfyUI console log")
    ap.add_argument("--url", help="local ComfyUI, e.g. http://127.0.0.1:8188 (reads /internal/logs)")
    ap.add_argument("--free-mem-test", type=float, default=0, metavar="GIB")
    a = ap.parse_args()

    import torch
    section("versions")
    print("python", sys.version.split()[0], "| torch", torch.__version__, "| hip", torch.version.hip)
    from importlib.metadata import version, PackageNotFoundError
    req = required_versions(a.comfy)
    for name in sorted(set(req) | {"comfy-kitchen", "comfy-aimdo", "comfyui-frontend-package", "onnxruntime", "onnxruntime-directml", "insightface", "triton-windows"}):
        try:
            inst = version(name)
        except PackageNotFoundError:
            inst = "-"
        flag = "  <-- differs from requirements.txt" if name in req and inst != req[name] else ""
        print(f"  {name:28s} installed {inst:14s} required {req.get(name, '-')}{flag}")

    section("device visibility")
    for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "COMFYUI_ENABLE_MIOPEN", "PYTORCH_HIP_ALLOC_CONF"):
        print(f"  {k}={os.environ.get(k)}")
    n = torch.cuda.device_count()
    print("  torch.cuda.device_count() =", n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        bus = "{:04x}:{:02x}:{:02x}".format(getattr(p, "pci_domain_id", 0), getattr(p, "pci_bus_id", 0), getattr(p, "pci_device_id", 0))
        free, total = torch.cuda.mem_get_info(i)
        print(f"  cuda:{i} {p.name} arch={p.gcnArchName} pci={bus} uuid={getattr(p, 'uuid', '?')} free={free / 2**30:.1f}/{total / 2**30:.1f} GiB")
    print("  Note: on Windows HIP the visible-devices list filters but does not reorder (cuda:0 is always the lowest physical ordinal).")

    section("AOTriton flash-attention probe (what comfy/model_management.py aotriton_supported() decides)")
    for val in (None, "1"):
        print(f"  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL={val}: {json.dumps(run_probe(val))}")

    section("onnxruntime (insightface) providers, separate process")
    p = subprocess.run([sys.executable, "-c", "import onnxruntime as o; print(o.__version__, o.get_available_providers())"], capture_output=True, text=True, timeout=120)
    print("  " + (p.stdout.strip() or p.stderr.strip()[-300:]))

    if a.free_mem_test > 0 and n >= 2:
        section(f"free memory on cuda:1 while another process holds {a.free_mem_test} GiB")
        before = torch.cuda.mem_get_info(1)[0]
        child = subprocess.Popen([sys.executable, "-c", HOLD, str(a.free_mem_test), "20"], stdout=subprocess.PIPE, text=True)
        child.stdout.readline()
        time.sleep(1)
        during = torch.cuda.mem_get_info(1)[0]
        child.wait()
        print(f"  free before {before / 2**30:.1f} GiB, while held {during / 2**30:.1f} GiB "
              f"-> {'other processes ARE counted' if before - during > a.free_mem_test * 0.8 * 2**30 else 'other processes are NOT counted'}")

    text = ""
    if a.log and os.path.isfile(a.log):
        text = open(a.log, encoding="utf-8", errors="replace").read()
    elif a.url:
        text = json.loads(urllib.request.urlopen(a.url.rstrip("/") + "/internal/logs", timeout=10).read().decode())
    if text:
        section("ComfyUI startup resolution (from log)")
        for key, pat in LOG_PATTERNS:
            hits = sorted({m.group(0).strip()[:160] for m in re.finditer(pat, text)})
            print(f"  {key:18s} " + (" | ".join(hits[:4]) if hits else "(not found)"))


if __name__ == "__main__":
    main()
