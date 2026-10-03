"""Run vae_bench.py for each VAE under each MIOpen configuration, in separate processes
(MIOpen reads its environment once), and print one summary table.

  .venv-rocm-100\\Scripts\\python run_vae_matrix.py --comfy C:\\ComfyUI --gpu 1 ^
      --krea2-vae C:\\ComfyUI\\models\\vae\\wan_2.1_vae.safetensors ^
      --qwen21-vae C:\\ComfyUI\\models\\vae\\qwen_image_2.1_vae.safetensors ^
      --minimax-vae C:\\ComfyUI\\models\\vae\\minimax_h3_vae.safetensors ^
      --res 1024 --out C:\\temp\\a4_vae.jsonl

Configurations: off (ComfyUI default on RDNA3+), miopen (COMFYUI_ENABLE_MIOPEN=1, default find mode),
miopen_fast (MIOPEN_FIND_MODE=FAST), miopen_immediate (torch.backends.miopen.immediate=True),
miopen_cold/miopen_warm (fresh MIOPEN_USER_DB_PATH, run twice: Find cost and whether the user
find-db persists between processes). --miopen-log adds MIOPEN_LOG_LEVEL=5 to the miopen run and
counts naive-solver picks and zero-workspace warnings (ROCm/TheRock#3077).
Read-only: the TDR registry values are only printed. MIOpen Find on Windows can trip the 2 s TDR on
large shapes; run --res 512 first.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--comfy", required=True)
p.add_argument("--gpu", default="1")
p.add_argument("--krea2-vae")
p.add_argument("--qwen21-vae")
p.add_argument("--minimax-vae")
p.add_argument("--res", default="1024")
p.add_argument("--minimax-frames", default="1,85")
p.add_argument("--runs", default="3")
p.add_argument("--configs", default="off,miopen,miopen_fast,miopen_immediate,miopen_cold,miopen_warm")
p.add_argument("--miopen-log", action="store_true")
p.add_argument("--comfy-args", default="")
p.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "a4_vae.jsonl"))
a = p.parse_args()
here = os.path.dirname(os.path.abspath(__file__))


def tdr():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers")
        r = {}
        for name in ("TdrDelay", "TdrDdiDelay", "TdrLevel"):
            try:
                r[name] = winreg.QueryValueEx(k, name)[0]
            except OSError:
                r[name] = "unset (default: TdrDelay 2 s, TdrDdiDelay 5 s)"
        return r
    except Exception as e:
        return f"n/a ({e})"


def miopen_dirs():
    home = os.path.expanduser("~")
    r = {}
    for d in (os.environ.get("MIOPEN_USER_DB_PATH") or os.path.join(home, ".config", "miopen"),
              os.environ.get("MIOPEN_CUSTOM_CACHE_DIR") or os.path.join(home, ".cache", "miopen")):
        files = []
        for root, _, fs in os.walk(d) if os.path.isdir(d) else []:
            files += [(os.path.relpath(os.path.join(root, f), d), os.path.getsize(os.path.join(root, f))) for f in fs]
        r[d] = files
    # system find/perf db and kernel packages shipped in the venv for this arch
    sysdb = []
    for root, _, fs in os.walk(sys.prefix):
        if "miopen" in root.lower():
            sysdb += [os.path.join(root, f) for f in fs if "gfx120" in f.lower() or f.endswith((".kdb", ".fdb.txt", ".db"))]
    r["system_db_in_venv"] = sysdb[:40]
    return r


print("TDR:", tdr())
print("MIOpen dirs before:", json.dumps(miopen_dirs(), indent=1))

cold_db = tempfile.mkdtemp(prefix="miopen_a4_")
configs = {
    "off": ({}, []),
    "miopen": ({"COMFYUI_ENABLE_MIOPEN": "1"}, []),
    "miopen_fast": ({"COMFYUI_ENABLE_MIOPEN": "1", "MIOPEN_FIND_MODE": "FAST"}, []),
    "miopen_immediate": ({"COMFYUI_ENABLE_MIOPEN": "1"}, ["--miopen-immediate"]),
    "miopen_cold": ({"COMFYUI_ENABLE_MIOPEN": "1", "MIOPEN_USER_DB_PATH": cold_db, "MIOPEN_CUSTOM_CACHE_DIR": cold_db}, []),
    "miopen_warm": ({"COMFYUI_ENABLE_MIOPEN": "1", "MIOPEN_USER_DB_PATH": cold_db, "MIOPEN_CUSTOM_CACHE_DIR": cold_db}, []),
}
jobs = []
if a.krea2_vae:
    jobs.append(("krea2", a.krea2_vae, ["--res", a.res, "--op", "both"], "conv2d1f"))
if a.qwen21_vae:
    jobs.append(("qwen21", a.qwen21_vae, ["--res", a.res, "--op", "both"], "conv2d1f"))
if a.minimax_vae:
    for fr in a.minimax_frames.split(","):
        jobs.append(("minimax", a.minimax_vae, ["--res", a.res, "--frames", fr, "--op", "both"], "minimax1f"))

rows = []
for cname in a.configs.split(","):
    env_add, extra = configs[cname]
    for model, path, args, variant in jobs:
        env = {k: v for k, v in os.environ.items() if k not in ("COMFYUI_ENABLE_MIOPEN", "MIOPEN_FIND_MODE")}
        env.update(env_add)
        log = None
        if a.miopen_log and cname == "miopen":
            env.update(MIOPEN_ENABLE_LOGGING="1", MIOPEN_LOG_LEVEL="5")
            log = os.path.join(tempfile.gettempdir(), f"a4_miopen_{model}.log")
        variants = "baseline," + variant if cname == "off" else ("baseline,minimax1f" if model == "minimax" else "baseline")
        cmd = [sys.executable, os.path.join(here, "vae_bench.py"), "--comfy", a.comfy, "--gpu", a.gpu, "--model", model,
               "--vae", path, "--variants", variants, "--runs", a.runs, "--tag", cname, "--out", a.out, "--comfy-args=" + a.comfy_args] + args + extra
        print(f"\n### {cname} {model} {' '.join(args)}", flush=True)
        with open(log, "w") if log else open(os.devnull, "w") as errf:
            r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=errf if log else subprocess.STDOUT, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("{\"model\""):
                rows.append(json.loads(line) | {"config": cname})
                print(line)
        if r.returncode:
            print(f"!! exit code {r.returncode} (a TDR shows up as a device lost / hipErrorLaunchFailure)")
            print("\n".join(r.stdout.splitlines()[-15:]))
        if log:
            text = open(log, errors="replace").read()
            naive = len(re.findall(r"ConvDirectNaive", text))
            zero_ws = len(re.findall(r"provided ptr: 0|workspace_sz = 0", text))
            modes = sorted(set(re.findall(r"findMode: ?\w+", text)))[:3]
            print(f"MIOpen log {log}: naive-solver picks {naive}, zero-workspace warnings {zero_ws}, find modes {modes}")

print("\nMIOpen dirs after:", json.dumps(miopen_dirs(), indent=1))
print("\n| config | model | op | res | frames | variant | first s | median s | peak GiB | estimate GiB | tiled | rel diff |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['config']} | {r['model']} | {r['op']} | {r['res']} | {r['frames']} | {r['variant']} | {r['first_s']} | {r['median_s']} | "
          f"{r['peak_gib']} | {r['estimate_gib']} | {r['tiled']} | {r['rel_diff_vs_first_variant']} |")
print(f"\nraw rows appended to {a.out}")
