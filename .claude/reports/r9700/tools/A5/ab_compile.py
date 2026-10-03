"""A/B runner for the comfy compiler, HIP graph capture and TorchCompileModel.

Starts a separate ComfyUI instance per variant (default port 8199, so a production
instance is not touched), queues API-format workflows through the HTTP API, and
records per run: wall time, ComfyUI execution time, sampler s/it or it/s (from the
console progress bars), polled peak VRAM per GPU, the comfy compiler's
"graph breaks / rogues" line, and errors, hangs (watchdog) or crashes.

Run with the production venv's python (read-only use, nothing is installed):

  .venv-rocm-100\\Scripts\\python.exe ab_compile.py --comfy-dir C:\\ComfyUI ^
      --workflow krea2=wf\\krea2_api.json --workflow qi21=wf\\qi21_api.json ^
      --workflow mmh3=wf\\minimax_h3_api.json --workflow te_gen=wf\\textgen_qwen3vl_api.json ^
      --variants baseline,no_graphs,no_compiler,torch_compile --runs 3 --out ab_out

Keep the instance away from production state with --extra-args, e.g.
  --extra-args "--user-directory D:\\ab\\user --output-directory D:\\ab\\out --cuda-device 1"
(--cuda-device 1 runs on GPU 1 when GPU 0 is busy; it hides the other card from that instance).

Workflows must be exported with "Export (API)". Seeds (inputs named seed / noise_seed)
are changed every run so ComfyUI's cache does not skip the sampler.
te_gen: a "Generate Text" (TextGenerate) workflow on a Qwen3-VL text encoder, the only
path that uses HIP graph capture; compare its "Generating tokens" it/s across variants.

Built-in variants (name -> extra args / env / workflow change):
  baseline        nothing
  no_graphs       --disable-cuda-graphs        (comfy compiler stays on)
  no_compiler     --disable-comfy-compiler     (also disables graphs)
  torch_compile   TorchCompileModel(inductor) inserted before every sampler/guider model input
  hwq2            env GPU_MAX_HW_QUEUES=2      (ROCm/legacy-rocm-build#6685, Linux-measured)
  no_async        --disable-async-offload      (also disables prefetch and graph capture)
Custom: --variant "name|--flag1 --flag2|ENV=VAL;ENV2=VAL"
"""
import argparse
import copy
import csv
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BUILTIN = {
    "baseline": ([], {}, False),
    "no_graphs": (["--disable-cuda-graphs"], {}, False),
    "no_compiler": (["--disable-comfy-compiler"], {}, False),
    "torch_compile": ([], {}, True),
    "hwq2": ([], {"GPU_MAX_HW_QUEUES": "2"}, False),
    "no_async": (["--disable-async-offload"], {}, False),
}
SAMPLER_NODES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "CFGGuider", "BasicGuider", "DualCFGGuider"}
LOG_PATTERNS = {
    "compiler_stats": re.compile(r"Comfy model compiler graph breaks: (\d+), rogues: (\d+)"),
    "hip_error": re.compile(r"hipError\w*|HIP error|CUDA error[^\n]*|device-side assert|out of memory", re.I),
    "dynamo": re.compile(r"recompile_limit|cache_size_limit|Graph break in user code|torch\._dynamo hit"),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
}
BAR = re.compile(r"(\d+)/(\d+) \[[^\]]*?,\s*([\d.]+)(s/it|it/s)\]")


def http(port, path, data=None, timeout=10):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


def versions(py, comfy_dir):
    code = ("import json,importlib\nout={}\n"
            "import torch\nout['torch']=torch.__version__;out['hip']=torch.version.hip\n"
            "out['devices']=[torch.cuda.get_device_properties(i).gcnArchName for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []\n"
            "for m in ('triton','comfy_kitchen','comfy_aimdo'):\n"
            "  try:\n    mod=importlib.import_module(m);out[m]=getattr(mod,'__version__',None) or importlib.metadata.version(m.replace('_','-'))\n"
            "  except Exception as e:\n    out[m]='n/a: %s'%e\n"
            "print(json.dumps(out))")
    info = {}
    try:
        info = json.loads(subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300).stdout.strip().splitlines()[-1])
    except Exception as e:
        info["error"] = str(e)
    try:
        info["comfyui_commit"] = subprocess.run(["git", "-C", comfy_dir, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    if os.name == "nt":
        ps = "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*Radeon*' } | ForEach-Object { $_.Name + ' ' + $_.DriverVersion }) -join '; '"
        info["driver"] = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    return info


def mutate_seed(prompt, value):
    for node in prompt.values():
        for k, v in node.get("inputs", {}).items():
            if (k in ("seed", "noise_seed") or k.endswith(".seed")) and isinstance(v, int):
                node["inputs"][k] = value


def set_size(prompt, w, h):
    for node in prompt.values():
        ins = node.get("inputs", {})
        if node.get("class_type", "").startswith("Empty") and isinstance(ins.get("width"), int) and isinstance(ins.get("height"), int):
            ins["width"], ins["height"] = w, h


def insert_torch_compile(prompt, backend):
    prompt = copy.deepcopy(prompt)
    made = {}
    for nid, node in list(prompt.items()):
        link = node.get("inputs", {}).get("model")
        if node.get("class_type") in SAMPLER_NODES and isinstance(link, list):
            key = (str(link[0]), link[1])
            if key not in made:
                made[key] = f"ab_compile_{len(made)}"
                prompt[made[key]] = {"class_type": "TorchCompileModel", "inputs": {"model": list(link), "backend": backend}}
            node["inputs"]["model"] = [made[key], 0]
    if not made:
        raise SystemExit("torch_compile variant: no sampler/guider with a linked model input found")
    return prompt


class Instance:
    def __init__(self, a, name, extra, env_add, log_path):
        env = os.environ.copy()
        env.update(env_add)
        if a.compile_cache:
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath(os.path.join(a.compile_cache, "inductor")))
            env.setdefault("TRITON_CACHE_DIR", os.path.abspath(os.path.join(a.compile_cache, "triton")))
        cmd = [a.python, "main.py", "--port", str(a.port), "--listen", "127.0.0.1", "--disable-auto-launch"] + a.extra_args.split() + extra
        self.log = open(log_path, "w", encoding="utf-8", errors="replace")
        self.log.write("# " + " ".join(cmd) + "\n# env: " + json.dumps(env_add) + "\n")
        self.log.flush()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, cwd=a.comfy_dir, env=env, stdout=self.log, stderr=subprocess.STDOUT, creationflags=flags)
        self.port = a.port
        self.startup_s = None
        deadline = time.time() + a.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return
            try:
                http(self.port, "/system_stats", timeout=2)
                self.startup_s = time.perf_counter() - self.t0
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1)

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def run_prompt(inst, prompt, timeout):
    peak = {}
    stop = threading.Event()

    def poll_vram():
        while not stop.is_set():
            try:
                for d in http(inst.port, "/system_stats", timeout=2)["devices"]:
                    used = d["vram_total"] - d["vram_free"]
                    peak[d["name"]] = max(peak.get(d["name"], 0), used)
            except Exception:
                pass
            stop.wait(1.0)

    poller = threading.Thread(target=poll_vram, daemon=True)
    poller.start()
    t0 = time.perf_counter()
    status, exec_s = "ok", None
    try:
        pid = http(inst.port, "/prompt", {"prompt": prompt, "client_id": "ab_compile"}, timeout=60)["prompt_id"]
        while True:
            if not inst.alive():
                status = "CRASH"
                break
            if time.perf_counter() - t0 > timeout:
                status = "HANG"
                try:
                    http(inst.port, "/interrupt", {}, timeout=5)
                except Exception:
                    pass
                break
            h = http(inst.port, f"/history/{pid}", timeout=10).get(pid)
            if h:  # history is written once the prompt has finished
                msgs = dict((m[0], m[1]) for m in h["status"].get("messages", []))
                if h["status"].get("status_str") != "success":
                    status = "error"
                if "execution_start" in msgs:
                    end = msgs.get("execution_success") or msgs.get("execution_error") or msgs.get("execution_interrupted")
                    if end:
                        exec_s = (end["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000.0
                break
            time.sleep(0.5)
    except Exception as e:
        status = f"api_error: {e}"
    wall = time.perf_counter() - t0
    stop.set()
    poller.join(3)
    return status, wall, exec_s, {k: round(v / 2**30, 2) for k, v in peak.items()}


def scan_log(path, start):
    with open(path, encoding="utf-8", errors="replace") as f:
        f.seek(start)
        text = f.read()
    # one entry per progress bar: its last reported rate (early-stopped bars such as token generation included)
    bars, last_n = {}, {}
    for chunk in re.split(r"[\r\n]", text):
        m = BAR.search(chunk)
        if not m:
            continue
        label = re.sub(r"\s*\d+%\|.*$", "", chunk[:m.start()]).strip(" :|")[-40:] or "sampler"
        n, val = int(m.group(1)), float(m.group(3))
        s_per_it = round(val if m.group(4) == "s/it" else 1.0 / val, 5)
        if label not in bars or n <= last_n[label]:
            bars.setdefault(label, []).append(s_per_it)
        else:
            bars[label][-1] = s_per_it
        last_n[label] = n
    found = {k: p.findall(text)[-5:] for k, p in LOG_PATTERNS.items()}
    return bars, {k: v for k, v in found.items() if v}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy-dir", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workflow", action="append", required=True, help="name=path_to_api_json")
    p.add_argument("--variants", default="baseline,no_graphs,no_compiler")
    p.add_argument("--variant", action="append", default=[], help='custom "name|--flags|ENV=VAL;ENV2=VAL"')
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--extra-args", default="", help='args for every instance, e.g. "--cuda-device 1"')
    p.add_argument("--compile-backend", default="inductor", choices=["inductor", "cudagraphs"])
    p.add_argument("--compile-cache", default="ab_compile_cache", help="TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR root (persistent)")
    p.add_argument("--cold-cache", action="store_true", help="wipe --compile-cache before the first torch_compile variant")
    p.add_argument("--alt-size", type=int, nargs=2, metavar=("W", "H"), help="extra run per workflow at another latent size (recompile check)")
    p.add_argument("--timeout", type=float, default=1800, help="per-prompt watchdog seconds")
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--out", default="ab_out")
    a = p.parse_args()
    a.comfy_dir = os.path.abspath(a.comfy_dir)
    os.makedirs(a.out, exist_ok=True)

    variants = {}
    for name in [v for v in a.variants.split(",") if v]:
        variants[name] = BUILTIN[name]
    for spec in a.variant:
        name, flags, envs = (spec.split("|") + ["", ""])[:3]
        variants[name] = (flags.split(), dict(e.split("=", 1) for e in envs.split(";") if e), False)
    workflows = {}
    for w in a.workflow:
        name, path = w.split("=", 1)
        with open(path, encoding="utf-8") as f:
            workflows[name] = json.load(f)

    info = versions(a.python, a.comfy_dir)
    print("environment:", json.dumps(info))
    if a.cold_cache and a.compile_cache and os.path.isdir(a.compile_cache):
        shutil.rmtree(a.compile_cache)

    rows = []
    for vname, (extra, env_add, compile_model) in variants.items():
        log_path = os.path.join(a.out, f"{vname}.log")
        inst = Instance(a, vname, extra, env_add, log_path)
        print(f"[{vname}] startup {inst.startup_s if inst.startup_s is None else round(inst.startup_s, 1)} s")
        if inst.startup_s is None:
            rows.append({"variant": vname, "workflow": "-", "run": "startup", "status": "STARTUP_FAIL", "wall_s": None,
                         "exec_s": None, "peak_vram_gib": {}, "bars_s_per_it": {}, "log": {}})
            print(f"[{vname}] did not start, see {log_path}")
            inst.stop()
            continue
        for wname, wf in workflows.items():
            base = insert_torch_compile(wf, a.compile_backend) if compile_model else wf
            plan = [("run%d" % (i + 1), None) for i in range(a.runs)]
            if a.alt_size:
                plan.append(("alt_size", tuple(a.alt_size)))
            for label, size in plan:
                if not inst.alive():
                    inst.stop()
                    inst = Instance(a, vname, extra, env_add, log_path + f".restart{len(rows)}")
                prompt = copy.deepcopy(base)
                mutate_seed(prompt, random.randint(1, 2**31))
                if size:
                    set_size(prompt, *size)
                pos = os.path.getsize(inst.log.name)
                status, wall, exec_s, peak = run_prompt(inst, prompt, a.timeout)
                time.sleep(1)
                inst.log.flush()
                bars, found = scan_log(inst.log.name, pos)
                row = {"variant": vname, "workflow": wname, "run": label, "status": status, "wall_s": round(wall, 2),
                       "exec_s": exec_s, "peak_vram_gib": peak, "bars_s_per_it": bars, "log": found}
                rows.append(row)
                print(json.dumps(row))
                if status in ("HANG", "CRASH"):
                    inst.stop()
                    break
        inst.stop()

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"environment": info, "rows": rows}, f, indent=1)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "workflow", "run", "status", "wall_s", "exec_s", "peak_vram_gib", "bars_s_per_it", "log"])
        for r in rows:
            w.writerow([r["variant"], r["workflow"], r["run"], r["status"], r["wall_s"], r["exec_s"],
                        json.dumps(r["peak_vram_gib"]), json.dumps(r["bars_s_per_it"]), json.dumps(r["log"])])

    print("\nmedian of warm runs (run2..runN); run1 includes load/compile")
    print(f"{'workflow':10} {'variant':14} {'run1 s':>8} {'warm s':>8} {'warm s/it':>10} status")
    for wname in workflows:
        for vname in variants:
            rs = [r for r in rows if r["variant"] == vname and r["workflow"] == wname and r["run"].startswith("run")]
            if not rs:
                continue
            warm = [r for r in rs[1:] if r["status"] == "ok"]
            its = [v for r in warm for vals in r["bars_s_per_it"].values() for v in vals[-1:]]
            med = lambda xs: round(statistics.median(xs), 3) if xs else "-"
            print(f"{wname:10} {vname:14} {rs[0]['wall_s']:>8} {med([r['wall_s'] for r in warm]):>8} {med(its):>10} "
                  f"{','.join(sorted(set(r['status'] for r in rs)))}")


if __name__ == "__main__":
    main()
