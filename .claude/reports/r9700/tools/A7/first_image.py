"""Time to first image against a local ComfyUI (127.0.0.1 only, stdlib only).

usage:
  python first_image.py --workflow my_workflow_api.json --port 8199 [--t0 <epoch seconds>] [--out first_image.json]

The workflow must be in API format (ComfyUI menu: Workflow -> Export (API)).
Sequence:
  1. wait until the server answers (/system_stats); reports the wait from --t0
     (pass the time ComfyUI was launched to get launch -> server ready)
  2. run 1 "first":  models load from disk, kernels JIT/load for the first time
  3. run 2 "warm":   new seed, models still loaded (only sampling + decode)
  4. POST /free {unload_models, free_memory}, then run 3 "reload": new seed,
     models reload from the OS file cache (no disk), kernels already loaded
Each run reports wall time and ComfyUI's own execution_start -> execution_success span.
"""
import argparse
import json
import random
import time
import urllib.request
import uuid


def call(base, path, data=None, timeout=30):
    req = urllib.request.Request(base + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def reseed(prompt):
    for node in prompt.values():
        inputs = node.get("inputs", {})
        for k in ("seed", "noise_seed"):
            if isinstance(inputs.get(k), int):
                inputs[k] = random.randint(0, 2**48)


def run(base, prompt, label, timeout_s):
    reseed(prompt)
    t = time.time()
    pid = call(base, "/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = call(base, "/history/" + pid)
        if h and pid in h:
            entry = h[pid]
            break
        if time.time() - t > timeout_s:
            raise TimeoutError("{} did not finish in {} s".format(label, timeout_s))
        time.sleep(0.25)
    wall = time.time() - t
    status = entry.get("status", {})
    stamps = {m[0]: m[1].get("timestamp") for m in status.get("messages", []) if isinstance(m, list) and len(m) == 2}
    span = None
    if stamps.get("execution_start") and stamps.get("execution_success"):
        span = (stamps["execution_success"] - stamps["execution_start"]) / 1000.0
    res = {"run": label, "wall_s": round(wall, 2), "comfy_exec_s": None if span is None else round(span, 2),
           "status": status.get("status_str")}
    print(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--t0", type=float, default=None, help="epoch seconds when ComfyUI was launched")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default="first_image.json")
    a = ap.parse_args()
    base = "http://127.0.0.1:{}".format(a.port)
    with open(a.workflow, encoding="utf-8") as f:
        prompt = json.load(f)
    if "nodes" in prompt and "links" in prompt:
        raise SystemExit("This is the UI workflow format; export it with Workflow -> Export (API).")

    t0 = a.t0 if a.t0 is not None else time.time()
    while True:
        try:
            call(base, "/system_stats", timeout=2)
            break
        except Exception:
            if time.time() - t0 > a.timeout:
                raise SystemExit("server did not come up")
            time.sleep(0.2)
    ready = time.time() - t0
    print(json.dumps({"server_ready_s": round(ready, 2)}))

    results = {"server_ready_s": round(ready, 2), "runs": []}
    first = run(base, prompt, "first", a.timeout)
    results["runs"].append(first)
    results["launch_to_first_image_s"] = round(ready + first["wall_s"], 2) if a.t0 is not None else None
    results["runs"].append(run(base, prompt, "warm", a.timeout))
    call(base, "/free", {"unload_models": True, "free_memory": True})
    time.sleep(2)
    results["runs"].append(run(base, prompt, "reload", a.timeout))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
