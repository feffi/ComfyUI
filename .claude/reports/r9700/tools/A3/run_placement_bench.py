"""A3 dual-GPU placement benchmark for ComfyUI (Krea 2, Qwen-Image 2.1, MiniMax H3).

Starts a private ComfyUI instance (default port 8199, in-memory asset DB), runs the
model's default template graph three times and reports, per pass:
  - per-node wall time (text encode, sampling, VAE decode) from websocket events
  - per-GPU peak used VRAM, sampled from /system_stats every 0.2 s (includes
    DynamicVRAM/aimdo weight pages, unlike torch.cuda.max_memory_allocated)
  - load / unload / deepclone events from the server log

Passes: 1 cold, 2 new seed (text encode cached; shows whether the DiT stayed
resident), 3 new prompt (text encode reruns; shows whether encode evicts the DiT).

Placement:
  single  everything on the default device (GPU 0)
  split   text encoder + VAE(s) on gpu:1 via Select CLIP/VAE Device, DiT on GPU 0
Add --cfg-split to insert MultiGPU CFG Split (only useful with --cfg > 1).

Run from the production venv, with the production ComfyUI stopped or idle:
  .venv-rocm-100\\Scripts\\python run_placement_bench.py --comfy C:\\path\\ComfyUI --model minimax --placement single
  .venv-rocm-100\\Scripts\\python run_placement_bench.py --comfy C:\\path\\ComfyUI --model minimax --placement split
Run split twice: on the unpatched install (documents the Select CLIP Device issue:
GPU 1 stays near idle during encode) and with select_clip_device_sync_model.patch.
Only stdlib + aiohttp (already a ComfyUI dependency).
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid

import aiohttp

DEFAULTS = {
    "minimax": dict(unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                    clip="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    vae="minimax_h3_video_vae_int8_convrot.safetensors",
                    vae_audio="minimax_h3_audio_vae_fp32.safetensors",
                    width=864, height=480, length=56, steps=20, cfg=1.0),
    "krea2": dict(unet="krea2_turbo_fp8_scaled.safetensors", clip="qwen3vl_4b_fp8_scaled.safetensors",
                  vae="qwen_image_vae.safetensors", width=1024, height=1024, steps=8, cfg=1.0),
    "qi21": dict(unet="qwen_image_2.1_int8_convrot.safetensors", clip="qwen3vl_8b_int8_convrot.safetensors",
                 vae="qwen_image_2.1_vae_bf16.safetensors", width=1024, height=1024, steps=25, cfg=1.0),
}
PROMPTS = ["A fennec-eared anime girl with long blonde wavy hair and blue eyes walks through a rainy neon street at dusk, cinematic.",
           "A fennec-eared anime girl with a big fluffy tail reads a book in a sunlit library, soft morning light, film grain."]


def build_graph(a, prompt, seed):
    g = {}
    g["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": a.unet, "weight_dtype": "default"}}
    g["clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": a.clip, "type": {"minimax": "minimax", "krea2": "krea2", "qi21": "qwen_image"}[a.model], "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}}
    model, clip, vae = ["unet", 0], ["clip", 0], ["vae", 0]
    if a.placement == "split":
        g["clip_dev"] = {"class_type": "SelectCLIPDevice", "inputs": {"clip": clip, "device": "gpu:1"}}
        g["vae_dev"] = {"class_type": "SelectVAEDevice", "inputs": {"vae": vae, "device": "gpu:1"}}
        clip, vae = ["clip_dev", 0], ["vae_dev", 0]
    if a.cfg_split:
        g["cfg_split"] = {"class_type": "MultiGPU_WorkUnits", "inputs": {"model": model, "max_gpus": 2}}
        model = ["cfg_split", 0]

    if a.model == "minimax":
        g["vae_audio"] = {"class_type": "VAELoader", "inputs": {"vae_name": a.vae_audio}}
        vae_audio = ["vae_audio", 0]
        if a.placement == "split":
            g["vae_audio_dev"] = {"class_type": "SelectVAEDevice", "inputs": {"vae": vae_audio, "device": "gpu:1"}}
            vae_audio = ["vae_audio_dev", 0]
        g["encode"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": clip, "vae": vae, "prompt": prompt, "width": a.width, "height": a.height, "length": a.length}}
        g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
        g["sampler_sel"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
        g["sched"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": a.steps, "denoise": 1.0}}
        if a.cfg > 1.0:
            g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["encode", 0]}}
            g["guider"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["encode", 0], "negative": ["neg", 0], "cfg": a.cfg}}
        else:
            g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["encode", 0]}}
        g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler_sel", 0], "sigmas": ["sched", 0], "latent_image": ["encode", 1]}}
        g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": vae}}
        g["decode_audio"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": vae_audio}}
        g["out_audio"] = {"class_type": "PreviewAny", "inputs": {"source": ["decode_audio", 0]}}
    else:
        if a.model == "krea2":
            g["encode"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": prompt}}
            g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["encode", 0]}}
            pos, neg = ["encode", 0], ["neg", 0]
        else:
            g["encode"] = {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": clip, "prompt": prompt, "negative_prompt": "", "resolution": 1024}}
            pos, neg = ["encode", 0], ["encode", 1]
        g["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": a.width, "height": a.height, "batch_size": 1}}
        g["sample"] = {"class_type": "KSampler", "inputs": {"model": model, "seed": seed, "steps": a.steps, "cfg": a.cfg, "sampler_name": "euler", "scheduler": "simple",
                                                            "positive": pos, "negative": neg, "latent_image": ["latent", 0], "denoise": 1.0}}
        g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": vae}}
    g["out"] = {"class_type": "PreviewAny", "inputs": {"source": ["decode", 0]}}
    return g


EVENT_RES = [re.compile(p) for p in (
    r"Requested to load \S+", r"Model loaded: .*", r"loaded (?:completely|partially).*", r"\d+ models unloaded.*",
    r"Unloading \S+", r"Creating deepclone of .*", r"Reusing loaded multigpu .*", r"Select (?:CLIP|VAE|Model) Device: .*",
    r"VAE decode needs more than the free VRAM.*", r".*[Tt]iled.*", r".*WDDM.*", r".*[Oo]ut of memory.*", r"Prompt executed in .*")]


def read_events(path, offset):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        text = f.read()
        end = f.tell()
    events = []
    for line in text.splitlines():
        for r in EVENT_RES:
            m = r.search(line)
            if m:
                events.append(m.group(0)[:200])
                break
    return events, end


async def poll_stats(session, base, state):
    while not state["done"]:
        try:
            async with session.get(base + "/system_stats") as r:
                js = await r.json()
            node = state["node"]
            for d in js["devices"]:
                if d["type"] != "cuda":
                    continue
                used = (d["vram_total"] - d["vram_free"]) / 2 ** 30
                key = d["index"]
                state["peak"][key] = max(state["peak"].get(key, 0.0), used)
                if node is not None:
                    pk = state["node_peak"].setdefault(node, {})
                    pk[key] = max(pk.get(key, 0.0), used)
                state["last"][key] = used
        except Exception as e:  # server busy or restarting; keep sampling
            state["errors"] = state.get("errors", 0) + 1
        await asyncio.sleep(0.2)


async def run_once(session, base, ws, client_id, graph):
    state = {"done": False, "node": None, "peak": {}, "node_peak": {}, "last": {}}
    poller = asyncio.create_task(poll_stats(session, base, state))
    async with session.post(base + "/prompt", json={"prompt": graph, "client_id": client_id}) as r:
        resp = await r.json()
    if "prompt_id" not in resp:
        state["done"] = True
        await poller
        raise RuntimeError("prompt rejected: " + json.dumps(resp)[:2000])
    pid = resp["prompt_id"]
    times, cached, t_node, t0 = {}, [], None, time.perf_counter()
    while True:
        msg = await ws.receive()
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        m = json.loads(msg.data)
        data = m.get("data", {})
        if data.get("prompt_id") not in (None, pid):
            continue
        now = time.perf_counter()
        if m["type"] == "execution_cached":
            cached += data.get("nodes", [])
        elif m["type"] == "executing":
            if t_node is not None:
                times[state["node"]] = times.get(state["node"], 0.0) + now - t_node
            state["node"], t_node = data.get("node"), now
            if data.get("node") is None:
                break
        elif m["type"] == "execution_error":
            state["done"] = True
            await poller
            raise RuntimeError("execution error: " + json.dumps(data)[:3000])
    state["done"] = True
    await poller
    return {"total_s": time.perf_counter() - t0, "node_s": times, "cached": cached, "peak_gib": state["peak"], "node_peak_gib": state["node_peak"]}


async def main_async(a):
    base = f"http://127.0.0.1:{a.port}"
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.model}_{a.placement}{'_cfgsplit' if a.cfg_split else ''}"
    console_log = os.path.join(a.out, f"{tag}_server.log")
    proc = None
    if not a.no_launch:
        cmd = [a.python, "main.py", "--port", str(a.port), "--database-url", "sqlite:///:memory:"] + a.server_args.split()
        logf = open(console_log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=a.comfy, stdout=logf, stderr=subprocess.STDOUT)
        print("started:", " ".join(cmd), "->", console_log)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            t_start = time.perf_counter()
            while True:
                try:
                    async with session.get(base + "/system_stats") as r:
                        stats = await r.json()
                    break
                except Exception:
                    if proc is not None and proc.poll() is not None:
                        raise RuntimeError(f"server exited, see {console_log}")
                    if time.perf_counter() - t_start > a.startup_timeout:
                        raise RuntimeError("server did not come up")
                    await asyncio.sleep(1.0)
            print("server ready in %.1f s" % (time.perf_counter() - t_start))
            for d in stats["devices"]:
                used = (d["vram_total"] - d["vram_free"]) / 2 ** 30
                print(f"  {d['type']}:{d['index']} {d['name']} total {d['vram_total'] / 2 ** 30:.1f} GiB, used at idle {used:.2f} GiB")
                if d["type"] == "cuda" and used > 2.0:
                    print("  WARNING: >2 GiB already used on this GPU (production instance or other app?). Results will be skewed.")
            print(f"  RAM total {stats['system']['ram_total'] / 2 ** 30:.1f} GiB, free {stats['system']['ram_free'] / 2 ** 30:.1f} GiB")

            client_id = str(uuid.uuid4())
            ws = await session.ws_connect(f"ws://127.0.0.1:{a.port}/ws?clientId={client_id}", max_msg_size=0)
            offset = os.path.getsize(console_log) if not a.no_launch else 0
            results = []
            passes = [("1 cold", PROMPTS[0], a.seed), ("2 new seed", PROMPTS[0], a.seed + 1), ("3 new prompt", PROMPTS[1], a.seed + 2)]
            for name, prompt, seed in passes:
                res = await run_once(session, base, ws, client_id, build_graph(a, prompt, seed))
                if not a.no_launch:
                    await asyncio.sleep(0.5)
                    res["events"], offset = read_events(console_log, offset)
                res["pass"] = name
                results.append(res)
                print(f"\n== pass {name}: total {res['total_s']:.1f} s")
                for node in ("encode", "sample", "decode", "decode_audio"):
                    if node in res["node_s"]:
                        pk = res["node_peak_gib"].get(node, {})
                        print(f"   {node:13s} {res['node_s'][node]:7.2f} s   peak used " + "  ".join(f"gpu{k}={v:.2f}" for k, v in sorted(pk.items())) + " GiB")
                print("   cached nodes:", ",".join(res["cached"]) or "-")
                print("   run peak used: " + "  ".join(f"gpu{k}={v:.2f} GiB" for k, v in sorted(res["peak_gib"].items())))
                for e in res.get("events", []):
                    print("   log:", e)
            await ws.close()
            out = os.path.join(a.out, f"{tag}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"args": vars(a), "idle_stats": stats, "passes": results}, f, indent=1, default=str)
            print("\nwrote", out)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy", required=True, help="ComfyUI directory (contains main.py)")
    p.add_argument("--python", default=sys.executable, help="python used to start the server (default: this one)")
    p.add_argument("--model", choices=sorted(DEFAULTS), required=True)
    p.add_argument("--placement", choices=["single", "split"], default="split")
    p.add_argument("--cfg-split", action="store_true", help="insert MultiGPU CFG Split (needs --cfg > 1 to do anything)")
    p.add_argument("--cfg", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--length", type=int, default=None, help="MiniMax H3 frames (17k+5)")
    p.add_argument("--unet"); p.add_argument("--clip"); p.add_argument("--vae"); p.add_argument("--vae-audio")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--server-args", default="--verbose INFO", help="extra args for main.py, e.g. \"--verbose DETAIL\" for vram_mb per load")
    p.add_argument("--no-launch", action="store_true", help="use an already running server on --port (no log events)")
    p.add_argument("--startup-timeout", type=float, default=600)
    p.add_argument("--out", default="a3_results")
    a = p.parse_args()
    for k, v in DEFAULTS[a.model].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
