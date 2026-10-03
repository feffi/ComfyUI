"""ComfyUI startup profiler. Does not modify ComfyUI or the venv.

Runs ComfyUI's main.py in-process and records, without editing any tracked file:
  - interpreter start (process creation -> this script), from psutil
  - first-import wall time of the heavy packages (torch, rocm_sdk, comfy_kitchen,
    triton, transformers, onnxruntime, ...) and ComfyUI phases
  - HIP/CUDA runtime init (torch._C._cuda_init) and the aotriton_supported() probe
    in comfy.model_management (first flash SDPA launch + synchronize)
  - nodes.init_extra_nodes and its parts, per-file load times for comfy_extras,
    comfy_api_nodes and custom nodes
  - frontend resolution, PromptServer setup, database/asset startup
  - a timestamped copy of every log line, up to "To see the GUI go to"
  - process CPU time vs wall time (a large gap on Windows points at disk or
    Defender scanning, which runs in MsMpEng.exe, not in python.exe)

Usage (from the ComfyUI folder, with the venv python):
  python comfy_startup_profile.py --out startup_run1 -- --port 8199 <your usual ComfyUI flags>

Writes <out>.json and <out>.txt and exits as soon as the GUI line is logged
(pass --keep-running to keep the server up). Use a non-default port so the
production instance is not disturbed.
"""
import argparse
import json
import logging
import os
import runpy
import sys
import threading
import time

T0 = time.perf_counter()
WALL0 = time.time()

WATCH_PREFIXES = (
    "torch", "torch._rocm_init", "rocm_sdk", "_rocm_sdk", "comfy_kitchen", "comfy_aimdo", "triton", "transformers",
    "comfy.model_management", "comfy.quant_ops", "comfy.utils", "comfy.sd", "comfy.model_base",
    "comfy.ops", "comfy.memory_management", "comfy.model_patcher", "execution", "server", "nodes",
    "app.database.db", "app.assets.manager", "app.frontend_management", "latent_preview",
    "comfyui_manager", "comfyui_frontend_package", "comfyui_workflow_templates",
)
WATCH_EXACT_DEPTH = {
    "torch": 1, "comfy_kitchen": 3, "triton": 1, "transformers": 1, "rocm_sdk": 1, "_rocm_sdk": 2,
}

state = {
    "imports": [],      # (name, start, end, depth)
    "phases": [],       # (name, start, end)
    "node_loads": [],   # (parent, path, start, end, ok)
    "dlls": [],         # (kind, name, start, end): extension modules and ctypes loads
    "log": [],          # (t, level, msg)
    "events": {},
}
_depth = [0]
_lock = threading.Lock()


def now():
    return time.perf_counter() - T0


def record_phase(name, start, end):
    with _lock:
        state["phases"].append((name, start, end))


def wrap_sync(owner, attr, name, once=False):
    orig = getattr(owner, attr)

    def wrapper(*a, **kw):
        t = now()
        try:
            return orig(*a, **kw)
        finally:
            record_phase(name, t, now())
            if once:
                setattr(owner, attr, orig)
    wrapper.__wrapped__ = orig
    setattr(owner, attr, wrapper)


def wrap_async(owner, attr, name):
    orig = getattr(owner, attr)

    async def wrapper(*a, **kw):
        t = now()
        try:
            return await orig(*a, **kw)
        finally:
            record_phase(name, t, now())
    wrapper.__wrapped__ = orig
    setattr(owner, attr, wrapper)


def wrap_classmethod(cls, attr, name):
    orig = cls.__dict__[attr].__func__

    def wrapper(c, *a, **kw):
        t = now()
        try:
            return orig(c, *a, **kw)
        finally:
            record_phase(name, t, now())
    setattr(cls, attr, classmethod(wrapper))


# ---- hooks applied right after a module finishes executing -------------------------

def hook_torch(mod):
    c = mod._C
    if hasattr(c, "_cuda_init"):
        wrap_sync(c, "_cuda_init", "torch._C._cuda_init (HIP/CUDA runtime + device init)", once=True)


def hook_nodes(mod):
    for attr in ("init_extra_nodes", "init_public_apis", "init_builtin_extra_nodes",
                 "init_builtin_api_nodes", "init_external_custom_nodes"):
        if hasattr(mod, attr):
            wrap_async(mod, attr, "nodes." + attr)
    orig = mod.load_custom_node

    async def load_custom_node(module_path, ignore=set(), module_parent="custom_nodes"):
        t = now()
        ok = await orig(module_path, ignore, module_parent=module_parent)
        with _lock:
            state["node_loads"].append((module_parent, module_path, t, now(), ok))
        return ok
    mod.load_custom_node = load_custom_node


def hook_server(mod):
    ps = mod.PromptServer
    wrap_sync(ps, "__init__", "server.PromptServer.__init__ (routes, frontend resolve)")
    wrap_sync(ps, "add_routes", "server.PromptServer.add_routes (static/web dirs)")
    wrap_async(ps, "setup", "server.PromptServer.setup")
    wrap_async(ps, "start_multi_address", "server.PromptServer.start_multi_address")


def hook_frontend(mod):
    wrap_classmethod(mod.FrontendManager, "init_frontend", "FrontendManager.init_frontend")


def hook_db(mod):
    if hasattr(mod, "init_db"):
        wrap_sync(mod, "init_db", "app.database.db.init_db")


def hook_assets(mod):
    if hasattr(mod, "AssetManager"):
        wrap_sync(mod.AssetManager, "startup", "AssetManager.startup")


def hook_aimdo_control(mod):
    for attr in ("init", "init_devices"):
        if hasattr(mod, attr):
            wrap_sync(mod, attr, "comfy_aimdo.control." + attr)


def hook_rocm_sdk(mod):
    if hasattr(mod, "initialize_process"):
        wrap_sync(mod, "initialize_process", "rocm_sdk.initialize_process (ROCm DLL preload)")


def hook_comfyui_manager(mod):
    for attr in ("prestartup", "start"):
        if hasattr(mod, attr):
            wrap_sync(mod, attr, "comfyui_manager." + attr)


# comfy.model_management runs aotriton_supported() while it is being imported, so
# time the flash SDPA launch and the synchronize it does from inside that import.
def pre_model_management():
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    F = torch.nn.functional
    saved = (F.scaled_dot_product_attention, torch.cuda.synchronize)
    wrap_sync(F, "scaled_dot_product_attention", "aotriton probe: first flash SDPA launch (kernel image load)")
    wrap_sync(torch.cuda, "synchronize", "aotriton probe: torch.cuda.synchronize")
    return saved


def post_model_management(saved):
    if saved is None:
        return
    torch = sys.modules["torch"]
    torch.nn.functional.scaled_dot_product_attention, torch.cuda.synchronize = saved


POST_HOOKS = {
    "torch": hook_torch,
    "nodes": hook_nodes,
    "server": hook_server,
    "app.frontend_management": hook_frontend,
    "app.database.db": hook_db,
    "app.assets.manager": hook_assets,
    "comfy_aimdo.control": hook_aimdo_control,
    "rocm_sdk": hook_rocm_sdk,
    "comfyui_manager": hook_comfyui_manager,
}
PRE_HOOKS = {"comfy.model_management": (pre_model_management, post_model_management)}

TOPLEVEL_SEEN = set()


def watched(name):
    top = name.split(".")[0]
    if top not in TOPLEVEL_SEEN:
        return True  # first import of every top-level package is timed
    if name in POST_HOOKS or name in PRE_HOOKS:
        return True
    for p in WATCH_PREFIXES:
        if name == p or name.startswith(p + "."):
            return name.count(".") < WATCH_EXACT_DEPTH.get(p, 1) + p.count(".")
    return False


class TimingFinder:
    def find_spec(self, name, path=None, target=None):
        if not watched(name):
            return None
        TOPLEVEL_SEEN.add(name.split(".")[0])
        spec = None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None or isinstance(spec.loader, type):
            return spec
        loader = spec.loader
        if not hasattr(loader, "exec_module"):
            return spec
        orig = loader.exec_module
        orig_create = getattr(loader, "create_module", None)
        started = []

        # extension modules (.pyd/.so) do their DLL load in create_module
        def create_module(spec_):
            started.append(now())
            return orig_create(spec_)

        def exec_module(module):
            pre = PRE_HOOKS.get(name)
            saved = pre[0]() if pre else None
            t = started[0] if started else now()
            _depth[0] += 1
            try:
                orig(module)
            finally:
                _depth[0] -= 1
                with _lock:
                    state["imports"].append((name, t, now(), _depth[0]))
                if pre:
                    pre[1](saved)
            hook = POST_HOOKS.get(name)
            if hook is not None:
                try:
                    hook(module)
                except Exception as e:  # a profiler hook must never break startup
                    state["events"]["hook_error_" + name] = repr(e)
        try:
            loader.exec_module = exec_module
            if orig_create is not None:
                loader.create_module = create_module
        except AttributeError:
            return spec
        return spec


def install_dll_timing():
    """Time every extension-module (.pyd/.so) load, including the ones custom code loads
    through spec_from_file_location (comfy-kitchen _C), and every ctypes.CDLL load
    (rocm_sdk preloads amdhip64, hipblaslt, miopen, ... this way)."""
    import ctypes
    import importlib._bootstrap_external as be
    orig_create = be.ExtensionFileLoader.create_module

    def create_module(self, spec):
        t = now()
        try:
            return orig_create(self, spec)
        finally:
            with _lock:
                state["dlls"].append(("ext", spec.origin or spec.name, t, now()))
    be.ExtensionFileLoader.create_module = create_module
    orig_init = ctypes.CDLL.__init__

    def cdll_init(self, name, *a, **kw):
        t = now()
        try:
            orig_init(self, name, *a, **kw)
        finally:
            with _lock:
                state["dlls"].append(("ctypes", str(name), t, now()))
    ctypes.CDLL.__init__ = cdll_init


def install_log_capture(exit_text, out):
    orig = logging.Logger.callHandlers

    def callHandlers(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        t = now()
        with _lock:
            state["log"].append((t, record.levelname, msg.strip().splitlines()[0][:160] if msg.strip() else ""))
        orig(self, record)
        if "To see the GUI go to" in msg and "gui" not in state["events"]:
            state["events"]["gui"] = t
        if exit_text and exit_text in msg:
            state["events"]["exit_text"] = t
            finish(out)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    logging.Logger.callHandlers = callHandlers


def pyc_check():
    import importlib.util
    total = missing = 0
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        total += 1
        try:
            if not os.path.exists(importlib.util.cache_from_source(f)):
                missing += 1
        except Exception:
            pass
    return {"py_modules": total, "missing_pyc": missing, "dont_write_bytecode": sys.dont_write_bytecode,
            "pycache_prefix": sys.pycache_prefix}


def defender_cpu():
    """CPU seconds used so far by Defender's scanner process, if readable (Windows only)."""
    if os.name != "nt":
        return None
    try:
        import psutil
        for p in psutil.process_iter(["name"]):
            if (p.info["name"] or "").lower() == "msmpeng.exe":
                c = p.cpu_times()
                return c.user + c.system
    except Exception:
        return None
    return None


_finished = [False]


def finish(out):
    if _finished[0]:
        return
    _finished[0] = True
    import psutil
    p = psutil.Process()
    end = now()
    cpu = p.cpu_times()
    create = p.create_time()
    report = {
        "argv": sys.argv,
        "python": sys.version,
        "interpreter_start_ms": round((WALL0 - create) * 1000, 1),
        "script_to_end_ms": round(end * 1000, 1),
        "process_total_ms": round((WALL0 - create + end) * 1000, 1),
        "gui_line_ms_since_script": round(state["events"]["gui"] * 1000, 1) if "gui" in state["events"] else None,
        "process_cpu_ms": round((cpu.user + cpu.system) * 1000, 1),
        "rss_mb": round(p.memory_info().rss / 2**20, 1),
        "defender_cpu_s_delta": None,
        "pyc": pyc_check(),
        "modules_loaded": len(sys.modules),
        "events": state["events"],
    }
    d0 = state["events"].get("defender_cpu_start")
    d1 = defender_cpu()
    if d0 is not None and d1 is not None:
        report["defender_cpu_s_delta"] = round(d1 - d0, 2)
    report["phases"] = [(n, round(s * 1000, 1), round((e - s) * 1000, 1)) for n, s, e in state["phases"]]
    report["imports"] = [(n, round(s * 1000, 1), round((e - s) * 1000, 1), d) for n, s, e, d in sorted(state["imports"], key=lambda r: r[1])]
    report["node_loads"] = [(par, path, round((e - s) * 1000, 1), ok) for par, path, s, e, ok in state["node_loads"]]
    report["dlls"] = [(k, n, round(s * 1000, 1), round((e - s) * 1000, 1)) for k, n, s, e in state["dlls"]]
    report["log"] = [(round(t * 1000, 1), lvl, msg) for t, lvl, msg in state["log"]]

    lines = []
    w = lines.append
    w("python: {}".format(sys.version.split()[0]))
    w("interpreter start (process create -> profiler): {:8.0f} ms  (exact on Windows; up to 1 s off on Linux)".format(report["interpreter_start_ms"]))
    if report["gui_line_ms_since_script"] is not None:
        w("profiler start -> 'To see the GUI go to':      {:8.0f} ms".format(report["gui_line_ms_since_script"]))
    w("profiler start -> end of profile:              {:8.0f} ms".format(report["script_to_end_ms"]))
    w("process create -> end of profile:             {:8.0f} ms".format(report["process_total_ms"]))
    w("process CPU (user+sys):                       {:8.0f} ms  (wall much larger than CPU = waiting on disk/AV/GPU)".format(report["process_cpu_ms"]))
    if report["defender_cpu_s_delta"] is not None:
        w("MsMpEng.exe CPU during startup:               {:8.1f} s".format(report["defender_cpu_s_delta"]))
    w("pyc: {missing_pyc} of {py_modules} loaded .py modules have no .pyc, dont_write_bytecode={dont_write_bytecode}".format(**report["pyc"]))
    w("")
    w("phases (start ms, duration ms):")
    for n, s, dur in sorted(report["phases"], key=lambda r: r[1]):
        w("  {:8.0f} {:8.1f}  {}".format(s, dur, n))
    w("")
    w("first imports >= 20 ms (start ms, inclusive ms, nested entries overlap):")
    for n, s, dur, d in report["imports"]:
        if dur >= 20:
            w("  {:8.0f} {:8.1f}  {}{}".format(s, dur, "  " * d, n))
    w("")
    w("DLL / extension module loads >= 5 ms (start ms, ms) [{} loads, {:.0f} ms total]:".format(
        len(report["dlls"]), sum(d[3] for d in report["dlls"])))
    for k, n, st, dur in report["dlls"]:
        if dur >= 5:
            w("  {:8.0f} {:8.1f}  {:6s} {}".format(st, dur, k, n.split("site-packages")[-1]))
    w("")
    by_parent = {}
    for par, path, dur, ok in report["node_loads"]:
        by_parent.setdefault(par, []).append((dur, os.path.basename(path.rstrip("/\\")), ok))
    for par, items in by_parent.items():
        w("{}: {} modules, {:.0f} ms total; slowest:".format(par, len(items), sum(i[0] for i in items)))
        for dur, name, ok in sorted(items, reverse=True)[:12]:
            w("  {:8.1f}  {}{}".format(dur, name, "" if ok else " (FAILED)"))
    w("")
    w("log timeline (ms since profiler start):")
    for t, lvl, msg in report["log"]:
        w("  {:8.0f} {:7s} {}".format(t, lvl, msg))
    text = "\n".join(lines)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    with open(out + ".txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    sys.__stderr__.write("\n[startup profile] written to {}.txt / .json\n".format(out))
    sys.__stderr__.write("\n".join(lines[:8]) + "\n")


def main():
    if "--" in sys.argv:
        i = sys.argv.index("--")
        own, comfy_args = sys.argv[1:i], sys.argv[i + 1:]
    else:
        own, comfy_args = sys.argv[1:], []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="startup_profile")
    ap.add_argument("--comfy-dir", default=os.getcwd())
    ap.add_argument("--keep-running", action="store_true", help="do not exit; stop it with Ctrl+C")
    ap.add_argument("--exit-on", default="To see the GUI go to",
                    help="exit once a log line contains this text, e.g. '[ComfyUI-Manager] All startup tasks have been completed.'")
    a = ap.parse_args(own)
    out = os.path.abspath(a.out)
    comfy_dir = os.path.abspath(a.comfy_dir)
    main_py = os.path.join(comfy_dir, "main.py")

    state["events"]["defender_cpu_start"] = defender_cpu()
    for name in list(sys.modules):
        TOPLEVEL_SEEN.add(name.split(".")[0])
    sys.meta_path.insert(0, TimingFinder())
    install_dll_timing()
    install_log_capture(None if a.keep_running else a.exit_on, out)

    import atexit
    atexit.register(finish, out)

    sys.argv = [main_py] + comfy_args
    sys.path[0] = comfy_dir
    os.chdir(comfy_dir)
    runpy.run_path(main_py, run_name="__main__")


if __name__ == "__main__":
    main()
