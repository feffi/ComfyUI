"""Profile one denoising step (one diffusion-model call) with torch.profiler.

Diagnostic custom node for a test instance, not for production. Load it without touching the
production install by pointing a second instance at this folder's parent:

  extra_paths.yaml:
    a2_profile:
      custom_nodes: <path to the folder that contains comfy_profile_step>

  python main.py --port 8199 --cuda-device 1 --extra-model-paths-config extra_paths.yaml

Insert "Profile Model Step (A2)" between the model loader (after LoRAs) and the KSampler. The Nth
model call (default 3, so warmup, autotune and graph capture are past) runs under torch.profiler.
Results go to output/profile_step/<label>_<n>/:
  kernels.txt   top kernels by GPU time (the 15 the analysis asks for, and more)
  modules.txt   GPU time per module type, from record_function ranges around every submodule
  ops.txt       top aten ops by self GPU time with the Python stack they came from
  trace.json    chrome trace, open in https://ui.perfetto.dev (local file, nothing is uploaded)
"""
import os
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

import comfy.patcher_extension
import folder_paths


def _label(m):
    if hasattr(m, "in_features") and hasattr(m, "out_features"):
        return f"{type(m).__name__} {m.out_features}x{m.in_features}"
    return type(m).__name__


def _annotate(root):
    handles, stack = [], []

    def pre(m, args):
        rf = record_function(_label(m))
        rf.__enter__()
        stack.append(rf)

    def post(m, args, out):
        stack.pop().__exit__(None, None, None)

    for m in root.modules():
        if m is not root:
            handles.append(m.register_forward_pre_hook(pre))
            handles.append(m.register_forward_hook(post))
    return handles


def _write_tables(prof, out_dir, header):
    events = prof.key_averages()
    kernels = sorted((e for e in events if e.device_type == DeviceType.CUDA), key=lambda e: e.self_device_time_total, reverse=True)
    total = sum(e.self_device_time_total for e in kernels) or 1.0
    with open(os.path.join(out_dir, "kernels.txt"), "w", encoding="utf-8") as f:
        f.write(header + f"GPU kernel time total {total / 1000:.1f} ms\n\n")
        f.write(f"{'ms':>9} {'%':>6} {'calls':>6}  kernel\n")
        for e in kernels[:40]:
            f.write(f"{e.self_device_time_total / 1000:9.2f} {100 * e.self_device_time_total / total:6.1f} {e.count:6d}  {e.key[:200]}\n")
    labels = {_label(m) for m in _annotate_targets}
    mods = sorted((e for e in events if e.key in labels), key=lambda e: e.device_time_total, reverse=True)
    with open(os.path.join(out_dir, "modules.txt"), "w", encoding="utf-8") as f:
        f.write(header + "inclusive GPU time per module type (nested modules overlap)\n\n")
        f.write(f"{'ms':>9} {'calls':>6}  module\n")
        for e in mods[:60]:
            f.write(f"{e.device_time_total / 1000:9.2f} {e.count:6d}  {e.key}\n")
    with open(os.path.join(out_dir, "ops.txt"), "w", encoding="utf-8") as f:
        f.write(header)
        f.write(prof.key_averages(group_by_stack_n=6).table(sort_by="self_device_time_total", row_limit=40, max_name_column_width=60))


_annotate_targets = []


class ProfileModelStepA2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "call_index": ("INT", {"default": 3, "min": 1, "max": 10000, "tooltip": "Which diffusion-model call to profile (CFG batches count once)."}),
            "label": ("STRING", {"default": "model"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "debug"

    def patch(self, model, call_index, label):
        model = model.clone()
        state = {"calls": 0}

        def wrapper(executor, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] != call_index:
                return executor(*args, **kwargs)
            diffusion_model = executor.class_obj.diffusion_model
            out_dir = os.path.join(folder_paths.get_output_directory(), "profile_step", f"{label}_{int(time.time())}")
            os.makedirs(out_dir, exist_ok=True)
            _annotate_targets[:] = list(diffusion_model.modules())
            handles = _annotate(diffusion_model)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
                    out = executor(*args, **kwargs)
                    torch.cuda.synchronize()
            finally:
                for h in handles:
                    h.remove()
            wall = time.perf_counter() - t0
            x = args[0]
            dev = torch.cuda.current_device()
            header = (f"{label} call {call_index}  x {tuple(x.shape) if hasattr(x, 'shape') else type(x).__name__}  "
                      f"wall {wall * 1000:.1f} ms (profiled, includes overhead)  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB\n"
                      f"torch {torch.__version__} hip {torch.version.hip} device {dev} {torch.cuda.get_device_name(dev)}\n")
            _write_tables(prof, out_dir, header)
            prof.export_chrome_trace(os.path.join(out_dir, "trace.json"))
            _annotate_targets.clear()
            print(f"[A2 profile] wrote {out_dir}")
            return out

        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "a2_profile_step", wrapper)
        return (model,)


NODE_CLASS_MAPPINGS = {"ProfileModelStepA2": ProfileModelStepA2}
NODE_DISPLAY_NAME_MAPPINGS = {"ProfileModelStepA2": "Profile Model Step (A2)"}
