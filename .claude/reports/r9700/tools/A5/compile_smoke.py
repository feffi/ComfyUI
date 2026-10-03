"""torch.compile smoke test for Krea 2, Qwen-Image 2.1 and MiniMax H3 DiT code on this stack.

Builds reduced-depth models from the checkout's model code (random weights, real head_dim
128 and a configurable width), wraps them like the TorchCompileModel node does
(torch.compile(diffusion_model, backend, options={"guard_filter_fn": ...})), and runs a
few "sampling steps" (changing sigma) at two latent sizes. Reports per model: eager
s/call, first compiled call (= compile time), warm compiled s/call, graph breaks,
unique graphs, and the first errors. Use it to check inductor + triton-windows on
gfx1201 without loading full checkpoints, and to A/B the proposed patches:

  --fix-fp16-check   comfy/ops.py _fp16_linear_wanted reordered (bf16 skips the torch._C getter)
  --fix-prefetch     comfy/model_prefetch.py with the torch.compiler.is_compiling() guards
                     (needs patched/model_prefetch.py next to this script)

  set TORCHINDUCTOR_CACHE_DIR=D:\\cache\\inductor & set TRITON_CACHE_DIR=D:\\cache\\triton
  .venv-rocm-100\\Scripts\\python.exe compile_smoke.py --comfy-dir C:\\ComfyUI --device cuda:1
  (CPU check: --device cpu)
"""
import argparse
import os
import sys
import time
import traceback

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--comfy-dir", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--backend", default="inductor", choices=["inductor", "eager", "aot_eager", "cudagraphs"])
ap.add_argument("--mode", default=None, help='torch.compile mode, e.g. "max-autotune" (Triton GEMM templates, see num_stages note)')
ap.add_argument("--width", type=int, default=1024, help="hidden size (heads = width // 128)")
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--models", default="krea2,qi21,mmh3")
ap.add_argument("--fix-fp16-check", action="store_true")
ap.add_argument("--fix-prefetch", action="store_true")
a = ap.parse_args()

sys.argv = [sys.argv[0]] + (["--cpu"] if a.device == "cpu" else [])
sys.path.insert(0, a.comfy_dir)
import comfy.options
comfy.options.enable_args_parsing()
import torch
import torch._dynamo
from torch._dynamo.utils import counters
import comfy.ops
import comfy.model_prefetch
import comfy.ldm.krea2.model as krea2
import comfy.ldm.qwen_image21.model as qi21
import comfy.ldm.minimax.model as mm
from comfy_extras.nodes_torch_compile import skip_torch_compile_dict

if a.fix_fp16_check:
    comfy.ops._fp16_linear_wanted = lambda x: (x.dtype == torch.float16 and x.is_cuda
                                               and getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False))
if a.fix_prefetch:
    import importlib.util
    spec = importlib.util.spec_from_file_location("comfy.model_prefetch", os.path.join(os.path.dirname(os.path.abspath(__file__)), "patched", "model_prefetch.py"))
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    sys.modules["comfy.model_prefetch"] = mp
    comfy.model_prefetch = mp

device = torch.device(a.device)
dt = torch.bfloat16
ops = comfy.ops.disable_weight_init
W, H = a.width, a.width // 128


def init(m):
    m = m.to(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    for p in m.parameters():
        p.data = (torch.randn(p.shape, generator=g) * 0.02).to(device=device, dtype=p.dtype)
    for _, b in m.named_buffers():
        b.data = torch.rand(b.shape, generator=g).to(device=device, dtype=b.dtype)
    return m.eval().requires_grad_(False)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build(name):
    if name == "krea2":
        m = init(krea2.SingleStreamDiT(features=W, tdim=256, txtdim=256, heads=H, kvheads=max(1, H // 4), multiplier=4, layers=a.layers,
                                       patch=2, channels=16, txtlayers=3, txtheads=2, txtkvheads=2, dtype=dt, operations=ops))
        def call(f, size, s):
            g = torch.Generator(device="cpu").manual_seed(1)
            x = torch.randn((1, 16, 1) + size, generator=g).to(device, dt)
            ctx = torch.randn(1, 77, 3 * 256, generator=g).to(device, dt)
            return f(x, torch.full((1,), s, device=device), ctx, transformer_options={})
        return m, call, [(64, 64), (80, 48)]
    if name == "qi21":
        m = init(qi21.QwenImage21Transformer2DModel(num_layers=a.layers, num_attention_heads=H, context_in_dim=512, dtype=dt, operations=ops))
        def call(f, size, s):
            g = torch.Generator(device="cpu").manual_seed(1)
            x = torch.randn((1, 64) + size, generator=g).to(device, dt)
            ctx = torch.randn(1, 128, 512, generator=g).to(device, dt)
            return f(x, torch.full((1,), s, device=device), ctx, transformer_options={})
        return m, call, [(64, 64), (80, 48)]
    m = init(mm.MiniMaxH3Model(hidden_size=W, num_layers=a.layers, num_attention_heads=H, ffn_hidden_size=W * 8 // 3 // 128 * 128,
                               text_dim=512, time_embed_hidden_size=W, time_embed_dim=W // 2, dtype=dt, operations=ops))
    def call(f, size, s):
        g = torch.Generator(device="cpu").manual_seed(1)
        v = torch.randn((1, 24, 4) + size, generator=g).to(device, dt)
        au = torch.randn(1, 32, 2, 50, generator=g).to(device, dt)
        ctx = torch.randn(1, 128, 512, generator=g).to(device, dt)
        return f([v, au], torch.full((1,), s * 1000, device=device), ctx, transformer_options={})
    return m, call, [(32, 32), (40, 24)]


def timed(fn):
    sync()
    t = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t


print(f"torch {torch.__version__} hip {torch.version.hip} device {device} backend {a.backend} mode {a.mode} "
      f"width {W} layers {a.layers} fixes fp16={a.fix_fp16_check} prefetch={a.fix_prefetch}")
print(f"TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR', '(default: %TEMP%/torchinductor_<user>)')} "
      f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR', '(default: <inductor cache>/triton)')}")
sigmas = [0.9 - 0.8 * i / max(1, a.steps - 1) for i in range(a.steps)]
for name in a.models.split(","):
    torch._dynamo.reset()
    counters.clear()
    try:
        with torch.inference_mode():
            m, call, sizes = build(name)
            eager = [timed(lambda: call(m, sizes[0], s)) for s in sigmas][-1]
            compiled = torch.compile(m, backend=a.backend, mode=a.mode, options={"guard_filter_fn": skip_torch_compile_dict} if a.backend == "inductor" else None)
            per_size = []
            for size in sizes:
                times = [timed(lambda: call(compiled, size, s)) for s in sigmas]
                per_size.append(f"{size}: first {times[0]:.2f} s, warm {min(times[1:]) if len(times) > 1 else float('nan'):.4f} s")
            out = call(compiled, sizes[0], sigmas[-1])
            ref = call(m, sizes[0], sigmas[-1])
            pair = zip(out, ref) if isinstance(out, (list, tuple)) else [(out, ref)]
            diff = max(((o.float() - r.float()).norm() / r.float().norm()).item() for o, r in pair)
        print(f"== {name}: eager {eager:.4f} s/call | " + " | ".join(per_size) +
              f" | rel diff {diff:.2e} | graph_breaks {sum(counters['graph_break'].values())} unique_graphs {counters['stats']['unique_graphs']}")
        for reason, n in sorted(counters["graph_break"].items(), key=lambda kv: -kv[1])[:4]:
            print(f"     {n:3d} x {reason.splitlines()[0][:140]}")
    except Exception:
        print(f"== {name}: FAILED")
        traceback.print_exc(limit=6)
