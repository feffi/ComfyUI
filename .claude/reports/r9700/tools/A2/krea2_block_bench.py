"""Time one Krea 2 SingleStreamBlock at real width: installed model file vs branch (d0e865c) vs fused.

Run from the ComfyUI folder (it imports comfy from the current directory):
  python <path>\\krea2_block_bench.py --device 1 --tokens 4608 --batch 1
  python <path>\\krea2_block_bench.py --device 1 --tokens 4608 --batch 2    (CFG batch)

installed = comfy/ldm/krea2/model.py of this ComfyUI install
branch    = krea2_branch.py next to this script (native GQA, no fp32 RMSNorm upcast, no einops)
fused     = krea2_fused.py next to this script (branch + rms_adaln modulation, addcmul residual, fused rms_rope)
Same random weights for all three. Prints ms per block (median of 3 x 20 calls) and the output difference
against an fp32 run of the installed block. 28 blocks per model call at the default config.
"""
import argparse
import importlib.util
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--tokens", type=int, default=4608, help="text + image tokens (1024x1024 is 4096 image tokens)")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--features", type=int, default=6144)
ap.add_argument("--heads", type=int, default=48)
ap.add_argument("--kvheads", type=int, default=12)
a = ap.parse_args()

os.environ.setdefault("HIP_VISIBLE_DEVICES", str(a.device))
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.ops  # noqa: E402
import comfy.ldm.krea2.model as installed  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mods = {"installed": installed, "branch": load("krea2_branch"), "fused": load("krea2_fused")}
dev = torch.device("cuda")
D, H = a.features, a.heads
headdim = D // H
axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]


def block(mod, dtype, sd=None):
    b = mod.SingleStreamBlock(D, H, 4, False, a.kvheads, device=dev, dtype=dtype, operations=comfy.ops.disable_weight_init)
    if sd is None:
        g = torch.Generator(device=dev).manual_seed(0)
        for n, p in sorted(b.named_parameters()):
            p.data = (torch.randn(p.shape, device=dev, generator=g) * (0.02 if p.ndim > 1 else 0.1)).to(dtype)
    else:
        b.load_state_dict({k: v.to(dtype) for k, v in sd.items()})
    return b


g = torch.Generator(device=dev).manual_seed(1)
x32 = torch.randn(a.batch, a.tokens, D, device=dev, generator=g)
vec32 = torch.randn(a.batch, 1, 6 * D, device=dev, generator=g) * 0.1
side = int(a.tokens ** 0.5)
pos = torch.stack(torch.meshgrid(torch.zeros(1), torch.arange(side, dtype=torch.float32), torch.arange(a.tokens // side + 1, dtype=torch.float32), indexing="ij"), -1).reshape(-1, 3)[:a.tokens]
freqs = installed.EmbedND(dim=headdim, theta=1000, axes_dim=axes)(pos[None].repeat(a.batch, 1, 1).to(dev).float())

ref_block = block(installed, torch.float32)
sd = {k: v.detach().clone() for k, v in ref_block.state_dict().items()}
ref = ref_block(x32, vec32, freqs, None, transformer_options={}).float()
del ref_block

print(f"torch {torch.__version__} hip {torch.version.hip} {torch.cuda.get_device_name()}  B={a.batch} N={a.tokens} D={D} heads={H}/{a.kvheads}")
for name, mod in mods.items():
    b = block(mod, torch.bfloat16, sd)
    x, vec = x32.bfloat16(), vec32.bfloat16()
    out = b(x, vec, freqs, None, transformer_options={})
    err = ((out.float() - ref).norm() / ref.norm()).item()
    times = []
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(20):
            b(x, vec, freqs, None, transformer_options={})
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 20)
    torch.cuda.reset_peak_memory_stats()
    b(x, vec, freqs, None, transformer_options={})
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{name:10} {sorted(times)[1]:8.2f} ms/block  rel_err vs fp32 {err:.2e}  peak {peak:.2f} GiB")
    del b, out
    torch.cuda.empty_cache()
