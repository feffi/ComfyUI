import sys, os
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, os.environ.get("COMFYUI_DIR", "/home/user/ComfyUI"))  # set COMFYUI_DIR to your checkout
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.ops
import comfy.ldm.krea2.model
import comfy.ldm.qwen_image21.model
import comfy.ldm.minimax.model
import comfy.text_encoders.qwen3vl as q3

ops = comfy.ops.disable_weight_init
GiB = 1024 ** 3


def count(m):
    by_dtype = {}
    total = 0
    linear = 0
    for name, p in list(m.named_parameters()) + list(m.named_buffers()):
        by_dtype[p.dtype] = by_dtype.get(p.dtype, 0) + p.numel()
        total += p.numel()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.ndim == 2:
            linear += mod.weight.numel()
    return total, linear, by_dtype


def report(name, m):
    total, linear, by_dtype = count(m)
    bf16 = sum(n * (4 if d == torch.float32 else 2) for d, n in by_dtype.items())
    # fp8 scaled: Linear weights 1 byte, the rest stays as stored
    fp8 = bf16 - linear * 1
    print(f"{name:28s} params={total/1e9:6.2f}B linear={linear/1e9:6.2f}B bf16={bf16/GiB:6.2f}GiB fp8(linears)={fp8/GiB:6.2f}GiB dtypes={ {str(k): round(v/1e6,1) for k,v in by_dtype.items()} }")


with torch.device("meta"):
    report("Krea2 SingleStreamDiT", comfy.ldm.krea2.model.SingleStreamDiT(device="meta", dtype=torch.bfloat16, operations=ops))
    report("QwenImage21 DiT", comfy.ldm.qwen_image21.model.QwenImage21Transformer2DModel(device="meta", dtype=torch.bfloat16, operations=ops))
    report("MiniMaxH3 DiT (defaults)", comfy.ldm.minimax.model.MiniMaxH3Model(device="meta", dtype=torch.bfloat16, operations=ops))
    for t in ["qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b"]:
        cls = q3._make_qwen3vl_model(t)
        m = cls({}, dtype=torch.bfloat16, device="meta", operations=ops)
        report(f"TE {t}", m)
        report(f"TE {t} (no vision)", m.model)
