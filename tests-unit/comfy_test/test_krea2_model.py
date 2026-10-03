import einops
import pytest
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.common_dit
import comfy.model_management
import comfy.ops
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.krea2 import model as krea2

ops = comfy.ops.disable_weight_init


def _randomize(module, seed=0):
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.2
    return module


class _ResultDtypes(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.dtypes = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.dtypes.add(out.dtype)
        return out


def _freqs(tokens, headdim=16):
    pos = torch.stack(torch.meshgrid(torch.zeros(1), torch.arange(2.0), torch.arange(tokens / 2), indexing="ij"), -1).reshape(1, -1, 3)
    return EmbedND(dim=headdim, theta=1000, axes_dim=[4, 6, 6])(pos)


def _reference_attention(attn, x, freqs):
    def norm(t, scale):
        return F.rms_norm(t, (t.shape[-1],), weight=scale + 1.0, eps=1e-5)
    q = attn.wq(x).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
    k = attn.wk(x).unflatten(-1, (attn.kvheads, -1)).transpose(1, 2)
    v = attn.wv(x).unflatten(-1, (attn.kvheads, -1)).transpose(1, 2)
    q, k = norm(q, attn.qknorm.qnorm.scale), norm(k, attn.qknorm.knorm.scale)
    q, k = apply_rope(q, k, freqs)
    rep = attn.heads // attn.kvheads
    out = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1))
    return attn.wo(out.transpose(1, 2).flatten(2) * torch.sigmoid(attn.gate(x)))


def test_rmsnorm_stays_in_input_dtype():
    norm = _randomize(krea2.RMSNorm(64)).to(torch.bfloat16)
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16) * 30
    with _ResultDtypes() as mode:
        out = norm(x)
    assert mode.dtypes == {torch.bfloat16}
    ref = F.rms_norm(x.double(), (64,), weight=norm.scale.double() + 1.0, eps=1e-5)
    torch.testing.assert_close(out.double(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("use_patch", [False, True])
def test_attention_passes_unexpanded_gqa_kv(use_patch, monkeypatch):
    attn = _randomize(krea2.Attention(64, 4, kvheads=2, operations=ops))
    x = torch.randn(2, 12, 64)
    freqs = _freqs(12)

    seen = {}
    attention = krea2.optimized_attention_masked

    def capture(q, k, v, heads, **kwargs):
        seen["k_heads"] = k.peek().shape[1]
        seen["enable_gqa"] = kwargs.get("enable_gqa", False)
        return attention(q, k, v, heads, **kwargs)

    monkeypatch.setattr(krea2, "optimized_attention_masked", capture)
    options = {}
    if use_patch:  # attn1_patch keeps the unfused norm/patch/rope order
        options = {"block_index": 0, "patches": {"attn1_patch": [lambda q, k, v, **kw: {}]}}
    out = attn(x, freqs, transformer_options=options)
    assert seen == {"k_heads": 2, "enable_gqa": True}
    torch.testing.assert_close(out, _reference_attention(attn, x, freqs), rtol=1e-4, atol=1e-5)


def _tiny_dit(features=64, heads=4, kvheads=2, layers=2):
    return _randomize(krea2.SingleStreamDiT(features=features, tdim=32, txtdim=32, heads=heads, kvheads=kvheads,
                                            multiplier=2, layers=layers, patch=2, channels=4, txtlayers=3,
                                            txtheads=2, txtkvheads=2, operations=ops))


def test_process_img_matches_einops_patchify():
    dit = _tiny_dit()
    x = torch.randn(2, 4, 7, 9)
    img, img_ids, h, w = dit.process_img(x)
    padded = comfy.ldm.common_dit.pad_to_patch_size(x, (2, 2))
    assert (h, w) == (4, 5)
    torch.testing.assert_close(img, einops.rearrange(padded, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2), rtol=0, atol=0)
    assert img_ids.shape == (2, 20, 3)


def test_forward_unpatchify_inverts_patchify():
    # features == channels * patch**2, no blocks, identity first/last: forward is unpatchify(patchify(x))
    dit = _tiny_dit(features=16, heads=1, kvheads=1, layers=0)
    dit.first = torch.nn.Identity()
    dit.last.forward = lambda x, t: x
    x = torch.randn(2, 4, 7, 9)
    out = dit(x, torch.tensor([0.5, 0.5]), torch.randn(2, 5, 3 * 32))
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_fused_kernels_match_unfused_path(monkeypatch):
    dit = _tiny_dit()
    x = torch.randn(2, 4, 6, 8)
    t = torch.tensor([0.3, 0.7])
    context = torch.randn(2, 5, 3 * 32)
    fused = dit(x, t, context)
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    unfused = dit(x, t, context)
    torch.testing.assert_close(fused, unfused, rtol=1e-4, atol=1e-5)
