import pytest
import torch

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp16_linear_check_has_no_graph_break(dtype):
    # reading the cuBLAS fp16 accumulation flag first broke the graph in every linear_input_act
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: x * 2 if comfy.ops._fp16_linear_wanted(x) else x + 1, backend="eager", fullgraph=True)
    x = torch.ones(4, dtype=dtype)
    torch.testing.assert_close(compiled(x), x + 1)
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the SDPA backend priority wrapper only exists with CUDA/ROCm")
def test_unmasked_gqa_without_native_backend_expands_kv(monkeypatch):
    # fp32 GQA: flash takes only fp16/bf16 and mem-efficient rejects GQA, so native GQA would fall to the math kernel
    q = torch.randn(1, 8, 512, 64, device="cuda")
    k = torch.randn(1, 2, 512, 64, device="cuda")
    v = torch.randn(1, 2, 512, 64, device="cuda")
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, True)
    native = (torch.backends.cuda.can_use_flash_attention(params) or torch.backends.cuda.can_use_cudnn_attention(params)
              or torch.backends.cuda.can_use_efficient_attention(params))
    seen = {}
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def capture(q, k, v, *a, **kw):
        seen["k_heads"] = k.shape[-3]
        seen["enable_gqa"] = kw.get("enable_gqa", False)
        return sdpa(q, k, v, *a, **kw)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    out = comfy.ops.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert seen == ({"k_heads": 2, "enable_gqa": True} if native else {"k_heads": 8, "enable_gqa": False})
    torch.testing.assert_close(out, sdpa(q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)), rtol=1e-4, atol=1e-4)
