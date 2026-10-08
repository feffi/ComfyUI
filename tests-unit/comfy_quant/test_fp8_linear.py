import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy import ops
from comfy.quant_ops import QuantizedTensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestFp8Linear(unittest.TestCase):
    def test_input_quantize_matches_clamp_and_cast(self):
        bits = torch.arange(-32768, 32768, dtype=torch.int32, device=DEVICE).to(torch.int16)
        for dtype in (torch.bfloat16, torch.float16):
            x = bits.view(dtype)
            x = x[~x.isnan()]
            q = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
            want = torch.clamp(x, min=-448, max=448).to(torch.float8_e4m3fn)
            self.assertTrue(torch.equal(q._qdata.view(torch.uint8), want.view(torch.uint8)), dtype)

    def test_saturates_without_modifying_input(self):
        torch.manual_seed(0)
        lin = ops.fp8_ops.Linear(64, 32, device=DEVICE, dtype=torch.bfloat16)
        lin.weight = torch.nn.Parameter((torch.randn(32, 64, device=DEVICE) * 0.1).to(torch.float8_e4m3fn), requires_grad=False)
        lin.bias = torch.nn.Parameter(torch.randn(32, device=DEVICE, dtype=torch.bfloat16), requires_grad=False)
        x = (torch.randn(2, 5, 64, device=DEVICE) * 300).to(torch.bfloat16)
        x_before = x.clone()

        out = ops.fp8_linear(lin, x)
        self.assertTrue(torch.equal(x, x_before))
        self.assertTrue(torch.equal(out, ops.fp8_linear(lin, x.clamp(-448, 448))))


if __name__ == "__main__":
    unittest.main()
