import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.qwen_image21.model import ZeroCenteredRMSNorm


class _ResultDtypes(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.dtypes = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.dtypes.add(out.dtype)
        return out


def test_zero_centered_rmsnorm_stays_in_input_dtype():
    norm = ZeroCenteredRMSNorm(64, dtype=torch.bfloat16)
    norm.weight.data = torch.randn(64, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16) * 0.2
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16) * 30
    with _ResultDtypes() as mode:
        out = norm(x)
    assert mode.dtypes == {torch.bfloat16}
    ref = F.rms_norm(x.double(), (64,), weight=norm.weight.double() + 1.0, eps=1e-6)
    torch.testing.assert_close(out.double(), ref, rtol=2e-2, atol=2e-2)
