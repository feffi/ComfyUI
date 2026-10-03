import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.mmaudio.vae import alias_free_torch
from comfy.ldm.lightricks.vocoders import vocoder as ltx_vocoder
from comfy.ldm.minimax import audio_vae as minimax_audio_vae

UPSAMPLERS = {
    "mmaudio_r2": lambda: alias_free_torch.UpSample1d(2),
    "mmaudio_r3": lambda: alias_free_torch.UpSample1d(3),
    "ltx_kaiser_r2": lambda: ltx_vocoder.UpSample1d(2, 12),
    "ltx_hann_r2": lambda: ltx_vocoder.UpSample1d(2, persistent=False, window_type="hann"),
    "ltx_hann_r3": lambda: ltx_vocoder.UpSample1d(3, persistent=False, window_type="hann"),
    "ltx_hann_r4": lambda: ltx_vocoder.UpSample1d(4, persistent=False, window_type="hann"),
    "minimax_r2": lambda: minimax_audio_vae.UpSample1d(2, 12),
}


def _transposed_reference(up, x):
    # the BigVGAN formulation the polyphase conv replaces
    C, T = x.shape[1:]
    y = F.pad(x, (up.pad, up.pad), mode="replicate")
    y = up.ratio * F.conv_transpose1d(y, up.filter.to(x.dtype).expand(C, -1, -1), stride=up.ratio, groups=C)
    return y[..., up.pad_left:up.pad_left + up.ratio * T]


@pytest.mark.parametrize("length", [1, 7, 300])
@pytest.mark.parametrize("name", list(UPSAMPLERS))
def test_upsample_matches_transposed_conv(name, length):
    up = UPSAMPLERS[name]()
    x = torch.randn(2, 5, length, dtype=torch.float64)
    out = up(x)
    assert out.shape == (2, 5, up.ratio * length)
    torch.testing.assert_close(out, _transposed_reference(up, x), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name", list(UPSAMPLERS))
def test_upsample_avoids_grouped_transposed_conv(name, monkeypatch):
    # without cuDNN/MIOpen torch runs a grouped conv_transpose1d one channel at a time
    def fail(*args, **kwargs):
        raise AssertionError("grouped conv_transpose1d used")
    monkeypatch.setattr(F, "conv_transpose1d", fail)
    monkeypatch.setattr(torch, "conv_transpose1d", fail)
    up = UPSAMPLERS[name]()
    assert up(torch.randn(1, 3, 16)).shape == (1, 3, 16 * up.ratio)


def test_upsample_keeps_state_dict():
    assert list(alias_free_torch.UpSample1d(2).state_dict()) == ["filter"]
    assert list(minimax_audio_vae.UpSample1d(2, 12).state_dict()) == ["filter"]
    assert list(ltx_vocoder.UpSample1d(2, 12).state_dict()) == ["filter"]
    assert list(ltx_vocoder.UpSample1d(2, persistent=False, window_type="hann").state_dict()) == []
