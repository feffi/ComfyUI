import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.wan import vae2_2
from comfy.ldm.wan.vae import CausalConv3d


def _randomize(module, seed=0):
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.2
    return module


@pytest.mark.parametrize("kernel,stride,groups", [((3, 3, 3), (1, 1, 1), 1), ((3, 3, 3), (1, 2, 2), 2), ((1, 1, 1), (1, 1, 1), 1)])
def test_single_frame_causal_conv3d_runs_as_conv2d(kernel, stride, groups, monkeypatch):
    conv = _randomize(CausalConv3d(4, 6, kernel, stride=stride, padding=tuple(k // 2 for k in kernel), groups=groups))
    x = torch.randn(2, 4, 1, 9, 10)
    reference = conv(x)  # cuDNN flag on: the conv3d path

    calls = {"conv2d": 0}
    conv2d = F.conv2d

    def counting_conv2d(*a, **kw):
        calls["conv2d"] += 1
        return conv2d(*a, **kw)

    def no_conv3d(*a, **kw):
        raise AssertionError("conv3d used for a single frame with cuDNN off")

    monkeypatch.setattr(F, "conv2d", counting_conv2d)
    monkeypatch.setattr(F, "conv3d", no_conv3d)
    with torch.backends.cudnn.flags(enabled=False):
        out = conv(x)
    assert calls["conv2d"] == 1
    assert out.shape == reference.shape
    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-5)


def _tiny_wan22_vae():
    vae = vae2_2.WanVAE(dim=8, dec_dim=8, z_dim=4, dim_mult=[1, 2, 2], num_res_blocks=1,
                        temperal_downsample=[False, True], image_channels=3, patch_size=2)
    return _randomize(vae).eval()


def test_wan22_single_image_decoder_head_runs_in_strips(monkeypatch):
    vae = _tiny_wan22_vae()
    z = torch.randn(1, 4, 6, 5)
    full = vae.decode(z)

    heights = []
    head_conv = vae.decoder.head[-1]
    handle = head_conv.register_forward_pre_hook(lambda m, inp: heights.append(inp[0].shape[-2]))
    monkeypatch.setattr(vae2_2, "STRIP_ELEMS", 2 ** 10)
    stripped = vae.decode(z)
    handle.remove()

    full_height = full.shape[-2] // vae.patch_size
    assert len(heights) > 1
    assert max(heights) < full_height
    torch.testing.assert_close(stripped, full, rtol=0, atol=0)


def test_wan22_single_image_decode_matches_video_path():
    vae = _tiny_wan22_vae()
    z = torch.randn(1, 4, 6, 5)
    torch.testing.assert_close(vae.decode(z), vae.decode(z.unsqueeze(2)).squeeze(2), rtol=1e-5, atol=1e-5)
