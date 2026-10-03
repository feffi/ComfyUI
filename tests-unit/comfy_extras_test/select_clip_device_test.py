import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sd  # noqa: E402
from comfy_extras import nodes_multigpu  # noqa: E402


def _patcher():
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(torch.nn.Linear(2, 2), load_device=cpu, offload_device=cpu)


def _clip():
    clip = comfy.sd.CLIP(no_init=True)
    clip.patcher = _patcher()
    clip.cond_stage_model = clip.patcher.model
    clip.tokenizer = None
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    return clip


def test_select_clip_device_encodes_with_the_retargeted_model(monkeypatch):
    # moving to another GPU deepclones the model; encode runs cond_stage_model, so it has to follow
    retargeted = _patcher()
    monkeypatch.setattr(comfy.model_management, "resolve_gpu_device_option", lambda option: torch.device("cuda", 1))
    monkeypatch.setattr(nodes_multigpu, "_apply_patcher_device", lambda patcher, resolved: retargeted)
    clip = _clip()
    original_model = clip.cond_stage_model

    out = nodes_multigpu.SelectCLIPDeviceNode.execute(clip, "gpu:1").result[0]

    assert out.patcher is retargeted
    assert out.cond_stage_model is retargeted.model
    assert clip.cond_stage_model is original_model
