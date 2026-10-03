"""End-to-end VAE decode/encode benchmark through ComfyUI's own VAE wrapper (comfy.sd.VAE).

Times decode and/or encode for the Krea 2 (Wan 2.1), Qwen-Image 2.1 and MiniMax H3 VAEs on random
latents/pixels, with the proposed changes applied as runtime patches (no tracked file is edited):

  baseline   the checkout as is
  conv2d1f   A4-1: single-frame causal conv3d runs as conv2d when cuDNN/MIOpen is off (comfy/ops.py)
  minimax1f  A4-2: MiniMax H3 single-frame encode takes the kitchen NDHWC conv (comfy/ldm/minimax/vae.py)

Reports first run (load + MIOpen Find / kernel JIT), median of --runs, peak VRAM
(torch.cuda.max_memory_allocated), ComfyUI's own memory estimate, whether it fell back to tiled, and
the output difference of each variant against baseline.

  python vae_bench.py --comfy C:\\ComfyUI --gpu 1 --model krea2  --vae C:\\ComfyUI\\models\\vae\\wan_2.1_vae.safetensors --res 1024,1536 --variants baseline,conv2d1f
  python vae_bench.py --comfy C:\\ComfyUI --gpu 1 --model qwen21 --vae ...qwen_image_2.1_vae.safetensors --res 1024 --variants baseline,conv2d1f
  python vae_bench.py --comfy C:\\ComfyUI --gpu 1 --model minimax --vae ...minimax_h3_vae.safetensors --res 1024 --op encode --variants baseline,minimax1f
  python vae_bench.py ... --model minimax --frames 85 --res 720x1280 --op both
Extra ComfyUI flags (e.g. --fp16-vae, --disable-dynamic-vram) go in --comfy-args "...". MIOpen: set
COMFYUI_ENABLE_MIOPEN=1 (and MIOPEN_FIND_MODE) in the environment, or use run_vae_matrix.py.
"""
import argparse
import json
import logging
import os
import statistics
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--comfy", required=True)
p.add_argument("--vae", required=True)
p.add_argument("--model", required=True, choices=["krea2", "qwen21", "minimax"])
p.add_argument("--res", default="1024", help="comma list; NxM = HxW")
p.add_argument("--frames", type=int, default=1)
p.add_argument("--op", default="decode", choices=["decode", "encode", "both"])
p.add_argument("--variants", default="baseline,conv2d1f,minimax1f")
p.add_argument("--runs", type=int, default=3)
p.add_argument("--gpu", default=None)
p.add_argument("--miopen-immediate", action="store_true")
p.add_argument("--comfy-args", default="")
p.add_argument("--tag", default="")
p.add_argument("--out", default=None)
a = p.parse_args()

if a.gpu is not None:
    os.environ["HIP_VISIBLE_DEVICES"] = a.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
sys.argv = [sys.argv[0]] + a.comfy_args.split()
sys.path.insert(0, a.comfy)
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ldm.minimax.vae as mmvae  # noqa: E402

if a.miopen_immediate:
    torch.backends.miopen.immediate = True

# ---- proposed changes as patches ----
Conv3d = comfy.ops.disable_weight_init.Conv3d
orig_conv_forward = Conv3d._conv_forward
orig_mm_forward = mmvae.CausalConv3d.forward


def conv2d1f_conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
    if autopad == "causal_zero" and input.shape[2] == 1 and not torch.backends.cudnn.enabled:
        weight = weight[:, :, -1:, :, :]
        return F.conv2d(input[:, :, 0], weight[:, :, 0], bias, self.stride[1:], self.padding[1:], self.dilation[1:], self.groups).unsqueeze(2)
    return orig_conv_forward(self, input, weight, bias, autopad, *args, **kwargs)


def minimax1f_forward(self, x, pre_norm=None, spatial_pad=None, residual=None):
    pad_t, pad_h, pad_w = self.causal_padding
    if spatial_pad is None:
        spatial_pad = (pad_w, pad_w, pad_h, pad_h)
    front = 0 if x.shape[2] == 1 else pad_t * 2
    ndhwc = mmvae._kitchen_ndhwc(x)
    if ndhwc and self.weight.is_cuda and not self.weight.is_contiguous(memory_format=torch.channels_last_3d):
        self.weight.data = self.weight.data.contiguous(memory_format=torch.channels_last_3d)
    if pre_norm is not None or front or any(spatial_pad):
        fused = mmvae._fused_norm_pad(x, pre_norm, spatial_pad, front)
        if fused is not None:
            x = fused
        else:
            if pre_norm is not None:
                x = F.silu(pre_norm(x), inplace=True)
            if any(spatial_pad):
                x = F.pad(x, (*spatial_pad, 0, 0), mode="reflect")
            if front:
                x = F.pad(x, (0, 0, 0, 0, front, 0), mode="constant")
    single = x.shape[2] == 1 and pad_t
    if single and not ndhwc:
        out = super(mmvae.CausalConv3d, self).forward(x, autopad="causal_zero")
    elif ndhwc:
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, x, offloadable=True)
        try:
            if single:
                weight = weight[:, :, -1:]
            weight_cl = weight.contiguous(memory_format=torch.channels_last_3d)
            out = mmvae._fp16_accum_conv(self, x, weight_cl, bias, residual)
            if out is not None:
                return out
            out = F.conv3d(x, weight_cl, bias, self.stride)
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
    else:
        out = super(mmvae.CausalConv3d, self).forward(x)
    if residual is not None:
        out += residual
    return out


def apply(variant):
    Conv3d._conv_forward = conv2d1f_conv_forward if variant == "conv2d1f" else orig_conv_forward
    mmvae.CausalConv3d.forward = minimax1f_forward if variant == "minimax1f" else orig_mm_forward


class TiledWatch(logging.Handler):
    hit = False

    def emit(self, record):
        if "tiled" in record.getMessage().lower():
            TiledWatch.hit = True


logging.getLogger().addHandler(TiledWatch())


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(fn):
    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    TiledWatch.hit = False
    t = time.perf_counter()
    out = fn()
    sync()
    dt = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return out, dt, peak, TiledWatch.hit


def main():
    dev = mm.get_torch_device()
    env = {"torch": torch.__version__, "hip": torch.version.hip, "cudnn_enabled": torch.backends.cudnn.enabled,
           "miopen_immediate": a.miopen_immediate, "device": str(dev),
           "arch": torch.cuda.get_device_properties(dev).gcnArchName if dev.type == "cuda" and torch.version.hip else None,
           "MIOPEN_FIND_MODE": os.environ.get("MIOPEN_FIND_MODE"), "COMFYUI_ENABLE_MIOPEN": os.environ.get("COMFYUI_ENABLE_MIOPEN"),
           "comfy_args": a.comfy_args, "tag": a.tag}
    print(json.dumps(env))
    sd, metadata = comfy.utils.load_torch_file(a.vae, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    print(f"VAE {type(vae.first_stage_model).__name__} dtype {vae.vae_dtype} latent_channels {vae.latent_channels}")
    out_f = open(a.out, "a") if a.out else None
    g = torch.Generator().manual_seed(0)
    for res in a.res.split(","):
        h, w = (int(v) for v in res.split("x")) if "x" in res else (int(res), int(res))
        if a.model == "krea2":
            latent = torch.randn(1, 16, 1, h // 8, w // 8, generator=g)
        elif a.model == "qwen21":
            latent = torch.randn(1, 64, h // 16, w // 16, generator=g)
        else:
            t_lat = vae.downscale_ratio[0](a.frames)
            latent = torch.randn(1, 24, t_lat, h // 16, w // 16, generator=g)
        pixels = torch.rand(a.frames, h, w, 3, generator=g)
        refs = {}
        for variant in a.variants.split(","):
            if variant == "minimax1f" and a.model != "minimax":
                continue
            apply(variant)
            for op in (["decode", "encode"] if a.op == "both" else [a.op]):
                fn = (lambda: vae.decode(latent)) if op == "decode" else (lambda: vae.encode(pixels))
                shape = latent.shape if op == "decode" else None
                est = vae.memory_used_decode(latent.shape, vae.vae_dtype) / 2**30 if op == "decode" else None
                out, first, peak0, tiled0 = measure(fn)
                times, peaks, tiled = [], [], tiled0
                for _ in range(a.runs):
                    out, dt, pk, tl = measure(fn)
                    times.append(dt)
                    peaks.append(pk)
                    tiled = tiled or tl
                diff = None
                key = op
                if key in refs:
                    ref = refs[key]
                    diff = ((out.float() - ref).norm() / ref.norm()).item()
                else:
                    refs[key] = out.float()
                row = {"model": a.model, "op": op, "res": f"{h}x{w}", "frames": a.frames, "variant": variant,
                       "first_s": round(first, 3), "median_s": round(statistics.median(times), 3),
                       "peak_gib": round(max(peaks + [peak0]), 2), "estimate_gib": None if est is None else round(est, 2),
                       "tiled": tiled, "rel_diff_vs_first_variant": diff,
                       "latent": list(shape) if shape is not None else None, "tag": a.tag,
                       "cudnn_enabled": torch.backends.cudnn.enabled}
                print(json.dumps(row))
                if out_f:
                    out_f.write(json.dumps({**env, **row}) + "\n")
    apply("baseline")
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
