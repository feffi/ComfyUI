# ComfyUI on 2× Radeon AI PRO R9700: analysis report

Stand: 03.10.2026. Branch `claude/cool-euler-0g3imw`. Produced by the prompt in `.claude/prompts/amd-r9700-analysis.md`: 8 analysis agents, 2 verifiers (code and claims), synthesis.

**Read this first.** All analysis ran in a Linux cloud container without a GPU. Nothing here was timed on an R9700. "Measured" means measured in that container: CPU numerics, CPU import timing, disassembly of the comfy-kitchen gfx1201 code objects, and PyTorch/HIP source reads. Every GPU speed or VRAM effect is an estimate until you run the tools in `.claude/reports/r9700/tools/`.

Target: Windows, Adrenalin 26.8.1, TheRock ROCm 10.0 wheels (HIP 7.15), torch 2.13.0+rocm10.0.0, comfy-kitchen 0.2.36, comfy-aimdo 0.5.5. Workloads: Krea 2, Qwen-Image 2.1, MiniMax H3.

## 1. Top 10

| # | Action | Bucket | Models | Effect, confidence | Status |
|---|---|---|---|---|---|
| 1 | Check which attention backend your startup log reports: `Using pytorch attention` or `Using sub quadratic optimization` | launch | all | ComfyUI#16526 reports the AOTriton probe failing on this exact torch/HIP build, which silently switches every model to sub-quadratic attention. Potentially the largest single factor; high confidence that it must be checked | **do now** |
| 2 | Test for idle VRAM page-out on 26.8.1, soak-test Adrenalin 26.9.2 Optional | system | all | AMD KMD bug on 26.5.1–26.8.1 (incl. the "validated" 26.7.1) pages live HIP allocations to RAM after ~9.5 s idle (TheRock#7221). Reload stalls, BSOD reports at 32 GB RAM. Fixed in 26.9.2 Optional (32.0.32015.2008), not yet validated with ROCm 10.0 | **do now** |
| 3 | Make GPU 1 usable: both cards visible, text encoder really on GPU 1 | core | all, H3 most | GPU 1 was hidden by the Windows NVIDIA workaround, and Select CLIP Device encoded with the original model. High confidence (source + CPU repro) | done: `52ed61a`, `2ed15ee` |
| 4 | MiniMax H3 VAE encode through the kitchen HIP conv | core | H3 | Removes a ~7.7 GB im2col buffer per 256 px × 17-frame tile and ~300k launches per 1024² keyframe. Kernel accumulates in fp32 (disassembly) | done: `85d2562`, `da79103`, `84cfd41` |
| 5 | Replace SolAttn_triton with core **Model Sparse Attention** (method sol) | extension | H3 | SolAttn_triton is deprecated (26d816e is the deprecation commit), Triton JIT, NVIDIA-tested only; core node runs precompiled HIP `sol_attn`. Speed unmeasured | A/B, then remove |
| 6 | Conv fallbacks with MIOpen off: single-frame VAE convs as conv2d, Qwen 2.1 decoder head in strips, polyphase audio upsample | core | Krea 2, Qwen 2.1, H3 (video bf16, audio) | Removes ~8.7k (Krea 2) / ~68.6k (Qwen 2.1) bias-fill launches per 1024² decode; Qwen head scratch 2.5 GiB → 0.3 GiB at 1024²; 18,296 per-channel transposed convs per H3 audio decode (same loop in the LTX and MMAudio vocoders). Output identical within 1e-6 | done: `ded5499`, `ff0e009`, `91d15cb` |
| 7 | Krea 2 attention, norms and modulation | core | Krea 2 | Native GQA (no 4× K/V copies), no fp32 norm copies, GQA guard for fp32/no-flash, fused RMS AdaLN and RMS RoPE kernels (~3 % per step, est.) | done: `d0e865c`, `d7b3369`, `377a3f9` |
| 8 | No bare `--fast`, no `--fast fp16_accumulation` | launch | Krea 2 | torch 2.13 ignores `allow_fp16_accumulation` on ROCm (verified in `CUDABlas.cpp`). The flag only moves Krea 2 to fp16, which has no clamp. No speed gain, NaN risk | **do now** |
| 9 | One ComfyUI instance per card for Krea 2 / Qwen 2.1 | launch | Krea 2, Qwen 2.1 | Templates run cfg=1, so CFG Split leaves GPU 1 idle. Two queues ≈ 2× images/hour (est.) | try |
| 10 | Startup: `--disable-partner-nodes`, Manager offline, `.pyc`/Defender | launch/system | all | −0.77 s measured (Linux CPU) for partner nodes; Manager pip probes cost 1–1.5 s on Linux, more on Windows (est.) | try |

## 2. Recommended launch configuration

Cherry-pick the branch commits (section 4) first. Without them, add `--cuda-device all` to any instance that should see both cards.

```
.venv-rocm-100\Scripts\python.exe main.py --disable-partner-nodes
```

| Setting | Status | Why |
|---|---|---|
| `--disable-partner-nodes` | measured (Linux CPU, −0.77 s) | Skips 42 paid-API node modules. Drop it if you use API nodes |
| no `--fast` of any kind | verified in source | see Top 10 #8; `--fast autotune` only sets `cudnn.benchmark`, which does nothing with MIOpen off |
| no `--use-ck-attention`, `--use-sage-attention`, `--enable-triton-backend` | verified in source / open issues | kitchen INT8 attention has open corruption reports on Windows RDNA3 with the same kernel source (#226, #230); Sage is a trial; Triton backend adds nothing the HIP backend lacks and has gfx12 `num_stages` crash reports |
| keep DynamicVRAM and the comfy compiler (defaults) | unmeasured | `--highvram`, `--gpu-only`, `--novram` silently disable DynamicVRAM. A/B `--disable-dynamic-vram` only if you see corruption or slowdowns (open: #15993, #16437, #16502) |
| `--disable-cuda-graphs` only if Generate Text misbehaves | verified in source | graph capture only runs in autoregressive decode (Generate Text), never in CLIPTextEncode or the three DiTs |
| no `COMFYUI_ENABLE_MIOPEN` | open upstream issues | Windows MIOpen gets workspace=0 and picks naive kernels (TheRock#3077), find results are not cached (#7958); decide only with `run_vae_matrix.py` |
| no environment variables by default | n/a | A/B candidates: `PYTORCH_TUNABLEOP_ENABLED=1` + `PYTORCH_TUNABLEOP_FILENAME=<abs>\tunableop_results%d.csv` (bf16 Krea 2 / Qwen 2.1 only; first run per shape is slow); `GPU_MAX_HW_QUEUES=2` (Linux evidence only) |
| ComfyUI-Manager: `user\__manager\config.ini`, `[default] network_mode = offline` | verified in Manager 3.42 source | stops startup channel/registry fetches. Set `public` when installing nodes. The old `user\default\ComfyUI-Manager\config.ini` is migrated once and ignored afterwards; the startup line `** ComfyUI-Manager config path:` shows the live file |
| Manager `use_uv = True` | measured (Linux: prestartup 1.5 s → 0.45 s) | only if `python -m uv --version` works in the venv; Manager disables uv on Windows by default |

**Driver.** Until 26.9.2 is soak-tested: load a model, idle 60 s, watch "Dedicated GPU memory" on both cards in Task Manager, then run again. If memory drops and the next run stalls, you are hit. `tools/A6/idle_pageout_check.py --gib 8 --idle 20 --devices 0,1` measures it.

## 3. Two-GPU plan

**MiniMax H3** (one instance, both cards visible):
- GPU 0: the pruned int8 DiT (`minimax_h3_*_pruned_int8_convrot`, ~19.6 GiB). The full DiT is 33.1B parameters (61.7 GiB bf16) and does not fit one card.
- GPU 1: the Qwen3-VL-32B text encoder through **Select CLIP Device → gpu:1**, plus both VAEs through **Select VAE Device → gpu:1**. Use the int8_convrot encoder (25.3 GiB, native HIP int8 kernels) or nvfp4_awq (14.6 GiB, but NVFP4 runs on kitchen's eager dequant path on RDNA4). The bf16 encoder is 48 GiB and does not fit.
- Needs `52ed61a` and `2ed15ee`. Verify: GPU 1 memory rises by the encoder size during encode, the DiT is not unloaded between prompts (`tools/A3/run_placement_bench.py --model minimax --placement single|split`).
- Attention: Model Sparse Attention (sol) on the model.

**Krea 2 and Qwen-Image 2.1** (throughput):
```
main.py --cuda-device 0 --port 8188
main.py --cuda-device 1 --port 8189 --database-url sqlite:///C:/<ComfyUI>/user/comfyui_gpu1.db
```
Each instance may pin up to 40 % of RAM; with less than ~64 GB RAM add `--disable-pinned-memory` to the second one. Use **MultiGPU CFG Split** only for workflows with cfg > 1 (`e9c38f9` fixes its batching under DynamicVRAM).

**Do not** share a card with another GPU process during runs: HIP on Windows likely does not count other processes in free memory, so WDDM pages to RAM instead of raising OOM (unverified, `tools/A3/vram_probe.py` tests it).

## 4. Core changes

### On the branch (verified by V1, CPU-tested)

All ten core proposals that survived verification are applied, plus the audio VAE follow-up `91d15cb`. Cherry-pick in this order onto your install (8cfe5e1e); the branch is based on e9027f2, so expect conflicts where your checkout differs:
```
git cherry-pick d0e865c 85d2562 da79103 dd85c0c 52ed61a 2ed15ee 84cfd41 ff0e009 ded5499 d7b3369 377a3f9 e9c38f9 e997bcb 3781fb9 91d15cb
```

| Commit | Change | CPU check |
|---|---|---|
| `d0e865c` | Krea 2: native GQA (`enable_gqa`), torch reshapes, RMSNorm without fp32 upcast; sparse override expands GQA itself | fp32 bit-identical; bf16 err vs fp32 6.0e-3 → 6.4e-3 |
| `d7b3369` | SDPA wrapper checks native GQA support for unmasked calls too (fixes MATH fallback from `d0e865c` in fp32 / no-flash) | V1: expanded vs native bit-identical, ~1.2 µs/call |
| `377a3f9` | Krea 2: `ck.rms_adaln` for norm+(1+scale)+shift, `addcmul` gated residuals, `ck.rms_rope` for QK-norm+RoPE; training and `attn1_patch` keep the unfused order | fp32 ≤ 2.5e-7 vs previous; bf16 err vs fp32 6.4–6.7e-3 → 6.0–6.4e-3 |
| `85d2562`, `da79103` | MiniMax H3 VAE encoder: kitchen NDHWC fused norm/pad + HIP `fp16_conv3d`, default on RDNA3+ | fp16 paths equal error vs fp32 (1.49e-3); kernels use `v_wmma_f32_16x16x16_f16` |
| `84cfd41` | MiniMax H3 single-frame encode uses the kitchen conv on HIP | 11/11 convs to kitchen, error unchanged |
| `dd85c0c` | Qwen 2.1 text norm without fp32 upcast | fp32 identical, bf16 1.66e-3 → 1.85e-3 |
| `ff0e009` | Wan 2.2 / Qwen 2.1 decoder head in strips (single image) | bit-identical |
| `ded5499` | Single-frame causal conv3d as conv2d with cuDNN off | ≤ 8.7e-7 fp32, bf16 identical |
| `52ed61a` | No forced single-GPU mode for ROCm on Windows | condition truth table |
| `2ed15ee` | Select CLIP Device encodes with the retargeted model | repro 73/0 → 0/73 Linear calls |
| `e9c38f9` | MultiGPU CFG Split scheduler counts DynamicVRAM-evictable memory like the single-GPU path | identical to before without aimdo |
| `e997bcb` | Bare `try/except: pass` around the AMD init block removed; AOTriton probe skipped when `--use-pytorch-cross-attention` is set | simulator: healthy start unchanged; missing arch and Ctrl+C now surface; probe 1× → 0× with the flag |
| `3781fb9` | `_fp16_linear_wanted` checks dtype before the cuBLAS getter, removing a torch.compile graph break per `linear_input_act` | `fullgraph=True`: break before, single graph after (bf16 and fp16) |
| `91d15cb` | Anti-aliased audio upsample (MMAudio, LTX vocoder incl. Hann resampler, MiniMax H3 audio VAE) as one shared polyphase depthwise conv1d instead of a grouped transposed conv, which torch runs one channel at a time with MIOpen off | fp32 ≤ 2.5e-7, bf16 identical (Hann ratio 4: 1.3e-5); per-channel transposed convs 512 → 0 for 512 channels; state dict unchanged; 6 MiniMax tests pass |

### GPU checks still owed

The kitchen HIP kernels and multi-GPU paths behind these commits never ran here. In order of risk:

| Commit | Check on the R9700 | If it fails |
|---|---|---|
| `377a3f9` | `tools/A2/krea2_block_bench.py --device 1 --tokens 4608` with `--batch 1` and `--batch 2`: fused `rel_err` near the CPU numbers above, then image diff and s/it | `git revert 377a3f9` |
| `da79103`, `84cfd41` | MiniMax H3 encode of a 17+ frame clip and of a single keyframe: time and peak VRAM against the previous build (`tools/A4/vae_bench.py --model minimax --op encode`) | revert both |
| `2ed15ee`, `52ed61a` | H3 with Select CLIP Device → gpu:1: GPU 1 memory rises by the encoder size during encode, encode time drops (`tools/A3/run_placement_bench.py --model minimax --placement split`) | `--cuda-device all` as workaround for `52ed61a` |
| `ded5499`, `ff0e009` | VAE decode at 1024² and 2048² for Krea 2 and Qwen 2.1: time and peak VRAM (`tools/A4/vae_bench.py`) | revert |
| `91d15cb` | MiniMax H3 audio decode of a 10 s and a 60 s clip in both builds: time and waveform diff; LTX audio if you use it | revert |
| `e9c38f9` | CFG Split with cfg > 1 and batch 2: s/it (`run_placement_bench.py --cfg 4 --cfg-split`) | revert |
| `e997bcb` | startup log on both cards still reports the expected attention line | revert |
| `3781fb9` | only with TorchCompileModel (`tools/A5/ab_compile.py --variants baseline,torch_compile`) | none needed for eager runs |

Known limits:
- `da79103` relies on the HIP kernel accumulating in fp32. kitchen documents `fp16_conv3d` as fp16-accumulate, so this is fork-only under AGENTS.md. Re-check the disassembly when you bump comfy-kitchen.
- The `da79103` gate (`amd_min_version(..., 3)`) also matches gfx1170/1171, which kitchen 0.2.36 does not ship kernels for. Irrelevant on your cards.
- `--default-device` still does nothing on Windows ROCm (HIP on PAL ignores device order). Use `--cuda-device N`.
- The Wan 2.1 VAE head (Krea 2) is not stripped; ~1.7 GB columns at 1024² bf16, within its estimate.
- `torch._dynamo.explain` does not report the graph break `3781fb9` removes; use `fullgraph=True` to see it.
- Audio: the 42 dilated convs per H3 audio decode still cost 12,192 per-channel bias fills with MIOpen off (7,936 more per reference-audio encode). Measure their share before fixing.

### Upstream items worth filing

- comfy-kitchen: document per-backend accumulation (HIP fp16 conv/GEMM accumulate in fp32), bf16 HIP conv3d (would move Krea 2 / Qwen VAEs off torch's fallback), `sol_attn` with Lq≠Lk and native GQA (Qwen 2.1 sparse), hardware fp8 convert in the HIP quantize kernel (`v_cvt_pk_fp8_f32`), kitchen#184 (HIP sol paired query blocks, +18–20 % on gfx1201).
- PyTorch: `slow_conv_dilated` writes bias with one kernel per output channel and builds columns for 1×1 (`NaiveDilatedConvolution.cu`); port the `slow_conv2d` behaviour. Covers multi-frame video VAE decode, which `ded5499` does not. Grouped convs on the slow backends (`SlowTranspose2d`, `Slow2d`, `SlowDilated*`) run one conv per group (`Convolution.cpp` group loop); `91d15cb` avoids it for the audio upsample only.
- ROCm: HIP on PAL ignores `HIP_VISIBLE_DEVICES` order; AOTriton 0.14 gfx1201 kernels landed in release/2.13 after the 10.0.0 wheels.

## 5. Startup

Measured in the container (Linux, CPU, warm, median of 6): **7.9–8.2 s** to the "To see the GUI go to" line. Cold file cache 10.6 s, no `.pyc` 25.8 s.

| Cost | ms | Source |
|---|---|---|
| `import torch` | 1473 | via `comfy.utils` |
| `torch._dynamo` + sympy + triton | ~1280 | comfy-kitchen and torchvision both import it, so no single fix |
| transformers | 1277 | `comfy/sd1_clip.py` |
| partner/API nodes | 781 | `nodes.py`, removable with `--disable-partner-nodes` |
| comfy_extras (140 files) | 689 | spandrel 295, kornia ~200 |

Windows-only costs, not measurable here (the profiler times each one): TheRock preloads 13 ROCm DLLs on `import torch` (MIOpen included, although it is disabled), HIP init on both cards, the 130 MB kitchen HIP extension, the AOTriton probe launch, Defender scans of ~5,000 `.pyc` files and the DLLs.

Actions, in order of expected value:
1. Manager offline (and `use_uv` if uv is present): 1–6 s on Windows (est.).
2. `--disable-partner-nodes`: −0.77 s (measured, Linux).
3. After each update: `python -m compileall -q -j0 .venv-rocm-100\Lib\site-packages <ComfyUI>` or `UV_COMPILE_BYTECODE=1`; a missing `.pyc` cache cost +17.9 s here.
4. Your decision: Dev Drive or Defender exclusions for the venv, models and `%USERPROFILE%\.triton`. Trades malware scanning for speed.

Run `tools/A7/profile_startup.ps1 -Label cold` after a reboot and `-Label warm` afterwards to get the real Windows numbers.

## 6. Extensions

| Extension | Verdict | Reason |
|---|---|---|
| SolAttn_triton (installed) | **skip**, remove after A/B | deprecated by its author; core Model Sparse Attention + kitchen HIP `sol_attn` covers gfx1201. Check first: `python -c "import torch,comfy_kitchen as ck;print(ck.sol_attn_is_available(torch.device('cuda',0)))"` |
| Core Model Sparse Attention (sol) | **try** on H3 | precompiled HIP kernels; keep audio/conditioning rows exact with the sink option |
| [SageAttention-RDNA4 v0.2.0](https://github.com/IxMxAMAR/SageAttention-RDNA4) | **try** in a cloned venv, Krea 2 only | wheel built for exactly your stack (cp312, torch 2.13.0+rocm10.0.0, gfx1201); upstream reports 2.567 → 2.202 s/step for Krea 2 at 2 MP on RX 9070. Qwen 2.1 prefix attention lands on the PR #368 fallback (black-image reports). 1 day old, images change |
| Kitchen INT8 attention (Model Attention Backend node) | **try** only on kitchen 0.2.37, per model, never `--use-ck-attention` | open corruption reports #226, #230 on the shared kernel; check PSNR at >150 token prompts |
| ComfyUI-Manager 3.42 | keep, set offline | see section 2 |
| KJNodes | try, H3 previews only | its H3 Sage patch corrupts output above ~200k tokens (#763) |
| INT8-Fast-ROCM, patientx-cfz fork, rocm-ninodes, TeaCache, MagCache, H3 cache packs, city96 GGUF, pollockjj MultiGPU, buqi H3 multigpu, Nunchaku, TensorRT, MIGraphX | skip | core covers it, no support for these models, Windows unsupported, or needs NCCL/CUDA |

Core EasyCache on H3: a third-party measurement found `end_percent` ≤ 0.70 gives 1.14× without artefacts, the default 0.95 gives 1.44× with artefacts.

## 7. Benchmark protocol

`.claude/prompts/r9700-measurement-run.md` runs this whole protocol on the R9700 machine and returns a keep/revert verdict per commit. Run against a separate instance (`--port 8199`), on GPU 1 when GPU 0 is busy. 3 runs, report the median. Record driver, ROCm wheel, HIP, torch, comfy-kitchen, comfy-aimdo and ComfyUI commit with every number (`tools/A6/a6_stability_check.py` prints them).

| Question | Tool |
|---|---|
| Driver, TDR, device order, AOTriton probe reason, versions vs requirements | `tools/A6/a6_system_check.ps1` (runs `a6_stability_check.py`) |
| Idle VRAM page-out | `tools/A6/idle_pageout_check.py --gib 8 --idle 20 --devices 0,1` |
| NaN / flat output across drivers or `--fast` variants | `tools/A6/flat_output_check.py <output dir>` |
| Attention backends per real shape (SDPA native/expanded, flash/efficient/math, kitchen int8, sol, VAE slice) | `tools/A1/attn_bench.py --device 1 --explain --profile` |
| bf16 vs fp8 vs int8 linears, hipBLASLt vs rocBLAS, TunableOp | `tools/A2/gemm_bench.py --model <file> --tokens <n> --device 1` |
| Top kernels per denoising step | `tools/A2/comfy_profile_step` (custom node; load via `extra_paths_profile.yaml` in the test instance only) |
| Krea 2 fused kernels (`377a3f9`) | `tools/A2/krea2_block_bench.py --device 1 --tokens 4608` |
| Model placement across cards, per-GPU peak VRAM, evictions | `tools/A3/run_placement_bench.py --model minimax\|krea2\|qi21 --placement single\|split` |
| P2P, host bandwidth, free-memory visibility across processes, pinnable RAM | `tools/A3/vram_probe.py` |
| VAE decode/encode with MIOpen off/on/FAST/immediate, cold vs warm find-db | `tools/A4/run_vae_matrix.py --gpu 1 --res 512` first (TDR risk), then 1024 |
| Per conv shape: torch vs conv2d vs kitchen vs MIOpen | `tools/A4/vae_conv_bench.py --profile` |
| Compiler, graphs, TorchCompileModel, `GPU_MAX_HW_QUEUES=2` A/B | `tools/A5/ab_compile.py`, `tools/A5/hipgraph_probe.py` |
| Startup cold/warm, time to first image | `tools/A7/profile_startup.ps1` |

Workflows: your saved workflows per model if present, else the official templates at default settings. Metrics: s/it, total time, time to first image, VAE decode time, text-encode time, peak VRAM per GPU, cold and warm startup.

## 8. Corrections to statements made during this session

- "`--fast fp16_accumulation` makes fp16 matmuls accumulate in fp16" is wrong on ROCm: torch ignores the flag there, and the kitchen HIP fp16 kernels accumulate in fp32 anyway.
- MiniMax H3's text encoder is ~48 GiB in bf16 (50 layers kept), not ~64 GB.
- Single-frame MiniMax encodes were not "never the problem": memory was small, but they paid ~12k bias-fill launches per tile until `84cfd41`.
- The Windows single-GPU forcing hid GPU 1 by default; `--cuda-device 0,1` always worked.

## 9. Dropped ideas

- VAE SDPA on AMD (A1-4): ≤ 1–2 % of decode, the historic high-res crash is unexplained; needs GPU evidence first.
- `PRIORITIZE_FP16 = is_nvidia()` (A2-4 core): unmeasured policy change for all AMD archs; not passing the flag is enough.
- `model_prefetch.py` compile guards (rest of A5-3): would silently disable the comfy compiler under TorchCompileModel.
- `debug=True` in the AOTriton probe (rest of A6-3): prints torch warnings on every RDNA2/3 startup.
- TorchCompileModel on H3: `float(sigma)` guard recompiles every step until the recompile limit.
- Lazy imports for startup: AGENTS.md requires module-scope imports. Removing kitchen's `torch._dynamo` import: no gain, torchvision imports it anyway.
- "10–60 min int8 loads" (kitchen#188): ComfyUI calls the `_dtype` dequant variants HIP does register; not supported on this stack.
- `PYTORCH_TUNABLEOP_CACHE_DIR` (patientx-cfz launcher): torch has no such variable.
- MultiGPU CFG Split at cfg = 1, sequence-parallel H3 (needs NCCL, not on Windows ROCm), GGUF for these models, Nunchaku, TensorRT, MIGraphX, NVFP4/MXFP8 kernels on RDNA4.
