# ComfyUI on 2× Radeon AI PRO R9700: analysis report

Stand: 04.10.2026. Branch `claude/cool-euler-0g3imw`. Produced by the prompt in `.claude/prompts/amd-r9700-analysis.md`: 8 analysis agents, 2 verifiers (code and claims), synthesis.

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
git cherry-pick b09048c 37b7bce   # regression tests only, no runtime change
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

### Regression tests

`b09048c` adds CPU unit tests for the upstreamable commits; `37b7bce` pins the conv3d reference in the single-frame test to the cuDNN path, because ComfyUI turns `torch.backends.cudnn.enabled` off on AMD at import and the reference would otherwise run the path under test. On e9027f2 (before the fixes) 20 tests fail, each on its guard; on the branch 48 pass and 1 skips (no GPU here). The full `tests-unit/comfy_test` and `comfy_extras_test` suites pass with them (648 passed, 15 skipped).

| Test file | Commits | Guard (fails before the fix) |
|---|---|---|
| `comfy_test/test_krea2_model.py` | `d0e865c`, `377a3f9` | RMSNorm creates no fp32 intermediates; attention gets 2 K/V heads plus `enable_gqa` instead of 4 expanded heads. Patchify matches the einops layout and round trips; fused DiT forward equals the unfused one |
| `comfy_test/test_qwen_image21_norm.py` | `dd85c0c` | `ZeroCenteredRMSNorm` creates no fp32 intermediates |
| `comfy_test/test_vae_single_frame.py` | `ded5499`, `ff0e009` | single-frame causal conv3d with cuDNN off calls conv2d, never conv3d; Wan 2.2 head runs in strips, bit identical |
| `comfy_test/test_audio_upsample.py` | `91d15cb` | 7 upsamplers (MMAudio r2/r3, LTX kaiser and Hann r2–4, MiniMax) never call `conv_transpose1d` and match it at T = 1, 7, 300 |
| `comfy_test/test_ops_attention_compile.py` | `3781fb9`, `d7b3369` | `_fp16_linear_wanted` compiles with `fullgraph=True`; unmasked GQA SDPA expands K/V when no fused backend takes native GQA (GPU only, skipped on CPU) |
| `comfy_extras_test/select_clip_device_test.py` | `2ed15ee` | `cond_stage_model` follows the retargeted patcher |
| `main_rocm_windows_gpu_test.py` | `52ed61a` | ROCm on Windows is not forced to `CUDA_VISIBLE_DEVICES=0`; CUDA still is |

The equivalence tests (round trip, fused vs unfused, upsample vs transposed conv) also pass on the old code; they guard later changes, not the original bugs. Not covered: `e9c38f9` (needs two devices), `e997bcb` (runs at `model_management` import), the sparse attention part of `d0e865c` (rejects non-CUDA tensors first; phase 1b of the measurement prompt checks it on the R9700), and the fork-only HIP commits `85d2562`, `da79103`, `84cfd41`. The measurement prompt (phase 1a) runs the tests on the R9700 against both builds; the baseline run is the on-hardware evidence for `d7b3369`.

### comfy-kitchen patches

**Round 1 (fp8 GEMM).** The local session adopted the branch on 04.10.2026 (Krea 2 at 1280x2048, cfg 1.8: 183 s -> 77 s warm, together with fp8_e4m3fn_fast weights, SageAttention-RDNA4 and a res_2m refine) and measured kitchen's fp8 GEMM against hipBLASLt: 10-20 % slower for K <= 6144, 35-42 % faster for K = 16384, 47.0 ms per Krea 2 block vs 49.9 (hipBLASLt untuned) and 40.9 (TunableOp), and 5x slower at M ~ 2 (0.55 vs 0.11 ms). Four patches against kitchen v0.2.36 (`888b13e`, they also apply to v0.2.37) answer that. Nothing was timed here; details in `.claude/reports/kitchen-r9700.md`, patches, benchmark and Windows build steps in `.claude/kitchen-r9700/`.

| Patch | Change | Output | Expected | Measured (local session) |
|---|---|---|---|---|
| 0001 | WMMA GEMM core: tile loads without exec-masked branches and coalesced to whole 128-byte lines; the K-tail test only in kernels whose K is not a multiple of BKB | bit-identical | per K tile and wave 45 VALU, 42 SALU, 11 branches -> 24, 8, 3; a few % at large M; shared by the int8, int4 and fp16 GEMMs | byte-identical; **kept**, with 0002 in production as 0.2.36+r9700: Krea 2 block fp8 GEMMs 45.46 -> 42.78 ms |
| 0002 | fp8 GEMV (M <= 8): `v_cvt_pk_f32_fp8` decode instead of ~650 VALU per 16 bytes, all rows per wave so the weight is read once | bit-identical | M ~ 2, K 4096, N 16384: 0.55 ms -> towards the 0.10 ms bandwidth floor | byte-identical; **kept**: M = 1/2/8 from 0.23/0.40/1.48 to 0.08/0.08/0.18 ms |
| 0003 | six more tiles (64x64 wave tiles: a third fewer LDS reads per MMA; 16/32-row tiles for small M) and a measured tile choice per shape class instead of fixed thresholds | bit-identical | 0 to ~15 % at K <= 6144; first GEMM per class pays ~0.2-0.4 s of tuning; `COMFY_KITCHEN_HIP_FP8_TUNE=0` restores 0.2.36 | byte-identical; **dropped**: tiles gain only 1-3 % (256x128x128 best at M = 10264), and the tuner's timing is noise (it picked a tile 40-70 % slower at M = 77) |
| 0004 | draft: hardware e4m3 encode (`v_cvt_pk_fp8_f32`) in the per-tensor quantize | **unverified** | quantize from VALU-bound towards memory-bound | byte-identical; **dropped**: no gain, the quantize is already memory-bound at 560-600 GB/s |

0001-0003 feed every output the same WMMA K-steps in the same order; a CPU emulation of kitchen's kernel source showed 0 differing bytes against v0.2.36 in 17 cases, and the R9700 confirmed it for all four patches.

**Round 2 (norms, rope, fixed GEMM tile).** The local session then set a harder rule: renders must be bit-identical across process restarts, so no timing-based choices at run time and no reduction order that depends on scheduling or atomics. Targets in its order: `rms_rope` (Krea 2: 1.59 ms per q+k call, ~200 GB/s against a practical ~600), `rms_adaln` (0.85 ms per call, ~300 GB/s), and the remaining fp8 GEMM gap. Three patches on top of 0001 + 0002; details in `.claude/reports/kitchen-r9700-2.md`, benchmarks in `.claude/kitchen-r9700/`.

| Patch | Change | Output | Expected (Krea 2 at 1280x2048) |
|---|---|---|---|
| 0005 | `rms_adaln` / `adaln`: one wave per row with 16-byte accesses, for 2-byte rows with D % 256 == 0 | bit-identical | 0.85 -> ~0.45 ms per call, 2 calls per block |
| 0006 | `rms_rope`, head_dim 128: one wave per token, freqs and weights loaded once, looping over the token's heads; DPP instead of LDS for the row sums | bit-identical | 1.59 -> ~0.6 ms per q+k call; MiniMax H3 at 39.5k tokens ~3.8-4 ms per call |
| 0007 | fp8 GEMM: fixed 256x128x128 tile when K >= 4096 and the grid has at least 8 blocks per WGP | bit-identical | 1-3 % on the large GEMMs (the local session's tile sweep) |

That is about 50 ms per 28-block Krea 2 forward from 0005 and 0006, plus 12-36 ms from 0007. Identity was checked against the 0.2.36 code objects, not the source: their compiled `rms_rope` rounds differently from source order in four places (which square is fused into the running sum, the grouping `(rrms * w) * x`, which fp32 rotation product is fused, and the unfused add after bf16/fp16 rounding), and the fast paths write each of those steps out explicitly. A CPU emulation that records every element's fp32 value before its bf16 store found 0 differences in 1.66 M values (2.53 M output elements); a deliberately swapped fma changed 30 % of those fp32 values but only 2-15 output elements per case, so byte comparisons alone are a weak test. Not solved: the mlp down GEMM at K = 16384 (10.2 ms). DRAM re-reads of its weight do not explain it, and the 10.2 ms may include the activation quantize. The review of phase 7 turned up a larger lead there: with `fp8_e4m3fn_fast` weights, `fp8_linear` in `comfy/ops.py` quantizes every fp8 Linear's input with torch (an in-place clamp, then a cast), not with kitchen's quantize, and wq, wk, wv and the attention gate quantize the same input four times. Per Krea 2 block at 10264 tokens that is ~4.3 GB of traffic, ~7 ms next to 42.78 ms of fp8 GEMMs; kitchen's quantize once per shared input would be ~1.8 ms (estimates from the code, not timed). That is a ComfyUI change in `fp8_linear`, not done.

Phase 7 of the measurement prompt builds 0001 + 0002 + 0005-0007 into a separate folder (the production venv stays untouched) and compares it with the installed kitchen through `bench_rms_rope.py`, `bench_rms_adaln.py` and `bench_fp8_gemm.py` (`--save-ref`, then `--ref`, all with `--json` for the min-max spread). A second candidate run in a new process must match the first, since renders must not change across restarts. Krea 2, Qwen-Image 2.1 and H3 then run end to end four times each (installed twice, candidate twice, each in a fresh instance, `fp8_e4m3fn_fast` so the fp8 GEMM is reached), and one profiled step per workload counts the new and old kernels by name: the benches' `fast` column only re-implements the launcher test on their own inputs. Phase 7 also times both quantize paths for the mlp down input, records the WGP count that 0007's threshold reads, and checks that the build has no timing-based choice left (0003's tuner). A patch is kept only if it is identical, reproduces itself across a restart and is measurably faster on the rows it serves; an end-to-end difference that the benches miss is traced to one patch with single-patch builds. Two cases end in inconclusive rather than drop: production rounding its old kernels differently from the official 0.2.36 wheel (the fast paths match that wheel), and 0007 losing only on the rows outside the tile sweep (Qwen 2.1 at M = 4096, small-M M = 335 exactly at the threshold). Round 1 is not re-measured.

### GPU checks still owed

The kitchen HIP kernels and multi-GPU paths behind these commits never ran here. In order of risk:

| Commit | Check on the R9700 | If it fails |
|---|---|---|
| `377a3f9` | `tools/A2/krea2_block_bench.py --device 1 --tokens 4608` with `--batch 1` and `--batch 2`: fused `rel_err` near the CPU numbers above, then image diff and s/it | `git revert 377a3f9` |
| `d0e865c` (sparse override) | Krea 2 with Model Sparse Attention (sol-attn, `verbose`) in both builds, 1024² with `min_tokens` 4096 and 2048² at the default: the log shows `sparse (1, <tokens>, <heads>, 128)` for the joint sequence; patched vs baseline diverges no more with sparse than with dense; s/it and PSNR vs dense at 2048². Baseline already ran sparse because it expanded K/V before the override | keep Krea 2 dense and fix the override in `comfy_extras/nodes_sparse_attention.py`; a full revert also drops native GQA and conflicts with `377a3f9` |
| `da79103`, `84cfd41` | MiniMax H3 encode of a 17+ frame clip and of a single keyframe: time and peak VRAM against the previous build (`tools/A4/vae_bench.py --model minimax --op encode`) | revert both |
| `2ed15ee`, `52ed61a` | H3 with Select CLIP Device → gpu:1: GPU 1 memory rises by the encoder size during encode, encode time drops (`tools/A3/run_placement_bench.py --model minimax --placement split`) | `--cuda-device all` as workaround for `52ed61a` |
| `ded5499`, `ff0e009` | VAE decode at 1024² and 2048² for Krea 2 and Qwen 2.1: time and peak VRAM (`tools/A4/vae_bench.py`) | revert |
| `91d15cb` | MiniMax H3 audio decode of a 10 s and a 60 s clip in both builds: time and waveform diff; LTX audio if you use it | revert |
| `e9c38f9` | CFG Split with cfg > 1 and batch 2: s/it (`run_placement_bench.py --cfg 4 --cfg-split`) | revert |
| `e997bcb` | startup log on both cards still reports the expected attention line | revert |
| `3781fb9` | only with TorchCompileModel (`tools/A5/ab_compile.py --variants baseline,torch_compile`) | none needed for eager runs |
| kitchen 0006 (prompt phase 7) | `.claude/kitchen-r9700/bench_rms_rope.py --ref` against the installed kitchen and against a second candidate run: `identical` in every row, Krea 2 near 0.6 ms per call; the profiled step shows `rms_rope128_kernel` and no `rms_rope_kernel` for Krea 2, Qwen 2.1 and H3 | leave 0006 out. If only the fast-path rows differ, compare with the official 0.2.36 wheel; matching it makes 0006 inconclusive, since it would change production renders once |
| kitchen 0005 (prompt phase 7) | `bench_rms_adaln.py`, same checks, Krea 2 near 0.45 ms per call; `adaln_wave_kernel` for Krea 2 (57 per model call) and Qwen 2.1 | leave 0005 out (same exception) |
| kitchen 0007 (prompt phase 7) | `bench_fp8_gemm.py`, same checks; Krea 2 block total per M against this session's reference; at 1280x2048, Krea 2's 8 block Linears on the 256x128 tile | leave 0007 out; a loss only on the rows outside the sweep (Qwen 2.1, M = 335) is inconclusive, not a drop |

Known limits:
- `da79103` relies on the HIP kernel accumulating in fp32. kitchen documents `fp16_conv3d` as fp16-accumulate, so this is fork-only under AGENTS.md. Re-check the disassembly when you bump comfy-kitchen.
- The `da79103` gate (`amd_min_version(..., 3)`) also matches gfx1170/1171, which kitchen 0.2.36 does not ship kernels for. Irrelevant on your cards.
- SolAttn_triton (Patch Sol-Attn) on Krea 2: since `d0e865c` it receives the unexpanded K/V heads plus `enable_gqa`. Whether it reads the flag is unverified, and a Triton kernel can read out of bounds instead of raising. Don't apply it to Krea 2.
- `--default-device` still does nothing on Windows ROCm (HIP on PAL ignores device order). Use `--cuda-device N`.
- The Wan 2.1 VAE head (Krea 2) is not stripped; ~1.7 GB columns at 1024² bf16, within its estimate.
- `torch._dynamo.explain` does not report the graph break `3781fb9` removes; use `fullgraph=True` to see it.
- Audio: the 42 dilated convs per H3 audio decode still cost 12,192 per-channel bias fills with MIOpen off (7,936 more per reference-audio encode). Measure their share before fixing.

### Upstream items worth filing

- comfy-kitchen: 0001 and 0002 (measured on the R9700, kept); 0005-0007 once measured. Still open: document per-backend accumulation (proposed text in `kitchen-r9700.md`: HIP fp16 conv/GEMM accumulate in fp32), bf16 HIP conv3d (would move Krea 2 / Qwen VAEs off torch's fallback), `sol_attn` with Lq≠Lk and native GQA (Qwen 2.1 sparse), kitchen#184 (HIP sol paired query blocks, +18–20 % on gfx1201; moot while Sage handles attention).
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

`.claude/prompts/r9700-measurement-run.md` runs this whole protocol on the R9700 machine and returns a keep/revert verdict per commit and per comfy-kitchen patch. Run against a separate instance (`--port 8199`), on GPU 1 when GPU 0 is busy. 3 runs, report the median. Record driver, ROCm wheel, HIP, torch, comfy-kitchen, comfy-aimdo and ComfyUI commit with every number (`tools/A6/a6_stability_check.py` prints them).

| Question | Tool |
|---|---|
| Driver, TDR, device order, AOTriton probe reason, versions vs requirements | `tools/A6/a6_system_check.ps1` (runs `a6_stability_check.py`) |
| Idle VRAM page-out | `tools/A6/idle_pageout_check.py --gib 8 --idle 20 --devices 0,1` |
| NaN / flat output across drivers or `--fast` variants | `tools/A6/flat_output_check.py <output dir>` |
| Attention backends per real shape (SDPA native/expanded, flash/efficient/math, kitchen int8, sol, VAE slice) | `tools/A1/attn_bench.py --device 1 --explain --profile` |
| bf16 vs fp8 vs int8 linears, hipBLASLt vs rocBLAS, TunableOp | `tools/A2/gemm_bench.py --model <file> --tokens <n> --device 1` |
| Top kernels per denoising step | `tools/A2/comfy_profile_step` (custom node; load via `extra_paths_profile.yaml` in the test instance only) |
| comfy-kitchen round-2 patches against the installed kitchen (phase 7 of the measurement prompt) | from `.claude/kitchen-r9700/`, patched build on `PYTHONPATH`: `bench_rms_rope.py --ref`, `bench_rms_adaln.py --ref`, `bench_fp8_gemm.py --ref` |
| Regression tests, patched and unpatched build (phase 1a of the measurement prompt) | `python -m pytest -p no:cacheprovider -rs <the 7 files above>` with `CUDA_VISIBLE_DEVICES=1`, `HIP_VISIBLE_DEVICES=1` |
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
