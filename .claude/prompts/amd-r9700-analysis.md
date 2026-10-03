# Multi-agent analysis: ComfyUI on 2× AMD Radeon AI PRO R9700 (Windows, ROCm 10.0 wheels)

Paste everything below the line into Claude Code, started in the production ComfyUI install on the R9700 machine. It runs as an orchestrator with eight parallel analysis subagents, one verifier and a synthesis step. Line numbers were checked against ComfyUI 0.38.0 (commit e9027f2); agents re-verify them in the checkout they run in.

---

You are the orchestrator of a multi-agent review of this ComfyUI install. Goal: concrete changes that make ComfyUI faster, more stable and quicker to start on this machine, for these workloads: **Krea 2**, **Qwen-Image 2.1** and **MiniMax H3**. Proposals only: no agent edits tracked files, and nobody installs, upgrades or removes packages in the production venv.

## Target system

- 2× AMD Radeon AI PRO R9700: RDNA4, gfx1201, Wave32, 31.9 GiB each, 640 GB/s, PCIe 5.0 x16, 300 W blower cooler. WMMA for FP16, BF16, FP8 (OCP E4M3FN and E5M2) and INT8. No block-scaled (MXFP8, NVFP4) matrix path. CDNA-tuned Wave64 kernels (default Composable Kernel builds) do not apply.
- Windows. Driver 32.0.31041.1004 = Adrenalin 26.8.1, AMD's match for ROCm 10.0 but not yet soak-tested here; 26.7.1 is the last validated driver.
- Production venv `.venv-rocm-100` (selected via `config/rocm-stack.active` = `prod`). Use its Python for every command; read-only.
  - ROCm SDK pip wheels 10.0.0 (`rocm`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1201`). No system HIP SDK; `HIP_PATH` unset.
  - `torch.version.hip` = 7.15.26333, so ComfyUI parses `rocm_version` as (7, 15).
  - Python 3.12.12, torch 2.13.0+rocm10.0.0, torchvision 0.28.0, torchaudio 2.11.0.2, triton-windows 3.7.1.post27.
  - ComfyUI 0.38.0 + 8 commits (8cfe5e1e), frontend 1.53.6, templates 0.11.70, embedded docs 0.5.13, comfy-kitchen 0.2.36, comfy-aimdo 0.5.5.
  - Custom nodes: ComfyUI-Manager 3.42, Pixaroma 1.4.182, SolAttn_triton 26d816e (the active attention backend). sageattention, flash-attn and xformers are not installed.
  - Other libraries in the venv: transformers 4.57.6, diffusers 0.38.0.dev0, accelerate 1.14.0, safetensors 0.8.0, numpy 2.4.6, onnxruntime-directml 1.24.3, insightface 0.7.3, llama_cpp_python 0.3.17, gguf 0.19.0, av 18.0.0.

What that stack switches on in ComfyUI (verify each in the code):
- SDPA attention enabled for gfx1201 (needs ROCm >= 7.0), `SUPPORT_FP8_OPS` forced on (torch >= 2.7, ROCm >= 6.4).
- DynamicVRAM via comfy-aimdo, and with it the comfy compiler and graph capture (`main.py:267-273`, ROCm >= 7.14). These paths are new on AMD.
- MIOpen disabled (`torch.backends.cudnn.enabled = False`) unless `COMFYUI_ENABLE_MIOPEN=1`.
- `OCL_SET_SVM_SIZE` set because the torch version string contains `rocm`. `cudaMallocAsync` not enabled (only for `+cu` builds).
- Windows pinned-memory cap of 40 % of RAM.

## Workloads and their hot paths

- **Krea 2**: `comfy/ldm/krea2/model.py` (SingleStreamDiT). Own `RMSNorm` that upcasts to fp32 (`:24-35`); einops `rearrange` in the forward (`:81-83`, `:162`, `:289`, `:377`); GQA done by `repeat_interleave` of K and V before attention (`:94-97`); masked attention via `optimized_attention_masked` (`:99`); RoPE via `comfy.ldm.flux.math.apply_rope`, not a fused kernel. VAE: Wan 2.1 (3D causal conv, `latent_formats.Wan21`). Text encoder: Qwen3-VL-4B (`comfy/text_encoders/krea2.py`). Dtypes: bf16, fp16, fp32.
- **Qwen-Image 2.1**: `comfy/ldm/qwen_image21/model.py`. Fused comfy-kitchen ops `ck.rms_rope` (`:104`) and `ck.adaln` (`:121-163`); chunked prefix attention (`:202-211`). Latent 64 ch, 16× downscale. Text encoder: Qwen3-VL-8B. Dtypes: bf16, fp32 only. `memory_usage_factor` 6.0.
- **MiniMax H3** (audio + video): `comfy/ldm/minimax/model.py`, fused `ck.rms_rope_split_half` (`:179-189`), attention `:200`. VAE `comfy/ldm/minimax/vae.py`: the NDHWC kitchen path (`group_norm_silu_pad3d`, `fp16_conv3d`) is disabled on HIP (`:41-46`, `:63-69`), so every `CausalConv3d` runs plain `F.conv3d` while MIOpen is also disabled. Text encoder: Qwen3-VL-32B (about 64 GB in bf16, does not fit one card). Dtypes: bf16, fp32 only. Block-sparse attention node targets this model: `comfy_extras/nodes_sparse_attention.py` (`ck.sol_attn`, `sol_attn_is_available`).

## Rules for every agent

1. Read `AGENTS.md` first. Every core proposal must comply: smallest owner layer, fewest files, no new dependencies, no outbound network requests in core, no `torch.no_grad`/`inference_mode`, no model-specific options in shared helpers, no custom fp32-upcasting norms, no einops in inference code, no tensors for bookkeeping, no defensive casts.
2. Measure on this machine; never invent numbers. Check the device first: `python -c "import torch; print(torch.version.hip, [torch.cuda.get_device_properties(i).gcnArchName for i in range(torch.cuda.device_count())])"`. Run benchmarks against a ComfyUI instance on a non-default port (for example `--port 8199`) so the production instance is not disturbed. Use GPU 1 for microbenchmarks when GPU 0 is busy. Put scripts and logs in the session scratchpad, not the repo. Anything not measured is labelled `unmeasured` with the reasoning and the command that would measure it.
3. No proposal may regress other targets. Name the effect on other AMD generations (RDNA2/3/3.5, CDNA), NVIDIA, CPU and Linux ROCm. Gate by arch or version only where behaviour differs, in the style `comfy/model_management.py` already uses.
4. Cite `path:line` for every claim about the code. Quote at most five lines.
5. Bucket each finding: `core` (code change in ComfyUI), `launch` (CLI flag or environment variable), `extension` (custom node or optional package), `system` (driver, Windows setting), `upstream` (fix belongs in ROCm, PyTorch, comfy-kitchen, comfy-aimdo, a custom node).
6. Return at most eight findings, ranked. Fields: `id`; `title` (imperative); `category` speed | stability | startup | capability; `bucket`; `models` affected (Krea 2, Qwen-Image 2.1, MiniMax H3, all); `location` path:line; `current` (what happens today on this stack, with evidence); `proposal` (diff sketch of 30 lines or less, or exact flag/env/setting); `effect` with basis and `confidence` high | medium | low; `risk`; `verify` (command, workflow, metric); `size` (files, lines).

## Known AMD touchpoints

Verify and go deeper; restating them is not a finding.

- `comfy/model_management.py:481-544` AMD init block: MIOpen switch, `aotriton_supported()` (launches a flash-attention kernel and synchronizes at import), SDPA and fp8 gates. The whole block sits in a bare `try: ... except: pass`.
- `comfy/model_management.py:553-560` `--fast fp16_accumulation` on AMD. Only Krea 2 can run fp16; the other two are bf16-only.
- `comfy/model_management.py:1373-1385` async offload, two streams on AMD. `:1631-1644` pinned memory. `:1770-1773` VAE SDPA disabled on every AMD GPU. `:2020-2063` fp8/nvfp4/mxfp8 capability checks.
- `comfy/sd.py:500-503` `VAE_KL_MEM_RATIO = 2.73` for all AMD GPUs.
- `comfy/ops.py:70-97` SDPA backend priority; GQA heads repeated on non-NVIDIA when a mask is present. `comfy/ops.py:969-973` fp16-accumulate kitchen path requires fp16 activations.
- `comfy/ldm/modules/attention.py:572-576` `SDP_BATCH_LIMIT` lowered only on NVIDIA; `:891-916` attention dispatch order.
- `comfy/quant_ops.py:7-45` comfy-kitchen backend selection: HIP backend first on AMD, Triton only with `--enable-triton-backend`.
- `comfy/model_prefetch.py:30,178` comfy compiler and graph capture; `comfy/text_encoders/llama.py:1169` graph capture for text encoders.
- `comfy/multigpu.py`, `comfy_extras/nodes_multigpu.py`: MultiGPU CFG Split, Select Model/CLIP/VAE Device.
- `main.py:90` `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` only under `__main__`; `main.py:116-117` `OCL_SET_SVM_SIZE`.

## Phase 1: analysis agents

Launch all eight in one message so they run in parallel. Give each the target system, workloads, rules and touchpoints sections plus its own brief.

**A1 Attention.** Owns attention for the three DiTs, their text encoders and VAEs.
- Identify SolAttn_triton (source repo, how it hooks into ComfyUI, which calls it replaces, whether it bypasses `optimized_attention`). Compare it with AOTriton SDPA, comfy-kitchen `sol_attn` via the Model Sparse Attention node (is a HIP or Triton kernel available on gfx1201?), and comfy-kitchen int8 attention (`--use-ck-attention`). Speed, peak memory, output difference per model.
- Krea 2: is a mask present in normal sampling? With a mask, which SDPA backend runs on gfx1201, and what do the explicit K/V `repeat_interleave` copies cost against native GQA (`enable_gqa`)?
- Qwen-Image 2.1 chunked prefix attention and MiniMax H3 attention at their real sequence lengths and head_dim 128: which backend runs, and is `SDP_BATCH_LIMIT` needed on gfx1201?
- Is the AMD-wide VAE SDPA disable still needed on this stack? Cost of the split/sub-quad fallback in each model's VAE.
- Deliver a standalone microbenchmark script.

**A2 Model hot paths and GEMM.** Owns the per-step cost of each DiT outside attention.
- Profile one denoising step per model with `torch.profiler` (CUDA activities work on ROCm). Report the top 15 kernels by time and where they come from.
- Which comfy-kitchen fused ops used by these models (`rms_rope`, `adaln`, `rms_rope_split_half`, int8/fp8 linears) have HIP kernels for gfx1201 in 0.2.36, and which fall back to eager? Check what 0.2.37 changes.
- Krea 2: cost of the fp32-upcasting `RMSNorm` and einops reshapes; propose an `AGENTS.md`-compliant replacement (`operations.RMSNorm` / `comfy.rmsnorm.rms_norm` with the `1 + scale` convention, native reshapes) and a fused RoPE where comfy-kitchen has one.
- fp8: trace an fp8-scaled linear for each model on this stack (`torch._scaled_mm` via hipBLASLt, kitchen HIP kernel, or dequantize + bf16). Is it faster than bf16 here? Tensor-wise vs row-wise scaling.
- `PYTORCH_TUNABLEOP_ENABLED` on Windows ROCm: gain per model, first-run cost, results file location.

**A3 Memory, offload and dual GPU.** Owns VRAM accounting, offload, pinned memory and use of the second card.
- MiniMax H3 with a 32B text encoder: best placement across two 31.9 GiB cards (Select CLIP Device to GPU 1, fp8 text encoder, DynamicVRAM partial offload) so the DiT stays resident on GPU 0. Measure encode time and whether the DiT gets evicted.
- MultiGPU CFG Split for Krea 2 and Qwen-Image 2.1: speedup vs a single card, PCIe traffic, stability.
- What each model fits fully per card in bf16 and fp8. Where ComfyUI offloads or tiles although it would fit; whether `VAE_KL_MEM_RATIO = 2.73` and `MIN_WEIGHT_MEMORY_RATIO = 0.4` cause it.
- DynamicVRAM on Windows ROCm: correctness of free-memory reporting, the 40 % pinned-memory cap against installed RAM, async offload stream count, `PYTORCH_HIP_ALLOC_CONF` options and fragmentation over long MiniMax H3 runs.

**A4 VAE and convolutions.** Owns VAE encode and decode for all three models.
- With MIOpen disabled, which ATen conv backend runs `F.conv3d` and `F.conv2d` on gfx1201? Compare `COMFYUI_ENABLE_MIOPEN=1`: speed, peak memory, first-run find cost (`MIOPEN_FIND_MODE`, where the user find-db lives on Windows and whether it persists between runs). Does the TheRock Windows wheel ship MIOpen kernels for gfx1201?
- MiniMax H3 VAE: is the HIP exclusion of the NDHWC kitchen path still right when MIOpen is off? Does comfy-kitchen 0.2.36 have HIP `group_norm_silu_pad3d` / `fp16_conv3d`? channels_last_3d with each conv backend.
- Wan 2.1 VAE (Krea 2) and the Qwen-Image 2.1 VAE: decode time and peak memory at the resolutions in use, tiled vs untiled with 32 GB.
- Propose an arch-gated default only together with the benchmark that justifies it.

**A5 Compile and graph capture.** Owns `torch.compile`, the comfy compiler and graph capture, now active on this stack.
- Is graph capture in `comfy/model_prefetch.py` and `comfy/text_encoders/llama.py` running for these models? Gain per step, and failures or hangs on Windows with driver 26.8.1. Test `--disable-cuda-graphs` and `--disable-comfy-compiler` as A/B.
- triton-windows 3.7.1 inductor on gfx1201: does the built-in TorchCompileModel node work for each model, compile time, recompiles on resolution change, cache location and persistence. Known: Triton INT8 WMMA kernels fail at `num_stages >= 3` on gfx1201.
- Whether `--enable-triton-backend` for comfy-kitchen helps any op the HIP backend lacks.

**A6 Stability.** Owns failure modes on this Windows stack. Every finding needs a reproduction or a concrete trigger.
- What the bare `except: pass` in the AMD init block can hide: a silent fallback to sub-quadratic attention and no fp8, with nothing logged.
- Windows TDR (GPU timeout and driver reset) during long kernels such as large VAE decodes or 32B encoder layers. Read `TdrDelay`/`TdrDdiDelay` from the registry; report, do not change them.
- Driver 26.8.1 vs the validated 26.7.1: list any crash, hang or wrong output and which driver it occurred on.
- Device ordering and visibility with two identical cards (`HIP_VISIBLE_DEVICES`, `--cuda-device`, `--default-device`), and coexistence with onnxruntime-directml in the same process.
- OOM recovery, interrupt during a step, NaN/black output in bf16 and fp16 (Krea 2), behaviour when GPU 1 is busy.
- Installed versions vs the checkout's `requirements.txt` (comfy-kitchen 0.2.36, frontend 1.53.6): mismatches and their effect.

**A7 Startup and time to first image.** Owns everything between launch and the first finished image. Measure before proposing.
- Cold start (after reboot) and warm start: wall time to the "To see the GUI go to" line. `python -X importtime main.py --quick-test-for-ci 2> importtime.log` plus per-phase timing: torch and ROCm DLL loading from the wheels, `comfy.model_management` import including the `aotriton_supported()` kernel launch, `comfy_kitchen` and `triton` import, `init_builtin_extra_nodes`, partner/API nodes, custom nodes (Manager, Pixaroma, SolAttn_triton) and their prestartup scripts, asset database setup and scan, frontend package resolution.
- Windows specifics: Defender real-time scanning of the venv and model folders, DLL load time of `rocm-sdk-libraries`, `.pyc` cache state. Report only; exclusions are the user's decision.
- ComfyUI-Manager network activity at startup (registry and channel fetches): measure its time and name the local setting that disables it.
- First-run costs after startup: AOTriton kernel image load, MIOpen find, TunableOp tuning, Triton JIT, model load from disk (mmap, fast-disk path), text encoder load for MiniMax H3.
- Report the five largest costs in ms with their source. Proposals stay local: deferral, lazy import, persisted caches. No network.

**A8 External research.** Owns extensions and upstream status. Use the web; cite URLs.
- For each candidate: gfx1201 and Windows ROCm evidence, last commit date, licence, what it replaces in core and whether core already covers it, install risk, network calls it makes.
- Seeds: SolAttn_triton (origin and status), SageAttention ROCm ports (`EmbeddedLLM/SageAttention-rocm`, `boxwrench/SageAttention-RDNA4`, ROCm/aiter gfx1201 operators), flash-attention Triton AMD backend on Windows, `patientx/ComfyUI-INT8-Fast-ROCM`, `patientx-cfz/comfyui-rocm` (Windows ROCm packaging and its tweaks), `iGavroche/rocm-ninodes`, `kijai/ComfyUI-KJNodes`, step caches (MagCache, TeaCache; core EasyCache) and whether they support Krea 2, Qwen-Image 2.1 or MiniMax H3, `city96/ComfyUI-GGUF` for the 32B text encoder.
- Non-options to confirm or refute: Nunchaku, TensorRT, NVFP4/MXFP8, ComfyUI_MIGraphX (no support for these three models).
- Upstream items: hipBLASLt gfx1201 FP8 dispatch, PyTorch `_scaled_mm` on gfx120x, AOTriton gfx1201 coverage in the Windows wheels, MIOpen RDNA4 performance, known issues of Adrenalin 26.8.1 with ROCm 10.0, comfy-kitchen HIP op coverage roadmap.

## Phase 2: verification

After all eight return, one `general-purpose` verifier gets every finding. For each it re-reads the cited code, checks `AGENTS.md` compliance, checks regressions on other platforms, merges duplicates across agents, and checks whether the claim survives (for example an "always" that is gated elsewhere, or a measurement taken with GPU 0 busy). Verdict per finding: `keep`, `revise` (with the revision) or `drop` (with the reason). It drops anything that needs a new core dependency, adds network access to core, or adds a flag no current code reads.

## Phase 3: synthesis

Write `.claude/reports/r9700-analysis.md` and do not commit it. Lead every section with the recommendation. Stay under 400 lines.

1. Top 10 table: rank, title, category, bucket, models, location, effect and confidence, risk, size.
2. Recommended launch configuration for this machine: exact command line and environment variables, one justification per line, each marked measured or unmeasured. One variant per workload where they differ.
3. Two-GPU plan per workload: what runs where and why.
4. Core change candidates as small, independent PR sketches in priority order, each with the benchmark it must pass before merging.
5. Startup: cold and warm timings, the five largest costs, fixes with expected savings.
6. Extension shortlist with a verdict (install, try, skip) and the reason.
7. Benchmark protocol: the user's saved workflows for each model if present (`user/default/workflows`), otherwise the official templates at default settings. Metrics: s/it, total time, time to first image, VAE decode time, text-encode time, peak VRAM per GPU (`torch.cuda.max_memory_allocated`), cold and warm startup. Three runs, report the median. Record driver, ROCm wheel, HIP, torch, comfy-kitchen, comfy-aimdo and ComfyUI commit with every number.
8. Dropped ideas, one line each with the reason.
