# Multi-agent analysis: ComfyUI on AMD Radeon AI PRO R9700

Paste everything below the line into Claude Code at the repository root. It runs as an orchestrator with eight parallel analysis subagents, one verifier, and a synthesis step. Line numbers were checked against commit e9027f2; agents re-verify them.

---

You are the orchestrator of a multi-agent review of this ComfyUI checkout. Goal: find concrete changes that make ComfyUI faster, more stable, quicker to start, and more capable on the AMD Radeon AI PRO R9700. This run produces proposals only: no agent edits tracked files.

## Target hardware

- AMD Radeon AI PRO R9700: RDNA4, Navi 48, `gcnArchName` gfx1201, Wave32 (CDNA-tuned Wave64 kernels such as default CK builds do not apply).
- 64 CUs, 128 AI accelerators. WMMA for FP16, BF16, FP8 (OCP E4M3FN and E5M2) and INT8. Vendor figure: about 383 TFLOPS FP8 dense. No block-scaled (MXFP8, NVFP4) matrix path.
- 32 GB GDDR6 at 640 GB/s, PCIe 5.0 x16, 300 W blower cooler (sustained clocks during long video jobs matter).
- Software baseline. State it whenever a finding depends on a version. Linux first, ROCm 7.x (some paths below need >= 7.14), PyTorch >= 2.12 ROCm wheels (2.10 has known fp8 problems on gfx120X), comfy-kitchen 0.2.37 (HIP backend with RDNA4 WMMA + fp8), comfy-aimdo 0.5.5. Windows ROCm 7.2.x is a secondary target: note where results differ.

## Rules for every agent

1. Read `AGENTS.md` first. Every proposal must comply with it: smallest owner layer, fewest files, no new dependencies, no outbound network requests in core, no `torch.no_grad`/`inference_mode`, no model-specific options in shared helpers, no tensors for bookkeeping, no defensive casts, delete dead workarounds rather than adding new ones.
2. Never invent measurements. This environment may have no AMD GPU. Check with `python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"` and `rocminfo | grep gfx`. Without the hardware, mark every effect as `unmeasured`, give the reasoning behind the estimate, and supply the exact command or script that measures it. Put scripts in the session scratchpad, not the repo.
3. No proposal may regress other targets. Name the effect on RDNA2 (gfx103x), RDNA3/3.5 (gfx110x, gfx115x), CDNA (gfx90a, gfx942, gfx950), NVIDIA, CPU and Windows ROCm. Gate by arch or ROCm/torch version only where behaviour differs, in the style `comfy/model_management.py` already uses.
4. Cite `path:line` for every claim about the code. Quote at most five lines.
5. Sort each finding into one bucket: `core` (code change in this repo), `launch` (CLI flag or environment variable, no code change), `extension` (custom node or optional package), `upstream` (fix belongs in ROCm, PyTorch, comfy-kitchen, comfy-aimdo or similar).
6. Return at most eight findings, ranked, each with:
   - `id`, `title` (imperative, e.g. "Gate VAE SDPA on gfx1201 by ROCm version")
   - `category`: speed | stability | startup | capability
   - `bucket`: core | launch | extension | upstream
   - `location`: path:line
   - `current`: what happens today on gfx1201 and the evidence
   - `proposal`: diff sketch (30 lines max) or exact flag/env value
   - `effect`: expected gain with its basis (measured number, linked upstream benchmark, or reasoning) and `confidence`: high | medium | low
   - `risk`: what can break and on which platforms
   - `verify`: command, workflow and metric that proves it
   - `size`: files and lines touched

## Known AMD touchpoints

Verify these and go deeper. Restating them is not a finding.

- `comfy/model_management.py:433-453` `is_amd`, `amd_min_version` (RDNA generation parsed from the arch string).
- `comfy/model_management.py:481-544` AMD init block. Disables MIOpen (`torch.backends.cudnn.enabled = False`) for everything newer than RDNA2 unless `COMFYUI_ENABLE_MIOPEN=1`. `aotriton_supported()` launches a real flash-attention kernel and synchronizes during module import. SDPA is enabled on gfx1201 only with ROCm >= 7.0. `SUPPORT_FP8_OPS` is forced on for gfx1201 with torch >= 2.7 and ROCm >= 6.4. The whole block sits in a bare `try: ... except: pass`.
- `comfy/model_management.py:553-560` `--fast fp16_accumulation` enabled on AMD, `PRIORITIZE_FP16` TODO.
- `comfy/model_management.py:1373-1385` async weight offload, two streams on AMD by default.
- `comfy/model_management.py:1631-1644` pinned memory limits.
- `comfy/model_management.py:1770-1773` `pytorch_attention_enabled_vae()` returns False on every AMD GPU ("crash when doing high res").
- `comfy/model_management.py:2020-2063` `supports_fp8_compute`, `supports_nvfp4_compute`, `supports_mxfp8_compute`: NVIDIA-only unless `SUPPORT_FP8_OPS`.
- `comfy/sd.py:500-503` `VAE_KL_MEM_RATIO = 2.73` for all AMD GPUs.
- `comfy/ops.py:70-97` SDPA backend priority list; GQA heads repeated on non-NVIDIA when a mask is present.
- `comfy/ldm/modules/attention.py:572-576` `SDP_BATCH_LIMIT` lowered only on NVIDIA. `:891-916` attention dispatch order (sage, flash, xformers, pytorch, split/sub-quad, comfy-kitchen int8).
- `comfy/quant_ops.py:7-45` comfy-kitchen backend selection: HIP backend takes priority on AMD, Triton is opt-in.
- `comfy/model_prefetch.py:30,178` comfy compiler and graph capture, gated on `is_device_cuda` (True on ROCm) and aimdo. `comfy/text_encoders/llama.py:1169` graph capture for text encoders.
- `cuda_malloc.py:90-112` `cudaMallocAsync` is only enabled by default for `+cu` torch builds.
- `main.py:90` `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, set only when run as `__main__`. `main.py:116-117` `OCL_SET_SVM_SIZE`. `main.py:267-273` DynamicVRAM needs ROCm >= 7.14.
- `comfy/ldm/minimax/vae.py:41-46` fused comfy-kitchen pad path excluded on HIP because MIOpen converts layouts back.

## Phase 1: analysis agents

Launch all eight in one message so they run in parallel. Use `Explore` for A1-A7 when they only read code, `general-purpose` when they need to run scripts, and `general-purpose` with WebSearch/WebFetch for A8. Give each agent the hardware section, the rules section, the touchpoints list and its own brief below.

**A1 Attention.** Owns attention selection for diffusion models, text encoders and VAEs on gfx1201.
- Is the AMD-wide VAE SDPA disable still needed on gfx1201 with ROCm 7.x and AOTriton? What does the split/sub-quad fallback cost at 1024x1024 and 2048x2048?
- Which SDPA backend does PyTorch select on gfx1201 for head_dim 64, 128, 256 in fp16 and bf16? Does the `SDPA_BACKEND_PRIORITY` list help or hurt there?
- Does gfx1201 need a batch limit like `SDP_BATCH_LIMIT` on NVIDIA?
- Compare AOTriton SDPA, flash-attn with the Triton AMD backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `--use-flash-attention`), SageAttention ROCm builds (`--use-sage-attention`) and comfy-kitchen int8 attention (`--use-ck-attention`): correctness, speed, peak memory, and whether core needs any change to use each.
- Deliver a standalone attention microbenchmark the user can run.

**A2 GEMM and quantization.** Owns fp8 and int8 paths.
- Trace an `fp8_scaled` checkpoint linear from load to matmul on gfx1201. Which path runs: `torch._scaled_mm` via hipBLASLt, a comfy-kitchen HIP kernel, or dequantize plus bf16 matmul?
- Is the `SUPPORT_FP8_OPS` gate (torch >= 2.7, ROCm >= 6.4) right, given hipBLASLt only recently added gfx1200/gfx1201 to its FP8 arch list? Tensor-wise vs row-wise scaling cost on RDNA4.
- Is fp16 accumulation a measurable gain on RDNA4 WMMA, and does `PRIORITIZE_FP16` pick the faster dtype there?
- Which comfy-kitchen layouts (tensor-wise INT8, asymmetric W4A8, ConvRot W4A4, FP8) have HIP kernels for gfx1201, and does ComfyUI select them?
- `PYTORCH_TUNABLEOP_ENABLED` and related variables as a `launch` setting: gain, first-run cost, where results are written, interaction with dynamic shapes.

**A3 Memory and offload.** Owns VRAM accounting, offload, pinned memory and the allocator.
- What fits fully in 32 GB under current estimates: Flux.1-dev bf16 and fp8, Wan 2.2 14B fp8, Qwen-Image, LTX-Video. Where does ComfyUI offload or tile although the model would fit?
- Do `VAE_KL_MEM_RATIO = 2.73` and `MIN_WEIGHT_MEMORY_RATIO = 0.4` cause unneeded tiling or partial loads on gfx1201?
- DynamicVRAM and comfy-aimdo on ROCm < 7.14: what users lose, whether the gate is right, what the legacy patcher does instead.
- Async offload stream count over PCIe 5.0, pinned memory limits, and whether `hipMallocAsync` (`backend:cudaMallocAsync`) or `expandable_segments` via `PYTORCH_HIP_ALLOC_CONF` reduce fragmentation in long video runs. Side effects of `OCL_SET_SVM_SIZE`.

**A4 Convolutions and VAE.** Owns conv paths and VAE encode/decode.
- MIOpen off vs on for gfx1201: 2D VAE decode (SDXL, Flux) and causal 3D video VAEs (Wan, Hunyuan, LTX). Speed, peak memory, and first-run find cost (`MIOPEN_FIND_MODE`, user database location and persistence).
- Is the HIP exclusion of the fused `group_norm_silu_pad3d` path still justified on RDNA4?
- channels_last on RDNA4. Tiled decode defaults with 32 GB.
- Propose an arch-gated default only together with the benchmark that justifies it.

**A5 Compile and graphs.** Owns `torch.compile`, the comfy compiler and graph capture.
- Does the compiler/graph path in `comfy/model_prefetch.py` and `comfy/text_encoders/llama.py` work on gfx1201, and under which ROCm and comfy-aimdo versions?
- Inductor Triton codegen on gfx1201. Known: Triton INT8 WMMA kernels fail to compile at `num_stages >= 3` on gfx1201.
- Built-in `TorchCompileModel` node and `--fast autotune` on ROCm: compile time vs per-step gain, recompiles on resolution change, which caches persist across restarts and where.

**A6 Stability.** Owns failure modes. Every finding needs a reproduction or a concrete trigger.
- What the bare `except: pass` around the AMD init block hides: a silent fallback to sub-quadratic attention and no fp8, with nothing logged.
- Behaviour on HIP errors, amdgpu ring timeouts and GPU resets, OOM recovery, and interrupts during a sampling step.
- Black images and NaNs on RDNA4: fp16 VAE, attention upcasting, bf16 vs fp16 defaults.
- Environment variable ordering: the AOTriton flag is only set when `main.py` runs as `__main__`, so embedded and desktop launches can differ.
- Device selection: `HIP_VISIBLE_DEVICES` vs `ROCR_VISIBLE_DEVICES` on multi-GPU and iGPU+dGPU systems.
- Clock and thermal drop on the blower card in long jobs: local observability only, no telemetry.

**A7 Startup and time to first image.** Owns everything between `python main.py` and the first finished image. Measure before proposing.
- `python -X importtime main.py --quick-test-for-ci 2> importtime.log`, wall time to the "To see the GUI go to" log line, and per-phase timing: torch import and ROCm shared libraries, `comfy.model_management` import including the `aotriton_supported()` kernel launch, `comfy_kitchen` and `triton` import, `init_builtin_extra_nodes`, partner/API nodes, custom nodes and prestartup scripts, asset database setup and scan, frontend package resolution.
- First-run costs after startup: AOTriton kernel image load, MIOpen find, TunableOp tuning, Triton JIT, model load from disk (mmap, fast-disk path).
- Separate cold start (empty caches, fresh boot) from warm start. Report the five largest costs in ms with their source lines.
- Proposals must stay local: deferral, lazy import, or persisting caches on disk. No network.

**A8 External research.** Owns extensions and upstream status. Use the web; cite URLs.
- For each candidate verify on its repository: gfx1201/ROCm support claim and evidence, last commit date, licence, what it replaces in core and whether core already covers it, install risk (build from source, pinned torch, patched files), and any network calls it makes.
- Seeds: `pnikolic-amd/ComfyUI_MIGraphX`; `EmbeddedLLM/SageAttention-rocm`, `boxwrench/SageAttention-RDNA4`, ROCm/aiter gfx1201 Sage operators; flash-attention Triton AMD backend; `city96/ComfyUI-GGUF`; `patientx/ComfyUI-INT8-Fast-ROCM`; `iGavroche/rocm-ninodes`; `kijai/ComfyUI-KJNodes`; ComfyUI-MagCache, ComfyUI-TeaCache, WaveSpeed; ComfyUI-MultiGPU; Ultimate SD Upscale and TiledDiffusion. Find others with real gfx1201 evidence.
- Confirm or refute as non-options: Nunchaku, TensorRT, NVFP4/MXFP8 paths, xformers on ROCm.
- Upstream items that unblock gains: hipBLASLt gfx1201 FP8 dispatch, PyTorch `_scaled_mm` fixes for gfx120x, AOTriton gfx1201 kernel coverage, MIOpen RDNA4 performance, the ROCm version matrix for the R9700 on Linux and Windows, Linux kernel/amdgpu versions with known RDNA4 stability fixes.

## Phase 2: verification

After all eight return, give one `general-purpose` verifier agent every finding. For each finding it re-reads the cited code, checks `AGENTS.md` compliance, checks regressions on other platforms, merges duplicates across agents, and tests whether the claim survives (for example an "always" that is gated elsewhere). Verdict per finding: `keep`, `revise` (with the revision) or `drop` (with the reason). It drops anything that needs a new dependency, adds network access to core, or adds a flag no current code reads.

## Phase 3: synthesis

Write `.claude/reports/r9700-analysis.md` and do not commit it. Lead every section with the recommendation. Stay under 400 lines.

1. Top 10 table: rank, title, category, bucket, location, effect and confidence, risk, size.
2. Recommended R9700 launch configuration: exact command line and environment variables, one justification per line, each marked measured or unmeasured.
3. Core change candidates as small, independent PR sketches in priority order, each with the benchmark it must pass before merging.
4. Startup section: cold and warm timings, the five largest costs, proposed fixes with expected savings.
5. Extension shortlist with a verdict (install, try, skip) and the reason.
6. Benchmark protocol for the user's machine. Workflows: SDXL 1024x1024 30 steps; Flux.1-dev fp8 1024x1024 20 steps; Wan 2.2 14B fp8 832x480 33 frames; standalone VAE decode at 1024x1024 and 2048x2048. Metrics: s/it, time to first image, peak VRAM (`torch.cuda.max_memory_allocated` and `rocm-smi`), cold and warm startup. Three runs, report the median. Record `rocminfo` gfx target, ROCm, torch, comfy-kitchen, comfy-aimdo, kernel and amdgpu firmware versions.
7. Dropped ideas, one line each with the reason.
