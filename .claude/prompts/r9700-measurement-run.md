# Measurement run: ComfyUI on 2× R9700 (Windows)

Paste everything below the line into Claude Code on the R9700 machine, started in the directory that holds the production ComfyUI checkout. It measures what the static analysis in `.claude/reports/r9700-analysis.md` (branch `claude/cool-euler-0g3imw`) could only estimate, and returns a keep/revert verdict for each of the 15 branch commits and the three round-2 comfy-kitchen patches (round 1 is already decided).

---

You are running a measurement session on a Windows machine with two AMD Radeon AI PRO R9700 (gfx1201). An earlier static analysis produced 15 ComfyUI commits, a commit of regression tests for them, comfy-kitchen HIP kernel patches in two rounds and a list of launch and system recommendations, all verified on CPU only except kitchen round 1, which is measured and decided. Your job is to measure them on this hardware, decide keep/revert per commit, and turn the estimates into numbers. Correctness first, then speed.

## Known stack (verify, report any drift)

- 2× R9700, 31.9 GiB each. Adrenalin 26.8.1 (32.0.31041.1004).
- Production venv `.venv-rocm-100` (selected by `config/rocm-stack.active` = `prod` in the stack repo): ROCm SDK wheels 10.0.0, `torch.version.hip` 7.15.26333, torch 2.13.0+rocm10.0.0, triton-windows 3.7.1.post27, Python 3.12.12, comfy-kitchen 0.2.36+r9700 (v0.2.36 with round-1 patches 0001 and 0002), comfy-aimdo 0.5.5.
- Production ComfyUI 0.38.0 + 8 commits (`8cfe5e1e`), frontend 1.53.6. Custom nodes: ComfyUI-Manager 3.42, Pixaroma 1.4.182, SolAttn_triton 26d816e. Also in the venv: onnxruntime-directml, insightface, llama_cpp_python.
- Workloads: Krea 2, Qwen-Image 2.1, MiniMax H3 (video + audio, Qwen3-VL-32B text encoder).

## Hard rules

1. **Never modify the production install.** No `pip`/`uv` installs, upgrades or removals in `.venv-rocm-100`. No edits to the production ComfyUI checkout. All code under test lives in git worktrees (below).
2. **Never change system settings.** No driver installs, registry edits (TDR), Defender exclusions, power plans or BIOS. Report them as recommendations only.
3. **Do not stop or restart the production ComfyUI instance yourself.** If it is running and holding a GPU, ask me to stop it before the GPU phases. Test instances use ports 8198 and 8199 only.
4. **One GPU workload at a time per card.** Never run two benchmarks on the same GPU concurrently, and never benchmark while another process uses that card (check with the device query in phase 0 before every block of runs). Subagents may only do work that does not touch the GPUs: reading code, parsing logs, writing the report.
5. **Ask before:** creating a cloned venv (disk heavy), installing the ROCm devel SDK into a build venv (several GB, phase 7), anything above ~50 GB of new disk use, running a test that the tool marks as TDR-risky above 512×512, or resolving a cherry-pick conflict that is not purely mechanical.
6. **Never invent numbers.** Every figure in the results comes from a run you did in this session, with its log file. Anything you could not run is listed as "not measured" with the reason.
7. **Checkpoint after every phase:** append results to `<RESULTS>\results.jsonl` and update `<RESULTS>\r9700-measurements.md`, so the run can resume after a crash or reboot. On resume, read both files and continue at the first incomplete phase.

## Setup

- `<RESULTS>` = `measurements\r9700-<YYYYMMDD-HHMM>` next to the production checkout.
- Fetch the branch from the fork without changing the production checkout's branch:
  ```
  git -C <PROD> remote add feffi https://github.com/feffi/ComfyUI.git   # skip if it exists
  git -C <PROD> fetch feffi claude/cool-euler-0g3imw
  git -C <PROD> worktree add <WT>\baseline 8cfe5e1e
  git -C <PROD> worktree add <WT>\patched 8cfe5e1e
  git -C <WT>\patched cherry-pick d0e865c 85d2562 da79103 dd85c0c 52ed61a 2ed15ee 84cfd41 ff0e009 ded5499 d7b3369 377a3f9 e9c38f9 e997bcb 3781fb9 91d15cb
  git -C <WT>\patched cherry-pick b09048c 37b7bce        # regression tests only, no runtime change
  git -C <PROD> worktree add <WT>\tests-baseline 8cfe5e1e
  git -C <WT>\tests-baseline cherry-pick b09048c 37b7bce  # the same tests against unpatched code
  git -C <PROD> show feffi/claude/cool-euler-0g3imw:.claude/reports/r9700-analysis.md > <RESULTS>\analysis.md
  git -C <PROD> archive feffi/claude/cool-euler-0g3imw .claude/reports/r9700/tools | tar -x -C <RESULTS>
  git -C <PROD> archive feffi/claude/cool-euler-0g3imw .claude/kitchen-r9700 .claude/reports/kitchen-r9700.md .claude/reports/kitchen-r9700-2.md | tar -x -C <RESULTS>
  ```
  The branch is based on e9027f2, not 8cfe5e1e. If a cherry-pick conflicts, resolve it only when the conflict is mechanical (context drift) and log the resolution; otherwise skip that commit, mark it "not applied", and ask me.
- Both worktrees run with the production venv's python, with `--base-directory` or `--extra-model-paths-config` pointing at the production models, `--disable-auto-launch`, and their own `--database-url sqlite:///<RESULTS>/<name>.db`. Custom nodes: load the same set as production (copy or junction `custom_nodes` read-only) so baseline and patched differ only by the commits.
- The tools in `<RESULTS>\.claude\reports\r9700\tools\` were only smoke-tested on CPU. Fix Windows breakage in your copy if needed and log each change.
- Tools that take `--comfy` / `--comfy-root` (`vae_bench.py`, `run_vae_matrix.py`, `run_placement_bench.py`, `gemm_bench.py`) run once per worktree, with `--python` pointing at the production venv where the tool asks for it. `krea2_block_bench.py` imports Krea 2 from the current directory and compares it with its bundled copies `krea2_branch.py` (= `d0e865c`) and `krea2_fused.py` (= `377a3f9`), so run it once from `<WT>\baseline`.
- Workflows: my saved workflows per model in `<PROD>\user\default\workflows` if present (ask which ones to use if there are several), else the official templates at default settings. Fixed seeds, fixed prompts. Save the exact workflow JSON used into `<RESULTS>\workflows\`.

## Method

- Every timed configuration: 1 warm-up run (discarded), then 3 measured runs; report median and min–max. Treat differences under 3 % as noise unless all runs separate cleanly.
- Measure the noise floor first: baseline vs baseline output (same seed, two runs) as PSNR and max abs diff. Judge patched vs baseline against that floor, not against zero.
- Record per run: s/it, total time, time to first image, VAE decode/encode time, text-encode time, peak VRAM per GPU (ComfyUI `/system_stats` polling plus torch peak where a tool reports it), and any log warning.
- The driver bug can page idle VRAM to RAM after ~9.5 s (TheRock#7221). Keep runs back to back, record idle gaps, and flag any run whose first step is much slower than the rest.
- Log versions with every result block: driver, ROCm wheel, HIP, torch, comfy-kitchen, comfy-aimdo, worktree commit.

## Phase 0: preflight (no heavy GPU load)

1. `tools\A6\a6_system_check.ps1`: driver, TDR values (read only), device order and `gcnArchName` per device, AOTriton probe with torch's failure reason, onnxruntime providers, versions against `requirements.txt`.
2. Start each worktree once (`--cuda-device 0 --port 8199`) and capture the startup log. Record the attention line: `Using pytorch attention` or `Using sub quadratic optimization`, and any `Could not run flash attention` reason. Also record `AMD arch`, `ROCm version`, the kitchen backend lines and `DynamicVRAM`.
3. Device visibility: baseline without flags vs patched without flags (`52ed61a`): how many devices `/system_stats` lists.
4. `tools\A6\idle_pageout_check.py --gib 8 --idle 20 --devices 0,1`: does the idle page-out reproduce on 26.8.1?

**Gate.** If the startup log shows sub-quadratic attention, this is the top finding. Run `tools\A1\attn_bench.py --device 1 --explain` for the three models' shapes and add a launch variant with `--use-pytorch-cross-attention` to phases 2 and 4. ComfyUI#16526 crashed only on Anima's direct SDPA call, so SDPA may work for these three models even though the probe fails. Check the output correctness of that variant before trusting its speed.

## Phase 1: correctness of the commits

### 1a. Regression tests

Run these before any workflow. They are CPU unit tests plus one small SDPA call on the GPU, so card 1 only has to be idle (phase 0 device query). Use the production venv's python, set `CUDA_VISIBLE_DEVICES=1` and `HIP_VISIBLE_DEVICES=1` for the test process only, and run from `<WT>\patched`, then from `<WT>\tests-baseline`:
```
python -m pytest -p no:cacheprovider -rs tests-unit\comfy_test\test_krea2_model.py tests-unit\comfy_test\test_qwen_image21_norm.py tests-unit\comfy_test\test_vae_single_frame.py tests-unit\comfy_test\test_audio_upsample.py tests-unit\comfy_test\test_ops_attention_compile.py tests-unit\comfy_extras_test\select_clip_device_test.py tests-unit\main_rocm_windows_gpu_test.py
```
Save the output to `<RESULTS>\logs\unit-tests-<worktree>.log`. If `import pytest` fails in the production venv, run `python -m pip install --target <RESULTS>\pydeps pytest` and put `<RESULTS>\pydeps` on `PYTHONPATH` for these runs. Never install into the venv.

| Test file | Guards |
|---|---|
| `test_krea2_model.py` | `d0e865c` (norm dtype, unexpanded GQA K/V, patchify layout), `377a3f9` (fused = unfused) |
| `test_qwen_image21_norm.py` | `dd85c0c` |
| `test_vae_single_frame.py` | `ded5499`, `ff0e009` |
| `test_audio_upsample.py` | `91d15cb` |
| `test_ops_attention_compile.py` | `3781fb9`; `d7b3369` (the only GPU test) |
| `select_clip_device_test.py` | `2ed15ee` |
| `main_rocm_windows_gpu_test.py` | `52ed61a` |

Pass conditions:
- **patched:** everything passes and nothing is skipped. A skipped `test_unmasked_gqa_without_native_backend_expands_kv` means the test process sees no GPU: fix the environment and rerun. A failure marks the guarded commit **revert**, unless it is a Windows harness problem (path, shell, encoding), which you fix in the worktree copy and log as with the tools. A commit you did not apply fails its tests by design; list those failures as such.
- **tests-baseline:** each commit's guard fails and the equivalence tests pass. The GQA SDPA test failing here is the on-hardware evidence for `d7b3369`: with the baseline wrapper, unmasked fp32 GQA reaches torch SDPA unexpanded although no fused backend takes it, so it runs on the math kernel. If that test passes on baseline, record why from the log; `d7b3369` then has no effect on this stack.

Commits without a unit test (`85d2562`, `da79103`, `84cfd41`, `e9c38f9`, `e997bcb`) rely on the GPU checks below.

### 1b. GPU checks

Per commit group, compare patched against baseline (and against the noise floor):

| Commits | Test | Pass condition |
|---|---|---|
| `377a3f9` (fused Krea 2 kernels) | from `<WT>\baseline`: `tools\A2\krea2_block_bench.py --device 1 --tokens 4608`, `--batch 1` and `--batch 2` | fused block `rel_err` vs fp32 in line with the unfused block (CPU reference: ~6e-3 in bf16) |
| `d0e865c`, `d7b3369` | Krea 2 workflow, same seed | PSNR vs baseline within the noise floor band; no NaN/flat image |
| `d0e865c` (sparse override now expands GQA K/V itself) | Krea 2 workflow, same seed, with core Model Sparse Attention (method sol-attn, defaults, `verbose` on), both worktrees: 1024² with `min_tokens` 4096, then 2048² with default `min_tokens`. Check `ck.sol_attn_is_available` first. Baseline Krea 2 expanded K/V before the override, so it already ran sparse and is the reference. Never apply SolAttn_triton (Patch Sol-Attn) to Krea 2: whether it reads `enable_gqa` is unverified | patched log shows `BlockSparseAttention: sparse (1, <tokens>, <heads>, 128)` for the joint text+image sequence and no `kept dense` line for that shape (the text-fusion layers staying dense below `min_tokens` is expected); at 1024², PSNR patched sparse vs baseline sparse no worse than patched dense vs baseline dense from the row above; no NaN/flat image. At 2048², report s/it and PSNR of sparse vs dense in both worktrees |
| `85d2562`, `da79103`, `84cfd41` | MiniMax H3 VAE encode, 17+ frame clip and a single keyframe (`tools\A4\vae_bench.py --model minimax --op encode --gpu 1 --comfy <worktree>`, both worktrees) | latent rel diff vs baseline ≤ 2e-3, no NaN |
| `ff0e009`, `ded5499` | VAE decode Krea 2 (Wan 2.1) and Qwen 2.1 at 1024² (`tools\A4\vae_bench.py --gpu 1 --comfy <worktree>`, both worktrees) | output rel diff ≤ 1e-5 in fp32, identical or near-identical in bf16 |
| `dd85c0c` | Qwen-Image 2.1 workflow, same seed | within noise floor |
| `91d15cb` (polyphase audio upsample) | MiniMax H3 audio decode of a 10 s clip, both worktrees; LTX audio if you use it | waveform rel diff ≤ 1e-5 (fp32), no NaN; decode time against baseline |
| `2ed15ee`, `52ed61a` | MiniMax H3 with Select CLIP Device → gpu:1 (`tools\A3\run_placement_bench.py --model minimax --placement split`) | during encode GPU 1 peak rises by about the encoder size and GPU 0 does not load the encoder |
| `e997bcb` | startup logs from phase 0 | same attention/fp8 state as baseline, or a clearer error |
| all | `tools\A6\flat_output_check.py <output dir>` over every output | exit code 0 |

A commit that fails correctness is marked **revert** immediately and excluded from the speed phase (build a `patched-minus-<commit>` worktree with `git revert` if needed).

## Phase 2: speed, baseline vs patched

For each workload (Krea 2, Qwen-Image 2.1, MiniMax H3 short clip), on GPU 1 alone unless the plan needs both:

1. End-to-end workflow: baseline vs patched (plus the `--use-pytorch-cross-attention` variant if the phase 0 gate fired).
2. Targeted micro-benchmarks for attribution:
   - VAE decode/encode, baseline vs patched: `tools\A4\vae_bench.py`, `tools\A4\vae_conv_bench.py --profile` (kernel counts per call).
   - Krea 2 block, unfused vs fused: `tools\A2\krea2_block_bench.py`.
   - GEMM paths: `tools\A2\gemm_bench.py --model <diffusion model file> --tokens <tokens> --device 1` per model. This also decides whether the fp8, int8 or bf16 checkpoint is fastest per model.
   - Top kernels per denoising step: `tools\A2\comfy_profile_step` (load only into the patched test instance via `extra_paths_profile.yaml`).
3. Only if a workload is slower on patched beyond noise: bisect over the commit list with additional worktrees until the responsible commit is found, and mark it.

## Phase 3: dual-GPU plans

1. MiniMax H3, single card vs split (DiT on GPU 0; text encoder and VAEs on GPU 1): `run_placement_bench.py --model minimax --placement single|split`, three prompts in a row. Report pass 2 and 3 sampling time, whether the DiT is evicted between prompts, GPU peaks, and text-encoder load time with the int8_convrot encoder.
2. Two instances, one per card, Krea 2 and Qwen 2.1: images per hour over 10 minutes, compared with one instance on one card. Watch system RAM (pinned memory up to 40 % per instance).
3. CFG Split at cfg 4, batch 2, baseline vs patched (`e9c38f9`).
4. `tools\A3\vram_probe.py`: P2P, host bandwidth, whether `hipMemGetInfo` sees another process's allocation, how much RAM pins.

## Phase 4: launch and config A/Bs (patched worktree)

Each against the phase 2 patched result, same workflows:

- MIOpen: `tools\A4\run_vae_matrix.py --gpu 1 --res 512` first, then 1024 if no TDR. Off (default) vs on vs `MIOPEN_FIND_MODE` variants, cold vs warm find-db.
- TunableOp for bf16 Krea 2 and Qwen 2.1: one tuning pass with `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=1`, `PYTORCH_TUNABLEOP_FILENAME=<RESULTS>\tunableop\results%d.csv` and `--disable-cuda-graphs`, clean exit; then measured runs with `PYTORCH_TUNABLEOP_TUNING=0`.
- Compiler and graphs: `tools\A5\ab_compile.py` variants `baseline`, `no_compiler`, `no_graphs`, `hwq2` (`GPU_MAX_HW_QUEUES=2`). TorchCompileModel only for Krea 2 and Qwen 2.1, never H3.
- MiniMax H3 attention: dense vs core Model Sparse Attention (sol) vs SolAttn_triton (Patch Sol-Attn). Check `ck.sol_attn_is_available` first. Report s/it, peak VRAM, PSNR vs dense, and an audio sanity check (RMS and a listen note from me if you need one).
- `--fast fp16_accumulation` on Krea 2: confirm no speed gain and check for NaN/flat output (expected: Krea 2 switches to fp16, no clamp).
- Optional, ask first: SageAttention-RDNA4 v0.2.0 in a cloned venv (`--no-deps` wheel install into the clone only), Krea 2 only, `--use-sage-attention`.

## Phase 5: startup

`tools\A7\profile_startup.ps1 -Runs 3 -Label warm` for both worktrees, plus `-Runs 1 -Label cold` after I reboot (ask me when you are ready; do not reboot yourself). Then the same with the Manager in offline mode, applied in a copy of the Manager config used only by the test instance, and with `--disable-partner-nodes`. Report the five largest costs with their source, Windows DLL preload and AOTriton probe rows included.

## Phase 6: gaps the static analysis did not cover

1. **MiniMax H3 audio VAE** (`comfy\ldm\minimax\audio_vae.py`). `91d15cb` replaced the per-channel transposed-conv upsample (18,296 single-channel convs per decode with MIOpen off). Time audio encode and decode for a 10 s and a 60 s clip in both worktrees, MIOpen off vs on, and profile the top kernels. Expected remaining cost: 12,192 per-channel bias fills from the 42 dilated convs per decode (7,936 more per reference-audio encode). Report their share; describe a fix, do not change code in this run.
2. **Text-encoder speed:** encode time per prompt for Qwen3-VL-4B (Krea 2), 8B (Qwen 2.1) and 32B int8_convrot (H3), on GPU 0 vs GPU 1, cold and warm.
3. **Power and thermals:** if a local tool exposes GPU clock, power and temperature (look for `amd-smi` in the ROCm wheels or any vendor CLI already installed; install nothing), log them at 1 Hz during a 10-minute H3 run on both cards. Report whether clocks sag over time.
4. **Coexistence:** does importing onnxruntime-directml or insightface in the same process change device enumeration, VRAM or startup time? Compare a test instance with and without the custom nodes that pull them in.

## Phase 7: comfy-kitchen patches

Round 1 (`<RESULTS>\.claude\reports\kitchen-r9700.md`) is decided; do not re-measure it: 0001 (WMMA GEMM tile loads) and 0002 (small-M fp8 GEMV) are kept and in production as 0.2.36+r9700 (Krea 2 block fp8 GEMMs 45.46 -> 42.78 ms), 0003 (tile tuner) and 0004 (quantize encode) are dropped. Round 2 (`<RESULTS>\.claude\reports\kitchen-r9700-2.md`) is three patches on top of 0001 + 0002: 0005 rms_adaln/adaln fast path, 0006 rms_rope fast path for head_dim 128, 0007 a fixed 256x128x128 fp8 GEMM tile. All three must give byte-identical output, also across process restarts, and none chooses anything by timing. Each touches one kernel and each script isolates one patch: `bench_rms_adaln.py` measures 0005, `bench_rms_rope.py` 0006, `bench_fp8_gemm.py` 0007.

Where the kernels run: Qwen-Image 2.1 calls `rms_rope` and `adaln` and MiniMax H3 calls `rms_rope_split_half_` at the production commit. Krea 2 calls `rms_rope` and `rms_adaln` only with branch commit `377a3f9`; production Krea 2 uses `F.rms_norm` and `apply_rope`. If phases 1-2 revert `377a3f9` or it is not applied in `<WT>\patched`, say that the Krea 2 norm rows serve no production call and that Krea 2 checks only 0007 end to end; 0006 is then judged on Qwen 2.1 and H3, and 0005 on Qwen 2.1's LayerNorm path through `ck.adaln`, with no end-to-end check for its `rms_adaln` path.

How to run: from `<RESULTS>\.claude\kitchen-r9700\` with the production venv's python, on GPU 1 alone; the production instance must not use that card. Environment variables do not reliably survive between your shell calls, so set the build in the same command line as the process it applies to. In PowerShell: `Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue; python ...` for the installed kitchen, `$env:PYTHONPATH = "<RESULTS>\kitchen\patched"; python ...` for the candidate. The commands below are PowerShell; from Git Bash use `env -u PYTHONPATH python ...` and `PYTHONPATH=<RESULTS>/kitchen/patched python ...`, and let bash expand the globs that step 2 passes through `Get-ChildItem`. Every script prints a `kitchen_path` line; check it on every run. The scripts repeat each shape internally and print the median; the min-max spread is only in their `--json` output, which replaces the table. So pass `--json` to every script run below and save each output under `<RESULTS>\kitchen\` (append it to `results.jsonl` too). `--json` also drops the derived columns and the block total lines, so compute them from the JSON rows: % of peak is `gbs / 6`, k/blas is `kitchen_ms / hipblaslt_ms`, and a Krea 2 block total is the sum of `kitchen_ms * uses` over one `krea2 M=...` group. Run a script once more if a row's (max - min) / median exceeds 3 %. `bench_fp8_gemm.py`'s `"tile": "tuned"` field and its `--sweep-tiles` option are round-1 leftovers; this build has no tuner, so ignore both.

1. **Reference:** with the installed kitchen, record what the production venv imports: `python -c "import importlib.metadata as md, comfy_kitchen, os; print(md.version('comfy-kitchen'), os.path.dirname(comfy_kitchen.__file__))"` (kitchen has no `__version__`, which is why the scripts' info lines print `"kitchen": "?"`). Expected: 0.2.36+r9700, but a local build need not carry the label. Settle it with step 3's reference GEMM rows: the small-M row M = 2, 4096 -> 16384 near 0.08 ms means 0002 is in, near 0.40 ms means plain 0.2.36. In that case the candidate also carries 0001 + 0002, so the GEMM rows measure 0001 + 0002 + 0007 together; say so in the results. Also record `python -c "import torch; print(torch.cuda.get_device_properties(1).multi_processor_count)"`: 0007's threshold reads this count, expected 32.
2. **Build** as in `<RESULTS>\.claude\kitchen-r9700\README.md`, with every path under `<RESULTS>\kitchen\` instead of `C:\kt`. Native programs get no wildcard expansion in PowerShell, so pass the files explicitly:
   ```
   git clone https://github.com/Comfy-Org/comfy-kitchen <RESULTS>\kitchen\comfy-kitchen
   git -C <RESULTS>\kitchen\comfy-kitchen checkout -b r9700-2 888b13e2c0e721f6576fe351a2ad79894b1c451f
   git -C <RESULTS>\kitchen\comfy-kitchen am (Get-ChildItem <RESULTS>\.claude\kitchen-r9700\patches\000[12567]-*.patch).FullName
   git -C <RESULTS>\kitchen\comfy-kitchen log --oneline 888b13e..HEAD
   git -C <RESULTS>\kitchen\comfy-kitchen grep -n -E "hipEventElapsedTime|FP8_TUNE" -- comfy_kitchen/backends/hip
   ```
   The log must list exactly five commits (0001, 0002, 0005, 0006, 0007; not 0003 or 0004) and the grep must print nothing: 0003's tuner was the only timing-based choice. Then a gfx1201-only wheel as in the README, installed with `pip install --no-deps --target <RESULTS>\kitchen\patched (Get-ChildItem <RESULTS>\kitchen\comfy-kitchen\dist\comfy_kitchen-0.2.36-*.whl).FullName`. Only the build venv gets packages; ask before installing the ROCm devel SDK there. Log the build to `<RESULTS>\logs\kitchen-build.log`. Fix Windows-only build breakage in the clone if it is mechanical and log it; anything else marks the phase "not measured" with the error.
3. **Identity:** with the installed kitchen, write the references:
   ```
   python bench_rms_rope.py --device 1 --json --save-ref <RESULTS>\kitchen\rope_ref.pt
   python bench_rms_adaln.py --device 1 --json --save-ref <RESULTS>\kitchen\adaln_ref.pt
   python bench_fp8_gemm.py --device 1 --json --save-ref <RESULTS>\kitchen\gemm_ref.pt
   ```
   Then with the candidate, the same three with `--ref <RESULTS>\kitchen\<name>_ref.pt --save-ref <RESULTS>\kitchen\<name>_cand.pt`, and once more, each in a new process, with `--ref <RESULTS>\kitchen\<name>_cand.pt`. Pass: `identical` is `True` in every row of both candidate runs, and in the two norm scripts `err64` (max error against fp64 on the first 64 tokens; the GEMM script has none) is the same in all runs of a row. A `False` against the reference belongs to the patch its script measures; a `False` between the two candidate runs means that patch's output changes across restarts, which is a drop.
   Two exceptions before calling a reference mismatch a drop. (a) Fallback rows differ as well (norm rows whose `fast` column reads `--` or `-`, or GEMM rows that 0007 does not serve, see step 4): your toolchain compiles the unchanged kernels differently from the one that built production, so the reference is unusable, not the patch. Build 0001 + 0002 alone the same way into `<RESULTS>\kitchen\ref-build` and repeat this step with it as the reference (`PYTHONPATH=<RESULTS>\kitchen\ref-build` for the reference runs). (b) Every fallback row is `True` and only rows showing the new path (`qk`, `y`) are `False`: the fast paths reproduce the rounding of the official 0.2.36 Windows wheel, and a production build compiled differently rounds its old kernels differently. Fetch the official wheel with the build venv's pip (`pip download comfy-kitchen==0.2.36 --no-deps --only-binary :all: -d <RESULTS>\kitchen\v0236-whl`, then `pip install --no-deps --target <RESULTS>\kitchen\v0236` that wheel), run that script with `PYTHONPATH=<RESULTS>\kitchen\v0236` and `--save-ref <RESULTS>\kitchen\<name>_ref_v0236.pt` (check that `kitchen_path` shows `v0236`; do not overwrite the references above), then the candidate with `--ref` on that file. If every new-path row is identical there, the patch reproduces 0.2.36 but would change production renders once, by about one ulp per call: mark it inconclusive for me to decide and report step 6's diff against the installed kitchen. Anything else: drop.
4. **Speed:** ms per row for reference and candidate, plus GB/s and % of the ~600 GB/s peak for the two norm scripts and hipBLASLt for the GEMM. Expected: Krea 2 `rms_rope` 1.59 -> ~0.6 ms per q+k call, the `h3 S=39520` row (`rms_rope_split_half_` in place) ~3.8-4 ms with no earlier time to compare, Krea 2 `rms_adaln` 0.85 -> ~0.45 ms, and the 256x128 tile 1-3 % on the Krea 2 GEMMs. Compare each Krea 2 block total (one `krea2 M=...` group) with this session's reference for the same M; 42.78 ms only checks that the reference is the production build. The `fast` column is the launcher's test applied to the bench's own inputs and reads the same in both builds; it must show `qk` on every Krea 2 and H3 `rms_rope` row and `y` on the Krea 2 `rms_adaln` rows, and whether model calls reach the new kernels is step 6's kernel count. Rows that keep the old kernels must not get slower beyond noise: norm rows with `--` or `-`, and the GEMM rows that 0007 does not serve. 0007 takes a GEMM when K >= 4096 and ceil(M/256) * ceil(N/128) >= 8 x the step 1 count (256 blocks at 32). That puts every Krea 2 and Qwen 2.1 row and small-M M = 335 (exactly 256 blocks) on the new tile, M = 16 and 77 (128 blocks) on the old one, and M <= 8 including the 2 x 6144 -> 36864 row on 0002's GEMV. With another count, recompute the split and say so. Report M = 335 and the Qwen 2.1 rows apart from Krea 2, each with its block count. M = 335 sits exactly at the threshold with 256 blocks and pads its 335 rows to 512, where the old tile pads to 384. The Qwen 2.1 rows (512, 512 and 3072 blocks; Krea 2's k/v rows have 492) run at M = 4096, and the tile sweep that chose 0007 ran only at M = 10264. If 0007 is slower beyond noise only on those rows and faster on Krea 2, mark it inconclusive, not drop, and list both builds' ms per row.
5. **The K = 16384 question:** from the candidate's GEMM rows, the mlp down row (10264 x 16384 -> 6144) in ms and TFLOP/s; the GEMM runs alone there, without the activation quantize. Then time both quantize paths for that GEMM's input, with the installed kitchen:
   ```
   python -c "import torch; torch.cuda.set_device(1); import comfy_kitchen; from bench_fp8_gemm import time_ms; from comfy_kitchen.backends import hip as ck; print(comfy_kitchen.__file__); x = torch.randn(10264, 16384, device='cuda:1', dtype=torch.bfloat16); s = torch.tensor(0.01, device='cuda:1'); print('kitchen', time_ms(lambda: ck.quantize_per_tensor_fp8(x, s, torch.float8_e4m3fn), 3, 20)); print('torch', time_ms(lambda: torch.clamp(x, min=-448, max=448, out=x).to(torch.float8_e4m3fn), 3, 20))"
   ```
   With `fp8_e4m3fn_fast` weights the model takes the torch line (`fp8_linear` in `comfy\ops.py`: an in-place clamp, then a cast, ~1.2 GB of traffic for this input); a scaled fp8 checkpoint with a quant config takes kitchen's quantize (`comfy\quant_ops.py`, ~0.5 GB). Report the time of the path step 6's weights take next to the GEMM time, and say whether the remaining gap at K = 16384 is the GEMM or the quantize.
6. **End to end:** Krea 2 and Qwen-Image 2.1 loaded with weight_dtype `fp8_e4m3fn_fast`. bf16 weights never reach the fp8 GEMM, and a plain fp8 checkpoint loaded with `default` runs `manual_cast` in bf16 unless the launch flags include `--fast` (or `--fast fp8_matrix_mult`), which phase 2's configuration does not. A scaled fp8 checkpoint (startup log `Using mixed precision operations`) reaches it only in some layers, so use one only if the kernel count below shows kitchen's fp8 GEMM kernels. `rms_rope`, `rms_adaln` and `adaln` run with any weight dtype. Plus the MiniMax H3 short clip from phase 2. All on `<WT>\patched` with phase 2's launch configuration (MIOpen off, no TunableOp) on GPU 1 (`--cuda-device 1 --port 8199`). Run each workload four times, each in a freshly started instance, same seeds: installed, installed, candidate, candidate. For every instance log the loaded extension, `(Get-Process -Id <pid>).Modules | Where-Object FileName -like '*comfy_kitchen*' | Select-Object -ExpandProperty FileName`: the production venv for the installed runs, `<RESULTS>\kitchen\patched` for the candidate. Pass: max abs diff 0 on the output images, decoded frames and H3 audio within each pair (renders must not change across restarts) and between installed and candidate. If the installed pair already differs, the stack is not restart-deterministic outside kitchen: report what differs, mark end-to-end identity inconclusive and judge the patches on the bench rows. Under step 3's exception (a), run `ref-build` in place of the installed kitchen; under exception (b), installed against candidate is expected to differ: report max abs diff and PSNR. Report s/it and total for every run.
   If the candidate reproduces itself but differs from the installed kitchen while every bench row was identical, attribute it before dropping anything: build 0001 + 0002 + one of 0005, 0006, 0007 into `<RESULTS>\kitchen\only-0005`, `only-0006` and `only-0007` (the README says each applies alone), rerun the differing workload with each, and drop only the patch whose build reproduces the difference. Krea 2 and Qwen 2.1 exercise all three (Qwen 2.1 reaches 0005 through `ck.adaln`; both reach 0007 only with fp8 weights), H3 runs 0006 (and 0007 only with fp8 DiT weights).
   Then count which kernels the model calls reach. With the candidate, run each workload once more with `tools\A2\comfy_profile_step` loaded as in phase 2 (a separate run, not one of the timed or identity runs), on Krea 2 without sparse attention or any other attn1 patch and without reference latents, since those paths skip `rms_rope` and `rms_adaln`. `kernels.txt` lists only the top 40 kernels, so count by name in `trace.json`: `rms_rope128_kernel` against `rms_rope_kernel`, `adaln_wave_kernel` against `adaln_kernel`, and the `gemm_wmma_kernel` instance with `256, 128, 128` in its template arguments against the other GEMM kernels, hipBLASLt (`Cijk_`) included. Expected per Krea 2 model call: 56 `rms_rope128_kernel` (28 blocks, q and k launched separately), 57 `adaln_wave_kernel` (two per block plus the last layer), no `rms_rope_kernel` or `adaln_kernel`, and, at 1280x2048 (M = 10264 per image), the 8 block Linears of each block on the 256x128 tile (224 launches). At other sizes apply step 4's rule to M = batch x tokens from the trace: at 1024x1024 with batch 1, for example, wk and wv have 204 blocks and stay on the old tile, which is expected and not a fallback. Qwen 2.1 has no bench row, so this count is the only evidence that 0005 and 0006 serve it. Report every old-kernel launch left in the candidate with its count and call site. A patch whose kernel does not run in a workload has no end-to-end evidence there: report it as not exercised instead of counting that identity as a pass.

Verdict per patch (0005, 0006, 0007): keep / drop / inconclusive. Keep needs `identical` in every row of its script in both candidate runs, max abs diff 0 end to end in every workload where step 6 counted its kernel, a gain on the rows it serves by the Method's rule (beyond 3 %, or smaller with min-max ranges of the two builds that do not overlap), and no row slower beyond noise. Identical but not measurably faster is inconclusive, as are step 3's exception (b), a non-deterministic installed pair, and 0007's case in step 4 where only the rows outside the sweep lose.

## Results

Write `<RESULTS>\r9700-measurements.md` (under 300 lines) and keep `results.jsonl` with every raw run. Lead each section with the decision:

1. **Verdict table per commit:** keep / revert / inconclusive, with the unit test result (patched and tests-baseline), the correctness numbers and the speed effect in its workload.
2. **Gates:** attention backend state and reason, idle page-out yes/no, device visibility.
3. **Per workload:** baseline vs patched for s/it, total, VAE, text encode, peak VRAM per GPU, PSNR vs noise floor.
4. **Recommended launch configuration and two-GPU plan**, updated with measured numbers, each line marked measured or not measured.
5. **Config A/B outcomes:** MIOpen, TunableOp, compiler/graphs, `GPU_MAX_HW_QUEUES`, sparse attention variants, Sage if run.
6. **Startup:** cold and warm, top costs, savings per action.
7. **New findings** from phase 6, with proposed fixes described in the same format as the analysis report (location, change, effect, risk).
8. **Corrections** to the analysis report: every estimate that the measurements contradict.
9. **comfy-kitchen patches:** verdict per round-2 patch; the reference kitchen (version, path, whether it carries 0001 + 0002) and the device's WGP count; `identical` per script in both candidate runs and `err64` for the two norm scripts (and the ref-build or official-wheel comparison if step 3 needed one); ms per row for reference and candidate (GB/s for the norm scripts, hipBLASLt for the GEMM), with M = 335 and the Qwen 2.1 GEMM rows apart from Krea 2; the Krea 2 block GEMM total per M; mlp down TFLOP/s, both quantize times, the path the model takes and whether the K = 16384 gap is the GEMM or the quantize; end to end: restart identity per build, installed against candidate, s/it; kernel counts per workload and every patch not exercised; whether the Krea 2 gains depend on `377a3f9`.
10. **Not measured:** what you skipped and why.

Do not commit or push anything. When finished, leave all worktrees in place (baseline, patched, tests-baseline and any bisect ones) and `<RESULTS>\kitchen\` (clone, build venv, patched build, and any ref-build, only-000x builds or official 0.2.36 wheel that phase 7 created), and tell me the cleanup commands (`git worktree remove`, the folders to delete) instead of running them.
