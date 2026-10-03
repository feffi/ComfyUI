# Measurement run: ComfyUI on 2× R9700 (Windows)

Paste everything below the line into Claude Code on the R9700 machine, started in the directory that holds the production ComfyUI checkout. It measures what the static analysis in `.claude/reports/r9700-analysis.md` (branch `claude/cool-euler-0g3imw`) could only estimate, and returns a keep/revert verdict for each of the 15 branch commits.

---

You are running a measurement session on a Windows machine with two AMD Radeon AI PRO R9700 (gfx1201). An earlier static analysis produced 15 ComfyUI commits, a commit of regression tests for them and a list of launch and system recommendations, all verified on CPU only. Your job is to measure them on this hardware, decide keep/revert per commit, and turn the estimates into numbers. Correctness first, then speed.

## Known stack (verify, report any drift)

- 2× R9700, 31.9 GiB each. Adrenalin 26.8.1 (32.0.31041.1004).
- Production venv `.venv-rocm-100` (selected by `config/rocm-stack.active` = `prod` in the stack repo): ROCm SDK wheels 10.0.0, `torch.version.hip` 7.15.26333, torch 2.13.0+rocm10.0.0, triton-windows 3.7.1.post27, Python 3.12.12, comfy-kitchen 0.2.36, comfy-aimdo 0.5.5.
- Production ComfyUI 0.38.0 + 8 commits (`8cfe5e1e`), frontend 1.53.6. Custom nodes: ComfyUI-Manager 3.42, Pixaroma 1.4.182, SolAttn_triton 26d816e. Also in the venv: onnxruntime-directml, insightface, llama_cpp_python.
- Workloads: Krea 2, Qwen-Image 2.1, MiniMax H3 (video + audio, Qwen3-VL-32B text encoder).

## Hard rules

1. **Never modify the production install.** No `pip`/`uv` installs, upgrades or removals in `.venv-rocm-100`. No edits to the production ComfyUI checkout. All code under test lives in git worktrees (below).
2. **Never change system settings.** No driver installs, registry edits (TDR), Defender exclusions, power plans or BIOS. Report them as recommendations only.
3. **Do not stop or restart the production ComfyUI instance yourself.** If it is running and holding a GPU, ask me to stop it before the GPU phases. Test instances use ports 8198 and 8199 only.
4. **One GPU workload at a time per card.** Never run two benchmarks on the same GPU concurrently, and never benchmark while another process uses that card (check with the device query in phase 0 before every block of runs). Subagents may only do work that does not touch the GPUs: reading code, parsing logs, writing the report.
5. **Ask before:** creating a cloned venv (disk heavy), anything above ~50 GB of new disk use, running a test that the tool marks as TDR-risky above 512×512, or resolving a cherry-pick conflict that is not purely mechanical.
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
9. **Not measured:** what you skipped and why.

Do not commit or push anything. When finished, leave all worktrees in place (baseline, patched, tests-baseline and any bisect ones) and tell me the cleanup commands (`git worktree remove`) instead of running them.
