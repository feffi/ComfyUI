# comfy-kitchen HIP kernels on the R9700 (gfx1201): fp8 GEMM first

Stand: 04.10.2026. Base: comfy-kitchen v0.2.36 (`888b13e2c0e7`); the touched files are
unchanged in v0.2.37. Patches, benchmark, quantize check and build steps:
`.claude/kitchen-r9700/`.

**Read this first.** No GPU here. Nothing below was timed. "Verified" means one of
three things done in this container: the ISA from compiling kitchen's sources for
gfx1201 with clang 20 (calibrated against the gfx1201 code object in the 0.2.36
Windows wheel, built with AMD clang 22: same VGPR count and LDS size, loop mix within
a few instructions); a CPU emulation that runs kitchen's real kernel source; or a
read of the source. Every speed effect is an estimate until `bench_fp8_gemm.py` runs.

## Result

| Patch | What | Output | Expected effect |
|---|---|---|---|
| 0001 | GEMM core: branch-free, coalesced tile loads | bit-identical (emulated) | big M: a few % (per K tile and wave, same compiler: 45 VALU, 42 SALU, 11 branches -> 24, 8, 3); applies to every WMMA GEMM user |
| 0002 | fp8 GEMV, M <= 8: hardware decode, weight read once | bit-identical (emulated, FMA order checked in ISA) | M~2, K=4096, N=16384: from 0.55 ms towards the 0.10 ms bandwidth floor (hipBLASLt 0.11) |
| 0003 | six more tiles, measured tile choice per shape class | bit-identical by construction; tiles 0-5 emulated, 6-10 pending | big M, K <= 6144: 0 to ~15 % depending on whether LDS bandwidth limits today's tile; never slower than today's choice by more than timing noise |
| 0004 | per-tensor quantize: hardware e4m3 encode | **unverified** | quantize goes from VALU-bound (~55 VALU and ~7 branches per element) towards memory-bound; run `check_fp8_quantize.py` first |

Bit identity of 0001-0003 holds by construction: every output element still runs
through one accumulator that receives the same WMMA K-steps in the same order, so a
tile's shape changes only scheduling. The emulator confirms it on the source: 17
shapes (GEMV, every tile path, partial M and N tiles, K tails, f32/f16/bf16 out,
with and without bias) for v0.2.36 and for 0001 and 0002: 0 differing bytes. For
0003, tiles 0-5 forced in turn: 0 differing bytes; tiles 6-10 forced and the tuned
default were still running in the emulator at this commit (the tuned run had passed
13 of 17 cases).

## Why kitchen loses to hipBLASLt at K <= 6144

The 0.2.36 kernel for these shapes is 128x128 with BKB 128, 8 waves of 32x64. Its
K loop, per 128-wide K tile and wave, from the shipped gfx1201 code object: 64 WMMA,
49 VALU, 33 SALU, 12 branches, 53 waits, 24 LDS fragment loads, 8 global loads,
2 barriers; 169 VGPRs, 34 KB LDS, so 3 blocks per WGP.

- **Bounds checks as branches.** Each 16-byte load sits behind `s_and_saveexec` /
  `s_cbranch_execz` with zero-filled registers (8 such blocks per tile, 24 VALU moves).
  Rows past M or N only feed output rows and columns the writeback already skips, so
  0001 clamps them to the last valid row and loads unconditionally. Only the K tail
  needs zeros, and only kernels launched with `K % BKB != 0` keep that test (a kernel
  template flag). Result with the same compiler (clang 20) for the whole K loop:
  VALU 45 -> 24 (eight 64-bit address adds for the loads, eight LDS address adds
  for the stores), SALU 42 -> 8, branches 11 -> 3, waits 52 -> 31. VGPRs rise to 184, below the 256 that 6 waves per SIMD allow, so occupancy is
  unchanged (LDS still limits it to 3 blocks).
- **Load coalescing.** A thread loaded 4 consecutive 16-byte chunks, so one wave
  instruction touched 16 rows with 32 bytes each. 0001 assigns chunks round-robin: one
  wave instruction covers 4 whole 128-byte lines.
- **LDS reads per MMA.** A 32x64 wave tile needs 6 fragment loads per 8 WMMA. 0003
  adds 64x64 wave tiles (256x128 and 128x256 blocks, 233-234 VGPRs, no spills,
  27 KB LDS): 8 loads per 16 WMMA, a third fewer LDS reads per MMA and a quarter less
  L2 traffic per FLOP. Whether LDS bandwidth or issue limits the current kernel is
  exactly what the tile sweep will show.

## Small M (text tokens, modulation)

The M <= 8 GEMV decoded every fp8 byte in software: about 650 VALU and 32 exec-masked
branches per 16-byte chunk per row, and it re-read the weight once per row. It was
VALU-bound, which is why hipBLASLt (bandwidth-bound) won 5x. 0002 decodes with
`v_cvt_pk_f32_fp8` (8 instructions per 16 bytes) and covers all M rows per wave:
about 34 instructions per row and chunk. The per-row arithmetic is the same serial
chain of 16 FMAs in byte order followed by the same 5-step wave reduction (checked
in the ISA of both versions), and the fp8-to-f32 conversion is exact, so outputs are
unchanged. For 9 <= M <= 64, 0003 adds 16- and 32-row tiles to the tuner.

## The tile choice (0003)

No fixed K/N thresholds were added. The first fp8 GEMM of a shape class, keyed on
(device, M in quarter-octave classes, N, K, output dtype, bias), runs each eligible
tile once to warm up and twice under HIP events, and keeps the fastest for the
process. 10264 and 10347 tokens share a class, so prompt-length changes do not
retune. Eligible: K divisible by the tile's BKB and the tile less than twice M and N.
The five 0.2.36 tiles are candidates, so the tuner can always fall back to today's
choice. During stream capture it uses 0.2.36's heuristic, since capture forbids the
host synchronization.

Cost: one host synchronization and up to 33 extra GEMM runs per class, about
0.2-0.4 s per large Krea 2 shape once per process, inside the first sampling step.
`COMFY_KITCHEN_HIP_FP8_TUNE=0` restores 0.2.36's heuristic; `=verbose` logs every
timing; `COMFY_KITCHEN_HIP_FP8_TILE=<i>` forces tile i for benchmarks.

## Per shape, what to expect

| Shape (M x K -> N) | Today (your log) | Expected after 0001-0003 |
|---|---|---|
| Krea 2 q/gate/o 10264 x 6144 -> 6144 | 10-20 % behind hipBLASLt | gap narrows; target is the tuned hipBLASLt block time of 40.9 ms vs 47.0 |
| Krea 2 k/v 10264 x 6144 -> 1536 | behind | same mechanism; 128x256 is the likely pick |
| Krea 2 MLP up/gate 10264 x 6144 -> 16384 | behind | same |
| Krea 2 MLP down 10264 x 16384 -> 6144 | 35-42 % ahead | 0001 applies; tuner may keep 128x128 |
| Qwen 2.1 4096 x 4096 -> 24576 / 12288 -> 4096 / 4096 -> 4096 | 4.01 / 2.03 / 0.75 ms vs 3.43 / 1.97 / 0.63 | same mechanism as Krea 2 |
| M <= 8, e.g. 2 x 4096 -> 16384 | 0.55 ms vs 0.11 | ~0.11-0.15 ms |
| 9 <= M <= 335 | hipBLASLt ahead | tuner picks 16/32/64-row tiles at small M; unmeasured |

If kitchen wins every shape after this, `rocm_amd_perf._pick_fp8_gemm_backend` can go.
If not, the sweep table says which tile shape is missing.

## Risks

- **Accuracy:** none for 0001-0003 (bit-identical). 0004 is unverified: whether the
  hardware convert matches the software encoder for fp8 subnormals, -0.0 and values
  rounding to zero is not documented precisely enough to rely on. The check covers
  every bf16 and fp16 pattern and every e4m3 rounding midpoint; one differing byte
  means 0004 stays out.
- **Other GPUs:** 0001 changes the loader shared by the int8, int4 (convrot,
  svdquant), fp16 GEMM and fp16 conv3d kernels on gfx11 and gfx12. They compile for
  gfx1201, gfx1100 and gfx1030; only the fp8 path was emulated. The argument for the
  others is the same (garbage rows never reach a written output). On gfx11, 0002
  keeps the software decode and its VGPRs rise from 42 to 161 per GEMV; correct, but
  unmeasured there.
- **Tuner:** a noisy first measurement can lock in a tile a few percent slower than
  the best for the process. It synchronizes the host once per new class.
- **0004 draft:** the dtype test and the NaN test still branch per element; a
  branch-free version with the dtype test hoisted out of the 16-element loop was
  planned but not written in this session.

## Priority 2 and lower

- **bf16 conv3d for the Wan-class VAEs:** not done. Kitchen's implicit-GEMM conv3d is
  fp16 only (`MmaF16`, fp16 epilogue). A bf16 variant needs a byte-addressed
  `MmaBf16` policy in the GEMM core (`kStepBytes = 32`, the `MmaF16` load), a bf16
  epilogue and ComfyUI wiring for the NDHWC layout in the Wan 2.1/2.2 VAEs (as
  `_kitchen_ndhwc` does for MiniMax H3). Output would not be bit-identical to torch's
  conv (different summation order), so it needs the VAE error check from the R9700
  measurement prompt.
- **Hardware fp8 convert in the quantize kernel:** patch 0004 (draft, unverified).
- **kitchen#184 (sol paired query blocks):** skipped; you skip sol attention.
- **Accumulation docs:** proposed text for kitchen's README HIP section: "WMMA
  matmuls and convolutions accumulate in fp32 on gfx11 and gfx12, fp16 operands
  included (`v_wmma_f32_16x16x16_f16`); int8 and int4 accumulate in int32. No HIP
  kernel accumulates in fp16." Verified earlier from the disassembly of the 0.2.36
  wheel's gfx1201 code objects.
- **int8 convrot:** skipped (lower priority, behind fp8).

## Not verified

- Any timing, on any shape.
- That the hardware runs the new tiles without problems the ISA does not show
  (e.g. LDS bank conflicts in the 64x64 wave tiles, instruction cache pressure from
  the extra kernels).
- The Windows host build of the patched sources.
- The int8, int4 and fp16 GEMM paths under 0001 beyond compilation.
