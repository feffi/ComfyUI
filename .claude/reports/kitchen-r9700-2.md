# comfy-kitchen HIP kernels on the R9700 (gfx1201), round 2: rms_rope, rms_adaln, fp8 GEMM tile

Stand: 08.10.2026. Base: the production build 0.2.36+r9700 (v0.2.36 `888b13e2c0e7` +
0001 + 0002). Patches 0005-0007 apply there and on v0.2.37 (checked with `git am`);
benchmarks, tools and build steps: `.claude/kitchen-r9700/`. Round 1: `kitchen-r9700.md`.

**Read this first.** No GPU here, nothing below was timed. "Verified" means: the gfx1201
ISA from clang 20 compared against the 0.2.36 Windows code objects (AMD clang 22), a CPU
emulation that runs kitchen's real kernel source, or a read of the source. Speed numbers
are estimates from bytes moved at your ~600 GB/s practical peak, until the benches run.

## Result

| Patch | Kernel | Output | Expected (Krea 2 at 1280x2048) |
|---|---|---|---|
| 0005 | rms_adaln / adaln: one wave per row, 16-byte accesses | bit-identical | 0.85 -> ~0.45 ms per call (252 MB), 2 calls per block |
| 0006 | rms_rope: head_dim 128, one wave per token looping over its heads | bit-identical | 1.59 -> ~0.6 ms per q+k call (336 MB); H3 in place at 39.5k tokens: ~3.8-4 ms (2.27 GB), old time unknown |
| 0007 | fp8 GEMM: fixed 256x128x128 tile for large grids | bit-identical | your sweep: 1-3 % at M = 10264 |

Per Krea 2 forward (28 blocks, the model default): about 28 x 1.0 ms (rms_rope) + 56 x 0.4 ms (rms_adaln)
= ~50 ms, plus 1-3 % of the fp8 GEMM time (28 x 42.78 ms) = 12-36 ms. Twice that per
step at cfg > 1.

**Determinism.** None of the three times anything, uses atomics or reduces in an order
that depends on scheduling. Each output comes from one fixed chain of operations. Which
path runs depends only on shapes, dtypes, strides and pointer alignment. Since each fast
path is bit-identical to its fallback, the path choice cannot change a result either.

## 0005: rms_adaln

The 0.2.36 kernel runs one 256-thread block per row with 2-byte loads, a runtime dtype
switch per element and eight barrier rounds of a shared-memory tree. That works out to
~300 GB/s on Krea 2's (10264, 6144). The fast path covers rows of fp16/bf16 with D % 256 == 0,
scale and shift in x's dtype, and 16-byte-aligned buffers. It runs one wave per row
(8 rows per block). Lane l stands in for threads 8l..8l+7 of the old kernel: it
accumulates their partial sums in the same order and adds the 256 partials in the
shared-memory tree's pairs, through xor shuffles.

- **Identity.** The 0.2.36 code object fuses three steps into fmas: each square into
  its sum, `sum * (1/D) + eps`, and `norm * (1 + scale) + shift`. The wave kernel writes
  those three as explicit `fmaf`, so it matches whatever the compiler would contract.
  It also hands the row sum back wave-uniform (`readlane`), so its rsqrt compiles to
  the same scalar sequence as the old kernel's: `s_fmac_f32`, the denormal scaling
  and `v_s_rsq_f32`. `1/D` is `v_rcp_iflag_f32` of the converted D in both.
  Emulation: 10 cases (6 fast path: RMS and LayerNorm, bf16 and fp16, D 256 to 6144,
  per-row and broadcast scale/shift; 4 fallbacks: D = 1000, fp32 scale, fp32 x,
  misaligned). 0 differing output bytes, 0 differing fp32 pre-store values.
- **Fallbacks** (old kernel, unchanged): fp32 x, scale/shift dtype different from x,
  D not a multiple of 256, buffers not 16-byte aligned. Krea 2's calls qualify.

## 0006: rms_rope

The 0.2.36 kernel runs one 64-thread block per (token, head) row. It does 2-byte
accesses, six barrier rounds and four freqs loads per pair, and reloads a token's freqs
for every head (~200 GB/s). The fast path gives one wave a token. The wave loads that
token's freqs and weights once, then loops over the heads four rows at a time with
8-byte accesses. It reduces each row with DPP row_xmask and `v_permlanex16` instead of
LDS (`flash_decode.hip` already uses both). Lane l stands in for threads 4l..4l+3,
whose second elements sit in lane l ^ 16.

- **Identity, and why bytes alone are not enough.** The 0.2.36 binary does not
  evaluate rms_rope_kernel in source order. Read off the bf16 and fp16 kernels in its
  code object:
  1. The sums of squares are `fma(x[t+64], x[t+64], x[t]^2)`.
  2. Rotated elements are scaled as `(rrms * w) * x`, while unrotated ones are scaled
     as `(rrms * x) * w`.
  3. The fp32-freqs rotation is `fma(f_b, x_b, f_a * x_a)`.
  4. The bf16/fp16-freqs rotation adds without fusing.

  A first version that left these to the compiler compiled every one of the four
  differently for at least some elements (its ISA fused the other square, the odd
  elements' other product, and the interleaved bf16 add). The fast path now spells all
  four out (`__builtin_fmaf`, contraction off), so its rounding no longer depends on
  contraction. The rsqrt and `1/D` compile to the same scalar sequence as in the
  shipped kernel. The emulator models the shipped
  rounding for the old kernel and records every element's fp32 value before its bf16
  store. Negative control: swapping the fp32 rotation's fma changed 30 % of those fp32
  values but only 2-15 output elements per case. A bytes-only comparison would have
  passed it most of the time.
- **Emulation:** 13 cases. 8 fast path: Krea 2's BHND views with and without k; H3's
  in-place BNHD qkv slices; fp16 x with fp16 freqs and fp32 weights; per-batch bf16
  freqs with partial rot 64; freqs broadcast over both dims; rot 2 interleaved; rot 8
  split-half. 5 fallbacks. Result: 0 of 2.53 M output elements and 0 of 1.66 M fp32
  pre-store values differ (adaln included).
- **Cost:** 69-79 VGPRs, no LDS. Per row, ~91 VALU for Krea 2's instance (bf16 x, fp32
  freqs, interleaved) and ~142 for H3's (bf16 freqs, split-half). Both stay
  memory-bound: H3 has 4.4 M rows x 142 VALU ≈ 2 ms of VALU against 3.8 ms of memory
  traffic. It compiles for gfx1030, gfx1100 and gfx1201.
- **Fast-path conditions:** head_dim 128; fp16/bf16 x; dense rows in and out; freqs
  broadcast over the head axis (BHND with freqs (B,1,T,...) or BNHD with (B,T,1,...));
  split-half rot_dim % 8 == 0; 8-byte-aligned pointers and strides a multiple of 4
  elements. Krea 2 (both launches) and H3 qualify; Qwen 2.1's call (BNHD, head_dim
  128, freqs (1,N,1,...)) should too. Fallbacks: other head_dims, per-head freqs,
  fp32 x, split-half rot_dim not a multiple of 8, misaligned views. A weight that is not 1-D of length head_dim still goes to eager in
  the Python wrapper, as before.

## 0007: fixed 256x128 tile

When K >= 4096 and ceil(M/256) * ceil(N/128) >= 8 x WGPs (256 on the R9700), the fp8
launcher runs 0003's 256x128x128 tile: 16 waves, 32x64 per wave, 171 VGPRs and 52 KB of
LDS, so 2 blocks per WGP. Per K tile and wave its loop has 64 WMMA, 12 VALU, 8 SALU and
3 branches. Every output gets the same WMMA K-steps in the same order.

- **Emulation:** 5 cases, all identical. 3 run the new tile: partial M and N, a K
  tail, f32/f16/bf16 out and bias variants; a probe confirms 512-thread blocks on the
  expected grid. 2 sit just below the threshold.
- **Coverage:** all Krea 2 shapes qualify, from 492 blocks (k/v, N = 1536) to 5248
  (MLP up). Qwen 2.1's M = 4096 shapes qualify too (512 blocks for N = 4096), but your
  sweep did not cover them. If the bench shows a loss there, phase 7 marks 0007
  inconclusive; the fix is then a fixed rule that leaves those shapes out, and a higher
  block threshold alone would not do it (4096 x 4096 -> 24576 has 3072 blocks).

## Not done: the K = 16384 shape and the rest of the K = 6144 gap

I found no deterministic change for mlp.down (10264 x 16384 -> 6144, 10.2 ms) that I
could justify without a measurement.

- **DRAM re-reads do not explain it.** B (6144 x 16384 fp8, 100.7 MB) exceeds the 64 MB
  MALL, so it streams from DRAM once per group of 4 block-rows. With 0007's 256-row
  tiles that is 11 passes, ~1.1 GB per GEMM; 128-row tiles need 21 passes, 2.1 GB.
  That is 110-210 GB/s, well below DRAM bandwidth. The MLP up GEMM has the same B
  size and pass count and still runs at the K = 6144 rate (209 TOP/s).
- **Question.** By 2·M·N·K, 10.2 ms is ~203 TOP/s, not 193; which number is right?
  And does the 10.2 ms include quantizing the 10264 x 16384 bf16 activation to fp8?
  With `fp8_e4m3fn_fast` weights that quantize is not kitchen's: `fp8_linear` in
  ComfyUI's `comfy/ops.py` clamps the input in place and casts it with torch, two
  kernels and ~1.2 GB of traffic for this input (~2 ms at 600 GB/s). Kitchen's
  quantize would move ~0.5 GB (~0.85 ms). Without the quantize the GEMM would be
  ~8.2-9.3 ms, at or above the other shapes' rate, and the lever would be the quantize.
- **Lead, not done: the activation quantize per block.** Every fp8 Linear quantizes
  its own input in `fp8_linear`, so wq, wk, wv and the attention gate quantize the
  same normed input four times, and the MLP gate and up twice. Per Krea 2 block at
  10264 tokens that is ~4.3 GB of torch clamp and cast traffic, ~7 ms at 600 GB/s
  next to 42.78 ms of fp8 GEMMs (estimate from the code, not timed). Kitchen's
  quantize would make it ~1.8 GB (~3 ms), and quantizing each shared input once
  ~1.1 GB (~1.8 ms). That is a ComfyUI change in `fp8_linear`, outside these
  patches; phase 7 of the measurement prompt times both quantize paths and its
  kernel profile shows the clamp and cast kernels.
- **If the GEMM alone is that slow,** the next bit-identical experiments are:
  1. Sweep the block-order group size (kGroupM 4 -> 8/16) to keep B's working set in
     MALL.
  2. Prefetch two K tiles ahead. On the 16-wave tile this needs ~195 VGPRs; above 192
     the WGP drops from 2 blocks to 1, so it only pays if the compiler stays under.
- **K = 6144 gap.** 0007 should close 1-3 % of the ~6 %. The next candidate is one
  barrier per K tile instead of two, through double-buffered LDS. That needs <= 64 KB
  per block: an unpadded 128x128x128 with an XOR swizzle fits exactly, at 2 blocks per
  WGP instead of 3. Not started; it is a structural change to the shared GEMM core,
  so it needs the bench first.

## How it gets measured

Phase 7 of `.claude/prompts/r9700-measurement-run.md` builds 0001 + 0002 + 0005-0007
and compares it with the installed kitchen through the three scripts. Two candidate
runs in separate processes must also match each other (restart identity). It then runs
Krea 2, Qwen-Image 2.1 and H3 end to end, twice per build, each in a fresh instance,
and counts the new and old kernels in one profiled step per workload. That count is
what shows the model calls reach the fast paths, which the `fast` column cannot: it
re-implements the launcher test on the bench's own inputs. If production rounds its
old kernels differently from the official 0.2.36 wheel, the fast-path rows match that
wheel instead; phase 7 then marks the patch inconclusive, since it would change
production renders once. A patch is kept only if it is also measurably faster on the
rows it serves; an end-to-end difference the benches miss is traced to one patch with
single-patch builds, and the build is checked for any timing-based choice left.

## Not verified

- Any timing.
- The Windows host build of 0005-0007. They change no build files, and the device
  code compiles with clang 20 for gfx1201, gfx1100 and gfx1030.
- The emulator's model of the shipped rounding comes from reading the bf16 and fp16
  kernels of the 0.2.36 code object. A future kitchen build compiled differently
  changes the fallback's rounding, not the fast paths'. In that case the
  `--save-ref` / `--ref` rows of the benches show it.
