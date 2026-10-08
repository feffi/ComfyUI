# comfy-kitchen HIP kernels for the R9700: patches, benchmarks, build

Patches against comfy-kitchen **v0.2.36** (`888b13e2c0e7`, the version installed in
`.venv-rocm-100`). The files they touch are identical in v0.2.37 (`be003b7`), and the
series applies there too (checked). Findings: `.claude/reports/kitchen-r9700.md` (round 1,
0001-0004) and `.claude/reports/kitchen-r9700-2.md` (round 2, 0005-0007).

| Patch | Change | Output | Verified here | Status |
|---|---|---|---|---|
| `0001` | WMMA GEMM core: coalesced, branch-free tile loads; K-tail test only in kernels whose K is not a multiple of BKB | bit-identical | CPU emulation of the kernel source, ISA | in production (0.2.36+r9700) |
| `0002` | fp8 GEMV (M <= 8): hardware fp8 decode on gfx12, all M rows per wave so the weight is read once | bit-identical | CPU emulation, ISA (FMA order) | in production |
| `0003` | fp8 GEMM: six more tiles plus a measured tile choice per shape class | bit-identical | CPU emulation of every tile and the tuned path, ISA (no spills) | dropped: 1-3 %, tuner noise |
| `0004` | per-tensor fp8 quantize: hardware e4m3 encode on gfx12 (draft) | **unverified** | compile only | dropped: no gain |
| `0005` | adaln/rms_adaln: one wave per row, 16-byte accesses, for 2-byte rows with D % 256 == 0 | bit-identical | CPU emulation with fp32 shadows, ISA against the 0.2.36 code object | new |
| `0006` | rms_rope: head_dim 128 fast path, one wave per token over its heads | bit-identical | same | new |
| `0007` | fp8 GEMM: fixed 256x128x128 tile when K >= 4096 and the grid has >= 8 blocks per WGP | bit-identical | CPU emulation, ISA | new |

0005-0007 apply on top of 0001 and 0002 (the production build) and do not need 0003 or
0004. Each is independent of the other two.

## Build on Windows with the TheRock ROCm 10.0 wheels

Never install into `.venv-rocm-100`. Everything below lives under `C:\kt`.

1. Sources:
   ```
   git clone https://github.com/Comfy-Org/comfy-kitchen C:\kt\comfy-kitchen
   git -C C:\kt\comfy-kitchen checkout -b r9700 888b13e2c0e721f6576fe351a2ad79894b1c451f
   cd C:\kt\comfy-kitchen
   $p = "<ComfyUI>\.claude\kitchen-r9700\patches"
   git am "$p\0001-*.patch" "$p\0002-*.patch" "$p\0005-*.patch" "$p\0006-*.patch" "$p\0007-*.patch"
   ```
2. Compiler: the ROCm clang of the same SDK version as the production torch. Check
   whether the production venv already carries it (read only):
   ```
   .venv-rocm-100\Scripts\python -m rocm_sdk path --root
   dir <root>\lib\llvm\bin\clang++.exe
   ```
   If clang is missing there, create a separate build venv and install the devel
   package of the *same* version from the index your stack repo uses:
   ```
   py -3.12 -m venv C:\kt\buildenv
   C:\kt\buildenv\Scripts\python -m pip install "rocm[devel]==<version of rocm in .venv-rocm-100>" --index-url <TheRock index>
   C:\kt\buildenv\Scripts\python -m rocm_sdk init
   ```
3. Build prerequisites: Visual Studio 2022 Build Tools (v143 toolset) and a Windows
   SDK. CMake >= 3.26, Ninja and nanobind come in through the PEP 517 build env.
4. Build a gfx1201-only wheel (kitchen's README: the ROCm clang finds MSVC itself, no
   developer prompt needed):
   ```
   $env:ROCM_HOME = "<root from step 2>"
   $env:COMFY_HIP_ARCHS = "gfx1201"
   $env:COMFY_KITCHEN_BUILD_HIP = "1"
   cd C:\kt\comfy-kitchen
   C:\kt\buildenv\Scripts\python -m pip wheel . --no-deps -w dist
   ```
5. Install it next to, not into, the production venv:
   ```
   C:\kt\buildenv\Scripts\python -m pip install --no-deps --target C:\kt\kitchen-patched dist\comfy_kitchen-0.2.36-*.whl
   ```
   A process started with `$env:PYTHONPATH = "C:\kt\kitchen-patched"` imports the
   patched build ahead of the installed one; everything else keeps 0.2.36. The
   benchmark prints `kitchen_path`, so check it says `C:\kt\kitchen-patched`.

Not verified: that the wheel builds on Windows with these patches. They change no
build files and compile with clang 20 for gfx1201, gfx1100 and gfx1030 (device code
only); the Windows host compile is untested.

## Measure

One GPU workload per card, nothing else on that card (production instance stopped
or on the other GPU). The reference is the production build (0.2.36 + 0001 + 0002);
0005-0007 must reproduce it byte for byte.

```
# production build: reference hashes and timings
.venv-rocm-100\Scripts\python bench_rms_rope.py --device 1 --save-ref C:\kt\rope_ref.pt
.venv-rocm-100\Scripts\python bench_rms_adaln.py --device 1 --save-ref C:\kt\adaln_ref.pt
.venv-rocm-100\Scripts\python bench_fp8_gemm.py --device 1 --save-ref C:\kt\gemm_ref.pt

# build with 0005-0007
$env:PYTHONPATH = "C:\kt\kitchen-patched"
.venv-rocm-100\Scripts\python bench_rms_rope.py --device 1 --ref C:\kt\rope_ref.pt
.venv-rocm-100\Scripts\python bench_rms_adaln.py --device 1 --ref C:\kt\adaln_ref.pt
.venv-rocm-100\Scripts\python bench_fp8_gemm.py --device 1 --ref C:\kt\gemm_ref.pt
```

`identical` must be `True` in every row of all three. `bench_rms_rope.py` and
`bench_rms_adaln.py` report GB/s against ~600 GB/s and a `fast` column that says which
launches 0005/0006 serve (rms_rope: `q` and `k` per launch, `-` for the old kernel), so
the fallback rows are visible; `err64` is the max error against fp64 on the first 64
tokens and must match between the two builds. `bench_fp8_gemm.py --sweep-tiles` needs
0003's tile knob and does not apply to this series; 0007 is measured by comparing the
two builds' rows.

## Tools

`tools/` holds what produced the "verified here" column, for repeating it after a
kitchen update:

- `emu/`: a host-side SIMT emulation of the HIP subset these kernels use (threads,
  block and wave barriers, shuffles, DPP row_xmask and v_permlanex16, the gfx12 wave32
  WMMA register layout). It compiles kitchen's unmodified sources for the CPU and runs
  the real kernel code.
  - fp8 GEMM: `build_emu.sh` + `suite.sh`, validated by running v0.2.36's kernels,
    which must reproduce a float64 GEMM. The cases cover every tile path, K tails,
    partial tiles, the GEMV, all three output dtypes and 0007's tile.
  - adaln and rms_rope: `build_norm.sh` + `test_norm.cpp` (23 cases, fast paths and
    fallbacks), compared with `compare.py`. `prep.py` models how the 0.2.36 code
    object rounds the old kernels (which steps it fuses into fmas, how it groups the
    rms_rope products; read off its ISA), and records the fp32 value of every element
    before its bf16/fp16 store. That second check matters: in a negative control, a
    swapped fma changed 30 % of the fp32 values but only 2-15 output elements per case.
- `kinfo.py`: per-kernel VGPRs, LDS and the instruction mix of the WMMA K-loop from a
  `clang -S` file; `build.sh` compiles a kitchen source for one gfx target with
  clang 20 and HIP headers, `ockl_shim.h` inlines the work-item queries and the f32
  rsqrt a `-nogpulib` build would otherwise call out of line.
