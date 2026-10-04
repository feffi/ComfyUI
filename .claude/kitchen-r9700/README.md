# comfy-kitchen HIP kernels for the R9700: patches, benchmark, build

Patches against comfy-kitchen **v0.2.36** (`888b13e2c0e7`, the version installed in
`.venv-rocm-100`). The files they touch are identical in v0.2.37 (`be003b7`), and they
apply there too. Findings and expected effects: `.claude/reports/kitchen-r9700.md`.

| Patch | Change | Output | Verified here |
|---|---|---|---|
| `0001` | WMMA GEMM core: coalesced, branch-free tile loads; K-tail test only in kernels whose K is not a multiple of BKB | bit-identical | CPU emulation of the kernel source, ISA |
| `0002` | fp8 GEMV (M <= 8): hardware fp8 decode on gfx12, all M rows per wave so the weight is read once | bit-identical | CPU emulation, ISA (FMA order) |
| `0003` | fp8 GEMM: six more tiles plus a measured tile choice per shape class | bit-identical | CPU emulation of every tile and the tuned path, ISA (no spills) |
| `0004` | per-tensor fp8 quantize: hardware e4m3 encode on gfx12 (draft) | **unverified** | compile only; run `check_fp8_quantize.py` first |

Patches 0001-0003 are independent of 0004; apply 0001-0003 alone if 0004 fails its check.
`git am` applies the series to v0.2.37 as well (checked).

## Build on Windows with the TheRock ROCm 10.0 wheels

Never install into `.venv-rocm-100`. Everything below lives under `C:\kt`.

1. Sources:
   ```
   git clone https://github.com/Comfy-Org/comfy-kitchen C:\kt\comfy-kitchen
   git -C C:\kt\comfy-kitchen checkout -b r9700 888b13e2c0e721f6576fe351a2ad79894b1c451f
   git -C C:\kt\comfy-kitchen am <ComfyUI>\.claude\kitchen-r9700\patches\000*.patch
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
or on the other GPU).

```
# installed 0.2.36: reference hashes and timings
.venv-rocm-100\Scripts\python bench_fp8_gemm.py --device 1 --save-ref C:\kt\ref_0236.pt
.venv-rocm-100\Scripts\python check_fp8_quantize.py --device 1 --save C:\kt\quant_0236.pt

# patched build
$env:PYTHONPATH = "C:\kt\kitchen-patched"
.venv-rocm-100\Scripts\python check_fp8_quantize.py --device 1 --ref C:\kt\quant_0236.pt
$env:COMFY_KITCHEN_HIP_FP8_TUNE = "verbose"
.venv-rocm-100\Scripts\python bench_fp8_gemm.py --device 1 --ref C:\kt\ref_0236.pt
.venv-rocm-100\Scripts\python bench_fp8_gemm.py --device 1 --sweep-tiles --only krea2
```

`identical` must be `True` for every shape. The sweep prints every tile per shape and
the fastest, which is what the tuner should have picked (its choices are in the
`verbose` stderr log). Knobs of patch 0003: `COMFY_KITCHEN_HIP_FP8_TUNE=0` keeps
0.2.36's tile heuristic, `COMFY_KITCHEN_HIP_FP8_TILE=<0-10>` forces one tile.

## Tools

`tools/` holds what produced the "verified here" column, for repeating it after a
kitchen update:

- `emu/`: a host-side SIMT emulation of the HIP subset these kernels use (threads,
  block and wave barriers, shuffles, the gfx12 wave32 WMMA register layout). It
  compiles kitchen's unmodified `gemm_fp8.hip` for the CPU and runs the real kernel
  code. Validated by running v0.2.36's kernels, which must reproduce a float64 GEMM.
  `suite.sh` covers every tile path, K tails, partial tiles, the GEMV and all three
  output dtypes; patched builds are compared byte for byte against v0.2.36's output.
- `kinfo.py`: per-kernel VGPRs, LDS and the instruction mix of the WMMA K-loop from a
  `clang -S` file; `build.sh` compiles a kitchen source for one gfx target with
  clang 20 and HIP headers, `ockl_shim.h` inlines the four work-item queries a
  `-nogpulib` build would otherwise call out of line.
