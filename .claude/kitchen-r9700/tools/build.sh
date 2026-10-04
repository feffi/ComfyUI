#!/bin/bash
# Device-only compile of a kitchen HIP source to gfx assembly, for kinfo.py.
# usage: build.sh <kitchen backends/hip dir> <src.hip> <out.s> [gfx1201]
# Needs clang-20 and HIP headers (Ubuntu 24.04: apt install clang-20 libamdhip64-dev).
T=$(cd "$(dirname "$0")" && pwd)
clang++-20 -x hip --offload-arch=${4:-gfx1201} --cuda-device-only -nogpulib -O3 -ffast-math -mno-wavefrontsize64 \
  -std=c++20 -D__HIP_PLATFORM_AMD__ -I"$1" -I"$T/gen" -include "$T/ockl_shim.h" -S "$2" -o "$3"
