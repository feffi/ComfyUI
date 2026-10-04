#!/bin/bash
# Build the host emulation of a kitchen tree's fp8 GEMM. usage: build_emu.sh <kitchen backends/hip dir> <out binary>
set -e
E=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d)
python3 "$E/prep.py" "$1" "$W/hip"
clang++-20 -std=c++20 -O2 -pthread -D__gfx1201__ -I"$E/include" -I"$W/hip" -I"$E/../gen" \
  -Wno-unknown-attributes -Wno-ignored-attributes -Wno-unused-value \
  "$E/test_fp8.cpp" "$E/emu.cpp" -o "$2"
rm -rf "$W"
