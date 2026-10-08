#!/bin/bash
# Build the host emulation of a kitchen tree's adaln and rms_rope launchers.
# usage: build_norm.sh <kitchen backends/hip dir> <out binary>
# -ffp-contract=off: the only fmas are the ones prep.py and the fast paths spell out.
set -e
E=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d)
python3 "$E/prep.py" "$1" "$W/hip"
clang++-20 -std=c++20 -O2 -ffp-contract=off -pthread -D__gfx1201__ -I"$E/include" -I"$W/hip" -I"$E/../gen" \
  -Wno-unknown-attributes -Wno-ignored-attributes -Wno-unused-value -Wno-unknown-pragmas \
  "$E/test_norm.cpp" "$E/emu.cpp" -o "$2"
rm -rf "$W"
