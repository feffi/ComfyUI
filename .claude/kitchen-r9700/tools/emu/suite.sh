#!/bin/bash
# usage: suite.sh <binary> <dump dir>; cases cover every launch_gemm_wmma path, the
# GEMV, partial M/N tiles, K tails and (last three) the 256x128 tile of patch 0007. Fields: M N K out bias seed wgps
BIN=$1; D=$2; mkdir -p $D; fail=0
while read M N K O B S W; do
  [ -z "$M" ] && continue
  timeout 900 $BIN $M $N $K $O $B $S $W $D/$M-$N-$K-$O-$B-$W.bin || fail=1
done <<'CASES'
3 40 64 2 2 1 32
1 72 4096 2 2 15 32
2 33 512 1 -1 16 32
5 64 1040 0 0 17 32
8 64 4096 0 -1 2 32
9 64 64 2 2 3 32
16 200 4096 0 0 4 32
40 100 2064 2 2 5 32
70 40 272 1 -1 6 32
130 136 272 2 -1 7 1
200 150 4112 2 2 8 1
300 260 4112 1 2 9 1
257 384 4096 2 -1 10 2
200 150 4096 2 2 11 1
40 100 2048 2 -1 12 32
70 40 256 0 2 13 32
129 129 128 2 2 14 1
600 400 4096 2 2 21 1
520 300 4112 1 -1 22 1
1024 256 4096 0 0 23 1
CASES
exit $fail
