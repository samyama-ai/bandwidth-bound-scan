#!/usr/bin/env bash
# evaluate.sh <candidate.c> — the AlphaEvolve evaluation cascade for one kernel.
# tier 0: compile (-O3 -march=native, same flags as the paper). fail -> {"compile":false}
# tier 1+2: run eval_driver (correctness oracle, then throughput). prints one JSON line.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CAND="${1:?usage: evaluate.sh <candidate.c>}"
CC="${CC:-cc}"
CFLAGS="${CFLAGS:--O3 -march=native -std=gnu11 -Wall}"
BIN="$HERE/.build/eval_$(basename "${CAND%.c}")"
mkdir -p "$HERE/.build"

ERR="$($CC $CFLAGS "$HERE/eval_driver.c" "$HERE/../src/codecs.c" "$CAND" -o "$BIN" -lm 2>&1)"
if [ $? -ne 0 ]; then
  # compact the compiler error onto one JSON line
  MSG="$(printf '%s' "$ERR" | tr '\n' ' ' | sed 's/"/\\"/g' | cut -c1-400)"
  printf '{"compile":false,"error":"%s"}\n' "$MSG"
  exit 0
fi
"$BIN" "${BBS_N:-16777216}" "${BBS_REPEATS:-7}"
