#!/usr/bin/env bash
# eval_mini.sh <candidate.c> — run the evaluation cascade on the Apple M4 (mini).
# Ships the candidate to mini and runs the same evaluate.sh there (NEON / -march=native).
# eval_driver.c + src/codecs.c are already synced under ~/projects/bandwidth-bound-scan.
set -u
CAND="${1:?usage: eval_mini.sh <candidate.c>}"
BN="$(basename "$CAND")"
rsync -az "$CAND" sandeep@mini:~/projects/bandwidth-bound-scan/evolve/ 2>/dev/null
ssh sandeep@mini "cd ~/projects/bandwidth-bound-scan/evolve && BBS_WIDTHS='${BBS_WIDTHS:-}' BBS_REPEATS=${BBS_REPEATS:-7} BBS_N=${BBS_N:-16777216} ./evaluate.sh $BN"
