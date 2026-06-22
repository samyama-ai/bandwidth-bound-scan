# Evolve-loop agent brief — make the FastLanes scan kernel faster

You are one member of an **ensemble** in an AlphaEvolve-style evolutionary loop. Your job:
produce a **correct, faster** descendant of the seed kernel `fl_scan_candidate`.

## The contract (must hold or you score -inf)
```c
uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target);
```
Counts how many values equal `target` in a FastLanes **bit-transposed** layout:
- `FL_LANES = 32` lanes, `FL_BLOCK = 1024` values per block, arbitrary bit-width `b` (1..32).
- Block `blk` starts at `planar + blk*b*FL_LANES`. Within a block, lane `l` (0..31) holds the
  32 values at positions `l, l+32, ..., l+992`, bit-packed little-endian. For output position
  `p` (0..31) the bits start at `bit = p*b`, word `wp = bit>>5`, offset `off = bit&31`, and the
  lane words are interleaved: word `w` of lane `l` is at `base[w*FL_LANES + l]`.
- The seed `candidate_gen0.c` is the reference implementation — read it, it is correct.

## The seed (generation 0), measured on THIS machine (Intel i9-9980HK, AVX2):
geomean **5.275 values/ns** (≈1.32 values/cycle @4GHz). Per-b values/ns:
b3=6.68 b5=6.59 b7=5.76 b8=6.81 b12=4.57 b16=5.56 b24=4.01 b32=3.41

## How to evaluate (the real fitness function — use it to hill-climb)
```bash
cd <repo>/evolve
./evaluate.sh your_candidate.c       # prints ONE json line
```
- `{"compile":false,...}` → fix the compile error in the `error` field.
- `{"correct":false,"error":"oracle mismatch ..."}` → your count is wrong at that (b,sigma). Fix it.
- `{"correct":true,...,"geomean_vpns":X, "per_b":{...}}` → X is your score. Higher is better.
- **Timing during development is noisy** (other ensemble members run concurrently and contend for
  memory bandwidth). Use it only for direction; the orchestrator re-times the winner SERIALLY and
  cleanly afterward. Optimize for a real algorithmic win, not a lucky sample.

## Rules
- Write ONE self-contained C file (only `<stdint.h>`/`<stddef.h>` + intrinsics). Define your own
  `FL_LANES 32`, `FL_BLOCK 1024`. Name it as instructed below so it doesn't collide.
- `-O3 -march=native` (AVX2 available; `<immintrin.h>` is fine). Single-threaded — NO threads/OpenMP.
- Correctness first: it must pass the oracle for ALL widths 3..32 in the sweep, including widths
  where every position crosses a 32-bit word boundary (`off != 0`).
- Return ONLY: (1) the final C source, (2) its last `evaluate.sh` JSON, (3) one paragraph on what
  made it faster (or why it didn't). Do not edit any file outside `evolve/`.
