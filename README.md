# bandwidth-bound-scan

**A predictive bandwidth-fraction law for columnar scans — and the open micro-benchmark that
fits it — validated on two ISAs (x86/AVX2 and Apple M4/NEON).**

> Reproducible baseline for the problem `29-hardware-conscious-db/bandwidth-bound-scan` in the
> [DBMS Research catalog](https://github.com/samyama-ai/dbms_research). **Not** a new SOTA kernel:
> a *closed-form model* of when a columnar scan is memory-bandwidth-bound vs compute-bound, plus
> the harness that measures it. Honest limitation up front: our portable-C decoders reach
> ~2 values/cycle, well below hand-tuned SOTA (FastLanes ~40 v/cyc); the law is parameterized by
> the *measured* decode throughput, so it holds regardless and **extrapolates to the SOTA regime**.

## The claim in one line

A columnar scan's achieved memory-bandwidth fraction is

```
f  =  min(1,  T_dec · b / (8 · β))
```

where `b` = bits/value, `β` = sustainable memory bandwidth (bytes/s), and **`T_dec` = the decoder's
value throughput (values/s), which we find is independent of bit-width** — set by the decode
*layout/strategy*, not by `b`. The compute→bandwidth **ridge** sits at `b* = 8β / T_dec`.

This single-parameter-per-strategy law fits measured bandwidth fractions with **median error 0.027
on x86 and 0.003 on the held-out Apple M4**, and the ridge `b*` shifts correctly with the machine.
Plugging FastLanes' own decode throughput (~140 Gv/s) into the law reproduces their headline
result ("decode is free; 52 GB/s at 3 bits ≈ RAM limit") as its large-`T_dec` limit.

## Results

| Strategy | x86 i9 `T_dec` (Gv/s) | x86 `b*` | M4 `T_dec` | M4 `b*` |
|---|---|---|---|---|
| scalar (row-major bit-unpack) | 0.27 | 426 | 2.06 | 273 |
| FastLanes-style (transposed, lane-parallel) | 3.9 | 29 | 8.4 | 67 |
| byte-aligned (b∈{8,16,32}) | 5.7 | 20 | 9.0 | 62 |

median \|f − f̂\| : **0.027 (x86, bar 0.10) · 0.003 (M4 held-out, bar 0.15)** → law generalizes across ISAs.

Two crossovers, also validated on both machines:
- **Branch:** on a materializing select, branch-free beats branchy in a mid-selectivity band
  (model vs measured edges agree within ±1 selectivity decade); the σ(1−σ) misprediction parabola
  is textbook. (Count aggregates auto-branchless at `-O3`, so the effect is real only for
  data-dependent/position-list scans — disclosed.)
- **Zone-map skip is clustering-gated, not selectivity-gated:** sorted data → reads 1/8192 of bytes;
  random data → reads everything (sharpens Zeng et al. PVLDB'24 "low-selectivity only").

Figures in [`results/`](results/): `fig_roofline_*`, `fig_branch_*`, `fig_zone_*`.

## Reproduce (one command)

```bash
make repro            # build + tests + beta + sweeps -> results/*.csv
python analyze.py     # fit the law, locate crossovers, render figures (needs numpy/pandas/matplotlib)
```

`make test` runs the correctness oracle + negative controls (encode/decode round-trips for all
widths 1..32 in both layouts; scan == brute force; byte-aligned == generic; random ⇒ no zone skip).
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment, hardware, and the exact commands used
for the two hardware points.

## Layout

```
src/        roofline (beta), codecs (bit-pack, FastLanes transposed, RLE, zone maps), scanbench driver
tests/      correctness oracle + negative controls
analyze.py  law fit + crossover location + figures
results/    committed CSVs + figures (regenerable by make repro)
paper/      preprint source
```

## What this is and isn't

- **Is:** a reproducible, cross-ISA *characterization* — a closed-form bandwidth-fraction law, the
  `b*=8β/T_dec` ridge, the branch and clustering crossovers, and an open harness.
- **Isn't:** a faster kernel. We do not beat FastLanes/BtrBlocks; we model the regime they operate in
  and reproduce their "decode is free" result as a limit of the law.

Builds on FastLanes (Afroozeh & Boncz, VLDB'23), BtrBlocks (Kuschewski et al., SIGMOD'23), Data
Blocks (Lang et al., SIGMOD'16), the Roofline model (Williams et al., CACM'09), Polychroniou & Ross
(DaMoN'15), and Zeng et al. (PVLDB'24). License: MIT (code); catalog content CC BY 4.0.
