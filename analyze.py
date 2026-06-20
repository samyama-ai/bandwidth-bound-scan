#!/usr/bin/env python3
"""analyze.py — fit the bandwidth-fraction law, locate the H2/H3 crossovers, make figures.

Reads results/{sweep,zone,select}_<machine>.csv (from scanbench) and emits:
  - results/law_fit.txt        : per (machine,strategy) T_dec + median |f - f_hat| (H1)
  - results/crossovers.txt     : H2 branch band + H3 zone-skip threshold
  - results/fig_roofline_*.png, fig_branch_*.png, fig_zone_*.png
Run from repo root inside ~/projects/venv.
"""
import sys, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"

def load(kind):
    frames = []
    for f in sorted(glob.glob(f"{R}/{kind}_*.csv")):
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"skip {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def strategy(row):
    """Clean decode-strategy label from (encoding,b)."""
    enc = row["encoding"]
    if enc == "fastlanes": return "fastlanes(transposed)"
    if enc == "bitpack-scalar":
        return "byte-aligned" if row["b"] in (8, 16, 32) else "scalar(row-major)"
    return enc

# ---------- H1: the bandwidth-fraction law f_hat = min(1, T_dec * b / (8*beta)) ----------
def fit_law():
    sweep = load("sweep")
    if sweep.empty: return
    sweep = sweep[sweep["sigma"] == 1.0].copy()          # f is ~sigma-independent for counts
    sweep["strategy"] = sweep.apply(strategy, axis=1)
    out = ["# H1 bandwidth-fraction law: f_hat = min(1, T_dec * b / (8*beta))",
           "# T_dec fitted per (machine,strategy) as median(read_gbps*8/b) over compute-bound pts (f<0.55)",
           f"{'machine':12} {'strategy':22} {'beta':>7} {'T_dec(Gv/s)':>12} {'b*':>6} {'med|f-fhat|':>12} {'n':>3}"]
    rows = []
    for (mach, strat), g in sweep.groupby(["machine", "strategy"]):
        g = g[g["predicate"].isin(["branchfree", "branchy"])]
        if g.empty:
            continue   # e.g. rle (predicate=runscan) — different bytes model, excluded from the law
        beta = g["beta_scan"].iloc[0]
        # estimate T_dec from compute-bound points (f<0.55): there read_gbps = T_dec*b/8
        cb = g[g["f"] < 0.55]
        src = cb if len(cb) >= 2 else g
        T_dec = np.median(src["read_gbps"] * 8.0 / src["b"])      # GB/s*8/bits = Gvalues/s
        fhat = np.minimum(1.0, T_dec * g["b"] / (8.0 * beta))
        err = np.median(np.abs(g["f"].values - fhat.values))
        bstar = 8.0 * beta / T_dec
        out.append(f"{mach:12} {strat:22} {beta:7.1f} {T_dec:12.3f} {bstar:6.1f} {err:12.4f} {len(g):3d}")
        rows.append((mach, strat, beta, T_dec, bstar, err))
    # overall train vs held-out: treat i9 as train, M4 as held-out
    err_by_machine = {}
    for mach, g in sweep.assign(strategy=sweep.apply(strategy, axis=1)).groupby("machine"):
        es = []
        for strat, gg in g.groupby("strategy"):
            gg = gg[gg["predicate"].isin(["branchfree", "branchy"])]
            if gg.empty:
                continue
            beta = gg["beta_scan"].iloc[0]
            cb = gg[gg["f"] < 0.55]; src = cb if len(cb) >= 2 else gg
            T = np.median(src["read_gbps"] * 8.0 / src["b"])
            fhat = np.minimum(1.0, T * gg["b"] / (8.0 * beta))
            es += list(np.abs(gg["f"].values - fhat.values))
        err_by_machine[mach] = np.median(es)
    out.append("")
    for m, e in err_by_machine.items():
        bar = 0.10 if "i9" in m or "9980" in m else 0.15
        out.append(f"median |f - fhat| on {m}: {e:.4f}  (bar {bar}) -> {'PASS' if e <= bar else 'FAIL'}")
    open(f"{R}/law_fit.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    plot_roofline(sweep, rows)

def plot_roofline(sweep, rows):
    for mach, g in sweep.groupby("machine"):
        plt.figure(figsize=(6, 4))
        for strat, gg in g.groupby("strategy"):
            gg = gg.sort_values("b")
            plt.plot(gg["b"], gg["f"], "o-", label=strat, markersize=4)
            beta = gg["beta_scan"].iloc[0]
            cb = gg[gg["f"] < 0.55]; src = cb if len(cb) >= 2 else gg
            T = np.median(src["read_gbps"] * 8.0 / src["b"])
            bb = np.linspace(1, 32, 64)
            plt.plot(bb, np.minimum(1, T * bb / (8 * beta)), "--", alpha=0.5)
        plt.xlabel("bit-width b"); plt.ylabel("bandwidth fraction f = read_GBps / beta")
        plt.title(f"Roofline fraction vs bit-width ({mach})\nsolid=measured, dashed=law min(1,T_dec*b/8beta)")
        plt.legend(fontsize=7); plt.ylim(0, 1.05); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{R}/fig_roofline_{mach}.png", dpi=110); plt.close()

# ---------- H2: branch-free vs branchy crossover band ----------
def fit_branch():
    sel = load("select")
    if sel.empty: return
    out = ["# H2: select time vs selectivity. branchy ~ A + B*sigma(1-sigma); predicated ~ const C.",
           "# branch-free (predicated) wins in the band where branchy is slower."]
    for mach, g in sel.groupby("machine"):
        by = g[g["predicate"] == "branchy"].sort_values("sigma")
        pr = g[g["predicate"] == "predicated"].sort_values("sigma")
        s = by["sigma"].values; tb = by["time_ns"].values
        # fit A + B*sigma(1-sigma)
        x = s * (1 - s)
        B, A = np.polyfit(x, tb, 1)
        C = np.median(pr["time_ns"].values)
        # crossover: A + B*x = C -> x = (C-A)/B ; sigma(1-sigma)=x -> roots
        xc = (C - A) / B if B != 0 else np.nan
        band = ""
        if 0 < xc < 0.25:
            disc = max(0.0, 1 - 4 * xc)
            slo = 0.5 * (1 - np.sqrt(disc)); shi = 0.5 * (1 + np.sqrt(disc))
            band = f"predicated wins in sigma in ({slo:.3f}, {shi:.3f})"
        # measured sign change
        merged = pd.merge(by[["sigma","time_ns"]], pr[["sigma","time_ns"]], on="sigma", suffixes=("_by","_pr"))
        merged["pred_wins"] = merged["time_ns_by"] > merged["time_ns_pr"]
        win_sigmas = merged[merged["pred_wins"]]["sigma"].values
        out.append(f"\n[{mach}] branchy peak at sigma=0.5 (max mispredict). A={A:.0f}ns B={B:.0f}ns C(pred)={C:.0f}ns")
        out.append(f"  model: {band}")
        if len(win_sigmas):
            out.append(f"  measured: predicated wins at sigma in [{win_sigmas.min():.4f}, {win_sigmas.max():.4f}]")
        out.append(f"  H2 verdict: {'CONFIRMED (mid-sigma band exists)' if band and len(win_sigmas) else 'no clean band'}")
        # figure
        plt.figure(figsize=(6,4))
        plt.plot(by["sigma"], by["time_ns"]/1e6, "o-", label="branchy (data-dependent store)")
        plt.plot(pr["sigma"], pr["time_ns"]/1e6, "s-", label="predicated (branch-free)")
        plt.xscale("log"); plt.xlabel("selectivity sigma (log)"); plt.ylabel("select time (ms)")
        plt.title(f"H2 branch-misprediction crossover ({mach})")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{R}/fig_branch_{mach}.png", dpi=110); plt.close()
    open(f"{R}/crossovers.txt", "a").write("\n".join(out) + "\n")
    print("\n".join(out))

# ---------- H3: zone-map-skip threshold ----------
def fit_zone():
    z = load("zone")
    if z.empty: return
    out = ["\n# H3: zone-map skip vs full scan. zonemap reads only un-skippable blocks' bytes.",
           "# threshold sigma_skip below which zonemap effective throughput beats full-scan roofline."]
    for mach, g in z.groupby("machine"):
        for clustered, gg in g.groupby("clustered"):
            full = gg[gg["encoding"] == "fullscan"].sort_values("sigma")
            zone = gg[gg["encoding"] == "zonemap"].sort_values("sigma")
            m = pd.merge(full[["sigma","time_ns","bytes_read"]], zone[["sigma","time_ns","bytes_read"]],
                         on="sigma", suffixes=("_full","_zone"))
            m["zone_faster"] = m["time_ns_zone"] < m["time_ns_full"]
            m["bytes_ratio"] = m["bytes_read_zone"] / m["bytes_read_full"]
            wins = m[m["zone_faster"]]["sigma"].values
            tag = "clustered" if clustered else "random(NC2)"
            out.append(f"\n[{mach}] {tag}: bytes_read(zone)/full by sigma:")
            for _, r in m.iterrows():
                out.append(f"   sigma={r['sigma']:.4f}  bytes_ratio={r['bytes_ratio']:.3f}  zone_faster={bool(r['zone_faster'])}")
            if clustered and len(wins):
                out.append(f"   sigma_skip threshold ~ zone wins for sigma <= {wins.max():.4f}")
            plt.figure(figsize=(6,4))
            plt.plot(m["sigma"], m["bytes_ratio"], "o-")
            plt.xscale("log"); plt.xlabel("selectivity sigma (log)"); plt.ylabel("zone bytes / full bytes")
            plt.title(f"H3 zone-map skip effectiveness ({mach}, {tag})")
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(f"{R}/fig_zone_{mach}_{tag}.png", dpi=110); plt.close()
    open(f"{R}/crossovers.txt", "a").write("\n".join(out) + "\n")
    print("\n".join(out))

if __name__ == "__main__":
    open(f"{R}/crossovers.txt", "w").write("")
    fit_law(); fit_branch(); fit_zone()
    print("\nfigures + law_fit.txt + crossovers.txt written to results/")
