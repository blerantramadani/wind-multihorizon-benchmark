#!/usr/bin/env python3
"""
05_make_figures.py
==================
Produce the paper figures from results/floor_vs_achieved.csv.

Figure 1 (main): two panels (one per site). Each shows, versus horizon,
    - the predictability floor (shaded lower bound),
    - the best non-ML benchmark error,
    - the best ML error,
so that the narrowing gap between achieved error and the floor is visible.

Figure 2 (optional): the gap-to-floor versus horizon for both sites, making the
convergence at long horizons explicit.

Outputs
-------
figures/fig1_floor_vs_achieved.png (+ .pdf)
figures/fig2_gap_vs_horizon.png    (+ .pdf)

Usage
-----
    python scripts/05_make_figures.py --table results/floor_vs_achieved.csv
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from windfloor import SITES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="results/floor_vs_achieved.csv")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.table):
        print(f"[error] {args.table} not found; run 04_floor_vs_achieved.py first")
        return 1
    df = pd.read_csv(args.table)
    sites = [s for s in SITES.values() if s.key in df["dataset"].unique()]

    # ---- Figure 1 ----
    fig, axes = plt.subplots(1, len(sites), figsize=(5.4 * len(sites), 4.3),
                             sharey=False)
    if len(sites) == 1:
        axes = [axes]
    for ax, site in zip(axes, sites):
        d = df[df["dataset"] == site.key].sort_values("horizon_h")
        h = d["horizon_h"].to_numpy()
        ax.fill_between(h, 0, d["floor_pct"], color="0.85",
                        label="predictability floor", zorder=0)
        ax.plot(h, d["floor_pct"], color="0.35", lw=1.6, zorder=1)
        ax.plot(h, d["best_nonML_pct"], "o-", color="#1f77b4", ms=4,
                lw=1.4, label="best non-ML benchmark")
        ax.plot(h, d["best_ML_pct"], "s--", color="#d62728", ms=4,
                lw=1.4, label="best ML model")
        ax.set_title(site.label)
        ax.set_xlabel("forecast horizon (h)")
        ax.set_ylabel("nRMSE (%)")
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted(h))
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"fig1_floor_vs_achieved.{ext}"),
                    dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2 ----
    fig2, ax = plt.subplots(figsize=(5.6, 4.0))
    markers = {"kelmarsh": "o-", "penmanshiel": "s--"}
    for site in sites:
        d = df[df["dataset"] == site.key].sort_values("horizon_h")
        ax.plot(d["horizon_h"], d["gap_to_best_pp"],
                markers.get(site.key, "o-"), ms=4, lw=1.4, label=site.label)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("forecast horizon (h)")
    ax.set_ylabel("gap to floor (percentage points)")
    ax.set_title("Achieved error approaches the floor at long horizons")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(os.path.join(args.outdir, f"fig2_gap_vs_horizon.{ext}"),
                     dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved figures to {args.outdir}/ (fig1, fig2; png + pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
