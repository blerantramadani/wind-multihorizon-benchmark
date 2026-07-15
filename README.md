# Wind Multi-Horizon Benchmark

Companion code and results for the paper:

> **A Cautionary Framework for Multi-Horizon Wind Power Forecasting: Shifting Benchmark Hierarchies at the Single-Turbine Scale**
> Blerant Ramadani, Vangel Fustic, 
> *Eng* (MDPI), Special Issue "Advances in AI-Enabled Integrated Renewable Energy Systems, Smart Grids and Their Applications", 2026 (under review).

## What this study does

We forecast the output of single-turbine wind power from 1 to 48 hours ahead using operational SCADA data from two UK wind farms (Kelmarsh WT1 and Penmanshiel WT01) and compare five learning architectures - MLP, LSTM, GRU, DLinear and PatchTST - with a full hierarchy of non-learning references: naive persistence, seasonal persistence, hour-of-day climatology, feature-based linear regression. Each model is trained with 10 random seeds at nine horizons and every comparison is validated using a stationary bootstrap (primary test) and Diebold-Mariano tests under the Harvey-Leybourne-Newbold correction.

The first finding is that the strongest simple reference is a feature-based linear regression across almost the entire horizon range (climatology overtakes only at the longest horizons on one site), and no learning model is better than the strongest reference, based on both statistical tests, at any of the eighteen site-horizon combinations.

## Repository structure

```
├── src/
│   ├── prepare_kelmarsh.py            # raw Zenodo SCADA -> hourly modelling CSV
│   ├── prepare_penmanshiel.py         # raw Zenodo SCADA -> hourly modelling CSV
│   ├── pipeline_corrected.py          # main study: 5 models x 9 horizons x 10 seeds
│   │                                  #   (legacy and corrected input construction)
│   ├── hyperparameter_sensitivity.py  # 54-cell lr x capacity sensitivity grid
│   ├── strict_mode_check_v2.py        # robustness: scalers fit on training period only
│   ├── make_figures.py                # regenerates Figures 2 and 3 from results
│   └── verify_pipeline_integrity.py   # audit of the corrected input construction
├── results/
│   ├── mh_summary_corrected.csv       # canonical summary: all models x horizons x sites
│   ├── 02_bootstrap_dm.csv            # DM (HLN) and stationary-bootstrap p-values
│   ├── 01_bug_cost.csv                # legacy vs corrected: measured cost of the defect
│   └── strict_mode_check_v2.csv       # train-only-scaling robustness check output
├── figures/                           # paper figures (PNG + PDF)
├── data/                              # see data/README.md (datasets NOT redistributed)
└── requirements.txt
```

## Reproducing the study

1. **Get the data** (not redistributed here — see `data/README.md`):
   download the Kelmarsh and Penmanshiel SCADA datasets from Zenodo and place the
   2018 turbine files in `data/`.

2. **Prepare the hourly modelling files:**
   ```bash
   python3 src/prepare_kelmarsh.py
   python3 src/prepare_penmanshiel.py
   ```

3. **Run the main study** in the corrected construction (approx. one night on a
   consumer machine, no GPU required — 900 model trainings):
   ```bash
   python3 src/pipeline_corrected.py --mode corrected
   ```
   Running without `--mode` executes both the legacy and corrected constructions and
   reports the legacy-vs-corrected delta (the measured cost of the input-construction
   defect; see below).

4. **Run the hyperparameter sensitivity grid** (optional, ~2–4 h):
   ```bash
   python3 src/hyperparameter_sensitivity.py
   ```

5. **Regenerate the figures:**
   ```bash
   python3 src/make_figures.py
   ```

Environment: Python 3.9+, TensorFlow 2.x (CPU is sufficient). Exact package versions
are pinned in `requirements.txt`. All experiments in the paper were executed on Apple
Silicon (M-series) without GPU acceleration.

## Input-construction note

An earlier version of the pipeline contained a one-hour information handicap against the
learning models: the lagged-power feature was shifted so that the most recent power value
in the input window was P(t−2), while the persistence reference conditioned on P(t−1).
The construction was corrected so that all methods condition on the same information set,
and the entire study was re-executed from a single unified protocol (including a gap
filter applied uniformly to both sites). All results in the paper derive from that
corrected run; the correction is documented in Section 3.6 of the paper.

`src/pipeline_corrected.py` implements both constructions (`--mode legacy` and
`--mode corrected`) and reports the measured cost of the defect
(`results/01_bug_cost.csv`). `src/verify_pipeline_integrity.py` audits the corrected
construction (a leakage guard asserts that every input-window row precedes its target).

## Robustness

`src/strict_mode_check_v2.py` re-runs representative configurations (1 h and 24 h) with
the Min–Max scalers fitted on the training period only, rather than on the full series.
As reported in `results/strict_mode_check_v2.csv`, the benchmark hierarchy and all
conclusions are unchanged: the non-learning references are identical by construction
(linear regression is affinely invariant to input scaling), and the learning models
shift by no more than seed-level variability.

## Data licence

The SCADA datasets are the property of their respective owners and are distributed by
C. Plumley on Zenodo under CC-BY-4.0. They are **not** redistributed in this repository;
see `data/README.md` for download instructions.

## Citing

If you use this code or the aggregated results, please cite the paper (citation will be
updated upon publication):

```bibtex
@article{ramadani2026cautionary,
  author  = {Ramadani, Blerant and Fustic, Vangel},
  title   = {A Cautionary Framework for Multi-Horizon Wind Power Forecasting:
             Shifting Benchmark Hierarchies at the Single-Turbine Scale},
  journal = {Eng},
  year    = {2026},
  note    = {under review}
}
```

## Licence

Code: MIT (see `LICENSE`). Result CSVs and figures: CC-BY-4.0.
