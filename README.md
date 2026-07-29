# Shifting Benchmark Hierarchies in Single-Turbine Wind Power Forecasting

Code and aggregated results for:

> Ramadani, B.; Fustic, V. *Shifting Benchmark Hierarchies and a Horizon-Specific
> Decision Rule for Single-Turbine Wind Power Forecasting.* Wind, 2026.

Five learning architectures (MLP, LSTM, GRU, DLinear, PatchTST) are evaluated at
nine horizons (1–48 h) on two operational UK wind farms against a tiered
hierarchy of four non-learning references, using a moving-block bootstrap and
HLN-corrected Diebold–Mariano tests.

**Main finding.** Across all eighteen site–horizon combinations, no learning
ensemble achieves an advantage over the strongest simple reference that is
significant under both tests. Sixteen of the ninety model–reference comparisons
are significant, and every one of them favours the simple reference.

---

## Data

SCADA data are not redistributed here. Download from Zenodo:

| Site | DOI |
|---|---|
| Kelmarsh | [10.5281/zenodo.5841834](https://doi.org/10.5281/zenodo.5841834) |
| Penmanshiel | [10.5281/zenodo.5946808](https://doi.org/10.5281/zenodo.5946808) |

Place the extracted CSV files under `data/kelmarsh/` and `data/penmanshiel/`.
This study uses turbine WT1 (Kelmarsh) and WT01 (Penmanshiel), resampled to
hourly resolution.

---

## Requirements

```
Python 3.9
tensorflow
numpy
pandas
scipy
scikit-learn
matplotlib
```

Install:

```bash
pip install -r requirements.txt
```

---

## Reproducing the results

Run in order:

```bash
python3 scripts/01_pipeline_corrected.py     # trains all models, writes summary + DM tests
python3 scripts/02_bootstrap_dm.py           # moving-block bootstrap on saved predictions
python3 scripts/03_ensemble_fragility.py     # seed-composition sensitivity of the DM statistic

```

Step 1 is the only expensive step (five architectures x nine horizons x ten
seeds x two sites). Steps 2–4 operate on the saved prediction files and take
seconds.

---

## Outputs

| File | Contents |
|---|---|
| `results/01_summary.csv` | nRMSE (mean, sd over seeds) and MAE for every model, horizon and site — 162 rows |
| `results/01_dm_vs_bench.csv` | HLN-corrected DM statistic and p-value for all 90 model–reference comparisons |
| `results/02_bootstrap_dm.csv` | Moving-block bootstrap p-values alongside the asymptotic ones |
| `results/03_ensemble_fragility.csv` | Jackknife and subsampling diagnostics for the seed ensemble |
| `results/preds_{site}_h{H}_corrected_filt.csv` | Per-seed predictions, ensemble mean, benchmarks and actuals |

`results/01_summary.csv` and `results/01_dm_vs_bench.csv` correspond exactly to
Tables 1–3 of the manuscript.

---

## Methodological notes

**Contiguity filter.** Because the Penmanshiel record contains an extended
maintenance outage, input–target pairs are built on the true hourly timestamp
index rather than on row positions. A pair at index `i` with horizon `h` and
look-back `L` is retained only if the whole span from `i - L` to `i + h - 1` is
contiguous at one-hour spacing. See `contiguous_sequence_ids` in
`scripts/01_pipeline_corrected.py`.

**Scaling.** Min–Max scalers are fitted on the training period only (all
observations preceding the test window, including those later held out for
validation) and applied unchanged to validation and test. No clipping is
performed.

**Lagged-power correction.** An earlier version of the feature construction
applied an additional one-step shift, leaving the learning models with a
strictly smaller information set than the persistence reference. This is
corrected here; see Section 3.6 of the manuscript. All results in this
repository reflect the corrected construction.

**Reproducibility.** TensorFlow training is not bit-reproducible across
executions even with fixed seeds. All learning results are therefore reported as
the mean and standard deviation over ten seeds, and all statistical tests are
applied to the seed-averaged forecast rather than to any individual run. Re-running
`01_pipeline_corrected.py` will reproduce the reported values to within
seed-level variation, not exactly.

---

## Citation

```bibtex
@article{ramadani2026shifting,
  author  = {Ramadani, Blerant and Fustic, Vangel},
  title   = {Shifting Benchmark Hierarchies and a Horizon-Specific Decision
             Rule for Single-Turbine Wind Power Forecasting},
  journal = {Wind},
  year    = {2026},
  doi     = {10.3390/xxxxx}
}
```

Archived release: [10.5281/zenodo.21472810](https://doi.org/10.5281/zenodo.21472810)

## License

MIT — see `LICENSE`.
