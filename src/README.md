# Source Code
 
Scripts to reproduce the study. Run from the repository root with the datasets
placed in `data/` (see `data/README.md` for download instructions).
 
| Script | Purpose |
|---|---|
| `prepare_kelmarsh.py` | Convert raw Kelmarsh SCADA (Zenodo) into the hourly modelling CSV. |
| `prepare_penmanshiel.py` | Convert raw Penmanshiel SCADA (Zenodo) into the hourly modelling CSV. |
| `pipeline_corrected.py` | Main study: 5 learning models × 9 horizons × 10 seeds, with both the legacy and corrected input construction (`--mode legacy` / `--mode corrected`). Produces the summary, bug-cost, and DM-vs-benchmark result files. |
| `hyperparameter_sensitivity.py` | 54-cell learning-rate × capacity sensitivity grid. |
| `strict_mode_check_v2.py` | Robustness check: scalers fitted on the training period only. |
| `make_figures.py` | Regenerate Figures 2 and 3 from `results/mh_summary_corrected.csv`. |
| `verify_pipeline_integrity.py` | Audit of the corrected input construction (leakage guard). |
 
## Typical workflow
```bash
python3 src/prepare_kelmarsh.py
python3 src/prepare_penmanshiel.py
python3 src/pipeline_corrected.py --mode corrected
python3 src/make_figures.py
```
 
Environment: Python 3.9+, TensorFlow 2.x (CPU is sufficient). Exact versions are
pinned in the root `requirements.txt`.
