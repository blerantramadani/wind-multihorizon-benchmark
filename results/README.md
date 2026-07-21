# Results

Aggregated result files produced by the pipeline. All values are traceable to
these CSVs; the manuscript tables and figures are generated from them.

| File | Contents |
|---|---|
| `mh_summary_corrected.csv` | Canonical summary: nRMSE (%) mean per method × horizon × site (wide format). Source of Tables 1-2 and Figures 2-3. |
| `02_bootstrap_dm.csv` | Diebold-Mariano (HLN) and stationary-bootstrap p-values for the best learning model vs the strongest non-learning reference, per site × horizon. Source of Table 3. |
| `01_bug_cost.csv` | Legacy vs corrected input-construction comparison per model × horizon × site: the measured cost of the defect documented in Section 3.6. |
| `strict_mode_check_v2.csv` | Robustness check with the scalers fitted on the training period only; benchmark hierarchy and conclusions unchanged. |

## Reproducing
Run `python3 src/pipeline_corrected.py --mode corrected` to regenerate
`mh_summary_corrected.csv`, then `python3 src/make_figures.py` for the figures.
See the root `README.md` for the full workflow.
