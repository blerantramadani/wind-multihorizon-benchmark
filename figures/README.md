# Figures
 
Figures used in the manuscript, provided in both PNG (600 dpi, for quick viewing
on GitHub) and PDF (vector, for reuse).
 
| File | Description |
|---|---|
| `fig1_workflow` | System workflow of the evaluation framework: SCADA inputs, preprocessing, nine-horizon forecasting with four non-learning references and five learning architectures, dual statistical testing, and the two principal outputs. |
| `fig2_nrmse` | Normalized RMSE versus forecast horizon for all methods at both sites (Kelmarsh, Penmanshiel), with one-standard-deviation error bars over ten seeds. |
| `fig3_bestml` | Best learning model versus the strongest non-learning reference at each horizon; shaded regions mark where the learning model is worse or marginally better. |
| `fig4_decisionrule` | Operational decision rule: horizon-specific baseline selection and conditional machine-learning deployment. |
 
## Regenerating
Figures 2 and 3 are reproduced from `results/mh_summary_corrected.csv`:
```bash
python3 src/make_figures.py
```
Figures 1 and 4 are schematic diagrams.
