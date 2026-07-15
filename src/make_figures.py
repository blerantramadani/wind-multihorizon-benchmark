#!/usr/bin/env python3
"""
================================================================================
FIGURE GENERATION — Figures 2 and 3 of the Eng multihorizon paper
================================================================================
Reproduces the two results figures directly from the canonical wide-format
summary CSV, so the plots are fully traceable to the source data.

  Figure 2 (fig2_nrmse.png):  nRMSE vs forecast horizon, all 8 methods, 2 panels.
  Figure 3 (fig3_bestml.png): best learning model vs strongest non-learning
                              reference at each horizon, shaded where ML is worse
                              (red) or better (green).

INPUT : results/mh_summary_corrected.csv
        columns: dataset, horizon_h, Persistence_naive, Persistence_seasonal,
                 Climatology, LinearRegression, MLP_mean, LSTM_mean, GRU_mean,
                 DLinear_mean, PatchTST_mean
OUTPUT: fig2_nrmse.png/.pdf, fig3_bestml.png/.pdf

USAGE : python3 make_figures.py
Author: B. Ramadani
================================================================================
"""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'results/mh_summary_corrected.csv'
if not os.path.exists(CSV):
    CSV = 'mh_summary_corrected.csv'
HOR = [1, 2, 3, 6, 12, 18, 24, 36, 48]
ML  = ['MLP', 'LSTM', 'GRU', 'DLinear', 'PatchTST']
NONML_ALL  = ['Persistence_naive', 'Persistence_seasonal', 'Climatology', 'LinearRegression']
NONML_SHOW = ['Persistence_naive', 'Climatology', 'LinearRegression']
LABEL = {'Persistence_naive': 'Persistence', 'Climatology': 'Climatology',
         'LinearRegression': 'Linear regression'}
COL = {'Persistence_naive': '#7f7f7f', 'Climatology': '#d99a00',
       'LinearRegression': '#2c6db5', 'MLP': '#3c8c40', 'LSTM': '#d55e00',
       'GRU': '#cc79a7', 'DLinear': '#56b4e9', 'PatchTST': '#c0392b'}

def col(m):
    return m if m in NONML_ALL else f'{m}_mean'

def figure2(s):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for a, site, lab in zip(ax, ['kelmarsh', 'penmanshiel'], ['(a) Kelmarsh', '(b) Penmanshiel']):
        d = s[s.dataset == site].set_index('horizon_h')
        for m in NONML_SHOW:
            lw = 2.4 if m != 'LinearRegression' else 2.0
            a.plot(HOR, [d.loc[h, m] for h in HOR], '--', color=COL[m], lw=lw,
                   label=LABEL[m], zorder=2)
        for m in ML:
            a.plot(HOR, [d.loc[h, f'{m}_mean'] for h in HOR], 'o-', ms=3, lw=1.2,
                   color=COL[m], label=m, zorder=3)
        a.set_title(lab, fontsize=11); a.set_xlabel('Forecast horizon (hours)')
        a.set_xticks([1, 6, 12, 18, 24, 36, 48]); a.grid(alpha=0.25)
    ax[0].set_ylabel('nRMSE (%)')
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=8, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig('fig2_nrmse.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig('fig2_nrmse.pdf', bbox_inches='tight', facecolor='white')
    print('  fig2_nrmse.png/.pdf written')

def figure3(s):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for a, site, lab in zip(ax, ['kelmarsh', 'penmanshiel'], ['(a) Kelmarsh', '(b) Penmanshiel']):
        d = s[s.dataset == site].set_index('horizon_h')
        bm = np.array([min(d.loc[h, f'{m}_mean'] for m in ML) for h in HOR])
        br = np.array([min(d.loc[h, m] for m in NONML_ALL) for h in HOR])
        a.fill_between(HOR, br, bm, where=(bm >= br), interpolate=True,
                       color='#f4b7bd', alpha=0.9, label='ML worse', zorder=1)
        a.fill_between(HOR, br, bm, where=(bm < br), interpolate=True,
                       color='#b7e2be', alpha=0.9, label='ML better', zorder=1)
        a.plot(HOR, br, '-o', ms=4, color='#2c6db5', lw=1.8,
               label='Strongest non-ML ref.', zorder=3)
        a.plot(HOR, bm, '-s', ms=4, color='#d55e00', lw=1.8,
               label='Best learning model', zorder=3)
        a.set_title(lab, fontsize=11); a.set_xlabel('Forecast horizon (hours)')
        a.set_xticks([1, 6, 12, 18, 24, 36, 48]); a.grid(alpha=0.25)
    ax[0].set_ylabel('nRMSE (%)'); ax[0].legend(fontsize=8, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    fig.savefig('fig3_bestml.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig('fig3_bestml.pdf', bbox_inches='tight', facecolor='white')
    for site in ['kelmarsh', 'penmanshiel']:
        d = s[s.dataset == site].set_index('horizon_h')
        for h in HOR:
            bm = min(d.loc[h, f'{m}_mean'] for m in ML)
            br = min(d.loc[h, m] for m in NONML_ALL)
            if bm < br:
                print(f'    ML numerically better: {site} h={h} by {br-bm:.2f} pts')
    print('  fig3_bestml.png/.pdf written')

if __name__ == '__main__':
    s = pd.read_csv(CSV)
    print('Generating figures from', CSV)
    figure2(s); figure3(s)
    print('Done.')
