#!/usr/bin/env python3
"""
================================================================================
HYPERPARAMETER SENSITIVITY STUDY — RESPONSE TO "UNDERTUNED MODELS" OBJECTION
================================================================================
Companion to multi_horizon_hq_v2.py (identical data pipeline & protocol).

PURPOSE
  A reviewer may object that the negative result (no ML model beats the
  strongest non-ML benchmark) is an artifact of a single fixed configuration
  (Adam lr=0.001, hidden 128/64). This script tests that objection directly:
  a 3x3 grid over learning rate x capacity, for the MLP and the LSTM, at three
  representative horizons on Penmanshiel (the site where the benchmark
  hierarchy shifts most).

DESIGN
  Site      : penmanshiel  (override with --site kelmarsh, or --site both)
  Horizons  : h in {1, 12, 48}   (persistence / linreg / climatology regimes)
  Models    : MLP, LSTM          (feedforward vs recurrent capacity question)
  Grid      : lr     in {0.0005, 0.001, 0.005}
              hidden in {(64,32), (128,64), (256,128)}   -- (128,64) = paper default
  Seeds     : 3 per configuration (mean reported)
  Everything else IDENTICAL to the main study: same features, LOOK_BACK=48,
  TEST_HOURS=1000, MinMaxScaler, direct multi-step, EarlyStopping(patience=5),
  validation_split=0.1, max 50 epochs, batch 32.

CLAIM TESTED
  If NO cell of the grid pushes the seed-mean prediction below the strongest
  non-ML benchmark (DM-HLN, p<0.05), the negative result is robust to tuning.

OUTPUT
  results_sensitivity/sensitivity_grid.csv   all cells (site,h,model,lr,hidden)
  results_sensitivity/sensitivity_summary.csv best cell vs default vs benchmark
  Console summary formatted for direct citation in the paper.

RUNTIME (MacBook Air M2, CPU): roughly 3-6 h for the default design
  (2 models x 9 cells x 3 horizons x 3 seeds = 162 trainings).
  Use --quick for a 30-60 min smoke test (2 seeds, lr grid only).

Author: B. Ramadani | Protocol consistent with multi_horizon_hq_v2.py
================================================================================
"""
import os, math, argparse, warnings, time, itertools
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy import stats

# --------- CONFIG (mirrors multi_horizon_hq_v2.py exactly) ---------
DATASETS = {
    'kelmarsh':    {'filepath': 'data/Kelmarsh_T1_2018_hourly.csv',     'rated_power_kw': 2050},
    'penmanshiel': {'filepath': 'data/Penmanshiel_T01_2018_hourly.csv', 'rated_power_kw': 2050},
}
LOOK_BACK, EPOCHS, BATCH_SIZE, TEST_HOURS = 48, 50, 32, 1000
TARGET_COL = 'ActivePower_kW'
FEATURES = ['WindSpeed_m_s', 'Temperature_C', 'ActivePower_lag1',
            'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'WindSpeed_Rolling3h']

HORIZONS_SENS = [1, 12, 48]
LR_GRID       = [0.0005, 0.001, 0.005]
HIDDEN_GRID   = [(64, 32), (128, 64), (256, 128)]   # (128,64) = paper default
DEFAULT_CFG   = (0.001, (128, 64))
SEEDS         = [0, 1, 2]

os.makedirs('results_sensitivity', exist_ok=True)


# --------------- DATA (identical to main study) ---------------
def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df['HourOfDay'] = df.index.hour
    df['Hour_sin']  = np.sin(2*np.pi*df.index.hour/24)
    df['Hour_cos']  = np.cos(2*np.pi*df.index.hour/24)
    df['Month_sin'] = np.sin(2*np.pi*df.index.month/12)
    df['Month_cos'] = np.cos(2*np.pi*df.index.month/12)
    df['WindSpeed_Rolling3h'] = df['WindSpeed_m_s'].rolling(window=3).mean()
    df['ActivePower_lag1'] = df[TARGET_COL].shift(1)
    return df.dropna()


def create_sequences_h(X_scaled, y_scaled, look_back, horizon):
    X_seq, y_seq, tgt_idx = [], [], []
    last = len(X_scaled) - horizon + 1
    for i in range(look_back, last):
        X_seq.append(X_scaled[i-look_back:i])
        y_seq.append(y_scaled[i + horizon - 1])
        tgt_idx.append(i + horizon - 1)
    return np.array(X_seq), np.array(y_seq), np.array(tgt_idx)


def nrmse(actual, pred, rated):
    return math.sqrt(mean_squared_error(actual, pred)) / rated * 100


# --------------- DM test (identical to main study) ---------------
def dm_test(actual, pred1, pred2, h):
    """HLN-corrected DM. Negative => pred1 better. Returns (stat, p)."""
    e1 = np.asarray(actual) - np.asarray(pred1)
    e2 = np.asarray(actual) - np.asarray(pred2)
    d = e1**2 - e2**2
    T = len(d); dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    gammas = [np.cov(d[k:], d[:-k], ddof=0)[0, 1] for k in range(1, h)] if h > 1 else []
    var_d = (gamma0 + 2*sum(gammas)) / T
    if var_d <= 0:
        return 0.0, 1.0
    dm = dbar / math.sqrt(var_d)
    corr = math.sqrt(max((T + 1 - 2*h + h*(h-1)/T) / T, 1e-9))
    dm_hln = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=T-1))
    return round(dm_hln, 4), round(p, 4)


# --------------- Parameterized model builders ---------------
def build_mlp(dim, seed, lr, hidden):
    tf.random.set_seed(seed)
    h1, h2 = hidden
    m = Sequential([Input(shape=(dim,)),
                    Dense(h1, activation='relu'), Dropout(0.05),
                    Dense(h2, activation='relu'), Dropout(0.05), Dense(1)])
    m.compile(optimizer=Adam(lr), loss='mse'); return m


def build_lstm(shape, seed, lr, hidden):
    tf.random.set_seed(seed)
    h1, h2 = hidden
    m = Sequential([Input(shape=shape),
                    LSTM(h1, return_sequences=True), Dropout(0.05),
                    LSTM(h2), Dropout(0.05), Dense(1)])
    m.compile(optimizer=Adam(lr), loss='mse'); return m


# --------------- Per-horizon setup + benchmarks ---------------
def prepare_horizon(dataset, horizon):
    cfg = DATASETS[dataset]; rated = cfg['rated_power_kw']
    df = load_and_prepare(cfg['filepath'])
    X_data = df[FEATURES].values
    y_data = df[TARGET_COL].values.reshape(-1, 1)
    hour_arr = df['HourOfDay'].values
    fscaler, tscaler = MinMaxScaler(), MinMaxScaler()
    X_scaled = fscaler.fit_transform(X_data)
    y_scaled = tscaler.fit_transform(y_data).flatten()
    full_real = y_data.flatten()

    X_seq, y_seq, tgt_idx = create_sequences_h(X_scaled, y_scaled, LOOK_BACK, horizon)
    split = len(X_seq) - TEST_HOURS
    Xtr_s, Xte_s = X_seq[:split], X_seq[split:]
    ytr, yte = y_seq[:split], y_seq[split:]
    tgt_te = tgt_idx[split:]
    Xtr_f, Xte_f = Xtr_s[:, -1, :], Xte_s[:, -1, :]
    yte_real = tscaler.inverse_transform(yte.reshape(-1, 1)).flatten()

    # Benchmarks (identical construction to main study)
    bench_preds = {}
    bench_preds['Persistence_naive'] = full_real[tgt_te - horizon]
    train_tgt_idx = tgt_idx[:split]
    train_hours = hour_arr[train_tgt_idx]
    train_vals = full_real[train_tgt_idx]
    clim = {hh: train_vals[train_hours == hh].mean() for hh in range(24)}
    bench_preds['Climatology'] = np.array([clim[h_] for h_ in hour_arr[tgt_te]])
    lin = LinearRegression().fit(Xtr_f, ytr)
    bench_preds['LinearRegression'] = tscaler.inverse_transform(
        lin.predict(Xte_f).reshape(-1, 1)).flatten()

    bench_scores = {k: nrmse(yte_real, v, rated) for k, v in bench_preds.items()}
    strongest = min(bench_scores, key=bench_scores.get)
    return dict(rated=rated, tscaler=tscaler,
                Xtr_s=Xtr_s, Xte_s=Xte_s, Xtr_f=Xtr_f, Xte_f=Xte_f,
                ytr=ytr, yte_real=yte_real,
                bench_preds=bench_preds, bench_scores=bench_scores,
                strongest=strongest)


# --------------- Grid runner ---------------
def run_cell(env, model_name, lr, hidden, seeds, horizon):
    preds = []
    for sd in seeds:
        np.random.seed(sd); tf.random.set_seed(sd)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        if model_name == 'MLP':
            m = build_mlp(env['Xtr_f'].shape[1], sd, lr, hidden)
            m.fit(env['Xtr_f'], env['ytr'], epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.1, callbacks=[es], verbose=0)
            p = env['tscaler'].inverse_transform(
                m.predict(env['Xte_f'], verbose=0)).flatten()
        else:
            m = build_lstm((env['Xtr_s'].shape[1], env['Xtr_s'].shape[2]), sd, lr, hidden)
            m.fit(env['Xtr_s'], env['ytr'], epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.1, callbacks=[es], verbose=0)
            p = env['tscaler'].inverse_transform(
                m.predict(env['Xte_s'], verbose=0)).flatten()
        preds.append(p)
        tf.keras.backend.clear_session()
    seed_scores = [nrmse(env['yte_real'], p, env['rated']) for p in preds]
    mean_pred = np.mean(preds, axis=0)   # seed-averaged prediction, as in main study
    cell_nrmse = nrmse(env['yte_real'], mean_pred, env['rated'])
    dm, pval = dm_test(env['yte_real'], mean_pred,
                       env['bench_preds'][env['strongest']], horizon)
    return dict(nRMSE_meanpred=round(cell_nrmse, 4),
                nRMSE_seedmean=round(np.mean(seed_scores), 4),
                nRMSE_seedstd=round(np.std(seed_scores), 4),
                DM_vs_strongest=dm, p_vs_strongest=pval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', default='penmanshiel',
                    choices=['penmanshiel', 'kelmarsh', 'both'])
    ap.add_argument('--quick', action='store_true',
                    help='2 seeds, learning-rate grid only (smoke test)')
    args = ap.parse_args()

    sites = ['penmanshiel', 'kelmarsh'] if args.site == 'both' else [args.site]
    seeds = SEEDS[:2] if args.quick else SEEDS
    hidden_grid = [(128, 64)] if args.quick else HIDDEN_GRID

    rows, t0 = [], time.time()
    for site in sites:
        for h in HORIZONS_SENS:
            env = prepare_horizon(site, h)
            print(f"\n=== {site} | h={h} | strongest non-ML: {env['strongest']} "
                  f"({env['bench_scores'][env['strongest']]:.2f} nRMSE%) ===")
            for model_name, lr, hidden in itertools.product(
                    ['MLP', 'LSTM'], LR_GRID, hidden_grid):
                r = run_cell(env, model_name, lr, hidden, seeds, h)
                is_default = (lr, hidden) == DEFAULT_CFG
                rows.append(dict(site=site, horizon_h=h, model=model_name,
                                 lr=lr, hidden=f"{hidden[0]}/{hidden[1]}",
                                 is_default=is_default,
                                 strongest_bench=env['strongest'],
                                 bench_nRMSE=round(env['bench_scores'][env['strongest']], 4),
                                 **r))
                flag = ' *default*' if is_default else ''
                sig = ' <-- ML significantly BETTER' if (
                    r['DM_vs_strongest'] < 0 and r['p_vs_strongest'] < 0.05) else ''
                print(f"  {model_name:5s} lr={lr:<7g} hid={hidden[0]}/{hidden[1]:<4d} "
                      f"nRMSE={r['nRMSE_meanpred']:6.2f}  DM={r['DM_vs_strongest']:+.2f} "
                      f"p={r['p_vs_strongest']:.4f}{flag}{sig}")
                pd.DataFrame(rows).to_csv(
                    'results_sensitivity/sensitivity_grid.csv', index=False)

    df = pd.DataFrame(rows)
    summ = []
    for (site, h), g in df.groupby(['site', 'horizon_h']):
        best = g.loc[g['nRMSE_meanpred'].idxmin()]
        dflt = g[g['is_default']]
        dflt_best = dflt.loc[dflt['nRMSE_meanpred'].idxmin()] if len(dflt) else best
        any_sig = ((g['DM_vs_strongest'] < 0) & (g['p_vs_strongest'] < 0.05)).any()
        summ.append(dict(site=site, horizon_h=h,
                         strongest_bench=best['strongest_bench'],
                         bench_nRMSE=best['bench_nRMSE'],
                         best_cell=f"{best['model']} lr={best['lr']} hid={best['hidden']}",
                         best_nRMSE=best['nRMSE_meanpred'],
                         default_nRMSE=dflt_best['nRMSE_meanpred'],
                         tuning_gain=round(dflt_best['nRMSE_meanpred']
                                           - best['nRMSE_meanpred'], 4),
                         any_cell_beats_bench_significantly=bool(any_sig)))
    sdf = pd.DataFrame(summ)
    sdf.to_csv('results_sensitivity/sensitivity_summary.csv', index=False)

    print("\n" + "="*74)
    print("SUMMARY (results_sensitivity/sensitivity_summary.csv)")
    print("="*74)
    print(sdf.to_string(index=False))
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    if not sdf['any_cell_beats_bench_significantly'].any():
        print("\nCONCLUSION: no configuration in the grid significantly beats the")
        print("strongest non-ML benchmark -> the negative result is robust to tuning.")
    else:
        print("\nATTENTION: at least one configuration significantly beats the benchmark.")
        print("Inspect sensitivity_grid.csv before drawing conclusions.")


if __name__ == '__main__':
    main()
