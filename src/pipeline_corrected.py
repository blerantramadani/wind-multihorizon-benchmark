#!/usr/bin/env python3
"""
================================================================================
CORRECTED PIPELINE — main study of the multi-horizon benchmark paper
================================================================================
Companion code for:
  Ramadani & Fustic, "A Cautionary Framework for Multi-Horizon Wind Power
  Forecasting: Shifting Benchmark Hierarchies at the Single-Turbine Scale",
  Eng (MDPI), 2026.

WHAT IT DOES:
  Runs the full study in two parallel modes with identical seeds:

    legacy    : power feature 'ActivePower_lag1' (shift 1) — the original
                construction. The last window row holds P(t-2): a one-hour
                information handicap against the learning models.
    corrected : the UNSHIFTED power column as a window feature. The last
                window row holds P(t-1) — exactly what persistence sees.
                No leakage: every window row j satisfies j <= i-1 < target.

  Both modes apply the GAP FILTER: only windows whose interval
  [window start ... target] is fully contiguous (1-hour steps) are accepted.
  This also keeps persistence/seasonal-persistence honest (the value
  "h hours before" is truly h hours before, not across a data gap).

  The legacy-vs-corrected delta per model/horizon/site is reported as the
  measured cost of the construction defect (documented in Section 3.6 of
  the paper).

ARCHITECTURES: MLP, LSTM, GRU, DLinear (Zeng et al. 2023),
               PatchTST (Nie et al. 2023)
BENCHMARKS:    Persistence, Seasonal persistence, Climatology,
               Linear regression

USAGE:
  python3 pipeline_corrected.py                     # h=1,2,3 | 10 seeds
  python3 pipeline_corrected.py --quick             # h=1     | 3 seeds
  python3 pipeline_corrected.py --horizons 1,6,24   # custom horizons
  python3 pipeline_corrected.py --mode corrected    # single mode

OUTPUT:
  results/01_summary.csv      all metrics: site x horizon x mode x model
  results/01_bug_cost.csv     legacy vs corrected per model: delta nRMSE + DM
  results/01_dm_vs_bench.csv  DM of every model vs the strongest non-ML
                              benchmark

Author: B. Ramadani
================================================================================
"""
import os, math, time, argparse, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, Layer,
                                     LayerNormalization, MultiHeadAttention,
                                     Flatten, Add, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ----------------------------- CONFIG -----------------------------
DATASETS = {
    'kelmarsh':    {'filepath': 'data/Kelmarsh_T1_2018_hourly.csv',     'rated_power_kw': 2050},
    'penmanshiel': {'filepath': 'data/Penmanshiel_T01_2018_hourly.csv', 'rated_power_kw': 2050},
}
LOOK_BACK, EPOCHS, BATCH_SIZE, LR, TEST_HOURS = 48, 50, 32, 0.001, 1000
TARGET_COL = 'ActivePower_kW'

BASE_FEATURES = ['WindSpeed_m_s', 'Temperature_C',
                 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
                 'WindSpeed_Rolling3h']
POWER_FEATURE = {'legacy': 'ActivePower_lag1',   # shifted P(t-1) -> window sees P(t-2)
                 'corrected': TARGET_COL}         # unshifted P(t)  -> window sees P(t-1)

os.makedirs('results', exist_ok=True)


# ----------------------------- DATA -----------------------------
def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    df['HourOfDay'] = df.index.hour
    df['Hour_sin']  = np.sin(2*np.pi*df.index.hour/24)
    df['Hour_cos']  = np.cos(2*np.pi*df.index.hour/24)
    df['Month_sin'] = np.sin(2*np.pi*df.index.month/12)
    df['Month_cos'] = np.cos(2*np.pi*df.index.month/12)
    df['WindSpeed_Rolling3h'] = df['WindSpeed_m_s'].rolling(window=3).mean()
    df['ActivePower_lag1'] = df[TARGET_COL].shift(1)
    return df.dropna()


def contiguous_sequence_ids(index, look_back, horizon, enabled=True):
    """Return positional target ids whose interval
    [i-look_back ... i+horizon-1] is fully contiguous (1-hour steps)."""
    n = len(index)
    if not enabled:
        return np.arange(look_back, n - horizon + 1, dtype=int)
    diffs_h = np.diff(index.to_numpy()).astype('timedelta64[m]').astype(float) / 60.0
    gap = (diffs_h > 1.0 + 1e-9).astype(int)          # gap[j] = hapi j -> j+1
    csum = np.concatenate([[0], np.cumsum(gap)])       # csum[k] = gaps before row k
    ids = []
    for i in range(look_back, n - horizon + 1):
        # hapat e mbuluar: (i-look_back .. i+horizon-2), d.m.th. csum[i+h-1]-csum[i-lb]==0
        if csum[i + horizon - 1] - csum[i - look_back] == 0:
            ids.append(i)
    return np.array(ids, dtype=int)


def build_sequences(X_scaled, y_scaled, ids, look_back, horizon):
    """Window [i-lb : i] -> target y[i+h-1]; accepted ids only."""
    X = np.stack([X_scaled[i - look_back:i] for i in ids])
    y = y_scaled[ids + horizon - 1]
    tgt = ids + horizon - 1
    return X, y, tgt


# ------------------ ARKITEKTURAT (verbatim nga multi_horizon_hq_v2) ------------------
def build_lstm(shape, seed):
    tf.random.set_seed(seed)
    m = Sequential([Input(shape=shape),
                    LSTM(128, return_sequences=True), Dropout(0.05),
                    LSTM(64), Dropout(0.05), Dense(1)])
    m.compile(optimizer=Adam(LR), loss='mse'); return m

def build_gru(shape, seed):
    tf.random.set_seed(seed)
    m = Sequential([Input(shape=shape),
                    GRU(128, return_sequences=True), Dropout(0.05),
                    GRU(64), Dropout(0.05), Dense(1)])
    m.compile(optimizer=Adam(LR), loss='mse'); return m

def build_mlp(dim, seed):
    tf.random.set_seed(seed)
    m = Sequential([Input(shape=(dim,)),
                    Dense(128, activation='relu'), Dropout(0.05),
                    Dense(64, activation='relu'), Dropout(0.05), Dense(1)])
    m.compile(optimizer=Adam(LR), loss='mse'); return m

class SeriesDecomposition(Layer):
    def __init__(self, kernel_size=25, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
    def call(self, x):
        x_pad = tf.concat([tf.repeat(x[:, :1, :], self.pad, axis=1), x,
                           tf.repeat(x[:, -1:, :], self.pad, axis=1)], axis=1)
        trend = tf.nn.avg_pool1d(x_pad, ksize=self.kernel_size, strides=1, padding='VALID')
        return x - trend, trend

def build_dlinear(shape, seed):
    tf.random.set_seed(seed)
    inp = Input(shape=shape)
    seasonal, trend = SeriesDecomposition(kernel_size=25)(inp)
    s_out = Dense(1, use_bias=True)(Flatten()(seasonal))
    t_out = Dense(1, use_bias=True)(Flatten()(trend))
    m = Model(inp, Add()([s_out, t_out]))
    m.compile(optimizer=Adam(LR), loss='mse'); return m

class PatchChannelEmbed(Layer):
    def __init__(self, patch_len=16, stride=8, d_model=64, n_channels=8, **kwargs):
        super().__init__(**kwargs)
        self.patch_len, self.stride = patch_len, stride
        self.d_model, self.n_channels = d_model, n_channels
        self.proj = Dense(d_model)
    def call(self, x):
        b = tf.shape(x)[0]; t = tf.shape(x)[1]; c = self.n_channels
        xt = tf.reshape(tf.transpose(x, [0, 2, 1]), [b * c, t, 1])
        patches = tf.image.extract_patches(
            images=tf.expand_dims(xt, 1),
            sizes=[1, 1, self.patch_len, 1], strides=[1, 1, self.stride, 1],
            rates=[1, 1, 1, 1], padding='VALID')
        return self.proj(tf.squeeze(patches, 1))

class RecombineChannels(Layer):
    def __init__(self, n_channels, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_channels, self.d_model = n_channels, d_model
    def call(self, x):
        b = tf.shape(x)[0] // self.n_channels
        return tf.reshape(x, [b, self.n_channels * self.d_model])

def transformer_block(x, d_model, n_heads, ff_dim):
    attn = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    ff = Dense(d_model)(Dense(ff_dim, activation='relu')(x))
    return LayerNormalization(epsilon=1e-6)(Add()([x, ff]))

def build_patchtst(shape, seed, patch_len=16, stride=8, d_model=64, n_heads=4,
                   ff_dim=128, n_layers=2):
    tf.random.set_seed(seed)
    look_back, n_feat = shape
    inp = Input(shape=shape)
    x = PatchChannelEmbed(patch_len, stride, d_model, n_channels=n_feat)(inp)
    for _ in range(n_layers):
        x = transformer_block(x, d_model, n_heads, ff_dim)
    x = GlobalAveragePooling1D()(x)
    x = RecombineChannels(n_feat, d_model)(x)
    out = Dense(1)(Dense(64, activation='relu')(x))
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR), loss='mse'); return m

ARCHITECTURES = {'MLP': build_mlp, 'LSTM': build_lstm, 'GRU': build_gru,
                 'DLinear': build_dlinear, 'PatchTST': build_patchtst}


# ------------------------- METRIKA & DM -------------------------
def nrmse(a, p, rated):
    return math.sqrt(mean_squared_error(a, p)) / rated * 100

def dm_test(actual, pred1, pred2, h):
    """HLN-corrected DM. Negative => pred1 better. Returns (stat, p)."""
    e1 = np.asarray(actual) - np.asarray(pred1)
    e2 = np.asarray(actual) - np.asarray(pred2)
    d = e1**2 - e2**2
    T = len(d); dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    gammas = [np.cov(d[k:], d[:-k], ddof=0)[0, 1] for k in range(1, h)] if h > 1 else []
    var_d = (gamma0 + 2 * sum(gammas)) / T
    if var_d <= 0:
        return 0.0, 1.0
    dm = dbar / math.sqrt(var_d)
    dm *= math.sqrt(max((T + 1 - 2*h + h*(h-1)/T) / T, 1e-9))
    return round(dm, 4), round(2 * (1 - stats.t.cdf(abs(dm), df=T - 1)), 4)


# ----------------------------- EKSPERIMENTI -----------------------------
def run_mode(dataset, horizon, mode, seeds, epochs, test_hours, arch_subset, gap_filter=True):
    cfg = DATASETS[dataset]; rated = cfg['rated_power_kw']
    df = load_and_prepare(cfg['filepath'])

    features = BASE_FEATURES + [POWER_FEATURE[mode]]
    X_data = df[features].values
    y_data = df[TARGET_COL].values.reshape(-1, 1)
    hours = df['HourOfDay'].values
    full_real = y_data.flatten()

    fs, ts_ = MinMaxScaler(), MinMaxScaler()
    Xs = fs.fit_transform(X_data)
    ys = ts_.fit_transform(y_data).flatten()

    ids = contiguous_sequence_ids(df.index, LOOK_BACK, horizon, enabled=gap_filter)
    n_all = len(df) - LOOK_BACK - horizon + 1
    X_seq, y_seq, tgt = build_sequences(Xs, ys, ids, LOOK_BACK, horizon)

    # leakage guard: every window row index < target index
    assert (ids - 1 < tgt).all(), "LEAKAGE: dritarja prek targetin!"

    split = len(X_seq) - test_hours
    Xtr, Xte = X_seq[:split], X_seq[split:]
    ytr = y_seq[:split]
    tgt_te = tgt[split:]
    Xtr_f, Xte_f = Xtr[:, -1, :], Xte[:, -1, :]
    yte_real = ts_.inverse_transform(y_seq[split:].reshape(-1, 1)).flatten()

    print(f"\n  [{dataset} | h={horizon} | {mode}] dritare: {len(X_seq)}/{n_all} "
          f"(dropped {n_all - len(X_seq)} with gaps), test: {test_hours}")

    # ---- Benchmarks (identical in both modes; same gap filter) ----
    bench = {}
    bench['Persistence'] = full_real[tgt_te - horizon]
    # Seasonal: the most recent same-hour value BEFORE the forecast origin.
    # shift = 24*ceil(h/24): for h<=24 -> 24; for h=36,48 -> 48.
    # NOTE: the naive formula (tgt-24) for h>24 would use data AFTER the origin!
    seas_shift = 24 * math.ceil(horizon / 24)
    bench['SeasonalPersistence'] = full_real[tgt_te - seas_shift]
    tr_tgt = tgt[:split]
    clim = pd.Series(full_real[tr_tgt]).groupby(hours[tr_tgt]).mean()
    bench['Climatology'] = clim.reindex(hours[tgt_te]).to_numpy()
    lin = LinearRegression().fit(Xtr_f, ytr)
    bench['LinearRegression'] = ts_.inverse_transform(
        lin.predict(Xte_f).reshape(-1, 1)).flatten()

    bscores = {k: nrmse(yte_real, v, rated) for k, v in bench.items()}
    strongest = min(bscores, key=bscores.get)

    rows = [{'dataset': dataset, 'horizon_h': horizon, 'mode': mode, 'model': k,
             'nRMSE_mean': round(v, 4), 'nRMSE_std': 0.0,
             'MAE_kW': round(mean_absolute_error(yte_real, bench[k]), 2),
             'is_benchmark': True} for k, v in bscores.items()]

    preds_avg, preds_seeds, dm_rows = {}, {}, []
    for name, build in ARCHITECTURES.items():
        if name not in arch_subset:
            continue
        t0 = time.time(); preds = []
        for sd in seeds:
            np.random.seed(sd); tf.random.set_seed(sd)
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            if name == 'MLP':
                m = build(Xtr_f.shape[1], sd)
                m.fit(Xtr_f, ytr, epochs=epochs, batch_size=BATCH_SIZE,
                      validation_split=0.1, callbacks=[es], verbose=0)
                p = ts_.inverse_transform(m.predict(Xte_f, verbose=0)).flatten()
            else:
                m = build((Xtr.shape[1], Xtr.shape[2]), sd)
                m.fit(Xtr, ytr, epochs=epochs, batch_size=BATCH_SIZE,
                      validation_split=0.1, callbacks=[es], verbose=0)
                p = ts_.inverse_transform(m.predict(Xte, verbose=0)).flatten()
            preds.append(p); tf.keras.backend.clear_session()
        seed_scores = [nrmse(yte_real, p, rated) for p in preds]
        mp = np.mean(preds, axis=0); preds_avg[name] = mp; preds_seeds[name] = preds
        dm, pv = dm_test(yte_real, mp, bench[strongest], horizon)
        rows.append({'dataset': dataset, 'horizon_h': horizon, 'mode': mode,
                     'model': name, 'nRMSE_mean': round(np.mean(seed_scores), 4),
                     'nRMSE_std': round(np.std(seed_scores), 4),
                     'MAE_kW': round(mean_absolute_error(yte_real, mp), 2),
                     'is_benchmark': False})
        dm_rows.append({'dataset': dataset, 'horizon_h': horizon, 'mode': mode,
                        'model': name, 'strongest_bench': strongest,
                        'bench_nRMSE': round(bscores[strongest], 4),
                        'DM': dm, 'p_value': pv})
        print(f"    {name:9s} {np.mean(seed_scores):6.2f} ± {np.std(seed_scores):4.2f} "
              f"| vs {strongest}({bscores[strongest]:.2f}): DM={dm:+.2f} p={pv:.4f} "
              f"| {time.time()-t0:.0f}s")

    # ---- Save predictions (for post-hoc bootstrap DM without retraining) ----
    pred_df = {'target_idx': tgt_te, 'actual': yte_real}
    pred_df.update({f'bench_{k}': v for k, v in bench.items()})
    pred_df.update({f'{k}_avg': v for k, v in preds_avg.items()})
    for k, plist in preds_seeds.items():
        for si, p in zip(seeds, plist):
            pred_df[f'{k}_seed{si}'] = p
    tagf = 'filt' if gap_filter else 'nofilt'
    pred_path = f'results/preds_{dataset}_h{horizon}_{mode}_{tagf}.csv'
    pd.DataFrame(pred_df).to_csv(pred_path, index=False)
    print(f'    predictions saved: {pred_path}')

    return rows, dm_rows, preds_avg, yte_real, tgt_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizons', default='1,2,3',
                    help='p.sh. "1,2,3" ose "1,6,24,48"')
    ap.add_argument('--datasets', default='kelmarsh,penmanshiel')
    ap.add_argument('--mode', default='both', choices=['both', 'legacy', 'corrected'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--quick', action='store_true', help='h=1, 3 seeds')
    ap.add_argument('--no-gap-filter', action='store_true',
                    help='disable the gap filter (ablation)')
    ap.add_argument('--smoke', action='store_true',
                    help='technical smoke test: 1 seed, 2 epochs, tiny test (NOT for results)')
    args = ap.parse_args()

    horizons = [1] if args.quick else [int(h) for h in args.horizons.split(',')]
    seeds = list(range(3 if args.quick else args.seeds))
    epochs, test_hours = EPOCHS, TEST_HOURS
    arch_subset = list(ARCHITECTURES)
    if args.smoke:
        horizons, seeds, epochs, test_hours = [1, 3], [0], 2, 50
    modes = ['legacy', 'corrected'] if args.mode == 'both' else [args.mode]
    datasets = [d.strip() for d in args.datasets.split(',')]

    print('#' * 70)
    print('# 01 — PIPELINE I KORRIGJUAR | legacy vs corrected')
    if args.no_gap_filter:
        print('# WARNING: gap filter DISABLED (ablation mode)')
    print(f'# datasets={datasets} | horizons={horizons} | modes={modes} | seeds={len(seeds)}')
    print('#' * 70)

    all_rows, all_dm, cost_rows = [], [], []
    for ds in datasets:
        if not os.path.exists(DATASETS[ds]['filepath']):
            print(f'\n[{ds}] skedari mungon: {DATASETS[ds]["filepath"]} — anashkaluar.')
            continue
        for h in horizons:
            mode_preds, mode_actual = {}, None
            for mode in modes:
                rows, dm_rows, preds, yte_real, tgt_te = run_mode(
                    ds, h, mode, seeds, epochs, test_hours, arch_subset,
                    gap_filter=not args.no_gap_filter)
                all_rows += rows; all_dm += dm_rows
                mode_preds[mode] = preds; mode_actual = (yte_real, tgt_te)

            # ---- KOSTOJA E GABIMIT: legacy vs corrected (test set identik) ----
            if len(modes) == 2:
                yte_real, _ = mode_actual
                rated = DATASETS[ds]['rated_power_kw']
                print(f"\n  KOSTOJA E GABIMIT ({ds}, h={h}):")
                for name in mode_preds['legacy']:
                    if name not in mode_preds['corrected']:
                        continue
                    p_leg = mode_preds['legacy'][name]
                    p_cor = mode_preds['corrected'][name]
                    n_leg = nrmse(yte_real, p_leg, rated)
                    n_cor = nrmse(yte_real, p_cor, rated)
                    dm, pv = dm_test(yte_real, p_cor, p_leg, h)
                    cost_rows.append({'dataset': ds, 'horizon_h': h, 'model': name,
                                      'nRMSE_legacy': round(n_leg, 4),
                                      'nRMSE_corrected': round(n_cor, 4),
                                      'delta_pp': round(n_leg - n_cor, 4),
                                      'DM_corr_vs_leg': dm, 'p_value': pv})
                    print(f"    {name:9s} legacy={n_leg:5.2f}  corrected={n_cor:5.2f}  "
                          f"delta={n_leg-n_cor:+5.2f} pp  DM={dm:+.2f} p={pv:.4f}")

    pd.DataFrame(all_rows).to_csv('results/01_summary.csv', index=False)
    pd.DataFrame(all_dm).to_csv('results/01_dm_vs_bench.csv', index=False)
    if cost_rows:
        pd.DataFrame(cost_rows).to_csv('results/01_bug_cost.csv', index=False)
    print('\nDONE. Outputet: results/01_summary.csv, results/01_dm_vs_bench.csv'
          + (', results/01_bug_cost.csv' if cost_rows else ''))


if __name__ == '__main__':
    main()
