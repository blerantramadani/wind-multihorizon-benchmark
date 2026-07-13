#!/usr/bin/env python3
"""
================================================================================
MULTI-HORIZON WIND POWER FORECASTING — HIGH-RIGOR EVALUATION
================================================================================
Third thesis contribution. Extends the 1-hour cross-site evaluation
(IEEE Access) to a systematic horizon study with a full benchmark hierarchy,
two recurrent architectures, and statistical identification of the horizon at
which deep learning begins to provide statistically significant value over the
strongest non-ML benchmark ("crossover horizon").

DESIGN (IEEE-grade):
  Horizons (dense): h in {1, 2, 3, 6, 12, 18, 24, 36, 48} hours
  Benchmarks (hierarchy, not just naive):
      - Persistence (naive):        P_hat(t+h) = P(t)
      - Seasonal persistence:       P_hat(t+h) = P(t+h-24)   (same hour, prev day)
      - Climatology:                hour-of-day mean from TRAIN set only
      - Linear regression
  ML architectures:
      - MLP   (feedforward)
      - LSTM  (recurrent)
      - GRU   (recurrent, alternative -> tests generality of the finding)
  Rigor:
      - 10 random seeds (tight confidence intervals)
      - Seed-averaged predictions for Diebold-Mariano testing
      - DM test with Harvey-Leybourne-Newbold small-sample correction,
        autocovariance lag = horizon (correct for multi-step)
      - Bootstrap 95% CI on nRMSE for each model/horizon
  Direct multi-step strategy: target = y[t+h], no recursion (no error accumulation).

OUTPUT:
  results_mh/mh_summary.csv      full metrics, all models x horizons x datasets
  results_mh/mh_dm.csv           DM tests (ML vs strongest non-ML benchmark)
  results_mh/mh_crossover.csv    identified crossover horizon per dataset
  results_mh/mh_bootstrap.csv    bootstrap 95% CIs

Author: B. Ramadani  |  Consistent protocol with IEEE Access submission.
================================================================================
"""

import os, math, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, Layer,
                                     LayerNormalization, MultiHeadAttention,
                                     Conv1D, Flatten, Add, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy import stats

# ----------------------------- CONFIG -----------------------------
DATASETS = {
    'kelmarsh':    {'filepath': 'data/Kelmarsh_T1_2018_hourly.csv',     'rated_power_kw': 2050},
    'penmanshiel': {'filepath': 'data/Penmanshiel_T01_2018_hourly.csv', 'rated_power_kw': 2050},
}
HORIZONS   = [1, 2, 3, 6, 12, 18, 24, 36, 48]   # dense horizon grid
LOOK_BACK  = 48
EPOCHS     = 50
BATCH_SIZE = 32
LR         = 0.001
N_SEEDS    = 10
TEST_HOURS = 1000
N_BOOTSTRAP = 1000
TARGET_COL = 'ActivePower_kW'
FEATURES = ['WindSpeed_m_s', 'Temperature_C', 'ActivePower_lag1',
            'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'WindSpeed_Rolling3h']

os.makedirs('results_mh', exist_ok=True)


# ----------------------------- DATA -----------------------------
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
    df = df.dropna()
    return df


def create_sequences_h(X_scaled, y_scaled, look_back, horizon):
    """Direct multi-step: window [i-look_back:i] -> target y[i+horizon-1]."""
    X_seq, y_seq, tgt_idx = [], [], []
    last = len(X_scaled) - horizon + 1
    for i in range(look_back, last):
        X_seq.append(X_scaled[i-look_back:i])
        y_seq.append(y_scaled[i + horizon - 1])
        tgt_idx.append(i + horizon - 1)   # original-series index of the target
    return np.array(X_seq), np.array(y_seq), np.array(tgt_idx)


def metrics(actual, pred, rated):
    rmse = math.sqrt(mean_squared_error(actual, pred))
    return {'RMSE_kW': round(rmse, 2),
            'nRMSE_pct': round(rmse/rated*100, 4),
            'MAE_kW': round(mean_absolute_error(actual, pred), 2)}


def bootstrap_ci(actual, pred, rated, n=N_BOOTSTRAP, seed=0):
    """Bootstrap 95% CI on nRMSE."""
    rng = np.random.default_rng(seed)
    a, p = np.asarray(actual), np.asarray(pred)
    N = len(a)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        rmse = math.sqrt(mean_squared_error(a[idx], p[idx]))
        vals.append(rmse/rated*100)
    return round(np.percentile(vals, 2.5), 4), round(np.percentile(vals, 97.5), 4)


# ----------------------------- MODELS -----------------------------
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


# ---------------- DLinear (Zeng et al., AAAI 2023) ----------------
class SeriesDecomposition(Layer):
    """Moving-average trend/seasonal decomposition (DLinear core)."""
    def __init__(self, kernel_size=25, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
    def call(self, x):
        # x: (batch, time, channels). Moving average over time -> trend.
        x_pad = tf.concat([
            tf.repeat(x[:, :1, :], self.pad, axis=1),
            x,
            tf.repeat(x[:, -1:, :], self.pad, axis=1)
        ], axis=1)
        trend = tf.nn.avg_pool1d(x_pad, ksize=self.kernel_size, strides=1, padding='VALID')
        seasonal = x - trend
        return seasonal, trend

def build_dlinear(shape, seed):
    """DLinear: decompose into trend+seasonal, apply separate linear maps, sum.
    shape = (look_back, n_features). Forecast = single value (h-step direct)."""
    tf.random.set_seed(seed)
    look_back, n_feat = shape
    inp = Input(shape=shape)
    seasonal, trend = SeriesDecomposition(kernel_size=25)(inp)
    # flatten each component and map linearly to scalar output
    s_flat = Flatten()(seasonal)
    t_flat = Flatten()(trend)
    s_out = Dense(1, use_bias=True)(s_flat)
    t_out = Dense(1, use_bias=True)(t_flat)
    out = Add()([s_out, t_out])
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR), loss='mse')
    return m


# ---------------- PatchTST (Nie et al., ICLR 2023) ----------------
class PatchChannelEmbed(Layer):
    """Channel-independent patching + linear embedding.
    Input  : (batch, time, channels)
    Output : (batch*channels, n_patches, d_model)  and stores channel count.
    """
    def __init__(self, patch_len=16, stride=8, d_model=64, n_channels=8, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_channels = n_channels
        self.proj = Dense(d_model)
    def call(self, x):
        b = tf.shape(x)[0]; t = tf.shape(x)[1]
        c = self.n_channels
        xt = tf.transpose(x, [0, 2, 1])              # (b, c, t)
        xt = tf.reshape(xt, [b * c, t, 1])           # (b*c, t, 1)
        patches = tf.image.extract_patches(
            images=tf.expand_dims(xt, 1),
            sizes=[1, 1, self.patch_len, 1],
            strides=[1, 1, self.stride, 1],
            rates=[1, 1, 1, 1], padding='VALID')     # (b*c, 1, n_patch, patch_len)
        patches = tf.squeeze(patches, 1)             # (b*c, n_patch, patch_len)
        return self.proj(patches)                    # (b*c, n_patch, d_model)

class RecombineChannels(Layer):
    """Reshape (batch*channels, d_model) back to (batch, channels*d_model)."""
    def __init__(self, n_channels, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.d_model = d_model
    def call(self, x):
        b_c = tf.shape(x)[0]
        b = b_c // self.n_channels
        return tf.reshape(x, [b, self.n_channels * self.d_model])

def transformer_block(x, d_model, n_heads, ff_dim):
    attn = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(d_model)(ff)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    return x

def build_patchtst(shape, seed, patch_len=16, stride=8, d_model=64, n_heads=4,
                   ff_dim=128, n_layers=2):
    """PatchTST: patching + channel-independent Transformer encoder.
    Direct single-value forecast head. shape=(look_back, n_features)."""
    tf.random.set_seed(seed)
    look_back, n_feat = shape
    inp = Input(shape=shape)
    x = PatchChannelEmbed(patch_len, stride, d_model, n_channels=n_feat)(inp)
    for _ in range(n_layers):
        x = transformer_block(x, d_model, n_heads, ff_dim)
    x = GlobalAveragePooling1D()(x)                   # (b*c, d_model)
    x = RecombineChannels(n_feat, d_model)(x)         # (b, c*d_model)
    x = Dense(64, activation='relu')(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR), loss='mse')
    return m


# ------------------------- DIEBOLD-MARIANO -------------------------
def dm_test(actual, pred1, pred2, h):
    """HLN-corrected DM. Negative => pred1 better. Returns (stat, p)."""
    e1 = np.asarray(actual) - np.asarray(pred1)
    e2 = np.asarray(actual) - np.asarray(pred2)
    d = e1**2 - e2**2
    T = len(d); dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    gammas = [np.cov(d[k:], d[:-k], ddof=0)[0,1] for k in range(1, h)] if h > 1 else []
    var_d = (gamma0 + 2*sum(gammas)) / T
    if var_d <= 0:
        return 0.0, 1.0
    dm = dbar / math.sqrt(var_d)
    corr = math.sqrt(max((T + 1 - 2*h + h*(h-1)/T) / T, 1e-9))
    dm_hln = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=T-1))
    return round(dm_hln, 4), round(p, 4)


# ----------------------------- EXPERIMENT -----------------------------
def run(dataset, horizon, seeds):
    cfg = DATASETS[dataset]; rated = cfg['rated_power_kw']
    df = load_and_prepare(cfg['filepath'])

    X_data = df[FEATURES].values
    y_data = df[TARGET_COL].values.reshape(-1,1)
    hour_arr = df['HourOfDay'].values
    fscaler, tscaler = MinMaxScaler(), MinMaxScaler()
    X_scaled = fscaler.fit_transform(X_data)
    y_scaled = tscaler.fit_transform(y_data).flatten()
    full_real = y_data.flatten()

    X_seq, y_seq, tgt_idx = create_sequences_h(X_scaled, y_scaled, LOOK_BACK, horizon)
    split = len(X_seq) - TEST_HOURS
    Xtr_s, Xte_s = X_seq[:split], X_seq[split:]
    ytr = y_seq[:split]
    yte = y_seq[split:]
    tgt_te = tgt_idx[split:]
    Xtr_f, Xte_f = Xtr_s[:, -1, :], Xte_s[:, -1, :]
    yte_real = tscaler.inverse_transform(yte.reshape(-1,1)).flatten()

    res = {}

    # ---- Benchmark 1: Persistence naive  P(t+h)=P(t) ----
    now_idx = tgt_te - horizon
    res['Persistence_naive'] = metrics(yte_real, full_real[now_idx], rated)
    pred_pers_naive = full_real[now_idx]

    # ---- Benchmark 2: Seasonal persistence  P(t+h)=P(t+h-24) ----
    seas_idx = tgt_te - 24
    valid = seas_idx >= 0
    if valid.all():
        pred_seas = full_real[seas_idx]
        res['Persistence_seasonal'] = metrics(yte_real, pred_seas, rated)
    else:
        pred_seas = full_real[np.clip(seas_idx, 0, None)]
        res['Persistence_seasonal'] = metrics(yte_real[valid], pred_seas[valid], rated)

    # ---- Benchmark 3: Climatology (hour-of-day mean from TRAIN target indices) ----
    train_tgt_idx = tgt_idx[:split]
    train_hours = hour_arr[train_tgt_idx]
    train_vals = full_real[train_tgt_idx]
    clim = {hh: train_vals[train_hours == hh].mean() for hh in range(24)}
    test_hours = hour_arr[tgt_te]
    pred_clim = np.array([clim[h_] for h_ in test_hours])
    res['Climatology'] = metrics(yte_real, pred_clim, rated)

    # ---- Benchmark 4: Linear regression ----
    lr = LinearRegression().fit(Xtr_f, ytr)
    pred_lr = tscaler.inverse_transform(lr.predict(Xte_f).reshape(-1,1)).flatten()
    res['LinearRegression'] = metrics(yte_real, pred_lr, rated)

    # ---- ML architectures (multi-seed) ----
    arch = {'MLP': [], 'LSTM': [], 'GRU': [], 'DLinear': [], 'PatchTST': []}
    preds_seeds = {'MLP': [], 'LSTM': [], 'GRU': [], 'DLinear': [], 'PatchTST': []}
    for sd in seeds:
        np.random.seed(sd); tf.random.set_seed(sd)
        es = lambda: EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        mlp = build_mlp(Xtr_f.shape[1], sd)
        mlp.fit(Xtr_f, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_split=0.1, callbacks=[es()], verbose=0)
        p = tscaler.inverse_transform(mlp.predict(Xte_f, verbose=0)).flatten()
        arch['MLP'].append(metrics(yte_real, p, rated)['nRMSE_pct']); preds_seeds['MLP'].append(p)

        ls = build_lstm((Xtr_s.shape[1], Xtr_s.shape[2]), sd)
        ls.fit(Xtr_s, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
               validation_split=0.1, callbacks=[es()], verbose=0)
        p = tscaler.inverse_transform(ls.predict(Xte_s, verbose=0)).flatten()
        arch['LSTM'].append(metrics(yte_real, p, rated)['nRMSE_pct']); preds_seeds['LSTM'].append(p)

        gr = build_gru((Xtr_s.shape[1], Xtr_s.shape[2]), sd)
        gr.fit(Xtr_s, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
               validation_split=0.1, callbacks=[es()], verbose=0)
        p = tscaler.inverse_transform(gr.predict(Xte_s, verbose=0)).flatten()
        arch['GRU'].append(metrics(yte_real, p, rated)['nRMSE_pct']); preds_seeds['GRU'].append(p)

        # DLinear (decomposition + linear) - uses the sequence input
        dl = build_dlinear((Xtr_s.shape[1], Xtr_s.shape[2]), sd)
        dl.fit(Xtr_s, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
               validation_split=0.1, callbacks=[es()], verbose=0)
        p = tscaler.inverse_transform(dl.predict(Xte_s, verbose=0)).flatten()
        arch['DLinear'].append(metrics(yte_real, p, rated)['nRMSE_pct']); preds_seeds['DLinear'].append(p)

        # PatchTST (patching + channel-independent Transformer)
        pt = build_patchtst((Xtr_s.shape[1], Xtr_s.shape[2]), sd)
        pt.fit(Xtr_s, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
               validation_split=0.1, callbacks=[es()], verbose=0)
        p = tscaler.inverse_transform(pt.predict(Xte_s, verbose=0)).flatten()
        arch['PatchTST'].append(metrics(yte_real, p, rated)['nRMSE_pct']); preds_seeds['PatchTST'].append(p)

    for name in ['MLP', 'LSTM', 'GRU', 'DLinear', 'PatchTST']:
        a = np.array(arch[name])
        res[name] = {'nRMSE_mean': round(a.mean(), 4), 'nRMSE_std': round(a.std(), 4)}

    # ---- strongest non-ML benchmark at this horizon ----
    nonml = {k: res[k]['nRMSE_pct'] for k in
             ['Persistence_naive', 'Persistence_seasonal', 'Climatology', 'LinearRegression']}
    best_nonml_name = min(nonml, key=nonml.get)
    best_nonml_pred = {'Persistence_naive': pred_pers_naive,
                       'Persistence_seasonal': pred_seas if valid.all() else pred_pers_naive,
                       'Climatology': pred_clim,
                       'LinearRegression': pred_lr}[best_nonml_name]

    # ---- DM: each ML (seed-avg) vs strongest non-ML benchmark ----
    ALL_ML = ['MLP', 'LSTM', 'GRU', 'DLinear', 'PatchTST']
    dm_out = {}
    best_ml_name = min(ALL_ML, key=lambda n: res[n]['nRMSE_mean'])
    for ml in ALL_ML:
        ml_avg = np.mean(preds_seeds[ml], axis=0)
        dm_out[f'{ml}_vs_{best_nonml_name}'] = dm_test(yte_real, ml_avg, best_nonml_pred, horizon)
    # recurrent vs feedforward, and modern vs recurrent
    dm_out['LSTM_vs_MLP'] = dm_test(yte_real, np.mean(preds_seeds['LSTM'],axis=0),
                                    np.mean(preds_seeds['MLP'],axis=0), horizon)
    dm_out['GRU_vs_MLP']  = dm_test(yte_real, np.mean(preds_seeds['GRU'],axis=0),
                                    np.mean(preds_seeds['MLP'],axis=0), horizon)
    dm_out['PatchTST_vs_LSTM'] = dm_test(yte_real, np.mean(preds_seeds['PatchTST'],axis=0),
                                    np.mean(preds_seeds['LSTM'],axis=0), horizon)
    dm_out['DLinear_vs_LSTM'] = dm_test(yte_real, np.mean(preds_seeds['DLinear'],axis=0),
                                    np.mean(preds_seeds['LSTM'],axis=0), horizon)

    # ---- bootstrap CI for best ML and best non-ML ----
    boot = {
        'best_ml': best_ml_name,
        'best_ml_ci': bootstrap_ci(yte_real, np.mean(preds_seeds[best_ml_name],axis=0), rated),
        'best_nonml': best_nonml_name,
        'best_nonml_ci': bootstrap_ci(yte_real, best_nonml_pred, rated),
    }

    meta = {'best_nonml_name': best_nonml_name, 'best_ml_name': best_ml_name}
    return res, dm_out, boot, meta


# ----------------------------- DRIVER -----------------------------
if __name__ == '__main__':
    seeds = list(range(N_SEEDS))
    t0 = time.time()
    summary_rows, dm_rows, boot_rows, crossover_rows = [], [], [], []

    for ds in DATASETS:
        print(f"\n{'#'*70}\n# DATASET: {ds.upper()}\n{'#'*70}")
        crossover_found = None
        for h in HORIZONS:
            th = time.time()
            print(f"\n--- {ds} | horizon {h}h ---")
            res, dm, boot, meta = run(ds, h, seeds)

            print(f"  Persistence naive   : {res['Persistence_naive']['nRMSE_pct']:.2f}%")
            print(f"  Persistence seasonal: {res['Persistence_seasonal']['nRMSE_pct']:.2f}%")
            print(f"  Climatology         : {res['Climatology']['nRMSE_pct']:.2f}%")
            print(f"  Linear Regression   : {res['LinearRegression']['nRMSE_pct']:.2f}%")
            print(f"  MLP     : {res['MLP']['nRMSE_mean']:.2f} ± {res['MLP']['nRMSE_std']:.2f}%")
            print(f"  LSTM    : {res['LSTM']['nRMSE_mean']:.2f} ± {res['LSTM']['nRMSE_std']:.2f}%")
            print(f"  GRU     : {res['GRU']['nRMSE_mean']:.2f} ± {res['GRU']['nRMSE_std']:.2f}%")
            print(f"  DLinear : {res['DLinear']['nRMSE_mean']:.2f} ± {res['DLinear']['nRMSE_std']:.2f}%")
            print(f"  PatchTST: {res['PatchTST']['nRMSE_mean']:.2f} ± {res['PatchTST']['nRMSE_std']:.2f}%")
            print(f"  Strongest non-ML: {meta['best_nonml_name']} | Best ML: {meta['best_ml_name']}")

            # crossover: best ML statistically better than strongest non-ML (p<0.05, ML superior)
            best_ml = meta['best_ml_name']
            dm_key = f"{best_ml}_vs_{meta['best_nonml_name']}"
            dmv, pv = dm[dm_key]
            ml_better_sig = (dmv < 0) and (pv < 0.05)
            print(f"  DM {dm_key}: stat={dmv}, p={pv} -> ML sig. better: {ml_better_sig}")
            if ml_better_sig and crossover_found is None:
                crossover_found = h
                print(f"  *** CROSSOVER at h={h}h: deep learning becomes statistically superior ***")

            row = {'dataset': ds, 'horizon_h': h,
                   'Persistence_naive': res['Persistence_naive']['nRMSE_pct'],
                   'Persistence_seasonal': res['Persistence_seasonal']['nRMSE_pct'],
                   'Climatology': res['Climatology']['nRMSE_pct'],
                   'LinearRegression': res['LinearRegression']['nRMSE_pct'],
                   'MLP_mean': res['MLP']['nRMSE_mean'], 'MLP_std': res['MLP']['nRMSE_std'],
                   'LSTM_mean': res['LSTM']['nRMSE_mean'], 'LSTM_std': res['LSTM']['nRMSE_std'],
                   'GRU_mean': res['GRU']['nRMSE_mean'], 'GRU_std': res['GRU']['nRMSE_std'],
                   'DLinear_mean': res['DLinear']['nRMSE_mean'], 'DLinear_std': res['DLinear']['nRMSE_std'],
                   'PatchTST_mean': res['PatchTST']['nRMSE_mean'], 'PatchTST_std': res['PatchTST']['nRMSE_std'],
                   'best_nonML': meta['best_nonml_name'], 'best_ML': meta['best_ml_name']}
            summary_rows.append(row)
            for comp, (dmv_, pv_) in dm.items():
                dm_rows.append({'dataset': ds, 'horizon_h': h, 'comparison': comp,
                                'DM_HLN': dmv_, 'p_value': pv_})
            boot_rows.append({'dataset': ds, 'horizon_h': h,
                              'best_ml': boot['best_ml'],
                              'best_ml_ci_low': boot['best_ml_ci'][0],
                              'best_ml_ci_high': boot['best_ml_ci'][1],
                              'best_nonml': boot['best_nonml'],
                              'best_nonml_ci_low': boot['best_nonml_ci'][0],
                              'best_nonml_ci_high': boot['best_nonml_ci'][1]})
            print(f"  [elapsed {time.time()-th:.0f}s]")

        crossover_rows.append({'dataset': ds,
                               'crossover_horizon_h': crossover_found if crossover_found else 'none_within_grid'})

    pd.DataFrame(summary_rows).to_csv('results_mh/mh_summary.csv', index=False)
    pd.DataFrame(dm_rows).to_csv('results_mh/mh_dm.csv', index=False)
    pd.DataFrame(boot_rows).to_csv('results_mh/mh_bootstrap.csv', index=False)
    pd.DataFrame(crossover_rows).to_csv('results_mh/mh_crossover.csv', index=False)

    print(f"\n{'='*70}")
    print("DONE. Total time: %.1f min" % ((time.time()-t0)/60))
    print("Saved 4 files to results_mh/:")
    print("  mh_summary.csv    — all metrics")
    print("  mh_dm.csv         — Diebold-Mariano tests")
    print("  mh_bootstrap.csv  — bootstrap 95% CIs")
    print("  mh_crossover.csv  — identified crossover horizon")
    print('='*70)
    for r in crossover_rows:
        print(f"  {r['dataset']}: crossover horizon = {r['crossover_horizon_h']}")
