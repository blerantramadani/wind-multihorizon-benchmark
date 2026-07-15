#!/usr/bin/env python3
"""
================================================================================
STRICT-MODE SPOT-CHECK v2 — corrected construction + scalers fit on TRAIN ONLY
================================================================================
Runs the CORRECTED-mode pipeline (same windows, same gap filter, same 1000-hour
test set, same architectures) with the only change being that the MinMax scalers
are fitted exclusively on the training period. If the benchmark hierarchy and
the ML-vs-strongest-reference conclusions are unchanged, Section 3 may state:
"Results are robust to fitting the scalers on the training period only."

PLACE THIS FILE NEXT TO pipeline_corrected.py (or 01_pipeline_corrected.py).
USAGE:   python3 strict_mode_check_v2.py          (~1-2 h, CPU fine)
OUTPUT:  results/strict_mode_check_v2.csv + console verdict.
================================================================================
"""
import os, math, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# ---- import the corrected pipeline module (works for either filename) ----
for fname in ('pipeline_corrected.py', '01_pipeline_corrected.py',
              'scripts/01_pipeline_corrected.py'):
    if os.path.exists(fname):
        spec = importlib.util.spec_from_file_location('pc', fname)
        pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
        print(f'>> imported corrected pipeline from {fname}')
        break
else:
    raise SystemExit('ERROR: pipeline_corrected.py / 01_pipeline_corrected.py not found here.')

HORIZONS = [1, 24]
SEEDS    = [0, 1, 2]
NETS     = {'MLP': pc.build_mlp, 'LSTM': pc.build_lstm, 'GRU': pc.build_gru}
os.makedirs('results', exist_ok=True)

def stationary_bootstrap_p(e1, e2, n_boot=2000, mean_block=50, seed=0):
    """Politis-Romano stationary bootstrap on d = e1^2 - e2^2 (two-sided)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(e1)**2 - np.asarray(e2)**2
    T = len(d); obs = d.mean(); p_geo = 1.0/mean_block; cnt = 0
    for _ in range(n_boot):
        idx = np.empty(T, dtype=int); i = rng.integers(0, T)
        for t in range(T):
            if t > 0 and rng.random() > p_geo: i = (i+1) % T
            else: i = rng.integers(0, T)
            idx[t] = i
        if abs((d[idx]-obs).mean()) >= abs(obs): cnt += 1
    return (cnt+1)/(n_boot+1)

def run_strict(dataset, horizon, seeds):
    cfg = pc.DATASETS[dataset]; rated = cfg['rated_power_kw']
    df = pc.load_and_prepare(cfg['filepath'])
    features = pc.BASE_FEATURES + [pc.POWER_FEATURE['corrected']]
    X_data = df[features].values
    y_data = df[pc.TARGET_COL].values.reshape(-1,1)
    hours = df['HourOfDay'].values
    full_real = y_data.flatten()

    ids = pc.contiguous_sequence_ids(df.index, pc.LOOK_BACK, horizon, enabled=True)
    # geometry first (value-independent), to find the training boundary row
    Xg, yg, tgt = pc.build_sequences(X_data, full_real, ids, pc.LOOK_BACK, horizon)
    split = len(Xg) - pc.TEST_HOURS
    train_last_row = tgt[split-1]

    # STRICT: fit scalers only on rows available during training
    fs, ts_ = MinMaxScaler(), MinMaxScaler()
    fs.fit(X_data[:train_last_row+1]); ts_.fit(y_data[:train_last_row+1])
    Xs = fs.transform(X_data); ys = ts_.transform(y_data).flatten()

    X_seq, y_seq, tgt2 = pc.build_sequences(Xs, ys, ids, pc.LOOK_BACK, horizon)
    assert np.array_equal(tgt, tgt2)
    Xtr, Xte = X_seq[:split], X_seq[split:]
    ytr = y_seq[:split]; tgt_te = tgt[split:]
    Xtr_f, Xte_f = Xtr[:,-1,:], Xte[:,-1,:]
    yte_real = full_real[tgt_te]

    res, preds = {}, {}
    preds['Persistence'] = full_real[tgt_te - horizon]
    res['Persistence'] = pc.nrmse(yte_real, preds['Persistence'], rated)
    tr_tgt = tgt[:split]
    clim = pd.Series(full_real[tr_tgt]).groupby(hours[tr_tgt]).mean()
    preds['Climatology'] = clim.reindex(hours[tgt_te]).values
    res['Climatology'] = pc.nrmse(yte_real, preds['Climatology'], rated)
    lr = LinearRegression().fit(Xtr_f, ytr)
    preds['LinearRegression'] = ts_.inverse_transform(lr.predict(Xte_f).reshape(-1,1)).flatten()
    res['LinearRegression'] = pc.nrmse(yte_real, preds['LinearRegression'], rated)

    for name, build in NETS.items():
        ss, ps = [], []
        for sd in seeds:
            np.random.seed(sd); tf.random.set_seed(sd)
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            if name == 'MLP':
                m = build(Xtr_f.shape[1], sd)
                m.fit(Xtr_f, ytr, epochs=pc.EPOCHS, batch_size=pc.BATCH_SIZE,
                      validation_split=0.1, callbacks=[es], verbose=0)
                p = ts_.inverse_transform(m.predict(Xte_f, verbose=0)).flatten()
            else:
                m = build((Xtr.shape[1], Xtr.shape[2]), sd)
                m.fit(Xtr, ytr, epochs=pc.EPOCHS, batch_size=pc.BATCH_SIZE,
                      validation_split=0.1, callbacks=[es], verbose=0)
                p = ts_.inverse_transform(m.predict(Xte, verbose=0)).flatten()
            ss.append(pc.nrmse(yte_real, p, rated)); ps.append(p)
            tf.keras.backend.clear_session()
        res[name] = float(np.mean(ss)); preds[name] = np.mean(ps, axis=0)
    return res, preds, yte_real

def main():
    ref = None
    for p in ('results/mh_summary_corrected.csv','mh_summary_corrected.csv'):
        if os.path.exists(p): ref = pd.read_csv(p); break
    rows = []
    for site in ['kelmarsh','penmanshiel']:
        for h in HORIZONS:
            print(f'\n=== {site} h={h} (corrected + strict scaling) ===')
            res, preds, yte = run_strict(site, h, SEEDS)
            bench = {k: res[k] for k in ['Persistence','Climatology','LinearRegression']}
            strongest = min(bench, key=bench.get); bestnet = min(NETS, key=lambda k: res[k])
            for k,v in res.items(): print(f'  {k:18s} {v:6.2f}')
            print(f'  strongest bench: {strongest} | best net: {bestnet}')
            row = {'dataset':site,'horizon_h':h,
                   **{k:round(v,4) for k,v in res.items()},
                   'strongest_bench':strongest,'best_net':bestnet}
            if ref is not None:
                r = ref[(ref.dataset==site)&(ref.horizon_h==h)]
                if len(r):
                    r = r.iloc[0]
                    for k,col in [('Persistence','Persistence_naive'),('Climatology','Climatology'),
                                  ('LinearRegression','LinearRegression'),('MLP','MLP_mean'),
                                  ('LSTM','LSTM_mean'),('GRU','GRU_mean')]:
                        if col in r.index: row[f'delta_{k}'] = round(res[k]-float(r[col]),3)
            if site=='penmanshiel' and h==1:
                dm, p_hln = pc.dm_test(yte, preds['GRU'], preds['LinearRegression'], h)
                p_boot = stationary_bootstrap_p(yte-preds['GRU'], yte-preds['LinearRegression'])
                print(f'  BORDERLINE GRU vs LR: DM={dm:+.3f} p_HLN={p_hln:.4f} p_boot={p_boot:.4f}')
                row.update(GRUvsLR_DM=round(dm,3), GRUvsLR_pHLN=round(p_hln,4),
                           GRUvsLR_pboot=round(p_boot,4))
            rows.append(row)
    out = pd.DataFrame(rows); out.to_csv('results/strict_mode_check_v2.csv', index=False)
    print('\n===================== VERDICT =====================')
    print([(r['dataset'],r['horizon_h'],r['strongest_bench']) for r in rows])
    print('Expected under the corrected construction: LinearRegression strongest at h=1')
    print('on both sites; LR (kelmarsh) / Climatology (penmanshiel) at h=24.')
    print('delta_LinearRegression should be ~0.000 (affine invariance).')
    print('Saved: results/strict_mode_check_v2.csv')

if __name__ == '__main__':
    main()
