#!/usr/bin/env python3
"""
================================================================================
02 — BOOTSTRAP DIEBOLD-MARIANO (post-hoc, mbi parashikimet e ruajtura)
================================================================================
Projekt: Journal indexed paper 1st (Paper 1 -> Q1) + verifikim Paper 2

Zbaton këshillën e recensentit/profesorit: verifikim shtesë i stabilitetit të
DM-testeve me moving-block bootstrap (default 1000 resamples), i cili nuk
mbështetet në supozimet asimptotike të HLN dhe respekton autokorrelacionin e
diferencialit të humbjeve përmes blloqeve.

METODA:
  d_t = e1_t^2 - e2_t^2 ;  t_obs = DM-stat HLN mbi d.
  Nën H0 qendërzohet d (d - dbar) dhe rindërtohen seri me blloqe rrethore
  me gjatësi L = max(2h, 10). p_boot = fraksioni |t*| >= |t_obs|.
  Raportohen krah për krah: p_HLN (asimptotik) dhe p_boot.

INPUT : results/preds_{dataset}_h{H}_{mode}_{filt|nofilt}.csv  (nga skripti 01)
OUTPUT: results/02_bootstrap_dm.csv

PËRDORIMI:
  python3 scripts/02_bootstrap_dm.py                # 1000 resamples, të gjitha preds
  python3 scripts/02_bootstrap_dm.py --n-boot 5000  # më i saktë, më i ngadaltë

Krahasimet për çdo skedar parashikimesh:
  - çdo model ML (kolonat *_avg) vs benchmark-u më i fortë jo-ML
  - LSTM vs MLP (pyetja e recurrence)
  - corrected vs legacy për çdo model (nëse ekzistojnë të dy skedarët)

Autor: B. Ramadani | Projekt Q1, korrik 2026
================================================================================
"""
import os, re, glob, math, argparse
import numpy as np
import pandas as pd
from scipy import stats


def hln_dm(d, h):
    """DM-stat me korrigjim HLN mbi diferencialin d. Kthen (stat, p_asimptotik)."""
    T = len(d); dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    gammas = [np.cov(d[k:], d[:-k], ddof=0)[0, 1] for k in range(1, h)] if h > 1 else []
    var_d = (gamma0 + 2 * sum(gammas)) / T
    if var_d <= 0:
        return 0.0, 1.0
    t = dbar / math.sqrt(var_d)
    t *= math.sqrt(max((T + 1 - 2*h + h*(h-1)/T) / T, 1e-9))
    return t, 2 * (1 - stats.t.cdf(abs(t), df=T - 1))


def block_bootstrap_p(d, h, n_boot=1000, seed=0):
    """Moving-block bootstrap nën H0 (d i qendërzuar). Kthen p_boot."""
    rng = np.random.default_rng(seed)
    T = len(d)
    L = max(2 * h, 10)
    d0 = d - d.mean()                      # H0: E[d] = 0
    t_obs, _ = hln_dm(d, h)
    n_blocks = math.ceil(T / L)
    # blloqe rrethore që të mos humbasë fundi i serisë
    d_ext = np.concatenate([d0, d0[:L]])
    count = 0
    for _ in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        db = np.concatenate([d_ext[s:s+L] for s in starts])[:T]
        t_b, _ = hln_dm(db, h)
        if abs(t_b) >= abs(t_obs):
            count += 1
    return t_obs, (count + 1) / (n_boot + 1)   # korrigjim +1 (Davison & Hinkley)


def compare(actual, p1, p2, h, n_boot, label1, label2, meta):
    d = (actual - p1)**2 - (actual - p2)**2
    t_hln, p_hln = hln_dm(d, h)
    _, p_boot = block_bootstrap_p(d, h, n_boot)
    return {**meta, 'comparison': f'{label1}_vs_{label2}',
            'DM_HLN': round(t_hln, 4), 'p_HLN': round(p_hln, 4),
            'p_bootstrap': round(p_boot, 4),
            'agree_at_5pct': (p_hln < 0.05) == (p_boot < 0.05)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--preds-dir', default='results')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.preds_dir, 'preds_*_h*_*.csv')))
    if not files:
        print('Asnjë skedar parashikimesh nuk u gjet në results/. '
              'Ekzekuto fillimisht skriptin 01.')
        return

    rows = []
    by_key = {}
    for f in files:
        m = re.match(r'preds_(\w+)_h(\d+)_(legacy|corrected)_(filt|nofilt)\.csv',
                     os.path.basename(f))
        if not m:
            continue
        ds, h, mode, filt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        df = pd.read_csv(f)
        by_key[(ds, h, mode, filt)] = df
        actual = df['actual'].to_numpy()
        bench_cols = [c for c in df.columns if c.startswith('bench_')]
        ml_cols = [c for c in df.columns if c.endswith('_avg')]
        meta = {'dataset': ds, 'horizon_h': h, 'mode': mode, 'gap_filter': filt}

        # benchmark-u më i fortë jo-ML
        bscores = {c: math.sqrt(np.mean((actual - df[c])**2)) for c in bench_cols}
        strongest = min(bscores, key=bscores.get)
        print(f'[{ds} h={h} {mode} {filt}] strongest bench: '
              f'{strongest.replace("bench_","")}')

        for c in ml_cols:
            rows.append(compare(actual, df[c].to_numpy(), df[strongest].to_numpy(),
                                h, args.n_boot, c.replace('_avg', ''),
                                strongest.replace('bench_', ''), meta))
        if 'LSTM_avg' in df.columns and 'MLP_avg' in df.columns:
            rows.append(compare(actual, df['LSTM_avg'].to_numpy(),
                                df['MLP_avg'].to_numpy(), h, args.n_boot,
                                'LSTM', 'MLP', meta))

    # corrected vs legacy për çdo model (test set identik nën të njëjtin filtër)
    for (ds, h, mode, filt), df in by_key.items():
        if mode != 'corrected':
            continue
        leg = by_key.get((ds, h, 'legacy', filt))
        if leg is None or len(leg) != len(df):
            continue
        actual = df['actual'].to_numpy()
        meta = {'dataset': ds, 'horizon_h': h, 'mode': 'corr_vs_leg', 'gap_filter': filt}
        for c in [c for c in df.columns if c.endswith('_avg')]:
            if c in leg.columns:
                rows.append(compare(actual, df[c].to_numpy(), leg[c].to_numpy(),
                                    h, args.n_boot, f'{c.replace("_avg","")}_corrected',
                                    f'{c.replace("_avg","")}_legacy', meta))

    out = pd.DataFrame(rows)
    out.to_csv('results/02_bootstrap_dm.csv', index=False)
    print(f'\n{out.to_string(index=False)}')
    n_dis = int((~out['agree_at_5pct']).sum())
    print(f'\nDONE. results/02_bootstrap_dm.csv | '
          f'Mospërputhje HLN vs bootstrap në 5%: {n_dis}/{len(out)}')
    if n_dis == 0:
        print('Të gjitha konkluzionet DM konfirmohen nga bootstrap — '
              'fjali e gatshme robustësie për letrën.')


if __name__ == '__main__':
    main()
