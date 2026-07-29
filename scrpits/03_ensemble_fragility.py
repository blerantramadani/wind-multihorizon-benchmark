#!/usr/bin/env python3
"""
================================================================================
03 — BRISHTËSIA E ANSAMBLIT: sa i ndjeshëm është DM ndaj përbërjes së farave
================================================================================
Pse: DM-ja llogaritet mbi parashikimin e mesatarizuar (mp = mean(preds, axis=0)),
ndërsa nRMSE_mean është mesatarja e rezultateve për farë. Ansambli lëviz më shumë
se mesatarja e anëtarëve, ndaj DM mund të ndryshojë ndjeshëm edhe kur nRMSE mezi
lëviz. Ky script e mat atë drejtpërdrejt, PA ritrajnuar asgjë.

TRI DIAGNOSTIKA:
  1) Jackknife (leave-one-seed-out): 10 ansamble me nga 9 fara.
     KUJDES: ansambli 9-farësh është sistematikisht më i zhurmshëm se ai
     10-farësh, ndaj një pjesë e çdo përkeqësimi vjen nga madhësia, jo nga
     brishtësia. Prandaj ekziston diagnostika 3.
  2) DM për farë individuale: shpërndarja e plotë, pa konfuzion madhësie.
  3) Nën-mostrim i rastësishëm me madhësi k: izolon efektin e madhësisë së
     ansamblit nga ai i përbërjes.

VETË-VERIFIKIM: DM-ja e ansamblit të plotë krahasohet me vlerën e ruajtur në
results/01_dm_vs_bench.csv. Nëse nuk përputhen, skedari i parashikimeve nuk i
përket të njëjtit run — scripti ndalon.

PËRDORIMI:
  python3 scripts/03_ensemble_fragility.py                      # të 18 kombinimet
  python3 scripts/03_ensemble_fragility.py --only penmanshiel:1 # një i vetëm
  python3 scripts/03_ensemble_fragility.py --subsample 5 --draws 200

OUTPUT:
  results/03_ensemble_fragility.csv
================================================================================
"""
import argparse
import importlib.util
import itertools
import os
import sys

import numpy as np
import pandas as pd

PIPELINE = 'scripts/01_pipeline_corrected.py'
DM_FILE = 'results/01_dm_vs_bench.csv'
PRED_TPL = 'results/preds_{ds}_h{h}_{mode}_filt.csv'
OUT = 'results/03_ensemble_fragility.csv'

HORIZONS = [1, 2, 3, 6, 12, 18, 24, 36, 48]
DATASETS = ['kelmarsh', 'penmanshiel']


def load_dm_test(path):
    """Importon dm_test nga pipeline-i — garanton funksion bit-identik."""
    if not os.path.exists(path):
        sys.exit(f'Nuk gjendet pipeline-i: {path}')
    spec = importlib.util.spec_from_file_location('pc', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # main() nuk ekzekutohet (__name__ != '__main__')
    return mod.dm_test


def analyse(d, model, bench_col, h, dm_test, dm_expected,
            subsample_k, n_draws, rng):
    y = d['actual'].values
    ref = d[bench_col].values
    cols = sorted(c for c in d.columns if c.startswith(f'{model}_seed'))
    if len(cols) < 3:
        return None
    P = d[cols].values                      # (T, n_seeds)
    n = P.shape[1]

    # --- ansambli i plotë + vetë-verifikim ---
    dm_full, p_full = dm_test(y, P.mean(axis=1), ref, h)
    ok = dm_expected is None or abs(dm_full - dm_expected) < 5e-3

    # --- 1) jackknife ---
    jk = [dm_test(y, np.delete(P, i, axis=1).mean(axis=1), ref, h) for i in range(n)]
    jk_p = np.array([p for _, p in jk])

    # --- 2) fara individuale ---
    seed_p = np.array([dm_test(y, P[:, i], ref, h)[1] for i in range(n)])

    # --- 3) nën-mostrim i rastësishëm me madhësi k ---
    k = min(subsample_k, n - 1)
    combos = list(itertools.combinations(range(n), k))
    if len(combos) > n_draws:
        idx = rng.choice(len(combos), size=n_draws, replace=False)
        combos = [combos[i] for i in idx]
    sub_p = np.array([dm_test(y, P[:, list(c)].mean(axis=1), ref, h)[1]
                      for c in combos])

    return {
        'n_seeds': n, 'dm_full': dm_full, 'p_full': p_full,
        'selfcheck_ok': ok, 'dm_in_csv': dm_expected,
        'jk_p_min': jk_p.min(), 'jk_p_max': jk_p.max(), 'jk_p_mean': jk_p.mean(),
        'jk_below_05': int((jk_p < 0.05).sum()),
        'seed_p_min': seed_p.min(), 'seed_p_max': seed_p.max(),
        'seed_p_median': float(np.median(seed_p)),
        'seed_below_05': int((seed_p < 0.05).sum()),
        f'sub{k}_p_min': sub_p.min(), f'sub{k}_p_max': sub_p.max(),
        f'sub{k}_below_05_frac': float((sub_p < 0.05).mean()),
        'subsample_k': k, 'n_subsets': len(combos),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='corrected')
    ap.add_argument('--only', default=None,
                    help='p.sh. "penmanshiel:1" — vetëm një kombinim')
    ap.add_argument('--subsample', type=int, default=5)
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    dm_test = load_dm_test(PIPELINE)
    rng = np.random.default_rng(args.seed)

    if not os.path.exists(DM_FILE):
        sys.exit(f'Nuk gjendet {DM_FILE}')
    dm_ref = pd.read_csv(DM_FILE)
    dm_ref = dm_ref[dm_ref['mode'] == args.mode]

    if args.only:
        ds_only, h_only = args.only.split(':')
        combos = [(ds_only, int(h_only))]
    else:
        combos = [(ds, h) for ds in DATASETS for h in HORIZONS]

    rows, failures = [], []
    for ds, h in combos:
        path = PRED_TPL.format(ds=ds, h=h, mode=args.mode)
        if not os.path.exists(path):
            print(f'  [mungon] {path}')
            continue
        d = pd.read_csv(path)

        sub = dm_ref[(dm_ref['dataset'] == ds) & (dm_ref['horizon_h'] == h)]
        if sub.empty:
            print(f'  [pa rresht ne CSV] {ds} h={h}')
            continue

        # modeli ML me DM-në më negative = kandidati më i fortë kundër referencës
        best = sub.loc[sub['DM'].idxmin()]
        model = best['model']
        bench_col = f"bench_{best['strongest_bench']}"
        if bench_col not in d.columns:
            print(f'  [pa kolone] {bench_col} ne {path}')
            continue

        r = analyse(d, model, bench_col, h, dm_test, float(best['DM']),
                    args.subsample, args.draws, rng)
        if r is None:
            continue
        r.update({'dataset': ds, 'horizon_h': h, 'model': model,
                  'strongest_bench': best['strongest_bench']})
        rows.append(r)

        flag = '' if r['selfcheck_ok'] else '  <<< VETË-VERIFIKIMI DËSHTOI'
        if not r['selfcheck_ok']:
            failures.append((ds, h))
        print(f"{ds:12s} h={h:<3d} {model:9s} vs {best['strongest_bench']:<17s} "
              f"p_full={r['p_full']:.4f} | jk [{r['jk_p_min']:.4f}, {r['jk_p_max']:.4f}] "
              f"{r['jk_below_05']}/{r['n_seeds']} | fara [{r['seed_p_min']:.4f}, "
              f"{r['seed_p_max']:.4f}] {r['seed_below_05']}/{r['n_seeds']}{flag}")

    if not rows:
        sys.exit('Asnjë rezultat.')

    df = pd.DataFrame(rows)
    front = ['dataset', 'horizon_h', 'model', 'strongest_bench', 'n_seeds',
             'dm_full', 'p_full', 'dm_in_csv', 'selfcheck_ok']
    df = df[front + [c for c in df.columns if c not in front]]
    os.makedirs('results', exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'\nU ruajt: {OUT}')

    if failures:
        print('\n!!! VETË-VERIFIKIMI DËSHTOI për: '
              + ', '.join(f'{d} h={h}' for d, h in failures))
        print('    Skedarët e parashikimeve nuk i përkasin të njëjtit run '
              'si 01_dm_vs_bench.csv. Mos i përdor këto rezultate.')
        sys.exit(1)

    # përmbledhje për rastet kufitare
    border = df[(df['p_full'] < 0.10)]
    if not border.empty:
        print('\nRastet kufitare (p_full < 0.10):')
        for _, r in border.iterrows():
            print(f"  {r['dataset']} h={int(r['horizon_h'])} {r['model']}: "
                  f"p={r['p_full']:.4f}, jackknife {r['jk_below_05']}/{int(r['n_seeds'])} "
                  f"nën 0.05, fara individuale {r['seed_below_05']}/{int(r['n_seeds'])} "
                  f"nën 0.05")


if __name__ == '__main__':
    main()
