"""E03f: partition-seed sensitivity.

Scope: 5f / 4f / 5d (the core conclusion chain). Main partition (seed 42)
plus regenerated seeds 43..46 (same two-step GroupShuffleSplit); for each
partition, RF homo test MAE and gini_charge.
Sanity: the regenerated seed-42 4f partition is compared against the
reference partition (logic equivalence check).
Outputs results/E03/partition_sensitivity.csv
"""
import sys
import json
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from common import RESULTS, PART_DIR, scaffold_key, EV_PER_HA
from five_series import SERIES_CFG, load_series, featurize_series

OUTDIR = RESULTS / 'E03'
SERIES_TARGETS = ['5f', '4f', '5d']
SEEDS = [42, 43, 44, 45, 46]


def rf(seed=42):
    return RandomForestRegressor(n_estimators=200, max_depth=10,
                                 random_state=seed, n_jobs=-1)


def regen_partition(structs, metal_range, seed):
    lo, hi = metal_range
    uniq = {}
    for d in structs:
        k = scaffold_key(d, (lo, hi))
        if k not in uniq:
            uniq[k] = len(uniq)
    g = np.array([uniq[scaffold_key(d, (lo, hi))] for d in structs])
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, rest = next(gss1.split(np.arange(len(structs)), groups=g))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    va_r, te_r = next(gss2.split(rest, groups=g[rest]))
    return tr, rest[va_r], rest[te_r]


def main():
    t0 = time.time()
    rows = []
    for name in SERIES_TARGETS:
        structs, part0, ys, cfg = load_series(name)
        X, n_elem = featurize_series(structs)
        y = ys['homo']
        # sanity: regenerated seed42 vs reference partition (reference available for 4f only)
        tr42, va42, te42 = regen_partition(structs, cfg['metal'], 42)
        if name == '4f':
            same = (len(tr42) == part0['n_train'] and
                    set(tr42.tolist()) == set(part0['train_idx']))
            print(f'  sanity: regenerated seed42 vs reference partition -- '
                  f'{"match" if same else "mismatch (recorded; seeds 43-46 remain self-consistent)"} '
                  f'({len(tr42)} vs {part0["n_train"]})', flush=True)
        for seed in SEEDS:
            tr, va, te = regen_partition(structs, cfg['metal'], seed)
            m = rf(); m.fit(X[tr], y[tr])
            mae = float(np.abs(m.predict(X[te]) - y[te]).mean())
            gini = float(m.feature_importances_[n_elem + 1])
            rows.append(dict(series=name, seed=seed, n_train=len(tr), n_test=len(te),
                             rf_test_mae=mae, rf_test_mae_eV=mae * EV_PER_HA,
                             gini_charge=gini))
            print(f'{name} seed {seed}: n {len(tr)}/{len(te)} | MAE {mae:.6f} Ha '
                  f'| gini {gini:.4f} | {time.time()-t0:.0f}s', flush=True)
        ds = [r for r in rows if r['series'] == name]
        print(f'  {name} across partitions: MAE {np.mean([r["rf_test_mae"] for r in ds]):.6f}'
              f'±{np.std([r["rf_test_mae"] for r in ds]):.6f} Ha | gini '
              f'{np.mean([r["gini_charge"] for r in ds]):.4f}'
              f'±{np.std([r["gini_charge"] for r in ds]):.4f}', flush=True)
    with open(OUTDIR / 'partition_sensitivity.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'DONE E03f | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
