"""E02e: error diagnostics and paired tests.

A) Error stratification: |err| for five series x homo stratified by
   {charge, natoms bins, unpaired electrons, per element}
B) LoMO-vs-CV paired bootstrap: confidence interval of the difference
   between leave-one-metal folds and CV folds (covering 4f and 5f)
Outputs results/E02/diagnostics_series.csv, paired_loeo_cv.csv
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from common import RESULTS, EV_PER_HA, scaffold_key
from five_series import SERIES_CFG, load_series, featurize_series

OUTDIR = RESULTS / 'E02'


def rf(seed=42):
    return RandomForestRegressor(n_estimators=200, max_depth=10,
                                 random_state=seed, n_jobs=-1)


def boot_ci(diffs, n_boot=1000, seed=0):
    rs = np.random.RandomState(seed)
    d = np.asarray(diffs)
    means = [d[rs.randint(len(d), size=len(d))].mean() for _ in range(n_boot)]
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    t0 = time.time()
    diag_rows, pair_rows = [], []
    for name, cfg in SERIES_CFG.items():
        structs, part, ys, _ = load_series(name)
        X, n_elem = featurize_series(structs)
        tr = np.array(part['train_idx']); te = np.array(part['test_idx'])
        y = ys['homo']
        m = rf(); m.fit(X[tr], y[tr])
        err = np.abs(m.predict(X[te]) - y[te])
        yte_structs = [structs[i] for i in te]
        # stratification dimensions
        charges = np.array([d['total_charge'] for d in yte_structs])
        natoms = np.array([len(d['atomic_numbers']) for d in yte_structs])
        unp = np.array([d.get('unpaired_e', 0) for d in yte_structs])
        lo, hi = cfg['metal']
        metal = np.array([next((z for z in d['atomic_numbers'] if lo <= z <= hi), -1)
                          for d in yte_structs])
        for dim, key in [('charge', charges), ('unpaired', unp), ('metal', metal)]:
            for val in sorted(set(key.tolist())):
                msk = key == val
                if msk.sum() < 10:
                    continue
                diag_rows.append(dict(series=name, dim=dim, value=val, n=int(msk.sum()),
                                      mae=float(err[msk].mean()), mae_eV=float(err[msk].mean()) * EV_PER_HA))
        # natoms binning (quartiles)
        q = np.percentile(natoms, [25, 50, 75])
        bins = np.digitize(natoms, q)
        for b in range(4):
            msk = bins == b
            if msk.sum() < 10:
                continue
            diag_rows.append(dict(series=name, dim='natoms_bin', value=b, n=int(msk.sum()),
                                  mae=float(err[msk].mean()), mae_eV=float(err[msk].mean()) * EV_PER_HA))
        print(f'{name}: diagnostics done | {time.time()-t0:.0f}s', flush=True)

        # ---- B) LoEO vs CV pairing (4f/5f; fold-level distribution comparison) ----
        if name in ('4f', '5f'):
            uniq = {}
            for d in structs:
                k = scaffold_key(d, (lo, hi))
                if k not in uniq:
                    uniq[k] = len(uniq)
            g = np.array([uniq[scaffold_key(d, (lo, hi))] for d in structs])
            folds = list(GroupKFold(n_splits=5).split(tr, groups=g[tr]))
            cv_folds = []
            for ftr, fte in folds:
                m2 = rf(); m2.fit(X[tr[ftr]], y[tr[ftr]])
                cv_folds.append(float(np.abs(m2.predict(X[tr[fte]]) - y[tr[fte]]).mean()))
            # LoEO: hold out each metal in turn
            zsel = np.array([next((z for z in d['atomic_numbers'] if lo <= z <= hi), -1)
                             for d in structs])
            loeo_folds = []
            for el in sorted(set(zsel[zsel >= 0])):
                te_e = np.where(zsel == el)[0]
                tr_e = np.where(zsel != el)[0]
                m3 = rf(); m3.fit(X[tr_e], y[tr_e])
                loeo_folds.append(float(np.abs(m3.predict(X[te_e]) - y[te_e]).mean()))
            loeo_folds = np.array(loeo_folds); cv_folds = np.array(cv_folds)
            diff = float(loeo_folds.mean() - cv_folds.mean())
            rs = np.random.RandomState(0)
            boot = [loeo_folds[rs.randint(len(loeo_folds), size=len(loeo_folds))].mean()
                    - cv_folds[rs.randint(len(cv_folds), size=len(cv_folds))].mean()
                    for _ in range(1000)]
            lo_ci, hi_ci = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            pair_rows.append(dict(series=name, n_loeo_folds=len(loeo_folds),
                                  loeo_mae=float(loeo_folds.mean()),
                                  loeo_std=float(loeo_folds.std()),
                                  cv_mae=float(cv_folds.mean()),
                                  diff=diff, ci_lo=lo_ci, ci_hi=hi_ci))
            print(f'  {name} LoEO {loeo_folds.mean():.5f}±{loeo_folds.std():.5f} '
                  f'({len(loeo_folds)} folds) vs CV {cv_folds.mean():.5f} '
                  f'| diff {diff:.5f} CI ({lo_ci:.5f}, {hi_ci:.5f})', flush=True)

    with open(OUTDIR / 'diagnostics_series.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(diag_rows[0].keys()))
        w.writeheader(); w.writerows(diag_rows)
    with open(OUTDIR / 'paired_loeo_cv.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader(); w.writerows(pair_rows)
    print(f'DONE E02e | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
