"""E03: five-series baselines and charge-importance trend.

Protocol: features = data-driven element table (fractions of sorted(set(Z)))
+ 5 global quantities (dimension adapts to the series); partition = single
source in partitions/; CV = GroupKFold(5) within train by scaffold; test set
evaluated once; energies reported in eV (x27.2114).
Unified cross-series targets (single-value convention): homo=max(up,dn),
lumo=min(up,dn), gap, dipole(D).

Outputs results/E03/:
  five_series_baseline.csv  series x target: RF/MLP test, RF CV(5)
  five_series_lomo.csv      series x fold (leave one metal out, homo target)
"""
import sys
import json
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from common import DATA, PART_DIR, RESULTS, scaffold_key, EV_PER_HA
from train_composition_models import train_mlp

OUTDIR = RESULTS / 'E03'
OUTDIR.mkdir(parents=True, exist_ok=True)

SERIES_CFG = {
    '3d': dict(props=[DATA / 'tmqm_properties.json'],
               part=PART_DIR / 'tmqm_scaffold_33086.json', metal=(21, 30), src='tmqm'),
    '4d': dict(props=[DATA / 'tmqm_properties_4d.json'],
               part=PART_DIR / 'tmqm4d_scaffold_37450.json', metal=(39, 48), src='tmqm'),
    '5d': dict(props=[DATA / 'tmqm_properties_5d.json'],
               part=PART_DIR / 'tmqm5d_scaffold_29292.json', metal=(72, 80), src='tmqm'),
    '4f': dict(props=[DATA / 'lnqm_properties_train.json', DATA / 'lnqm_properties_test.json'],
               part=PART_DIR / 'lnqm_scaffold_12444.json', metal=(57, 71), src='lnqm'),
    '5f': dict(props=[DATA / 'acqm_properties.json'],
               part=PART_DIR / 'acqm_scaffold_2394.json', metal=(89, 103), src='acqm'),
}
TARGETS = ['homo', 'lumo', 'gap', 'dipole']
TARGET_IS_ENERGY = {'homo': True, 'lumo': True, 'gap': True, 'dipole': False}


def load_series(name):
    cfg = SERIES_CFG[name]
    structs = []
    for p in cfg['props']:
        structs.extend(json.load(open(p)))
    part = json.load(open(cfg['part']))
    # unified targets (single-value convention)
    ys = {t: np.zeros(len(structs), dtype=np.float64) for t in TARGETS}
    for i, d in enumerate(structs):
        if cfg['src'] in ('lnqm', 'acqm'):
            hu, hd = d['homo_up'], d['homo_dn']
            lu, ld = d['lumo_up'], d['lumo_dn']
            ys['homo'][i] = max(hu, hd)
            ys['lumo'][i] = min(lu, ld)
            ys['gap'][i] = ys['lumo'][i] - ys['homo'][i]
        else:
            ys['homo'][i] = d['homo']
            ys['lumo'][i] = d['lumo']
            ys['gap'][i] = d['gap']
        ys['dipole'][i] = d['dipole_mag']
    return structs, part, ys, cfg


def featurize_series(structs):
    """Data-driven element table. Returns X and the charge slot index."""
    zs_all = sorted({int(z) for d in structs for z in d['atomic_numbers'] if z > 0})
    z2i = {z: i for i, z in enumerate(zs_all)}
    n_elem = len(zs_all)
    X = np.zeros((len(structs), n_elem + 5), dtype=np.float32)
    for i, d in enumerate(structs):
        zs = np.array(d['atomic_numbers'])
        n = len(zs)
        for z in zs:
            if int(z) in z2i:
                X[i, z2i[int(z)]] += 1.0
        X[i, :n_elem] /= max(n, 1)
        X[i, n_elem] = n
        X[i, n_elem + 1] = float(d['total_charge'])
        X[i, n_elem + 2] = float(d.get('unpaired_e', 0))
        X[i, n_elem + 3] = float(np.mean(zs))
        X[i, n_elem + 4] = float(np.std(zs)) if n > 1 else 0.0
    return X, n_elem


def rf(seed=42):
    return RandomForestRegressor(n_estimators=200, max_depth=10,
                                 random_state=seed, n_jobs=-1)


def main():
    t0 = time.time()
    base_rows, lomo_rows = [], []
    for name, cfg in SERIES_CFG.items():
        structs, part, ys, _ = load_series(name)
        X, n_elem = featurize_series(structs)
        tr = np.array(part['train_idx']); va = np.array(part['val_idx']); te = np.array(part['test_idx'])
        lo, hi = cfg['metal']
        uniq = {}
        for d in structs:
            k = scaffold_key(d, (lo, hi))
            if k not in uniq:
                uniq[k] = len(uniq)
        g = np.array([uniq[scaffold_key(d, (lo, hi))] for d in structs])
        folds = list(GroupKFold(n_splits=5).split(tr, groups=g[tr]))
        print(f'== {name}: n={len(structs)} feat={X.shape[1]} scaffolds={len(uniq)} '
              f'| {time.time()-t0:.0f}s', flush=True)

        for t in TARGETS:
            y = ys[t]
            m = rf(); m.fit(X[tr], y[tr])
            rf_test = float(np.abs(m.predict(X[te]) - y[te]).mean())
            # Feature layout: fractions [0,n_elem) + natoms [n_elem]
            #                + charge [n_elem+1] + unpaired + mean_Z + std_Z
            gini_charge = float(m.feature_importances_[n_elem + 1])
            gini_natoms = float(m.feature_importances_[n_elem])
            cvs = []
            for ftr, fte in folds:
                m2 = rf(); m2.fit(X[tr[ftr]], y[tr[ftr]])
                cvs.append(float(np.abs(m2.predict(X[tr[fte]]) - y[tr[fte]]).mean()))
            mlp_test = float(np.abs(train_mlp(X[tr], y[tr], X[va], y[va], X[te]) - y[te]).mean())
            base_rows.append(dict(series=name, target=t, n_total=len(structs),
                                  n_train=len(tr), n_test=len(te),
                                  rf_test=rf_test, rf_test_eV=rf_test * EV_PER_HA if TARGET_IS_ENERGY[t] else '',
                                  mlp_test=mlp_test, mlp_test_eV=mlp_test * EV_PER_HA if TARGET_IS_ENERGY[t] else '',
                                  rf_cv5_mean=float(np.mean(cvs)), rf_cv5_std=float(np.std(cvs)),
                                  gini_charge=gini_charge, gini_natoms=gini_natoms))
            print(f'  [{t:6s}] RF {rf_test:.6f} | MLP {mlp_test:.6f} | CV5 {np.mean(cvs):.6f}'
                  f' | gini_charge {gini_charge:.3f} | gini_natoms {gini_natoms:.3f} '
                  f'| {time.time()-t0:.0f}s', flush=True)

        # ---- LoMO/LoEO (leave one metal out, homo target) ----
        zsel = np.array([next((z for z in d['atomic_numbers'] if lo <= z <= hi), -1) for d in structs])
        for el in sorted(set(zsel[zsel >= 0])):
            te_e = np.where(zsel == el)[0]
            tr_e = np.where(zsel != el)[0]
            m = rf(); m.fit(X[tr_e], ys['homo'][tr_e])
            mae = float(np.abs(m.predict(X[te_e]) - ys['homo'][te_e]).mean())
            lomo_rows.append(dict(series=name, metal_z=int(el), n_test_fold=len(te_e),
                                  homo_mae=mae, homo_mae_eV=mae * EV_PER_HA))
        print(f'  LoMO {len([r for r in lomo_rows if r["series"]==name])} folds done '
              f'| {time.time()-t0:.0f}s', flush=True)

        with open(OUTDIR / 'five_series_baseline.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(base_rows[0].keys()))
            w.writeheader(); w.writerows(base_rows)
        with open(OUTDIR / 'five_series_lomo.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(lomo_rows[0].keys()))
            w.writeheader(); w.writerows(lomo_rows)

    print(f'DONE E03 five_series | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
