"""E02b: effect decomposition and per-target leave-one-element-out analysis.

Outputs in results/E02b/:
  ablation.csv        per-target full / no_charge / charge_only (RF test) delta%
  shap_summary.csv    charge importance by three measures: Gini / permutation / SHAP
  loeo_by_target.csv  per-target LoEO (leave one element out) MAE mean +- sigma
  decision_matrix.csv determinant matrix (target x metric, for plotting)
"""
import sys
import json
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from common import (LNQM_TRAIN, LNQM_TEST, LNQM_EXT, PART_LNQM, RESULTS,
                    featurize_global, EV_PER_HA)
from train_composition_models import TARGETS, build_xy

OUTDIR = RESULTS / 'E02b'
OUTDIR.mkdir(parents=True, exist_ok=True)


def rf(seed=42):
    return RandomForestRegressor(n_estimators=200, max_depth=10,
                                 random_state=seed, n_jobs=-1)


def run_loeo(structs, y, feat, elements):
    """Leave one element out: train on all structures without the element,
    test on all structures containing it. structs/y/feat share one pool
    (train+test). Returns the list of fold MAEs."""
    zs_metal = np.array([next((z for z in s['atomic_numbers'] if z in elements), -1)
                         for s in structs])
    maes = []
    for el in elements:
        te = np.where(zs_metal == el)[0]
        tr = np.where(zs_metal != el)[0]
        if len(te) == 0:
            continue
        m = rf()
        m.fit(feat[tr], y[tr])
        maes.append(float(np.abs(m.predict(feat[te]) - y[te]).mean()))
    return maes


def main():
    t0 = time.time()
    all_data = []
    for p in [LNQM_TRAIN, LNQM_TEST]:
        all_data.extend(json.load(open(p)))
    ext = json.load(open(LNQM_EXT))
    ext_by_index = {r['index']: r for r in ext}
    part = json.load(open(PART_LNQM))
    tr_idx = np.array(part['train_idx']); te_idx = np.array(part['test_idx'])
    tr = [all_data[i] for i in tr_idx]
    te = [all_data[i] for i in te_idx]

    Xtr_all, Xte_all = featurize_global(tr), featurize_global(te)
    X_all = featurize_global(all_data)
    Y = build_xy(all_data, ext_by_index)
    elements = list(range(57, 72))
    print(f'loaded | {time.time()-t0:.0f}s', flush=True)

    try:
        import shap
        HAS_SHAP = True
    except Exception as e:
        print('shap unavailable:', e, flush=True)
        HAS_SHAP = False

    abl_rows, shap_rows, loeo_rows, dm_rows = [], [], [], []
    for key, src, unit in TARGETS:
        mask, y = Y[key]
        m_tr, m_te = mask[tr_idx], mask[te_idx]
        ytr, yte = y[tr_idx][m_tr], y[te_idx][m_te]
        Xtr, Xte = Xtr_all[m_tr], Xte_all[m_te]

        # ---- 1. ablation ----
        m_full = rf(); m_full.fit(Xtr, ytr)
        mae_full = float(np.abs(m_full.predict(Xte) - yte).mean())
        Xtr_nc = np.delete(Xtr, 27, axis=1); Xte_nc = np.delete(Xte, 27, axis=1)
        m_nc = rf(); m_nc.fit(Xtr_nc, ytr)
        mae_nc = float(np.abs(m_nc.predict(Xte_nc) - yte).mean())
        Xtr_co = Xtr[:, 27:28]; Xte_co = Xte[:, 27:28]
        m_co = rf(); m_co.fit(Xtr_co, ytr)
        mae_co = float(np.abs(m_co.predict(Xte_co) - yte).mean())
        delta = (mae_nc - mae_full) / mae_full * 100 if mae_full > 0 else float('nan')
        abl_rows.append({'target': key, 'unit': unit, 'mae_full': mae_full,
                         'mae_no_charge': mae_nc, 'mae_charge_only': mae_co,
                         'delta_pct': delta})

        # ---- 2. charge importance by three measures (train pool) ----
        gini = float(m_full.feature_importances_[27])
        sub = np.random.RandomState(42).choice(len(Xtr), size=min(2000, len(Xtr)), replace=False)
        # small repeat count to limit runtime
        pi = permutation_importance(m_full, Xtr[sub], ytr[sub], n_repeats=3,
                                    random_state=42, n_jobs=-1)
        perm_share = float(pi.importances_mean[27] / np.abs(pi.importances_mean).sum())
        shap_share = float('nan')
        if HAS_SHAP:
            try:
                expl = shap.TreeExplainer(m_full)
                sv = expl.shap_values(Xtr[sub])
                if isinstance(sv, list):
                    sv = sv[0]
                sv = np.abs(sv).mean(0)
                shap_share = float(sv[27] / sv.sum())
            except Exception as e:
                print(f'  [warn] shap failed for {key}: {e}', flush=True)
        shap_rows.append({'target': key, 'gini': gini, 'perm_share': perm_share,
                          'shap_share': shap_share})

        # ---- 3. LoEO (full pool) ----
        m_all, y_all = mask, y
        loeo = run_loeo([all_data[i] for i in np.where(m_all)[0]],
                        y_all[m_all], X_all[m_all], elements)
        loeo_rows.append({'target': key, 'unit': unit,
                          'loeo_mean': float(np.mean(loeo)), 'loeo_std': float(np.std(loeo)),
                          'n_folds': len(loeo)})
        # CV ratio: use the main-chain RF CV (read from the E02 csv)
        dm_rows.append({'target': key, 'unit': unit, 'delta_pct': delta,
                        'gini': gini, 'perm_share': perm_share, 'shap_share': shap_share,
                        'loeo_mean': float(np.mean(loeo))})
        print(f'[{key:15s}] full {mae_full:.6f} | d% {delta:6.1f} | gini {gini:.3f} '
              f'perm {perm_share:.3f} shap {shap_share:.3f} | loeo {np.mean(loeo):.5f} '
              f'| {time.time()-t0:.0f}s', flush=True)

    for name, rows in [('ablation.csv', abl_rows), ('shap_summary.csv', shap_rows),
                       ('loeo_by_target.csv', loeo_rows), ('decision_matrix.csv', dm_rows)]:
        with open(OUTDIR / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f'DONE E02b | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
