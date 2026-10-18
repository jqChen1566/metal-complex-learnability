"""E02f: extended model families -- gradient-boosted trees (HistGB) + KRR.

Purpose: cross-family robustness check of the core conclusions (charge
dominance trend across five series, 5f position).
HistGB: gradient-boosted trees (CPU, to avoid competing for the GPU);
KRR: kernel family (train subsample <=8000 to bound the kernel matrix).
Outputs results/E02/model_families.csv
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from common import RESULTS, EV_PER_HA
from five_series import SERIES_CFG, load_series, featurize_series

OUTDIR = RESULTS / 'E02'
OUTDIR.mkdir(parents=True, exist_ok=True)
KRR_MAX = 8000


def main():
    t0 = time.time()
    rows = []
    for name in SERIES_CFG:
        structs, part, ys, _ = load_series(name)
        X, n_elem = featurize_series(structs)
        tr = np.array(part['train_idx']); te = np.array(part['test_idx'])
        y = ys['homo']

        # ---- HistGB (gradient-boosted trees) ----
        cb = HistGradientBoostingRegressor(max_iter=200, max_depth=6,
                                           learning_rate=0.08, random_state=42)
        cb.fit(X[tr], y[tr])
        cb_mae = float(np.abs(cb.predict(X[te]) - y[te]).mean())
        # no feature_importances_: use charge ablation instead (same measure as E03e)
        Xnc = np.delete(X, n_elem + 1, axis=1)
        cb_nc = HistGradientBoostingRegressor(max_iter=200, max_depth=6,
                                              learning_rate=0.08, random_state=42)
        cb_nc.fit(Xnc[tr], y[tr])
        cb_mae_nc = float(np.abs(cb_nc.predict(Xnc[te]) - y[te]).mean())
        cb_gini = (cb_mae_nc - cb_mae) / cb_mae * 100  # reused column: stores delta% here

        # ---- KRR (kernel family; RBF, train subsample) ----
        rs = np.random.RandomState(0)
        sub = rs.choice(tr, size=min(KRR_MAX, len(tr)), replace=False)
        sc = StandardScaler().fit(X[sub])
        krr = KernelRidge(kernel='rbf', alpha=1e-3, gamma=1.0 / X.shape[1])
        krr.fit(sc.transform(X[sub]), y[sub])
        krr_mae = float(np.abs(krr.predict(sc.transform(X[te])) - y[te]).mean())

        rows.append(dict(series=name, target='homo', cb_mae=cb_mae,
                         cb_mae_eV=cb_mae * EV_PER_HA, cb_charge_ablation_pct=cb_gini,
                         krr_mae=krr_mae, krr_mae_eV=krr_mae * EV_PER_HA))
        print(f'{name}: HistGB {cb_mae:.6f} (charge Δ {cb_gini:+.0f}%) '
              f'| KRR {krr_mae:.6f} Ha | {time.time()-t0:.0f}s', flush=True)

    with open(OUTDIR / 'model_families.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'DONE E02f | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
