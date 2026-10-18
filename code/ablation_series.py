"""E03e: five-series charge ablation (cross-series comparable measure).

Motivation: Gini importances are not directly comparable across feature
tables of different width (5f 48 dimensions vs 3d 29 dimensions).
The ablation delta% = (MAE_no_charge - MAE_full)/MAE_full is independent of
feature dimensionality -- a second measure for the cross-series "charge
dominance" determination. Also reports the charge-only ratio (charge_only
MAE / full MAE).
Outputs results/E03/ablation_series.csv.
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from common import RESULTS, EV_PER_HA
from five_series import SERIES_CFG, load_series, featurize_series

OUTDIR = RESULTS / 'E03'


def rf(seed=42):
    return RandomForestRegressor(n_estimators=200, max_depth=10,
                                 random_state=seed, n_jobs=-1)


def main():
    t0 = time.time()
    rows = []
    for name in SERIES_CFG:
        structs, part, ys, _ = load_series(name)
        X, n_elem = featurize_series(structs)
        tr = np.array(part['train_idx']); te = np.array(part['test_idx'])
        y = ys['homo']
        m = rf(); m.fit(X[tr], y[tr])
        mae_full = float(np.abs(m.predict(X[te]) - y[te]).mean())
        Xn = np.delete(X, n_elem + 1, axis=1)
        m2 = rf(); m2.fit(Xn[tr], y[tr])
        mae_nc = float(np.abs(m2.predict(Xn[te]) - y[te]).mean())
        Xc = X[:, n_elem + 1:n_elem + 2]
        m3 = rf(); m3.fit(Xc[tr], y[tr])
        mae_co = float(np.abs(m3.predict(Xc[te]) - y[te]).mean())
        rows.append(dict(series=name, target='homo',
                         mae_full=mae_full, mae_no_charge=mae_nc,
                         delta_pct=(mae_nc - mae_full) / mae_full * 100,
                         charge_only_ratio=mae_co / mae_full,
                         mae_full_eV=mae_full * EV_PER_HA))
        print(f'{name}: full {mae_full:.6f} | no_charge {mae_nc:.6f} '
              f'(Δ {rows[-1]["delta_pct"]:+.0f}%) | charge_only/full '
              f'{rows[-1]["charge_only_ratio"]:.2f} | {time.time()-t0:.0f}s', flush=True)
    with open(OUTDIR / 'ablation_series.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'DONE E03e | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
