"""E02d: uncertainty quantification and calibration.

RF: inter-tree sigma (std of single-tree predictions over estimators_);
MLP: 10-seed ensemble sigma.
Calibration: coverage (fraction with |err| <= z*sigma) vs nominal z in
{0.5..3}; Spearman(sigma, |err|).
Scope: five series x homo. Outputs results/E02/uq_calibration.csv,
uq_sigma.csv.
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from common import RESULTS, EV_PER_HA
from five_series import SERIES_CFG, load_series, featurize_series
from robustness import fit_mlp_w

OUTDIR = RESULTS / 'E02'
OUTDIR.mkdir(parents=True, exist_ok=True)
Z_GRID = [0.5, 0.6745, 1.0, 1.2816, 1.6449, 1.96, 2.3263, 2.5758]  # nominal confidence 38-99%


def main():
    t0 = time.time()
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    cal_rows, sig_rows = [], []
    for name in SERIES_CFG:
        structs, part, ys, _ = load_series(name)
        X, n_elem = featurize_series(structs)
        tr = np.array(part['train_idx']); va = np.array(part['val_idx']); te = np.array(part['test_idx'])
        y = ys['homo']
        yte = y[te]

        # ---- RF inter-tree sigma ----
        m = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr])
        tree_preds = np.stack([t.predict(X[te]) for t in m.estimators_])
        rf_sig = tree_preds.std(0)
        rf_pred = tree_preds.mean(0)
        rf_err = np.abs(rf_pred - yte)

        # ---- MLP 10-seed ensemble sigma ----
        mlp_preds = []
        for s in range(10):
            p = fit_mlp_w(X.shape[1], X[tr], y[tr], X[va], y[va], X[te],
                          seed=100 + s, device=device)
            mlp_preds.append(p)
        mlp_preds = np.stack(mlp_preds)
        mlp_sig = mlp_preds.std(0)
        mlp_pred = mlp_preds.mean(0)
        mlp_err = np.abs(mlp_pred - yte)

        for meth, sig, err in [('rf_tree', rf_sig, rf_err), ('mlp_ens', mlp_sig, mlp_err)]:
            rho = float(spearmanr(sig, err).statistic)
            sig_rows.append(dict(series=name, method=meth,
                                 sigma_mean=float(sig.mean()), sigma_median=float(np.median(sig)),
                                 err_mae=float(err.mean()), spearman=rho))
            for z in Z_GRID:
                cov = float((err <= z * sig).mean())
                # nominal coverage under the Gaussian assumption
                from scipy.stats import norm
                nominal = float(2 * norm.cdf(z) - 1)
                cal_rows.append(dict(series=name, method=meth, z=z,
                                     nominal=nominal, coverage=cov))
        print(f'{name}: RF σ̄ {rf_sig.mean():.4f} sp {sig_rows[-2]["spearman"]:.2f} | '
              f'MLP σ̄ {mlp_sig.mean():.4f} sp {sig_rows[-1]["spearman"]:.2f} '
              f'| {time.time()-t0:.0f}s', flush=True)

    with open(OUTDIR / 'uq_calibration.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal_rows[0].keys()))
        w.writeheader(); w.writerows(cal_rows)
    with open(OUTDIR / 'uq_sigma.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(sig_rows[0].keys()))
        w.writeheader(); w.writerows(sig_rows)
    print(f'DONE E02d | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
