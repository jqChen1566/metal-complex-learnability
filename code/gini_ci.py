"""E03d: multi-seed confidence interval of charge importance.

For each series and the homo target, 20 RF seeds -> distribution of
gini_charge; used to locate 5f (whether its CI is separated from 4f and 5d,
the statistical criterion for "which series actinides resemble").
Outputs results/E03/gini_ci.csv.
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from common import RESULTS
from five_series import SERIES_CFG, load_series, featurize_series

OUTDIR = RESULTS / 'E03'


def main():
    t0 = time.time()
    rows = []
    for name in SERIES_CFG:
        structs, part, ys, _ = load_series(name)
        X, n_elem = featurize_series(structs)
        tr = np.array(part['train_idx'])
        ginis = []
        for seed in range(42, 62):
            m = RandomForestRegressor(n_estimators=200, max_depth=10,
                                      random_state=seed, n_jobs=-1)
            m.fit(X[tr], ys['homo'][tr])
            ginis.append(float(m.feature_importances_[n_elem + 1]))
        g = np.array(ginis)
        rows.append(dict(series=name, target='homo', n_seeds=len(g),
                         gini_mean=float(g.mean()), gini_std=float(g.std()),
                         ci_lo=float(g.mean() - 2 * g.std()),
                         ci_hi=float(g.mean() + 2 * g.std())))
        print(f'{name}: gini {g.mean():.4f} ± {g.std():.4f} '
              f'(CI {g.mean()-2*g.std():.4f}, {g.mean()+2*g.std():.4f}) | {time.time()-t0:.0f}s',
              flush=True)
    with open(OUTDIR / 'gini_ci.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'DONE E03d | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
