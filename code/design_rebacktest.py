"""E04c: independent rebacktest comparison of design candidates.

Compares four methods: surrogate attainment (proxy |delta|/std) vs
independent RF attainment (rf |delta|/std), with the RF error at the start
point (a real library structure) as baseline, separating surrogate
exploitation from an unrealistic design space. 10 runs per target.
Outputs results/E04/rebacktest_compare.csv
"""
import sys
import json
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from common import (LNQM_TRAIN, LNQM_TEST, LNQM_EXT, PART_LNQM, RESULTS,
                    featurize_global)
from train_composition_models import build_xy
from transfer import make_model
from inverse_design import (PROXY_TARGETS, optimize, random_search,
                                ga_search, greedy_search)

OUTDIR = RESULTS / 'E04'
N_RUNS = 10


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_data = []
    for p in [LNQM_TRAIN, LNQM_TEST]:
        all_data.extend(json.load(open(p)))
    ext = json.load(open(LNQM_EXT))
    ext_by_index = {r['index']: r for r in ext}
    part = json.load(open(PART_LNQM))
    tr_idx, te_idx = np.array(part['train_idx']), np.array(part['test_idx'])
    tr = [all_data[i] for i in tr_idx]
    va = [all_data[i] for i in part['val_idx']]
    te = [all_data[i] for i in te_idx]
    Xtr = featurize_global(tr); Xva = featurize_global(va); Xte = featurize_global(te)
    Y = build_xy(all_data, ext_by_index)
    print(f'loaded | {time.time()-t0:.0f}s', flush=True)

    rows = []
    for key in PROXY_TARGETS:
        ck = torch.load(OUTDIR / f'proxy_{key}.pt', map_location='cpu', weights_only=False)
        xm, xs, ym, ys = ck['stats']
        model = make_model(31, device)
        model.load_state_dict(ck['state'])
        model.eval()
        xm_t = torch.tensor(np.asarray(xm), device=device)
        xs_t = torch.tensor(np.asarray(xs), device=device)
        ym_t = torch.tensor(float(np.asarray(ym)), device=device)
        ys_t = torch.tensor(float(np.asarray(ys)), device=device)

        def predict_grad(X):
            return model((X - xm_t) / xs_t).squeeze(1) * ys_t + ym_t

        def predict_np(X):
            with torch.no_grad():
                return predict_grad(torch.tensor(X, device=device)).cpu().numpy()

        mask, y = Y[key]
        std_y = float(y[mask].std())
        m_tr = mask[tr_idx]
        rf_ind = RandomForestRegressor(n_estimators=200, max_depth=10,
                                       random_state=42, n_jobs=-1)
        rf_ind.fit(Xtr[m_tr], y[tr_idx][m_tr])

        acc = {m: {'proxy': [], 'rf': []} for m in ['gradient', 'random', 'ga', 'greedy']}
        start_rf = []
        for run in range(N_RUNS):
            rng = np.random.RandomState(42 + run)
            i0, it = rng.choice(te_idx, size=2, replace=False)
            x0 = featurize_global([all_data[i0]])[0]
            target_value = float(y[it])
            start_rf.append(abs(float(rf_ind.predict(x0[None, :])[0]) - target_value) / std_y)
            bx, bp, _ = optimize(predict_grad, x0, target_value, budget=800, device=device)
            acc['gradient']['proxy'].append(abs(bp - target_value) / std_y)
            acc['gradient']['rf'].append(abs(float(rf_ind.predict(bx[None, :])[0]) - target_value) / std_y)
            for meth, fn in [('random', random_search), ('ga', ga_search), ('greedy', greedy_search)]:
                bx, bp, _ = fn(predict_np, x0, target_value, budget=800)
                acc[meth]['proxy'].append(abs(bp - target_value) / std_y)
                acc[meth]['rf'].append(abs(float(rf_ind.predict(bx[None, :])[0]) - target_value) / std_y)
        for meth in acc:
            rows.append(dict(target=key, method=meth, n_runs=N_RUNS,
                             proxy_dstd=float(np.mean(acc[meth]['proxy'])),
                             rf_dstd=float(np.mean(acc[meth]['rf'])),
                             rf_start_dstd=float(np.mean(start_rf))))
            print(f'[{key:12s}] {meth:8s} proxy {rows[-1]["proxy_dstd"]:.5f} | '
                  f'rf {rows[-1]["rf_dstd"]:.3f} | start-rf {rows[-1]["rf_start_dstd"]:.3f} '
                  f'| {time.time()-t0:.0f}s', flush=True)

    with open(OUTDIR / 'rebacktest_compare.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'DONE E04c | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
