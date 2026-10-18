"""E04b: full inverse-design benchmark.

50 independent runs per target (start/target = paired random test
structures); four methods with the same budget (800 calls);
metrics: hit rate (|delta|/std < 0.1) and best |delta|/std.
Independent rebacktest: gradient best_x -> independent RF (retrained on
train) prediction, guarding against surrogate exploitation; designability x
predictability data are written to disk.
Outputs: results/E04/design_benchmark.csv, design_rebacktest.csv,
designability.csv
"""
import sys
import json
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from common import (LNQM_TRAIN, LNQM_TEST, LNQM_EXT, PART_LNQM, RESULTS,
                    featurize_global)
from train_composition_models import build_xy
from transfer import make_model
from inverse_design import (PROXY_TARGETS, N_ELEM, optimize, random_search,
                                ga_search, greedy_search)

OUTDIR = RESULTS / 'E04'
N_RUNS = 50
BUDGET = 800
THRESH = 0.10  # hit threshold: |delta| / std(y)


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_data = []
    for p in [LNQM_TRAIN, LNQM_TEST]:
        all_data.extend(json.load(open(p)))
    ext = json.load(open(LNQM_EXT))
    ext_by_index = {r['index']: r for r in ext}
    part = json.load(open(PART_LNQM))
    tr_idx, va_idx, te_idx = (np.array(part['train_idx']), np.array(part['val_idx']),
                              np.array(part['test_idx']))
    tr = [all_data[i] for i in tr_idx]
    va = [all_data[i] for i in va_idx]
    te = [all_data[i] for i in te_idx]
    Xtr, Xva, Xte = featurize_global(tr), featurize_global(va), featurize_global(te)
    Y = build_xy(all_data, ext_by_index)
    print(f'loaded | {time.time()-t0:.0f}s', flush=True)

    bench_rows, rb_rows, design_rows = [], [], []
    for key in PROXY_TARGETS:
        ck = torch.load(OUTDIR / f'proxy_{key}.pt', map_location='cpu', weights_only=False)
        stats = ck['stats']
        xm, xs, ym, ys = stats
        model = make_model(31, device)
        model.load_state_dict(ck['state'])
        model.eval()
        xm_t = torch.tensor(np.asarray(xm), device=device)
        xs_t = torch.tensor(np.asarray(xs), device=device)
        ym_f, ys_f = float(np.asarray(ym)), float(np.asarray(ys))
        ym_t = torch.tensor(ym_f, device=device)
        ys_t = torch.tensor(ys_f, device=device)

        def predict_grad(X):
            return model((X - xm_t) / xs_t).squeeze(1) * ys_t + ym_t

        def predict_np(X):
            with torch.no_grad():
                return predict_grad(torch.tensor(X, device=device)).cpu().numpy()

        mask, y = Y[key]
        std_y = float(y[mask].std())
        # independent RF (for rebacktest, trained on train)
        m_tr = mask[tr_idx]
        rf_ind = RandomForestRegressor(n_estimators=200, max_depth=10,
                                       random_state=42, n_jobs=-1)
        rf_ind.fit(Xtr[m_tr], y[tr_idx][m_tr])

        # ---- 50 runs ----
        per_method = {m: [] for m in ['gradient', 'random', 'ga', 'greedy']}
        rb_deltas = []
        for run in range(N_RUNS):
            rng = np.random.RandomState(42 + run)
            i0, it = rng.choice(te_idx, size=2, replace=False)
            x0 = featurize_global([all_data[i0]])[0]
            target_value = float(y[it])
            bx, bp, _ = optimize(predict_grad, x0, target_value, budget=BUDGET, device=device)
            for name, d in [('gradient', abs(bp - target_value))]:
                per_method[name].append(d / std_y)
            # independent rebacktest (gradient method)
            prf = float(rf_ind.predict(bx[None, :])[0])
            rb_deltas.append((abs(prf - target_value) / std_y, abs(bp - target_value) / std_y))
            bx, bp, _ = random_search(predict_np, x0, target_value, budget=BUDGET)
            per_method['random'].append(abs(bp - target_value) / std_y)
            bx, bp, _ = ga_search(predict_np, x0, target_value, budget=BUDGET)
            per_method['ga'].append(abs(bp - target_value) / std_y)
            bx, bp, _ = greedy_search(predict_np, x0, target_value, budget=BUDGET)
            per_method['greedy'].append(abs(bp - target_value) / std_y)
        for name, ds in per_method.items():
            ds = np.array(ds)
            bench_rows.append(dict(target=key, method=name, n_runs=N_RUNS,
                                   hit_rate=float((ds < THRESH).mean()),
                                   dstd_mean=float(ds.mean()), dstd_median=float(np.median(ds)),
                                   dstd_std=float(ds.std())))
            print(f'[{key:12s}] {name:8s} hit {bench_rows[-1]["hit_rate"]:.2f} '
                  f'| d/std mean {ds.mean():.4f} median {np.median(ds):.4f} | {time.time()-t0:.0f}s',
                  flush=True)
        rb = np.array(rb_deltas)
        rb_rows.append(dict(target=key, proxy_dstd_mean=float(rb[:, 1].mean()),
                            rf_dstd_mean=float(rb[:, 0].mean()),
                            agree_rate=float((np.abs(rb[:, 0] - rb[:, 1]) < THRESH).mean())))
        # predictability (RF-ind test MAE) and designability (cross-method spread)
        mae_rf = float(np.abs(rf_ind.predict(Xte[mask[te_idx]]) - y[te_idx][mask[te_idx]]).mean())
        design_rows.append(dict(target=key, rf_test_mae=mae_rf, std_y=std_y,
                                grad_dstd=bench_rows[-4]['dstd_mean'],
                                rand_dstd=bench_rows[-3]['dstd_mean']))

    for name, rows in [('design_benchmark.csv', bench_rows),
                       ('design_rebacktest.csv', rb_rows),
                       ('designability.csv', design_rows)]:
        with open(OUTDIR / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f'DONE E04b | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
