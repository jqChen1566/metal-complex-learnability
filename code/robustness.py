"""E02c: robustness checks.

A) Multi-seed performance: five series x homo x {RF 20 seeds, MLP 10 seeds}
   -> test MAE mean +- sigma
B) Small MLP hyperparameter grid: 4f (+4d spot check) x lr{5e-4,1e-3,3e-3}
   x width{128,256,512}
Outputs results/E02/seed_robustness.csv, mlp_hpo.csv
Usage: python3 E02c_robustness.py [--part seeds|hpo|all]
"""
import sys
import time
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from common import RESULTS, EV_PER_HA
from five_series import SERIES_CFG, load_series, featurize_series

OUTDIR = RESULTS / 'E02'
OUTDIR.mkdir(parents=True, exist_ok=True)


def rf(seed):
    return RandomForestRegressor(n_estimators=200, max_depth=10,
                                 random_state=seed, n_jobs=-1)


def fit_mlp_w(w_in, Xtr, ytr, Xva, yva, Xte, lr=1e-3, width=256, seed=42,
              max_ep=250, patience=25, device=None):
    """MLP with configurable width (used for HPO)."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = 4096 if device == 'cuda' else 256
    torch.manual_seed(seed)
    np.random.seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = float(ytr.mean()), float(ytr.std()) + 1e-8
    Xtr_n, Xva_n, Xte_n = (Xtr - xm) / xs, (Xva - xm) / xs, (Xte - xm) / xs
    ytr_n = (ytr - ym) / ys
    model = nn.Sequential(
        nn.Linear(w_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width // 2), nn.ReLU(), nn.Linear(width // 2, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.L1Loss()
    Xtr_t = torch.tensor(Xtr_n, device=device)
    ytr_t = torch.tensor(ytr_n, device=device).unsqueeze(1)
    Xva_t = torch.tensor(Xva_n, device=device)
    yva_t = torch.tensor(yva, device=device)
    ym_t = torch.tensor(ym, device=device)
    ys_t = torch.tensor(ys, device=device)
    n, bs = len(Xtr_t), min(batch, max(len(Xtr_t), 1))
    best, best_state, bad = 1e18, None, 0
    for ep in range(1, max_ep + 1):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            lossf(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = ((model(Xva_t).squeeze(1) * ys_t + ym_t) - yva_t).abs().mean().item()
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = (model(torch.tensor(Xte_n, device=device)).squeeze(1) * ys_t + ym_t).cpu().numpy()
    return pred


def run_seeds():
    rows = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for name in SERIES_CFG:
        structs, part, ys, _ = load_series(name)
        X, n_elem = featurize_series(structs)
        tr = np.array(part['train_idx']); va = np.array(part['val_idx']); te = np.array(part['test_idx'])
        y = ys['homo']
        t_s = time.time()
        rf_maes = [float(np.abs(rf(s).fit(X[tr], y[tr]).predict(X[te]) - y[te]).mean())
                   for s in range(42, 62)]
        mlp_maes = []
        for s in range(42, 52):
            p = fit_mlp_w(X.shape[1], X[tr], y[tr], X[va], y[va], X[te], seed=s, device=device)
            mlp_maes.append(float(np.abs(p - y[te]).mean()))
        rows.append(dict(series=name, target='homo', n_seeds_rf=20, n_seeds_mlp=10,
                         rf_mean=float(np.mean(rf_maes)), rf_std=float(np.std(rf_maes)),
                         mlp_mean=float(np.mean(mlp_maes)), mlp_std=float(np.std(mlp_maes))))
        print(f'{name}: RF {np.mean(rf_maes):.6f}±{np.std(rf_maes):.6f} | '
              f'MLP {np.mean(mlp_maes):.6f}±{np.std(mlp_maes):.6f} Ha '
              f'({time.time()-t_s:.0f}s)', flush=True)
    with open(OUTDIR / 'seed_robustness.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_hpo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    for name in ['4f', '4d']:
        structs, part, ys, _ = load_series(name)
        X, n_elem = featurize_series(structs)
        tr = np.array(part['train_idx']); va = np.array(part['val_idx']); te = np.array(part['test_idx'])
        y = ys['homo']
        for width in [128, 256, 512]:
            for lr in [5e-4, 1e-3, 3e-3]:
                t_s = time.time()
                p = fit_mlp_w(X.shape[1], X[tr], y[tr], X[va], y[va], X[te],
                              lr=lr, width=width, device=device)
                mae = float(np.abs(p - y[te]).mean())
                rows.append(dict(series=name, width=width, lr=lr, test_mae=mae))
                print(f'{name}: width {width} lr {lr:g} -> {mae:.6f} Ha '
                      f'({time.time()-t_s:.0f}s)', flush=True)
    with open(OUTDIR / 'mlp_hpo.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['seeds', 'hpo', 'all'])
    args = ap.parse_args()
    t0 = time.time()
    if args.part in ('seeds', 'all'):
        run_seeds()
    if args.part in ('hpo', 'all'):
        run_hpo()
    print(f'DONE E02c ({args.part}) | {time.time()-t0:.0f}s', flush=True)
