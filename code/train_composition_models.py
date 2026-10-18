"""E02: unified-protocol ML pipeline over all targets.

Protocol: scaffold partition partitions/lnqm_scaffold_12444.json;
CV = GroupKFold(10) within train by scaffold; RF 200 trees / depth 10;
MLP = 256-256-128 hidden layers; energy-like targets are reported in eV
(x27.2114); the test set is evaluated once.

Outputs: results/E02/full_targets_test_<models>.csv and
         results/E02/full_targets_cv_<models>.csv
"""
import sys
import json
import time
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from common import (LNQM_TRAIN, LNQM_TEST, LNQM_EXT, PART_LNQM, RESULTS,
                    featurize_global, scaffold_key, EV_PER_HA)

OUTDIR = RESULTS / 'E02'
OUTDIR.mkdir(parents=True, exist_ok=True)

# (target_key, source, unit); source: prop=existing json, ext=E01-extended, calc=computed from json
TARGETS = [
    ('homo_up', 'prop', 'Ha-eV'), ('homo_dn', 'prop', 'Ha-eV'),
    ('lumo_up', 'prop', 'Ha-eV'), ('lumo_dn', 'prop', 'Ha-eV'),
    ('gap', 'prop', 'Ha-eV'), ('dipole_mag', 'prop', 'D'),
    ('exc', 'ext', 'Ha-eV'), ('ecnl', 'ext', 'Ha-eV'),
    ('pol_sum', 'ext', 'a.u.'), ('mayer_pop_sum', 'ext', 'e'),
    ('cn_mean', 'ext', ''), ('cn_max', 'ext', ''), ('cn_std', 'ext', ''),
    ('rot_mean', 'ext', '?'),
    ('hirshfeld_abs', 'calc', 'e'),
]
ANCHOR = {'homo_up': 0.419, 'homo_dn': 0.392, 'lumo_up': 0.332, 'lumo_dn': 0.392}


def build_xy(all_data, ext_by_index):
    """Return {key: (valid_mask, y)}; NaN samples are flagged by the mask."""
    out = {}
    n = len(all_data)
    for key, src, unit in TARGETS:
        y = np.full(n, np.nan, dtype=np.float64)
        if src == 'prop':
            y = np.array([float(d[key]) for d in all_data], dtype=np.float64)
        elif src == 'ext':
            y = np.array([float(ext_by_index[d['index']][key]) for d in all_data], dtype=np.float64)
        elif src == 'calc':
            y = np.array([np.abs(np.array(d['hirshfeld_charges'])).sum() for d in all_data], dtype=np.float64)
        mask = ~np.isnan(y)
        if mask.sum() < n:
            print(f'  [warn] {key}: {n - mask.sum()} NaN samples masked', flush=True)
        out[key] = (mask, y)
    return out


def train_mlp(Xtr, ytr, Xva, yva, Xte, seed=42, max_ep=400, patience=40,
              device=None, batch=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if batch is None:
        batch = 4096 if device == 'cuda' else 256
    torch.manual_seed(seed)
    np.random.seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    Xtr_n, Xva_n, Xte_n = (Xtr - xm) / xs, (Xva - xm) / xs, (Xte - xm) / xs
    ytr_n = (ytr - ym) / ys
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    lossf = nn.L1Loss()
    Xtr_t = torch.tensor(Xtr_n, device=device)
    ytr_t = torch.tensor(ytr_n, device=device).unsqueeze(1)
    Xva_t = torch.tensor(Xva_n, device=device)
    yva_raw = torch.tensor(yva, device=device)
    ym_t, ys_t = torch.tensor(ym, device=device), torch.tensor(ys, device=device)
    n, bs = len(Xtr_t), batch
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
            va = ((model(Xva_t).squeeze(1) * ys_t + ym_t) - yva_raw).abs().mean().item()
        sched.step(va)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='both', choices=['rf', 'mlp', 'both'])
    args = ap.parse_args()
    t0 = time.time()

    all_data = []
    for p in [LNQM_TRAIN, LNQM_TEST]:
        all_data.extend(json.load(open(p)))
    ext = json.load(open(LNQM_EXT))
    ext_by_index = {r['index']: r for r in ext}
    part = json.load(open(PART_LNQM))
    tr_idx = np.array(part['train_idx']); va_idx = np.array(part['val_idx']); te_idx = np.array(part['test_idx'])
    print(f'data {len(all_data)} | tr {len(tr_idx)} va {len(va_idx)} te {len(te_idx)} | {time.time()-t0:.0f}s', flush=True)

    tr = [all_data[i] for i in tr_idx]
    va = [all_data[i] for i in va_idx]
    te = [all_data[i] for i in te_idx]

    Xtr, Xva, Xte = featurize_global(tr), featurize_global(va), featurize_global(te)
    print(f'featurized | {time.time()-t0:.0f}s', flush=True)

    uniq = {}
    for d in tr:
        k = scaffold_key(d)
        if k not in uniq:
            uniq[k] = len(uniq)
    g_tr = np.array([uniq[scaffold_key(d)] for d in tr])
    folds = list(GroupKFold(n_splits=10).split(np.arange(len(tr)), groups=g_tr))
    print(f'train scaffolds {len(uniq)} | {time.time()-t0:.0f}s', flush=True)

    Y = build_xy(all_data, ext_by_index)

    test_rows, cv_rows = [], []
    for key, src, unit in TARGETS:
        mask, y = Y[key]
        m_tr = mask[tr_idx]; m_va = mask[va_idx]; m_te = mask[te_idx]
        ytr_f = y[tr_idx][m_tr]; yva_f = y[va_idx][m_va]; yte_f = y[te_idx][m_te]
        Xtr_f = Xtr[m_tr]; Xva_f = Xva[m_va]; Xte_f = Xte[m_te]
        row = {'target': key, 'unit': unit}

        if args.models in ('rf', 'both'):
            rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                       random_state=42, n_jobs=-1)
            rf.fit(Xtr_f, ytr_f)
            rf_te = float(np.abs(rf.predict(Xte_f) - yte_f).mean())
            row['rf_test'] = rf_te
            # Rebuild folds within the masked train subset (preserving scaffold groups)
            cvs = []
            g_tr_f = g_tr[m_tr]
            folds_f = list(GroupKFold(n_splits=10).split(np.arange(len(Xtr_f)), groups=g_tr_f))
            for ftr, fte in folds_f:
                rf2 = RandomForestRegressor(n_estimators=200, max_depth=10,
                                            random_state=42, n_jobs=-1)
                rf2.fit(Xtr_f[ftr], ytr_f[ftr])
                cvs.append(float(np.abs(rf2.predict(Xtr_f[fte]) - ytr_f[fte]).mean()))
            cv_rows.append({'target': key, 'unit': unit,
                            'rf_cv_mean': float(np.mean(cvs)), 'rf_cv_std': float(np.std(cvs)),
                            'n_folds': len(cvs)})
            print(f'[rf] {key:15s} test {rf_te:.6f} {unit} | cv {np.mean(cvs):.6f}±{np.std(cvs):.6f} | {time.time()-t0:.0f}s', flush=True)

        if args.models in ('mlp', 'both'):
            mlp_te = float(np.abs(train_mlp(Xtr_f, ytr_f, Xva_f, yva_f, Xte_f) - yte_f).mean())
            row['mlp_test'] = mlp_te
            print(f'[mlp] {key:15s} test {mlp_te:.6f} {unit} | {time.time()-t0:.0f}s', flush=True)

        if unit == 'Ha-eV':
            if 'rf_test' in row:
                row['rf_test_eV'] = row['rf_test'] * EV_PER_HA
            if 'mlp_test' in row:
                row['mlp_test_eV'] = row['mlp_test'] * EV_PER_HA
        test_rows.append(row)

    # ---- output ----
    sfx = args.models
    with open(OUTDIR / f'full_targets_test_{sfx}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['target', 'unit', 'rf_test', 'rf_test_eV', 'mlp_test', 'mlp_test_eV'])
        w.writeheader()
        for r in test_rows:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})
    if cv_rows:
        with open(OUTDIR / f'full_targets_cv_{sfx}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['target', 'unit', 'rf_cv_mean', 'rf_cv_std', 'n_folds'])
            w.writeheader()
            w.writerows(cv_rows)

    # ---- anchor reproduction assertion (regression test; anchors are in eV,
    #      rf_test is stored in Ha, converted before comparison) ----
    if args.models in ('rf', 'both'):
        print('== anchor reproduction assertion ==', flush=True)
        ok = True
        for k, ref in ANCHOR.items():
            got_ha = [r['rf_test'] for r in test_rows if r['target'] == k][0]
            got_ev = got_ha * EV_PER_HA
            dev = abs(got_ev - ref) / ref
            flag = 'PASS' if dev < 0.02 else 'FAIL'
            if dev >= 0.02:
                ok = False
            print(f'  {k:8s} rf_test {got_ha:.4f} Ha = {got_ev:.4f} eV vs anchor {ref} -> {flag} ({dev*100:.1f}%)')
        assert ok, 'anchor reproduction failed'
    print(f'\nDONE {args.models} | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
