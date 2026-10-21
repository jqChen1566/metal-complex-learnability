"""E03c: cross-series transfer matrix.

Convention: super-table features (fractions over the union of all elements
across the five series + 5 global quantities) -- same dimension across
domains, so model weights can legitimately be reused; unified target homo;
MLP (GPU). Zero-shot uses source-domain statistics for standardization
(no distribution-shift correction).
Outputs results/E03/:
  transfer_zeroshot.csv  5x5 zero-shot matrix (+ mean-shift corrected diagnostic column)
  transfer_curve.csv     focus pairs x budget{25,100,500} x 10 draws x 2 arms
                         (scratch / fine-tune lr 3e-4)
  transfer_srcscale.csv  source-size control (4f subsampled to the 5f train size)
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
from common import RESULTS, EV_PER_HA
from five_series import SERIES_CFG, load_series

OUTDIR = RESULTS / 'E03'
TARGET = 'homo'
N_DRAWS = 10
BUDGETS = [25, 100, 500]
FOCUS_PAIRS = [('4f', '5f'), ('5d', '5f'), ('3d', '4f'), ('4f', '4f')]


def featurize_super(structs, z2i):
    n_elem = len(z2i)
    X = np.zeros((len(structs), n_elem + 5), dtype=np.float32)
    for i, d in enumerate(structs):
        zs = np.array(d['atomic_numbers'])
        n = len(zs)
        for z in zs:
            j = z2i.get(int(z))
            if j is not None:
                X[i, j] += 1.0
        X[i, :n_elem] /= max(n, 1)
        X[i, n_elem] = n
        X[i, n_elem + 1] = float(d['total_charge'])
        X[i, n_elem + 2] = float(d.get('unpaired_e', 0))
        X[i, n_elem + 3] = float(np.mean(zs))
        X[i, n_elem + 4] = float(np.std(zs)) if n > 1 else 0.0
    return X


def make_model(d_in, device):
    return nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)


def fit_mlp(Xtr, ytr, Xva, yva, Xte, stats=None, init_state=None,
            lr=1e-3, max_ep=250, patience=25, device=None, seed=42, batch=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if batch is None:
        batch = 4096 if device == 'cuda' else 256
    torch.manual_seed(seed)
    np.random.seed(seed)
    if stats is None:
        xm, xs = Xtr.mean(0), Xtr.std(0)
        # guard against zero-variance columns: element columns absent from the
        # source domain have std ~0, so (x-mu)/sigma explodes across domains
        xs = np.where(xs < 1e-6, 1.0, xs)
        ym, ys = float(ytr.mean()), float(ytr.std()) + 1e-8
        stats = (xm, xs, ym, ys)
    xm, xs, ym, ys = stats
    Xtr_n = (Xtr - xm) / xs
    Xva_n = (Xva - xm) / xs
    ytr_n = (ytr - ym) / ys
    model = make_model(Xtr.shape[1], device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.L1Loss()
    Xtr_t = torch.tensor(Xtr_n, device=device)
    ytr_t = torch.tensor(ytr_n, device=device).unsqueeze(1)
    Xva_t = torch.tensor(Xva_n, device=device)
    yva_t = torch.tensor(yva, device=device)
    ym_t = torch.tensor(float(ym), device=device)
    ys_t = torch.tensor(float(ys), device=device)
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
    state = {k: v.clone() for k, v in model.state_dict().items()}
    pred = None
    if Xte is not None:
        with torch.no_grad():
            pred = (model(torch.tensor((Xte - xm) / xs, device=device)).squeeze(1)
                    * ys_t + ym_t).cpu().numpy()
    return pred, state, stats


def predict_with(state, stats, X, device):
    xm, xs, ym, ys = stats
    model = make_model(X.shape[1], device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return (model(torch.tensor((X - xm) / xs, device=device)).squeeze(1)
                * torch.tensor(float(ys), device=device)
                + torch.tensor(float(ym), device=device)).cpu().numpy()


def main():
    global TARGET
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='homo',
                        choices=['homo', 'lumo', 'gap', 'dipole'])
    parser.add_argument('--acqm-min-atoms', type=int, default=0)
    parser.add_argument('--charge-abs-max', type=int, default=-1)
    parser.add_argument('--unpaired-max', type=int, default=-1)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    TARGET = args.target
    suffix = '' if TARGET == 'homo' else f'_{TARGET}'
    if args.tag:
        suffix += f'_{args.tag}'
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device = {device} | target = {TARGET}', flush=True)

    # ---- load + super table ----
    data = {}
    all_z = set()
    for name in SERIES_CFG:
        structs, part, ys, _ = load_series(name)
        data[name] = (structs, part, ys)
        for d in structs:
            all_z.update(int(z) for z in d['atomic_numbers'] if z > 0)
    z2i = {z: i for i, z in enumerate(sorted(all_z))}
    print(f'super table: {len(z2i)} elements | {time.time()-t0:.0f}s', flush=True)
    Xd, yd, TR, TE, VA = {}, {}, {}, {}, {}
    for name, (structs, part, ys) in data.items():
        Xd[name] = featurize_super(structs, z2i)
        yd[name] = ys[TARGET]
        mask = np.ones(len(structs), dtype=bool)
        if name == '5f' and args.acqm_min_atoms > 0:
            mask &= np.array([len(d['atomic_numbers']) >= args.acqm_min_atoms
                              for d in structs])
        if args.charge_abs_max >= 0:
            mask &= np.array([abs(int(d['total_charge'])) <= args.charge_abs_max
                              for d in structs])
        if args.unpaired_max >= 0:
            mask &= np.array([int(d.get('unpaired_e', 0)) <= args.unpaired_max
                              for d in structs])
        tr = np.array(part['train_idx'])
        te = np.array(part['test_idx'])
        va = np.array(part['val_idx'])
        TR[name] = tr[mask[tr]]
        TE[name] = te[mask[te]]
        VA[name] = va[mask[va]]
        print(f'{name}: filter keeps train {len(TR[name])}/{len(tr)}, '
              f'val {len(VA[name])}/{len(va)}, test {len(TE[name])}/{len(te)}',
              flush=True)

    # ---- pretraining (full source) ----
    pre = {}
    for name in SERIES_CFG:
        _, state, stats = fit_mlp(Xd[name][TR[name]], yd[name][TR[name]],
                                  Xd[name][VA[name]], yd[name][VA[name]], None)
        pre[name] = (state, stats)
        print(f'pretrain {name} | {time.time()-t0:.0f}s', flush=True)

    # ---- zero-shot matrix 5x5 ----
    zs_rows = []
    preds_out = {}
    for src in SERIES_CFG:
        for tgt in SERIES_CFG:
            state, stats = pre[src]
            pred = predict_with(state, stats, Xd[tgt][TE[tgt]], device)
            y_true = yd[tgt][TE[tgt]]
            preds_out[f'{src}->{tgt}'] = (pred, y_true)
            mae = float(np.abs(pred - y_true).mean())
            pred_s = pred - pred.mean() + y_true.mean()
            mae_s = float(np.abs(pred_s - y_true).mean())
            zs_rows.append(dict(src=src, tgt=tgt, zeroshot_mae=mae,
                                zeroshot_mae_eV=mae * EV_PER_HA, shifted_mae=mae_s))
            print(f'  zs {src}->{tgt}: {mae:.5f} Ha ({mae*EV_PER_HA:.3f} eV)', flush=True)
    np.savez(OUTDIR / f'transfer_preds{suffix}.npz',
             **{k: np.vstack(v) for k, v in preds_out.items()})
    with open(OUTDIR / f'transfer_zeroshot{suffix}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(zs_rows[0].keys()))
        w.writeheader()
        w.writerows(zs_rows)

    # ---- fine-tuning curve ----
    curve_rows = []
    for srct, tgtt in FOCUS_PAIRS:
        Xva, yva = Xd[tgtt][VA[tgtt]], yd[tgtt][VA[tgtt]]
        Xte, yte = Xd[tgtt][TE[tgtt]], yd[tgtt][TE[tgtt]]
        for n_sample in BUDGETS:
            for draw in range(N_DRAWS):
                rs = np.random.RandomState(10000 + draw)
                sub = rs.choice(TR[tgtt], size=min(n_sample, len(TR[tgtt])), replace=False)
                Xtr, ytr = Xd[tgtt][sub], yd[tgtt][sub]
                p0, _, _ = fit_mlp(Xtr, ytr, Xva, yva, Xte, lr=1e-3)
                p1, _, _ = fit_mlp(Xtr, ytr, Xva, yva, Xte,
                                   stats=pre[srct][1], init_state=pre[srct][0], lr=3e-4)
                curve_rows.append(dict(src=srct, tgt=tgtt, n=n_sample, draw=draw,
                                       scratch_mae=float(np.abs(p0 - yte).mean()),
                                       finetune_mae=float(np.abs(p1 - yte).mean())))
            print(f'  curve {srct}->{tgtt} n={n_sample} done | {time.time()-t0:.0f}s', flush=True)
    with open(OUTDIR / f'transfer_curve{suffix}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        w.writerows(curve_rows)

    # ---- source-size control: 4f subsampled to the 5f train size -> 5f ----
    rs = np.random.RandomState(7)
    src_sub = rs.choice(TR['4f'], size=len(TR['5f']), replace=False)
    _, state_s, stats_s = fit_mlp(Xd['4f'][src_sub], yd['4f'][src_sub],
                                  Xd['4f'][VA['4f']], yd['4f'][VA['4f']], None)
    scale_rows = []
    for n_sample in [100, 500]:
        for draw in range(5):
            r2 = np.random.RandomState(20000 + draw)
            sub = r2.choice(TR['5f'], size=min(n_sample, len(TR['5f'])), replace=False)
            p, _, _ = fit_mlp(Xd['5f'][sub], yd['5f'][sub], Xd['5f'][VA['5f']],
                              yd['5f'][VA['5f']], Xd['5f'][TE['5f']],
                              stats=stats_s, init_state=state_s, lr=3e-4)
            scale_rows.append(dict(src='4f_sub1896', tgt='5f', n=n_sample, draw=draw,
                                   finetune_mae=float(np.abs(p - yd['5f'][TE['5f']]).mean())))
    with open(OUTDIR / f'transfer_srcscale{suffix}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(scale_rows[0].keys()))
        w.writeheader()
        w.writerows(scale_rows)
    print(f'DONE E03c | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
