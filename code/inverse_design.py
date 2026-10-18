"""E04: differentiable inverse-design framework.

Design variables = composition fractions (26, simplex) + charge
(1, continuous relaxation in [-1, 3]); natoms/unpaired are fixed at the start
point; mean_Z/std_Z are recomputed from the fractions. Holding charge fixed
while optimizing only the fractions makes the homo target trivially reachable
by random sampling, since composition alone has a much weaker lever on homo
than charge.

Components: surrogate checkpoints (5 targets) / differentiable projection /
gradient optimizer / random + GA + greedy baselines / unit tests.
Outputs: results/E04/proxy_<target>.pt and diagnostic reports.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
import torch.nn as nn
from common import (LNQM_TRAIN, LNQM_TEST, LNQM_EXT, PART_LNQM, RESULTS,
                    featurize_global, EV_PER_HA)
from train_composition_models import build_xy
from transfer import fit_mlp, make_model

OUTDIR = RESULTS / 'E04'
OUTDIR.mkdir(parents=True, exist_ok=True)

PROXY_TARGETS = ['homo_up', 'gap', 'dipole_mag', 'pol_sum', 'cn_mean']
N_ELEM = 26
N_VAR = N_ELEM + 1  # 26 fractions + charge
CHARGE_LO, CHARGE_HI = -1.0, 3.0
_ELEM_Z = np.array([1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53,
                    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71], dtype=np.float64)


def project_simplex_torch(v):
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, 0)
    idx = torch.arange(1, len(v) + 1, dtype=v.dtype, device=v.device)
    cond = u * idx > (css - 1.0)
    rho = torch.nonzero(cond)[-1].item()
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


def compose_full_x(theta_var, template):
    """theta_var = [26 unprojected fractions, charge] -> projected 31-D feature vector."""
    p = project_simplex_torch(theta_var[:N_ELEM])
    ch = torch.clamp(theta_var[N_ELEM], CHARGE_LO, CHARGE_HI)
    z = torch.tensor(_ELEM_Z, dtype=p.dtype, device=p.device)
    zmean = (p * z).sum()
    zstd = torch.sqrt(((p * (z - zmean) ** 2).sum()) + 1e-12)
    tail = template[N_ELEM:].clone()
    return torch.cat([p, tail[:1], ch.reshape(1), tail[2:3], zmean.reshape(1), zstd.reshape(1)])


def np_build(var, template):
    """NumPy assembly (for baselines): var = [26 fractions, charge]."""
    p = np.clip(var[:N_ELEM], 0, None)
    p = p / max(p.sum(), 1e-9)
    ch = float(np.clip(var[N_ELEM], CHARGE_LO, CHARGE_HI))
    x = template.copy()
    x[:N_ELEM] = p
    x[N_ELEM + 1] = ch
    z = _ELEM_Z
    x[N_ELEM + 3] = float((p * z).sum())
    x[N_ELEM + 4] = float(np.sqrt((p * (z - x[N_ELEM + 3]) ** 2).sum()))
    return x


def optimize(predict_fn_grad, x0, target_value, budget=800, lr=0.05, device='cpu'):
    template = torch.tensor(x0, dtype=torch.float32, device=device)
    theta = torch.tensor(np.concatenate([x0[:N_ELEM], [x0[N_ELEM + 1]]]),
                         dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    best_x, best_pred, best_d = None, None, 1e18
    trace = []
    for it in range(budget):
        opt.zero_grad()
        x = compose_full_x(theta, template)
        pred = predict_fn_grad(x.unsqueeze(0)).squeeze()
        d = (pred - target_value).abs()
        d.backward()
        opt.step()
        dv = float(d.item())
        if dv < best_d:
            best_d = dv
            best_x = x.detach().cpu().numpy()
            best_pred = float(pred.item())
        if it % 50 == 0:
            trace.append((it + 1, dv))
    return best_x, best_pred, trace


def random_search(predict_fn, x0, target_value, budget=800, seed=0):
    rs = np.random.RandomState(seed)
    template = np.array(x0, dtype=np.float32)
    best_x, best_pred, best_d = None, None, 1e18
    trace = []
    for it in range(budget):
        var = np.concatenate([rs.dirichlet(np.ones(N_ELEM)),
                              [rs.uniform(CHARGE_LO, CHARGE_HI)]]).astype(np.float32)
        x = np_build(var, template)
        pred = float(predict_fn(x[None, :])[0])
        d = abs(pred - target_value)
        if d < best_d:
            best_d, best_x, best_pred = d, x, pred
        if it % 50 == 0:
            trace.append((it + 1, d))
    return best_x, best_pred, trace


def ga_search(predict_fn, x0, target_value, budget=800, pop=40, seed=0):
    rs = np.random.RandomState(seed)
    template = np.array(x0, dtype=np.float32)
    P = np.concatenate([rs.dirichlet(np.ones(N_ELEM), size=pop),
                        rs.uniform(CHARGE_LO, CHARGE_HI, (pop, 1))], axis=1).astype(np.float32)

    def eval_pop(P):
        X = np.stack([np_build(v, template) for v in P])
        return predict_fn(X)

    calls = 0
    best_x, best_pred, best_d = None, None, 1e18
    trace = []
    while calls + pop <= budget:
        preds = eval_pop(P)
        calls += pop
        ds = np.abs(preds - target_value)
        j = int(ds.argmin())
        if ds[j] < best_d:
            best_d, best_pred = float(ds[j]), float(preds[j])
            best_x = np_build(P[j], template)
        order = np.argsort(ds)
        elite = P[order[: pop // 4]]
        kids = []
        while len(kids) < pop - len(elite):
            a, b = rs.randint(len(elite), size=2)
            w = rs.rand()
            kid = w * elite[a] + (1 - w) * elite[b]
            m = rs.rand(N_VAR) < 0.1
            kid[m] += rs.normal(0, 0.02, m.sum())
            kids.append(kid.astype(np.float32))
        P = np.vstack([elite, kids])
        trace.append((calls, best_d))
    return best_x, best_pred, trace


def greedy_search(predict_fn, x0, target_value, budget=800, seed=0, step=0.1):
    rs = np.random.RandomState(seed)
    template = np.array(x0, dtype=np.float32)
    var = np.concatenate([x0[:N_ELEM] / max(x0[:N_ELEM].sum(), 1e-9), [x0[N_ELEM + 1]]]).astype(np.float32)

    def ev(v):
        return float(predict_fn(np_build(v, template)[None, :])[0])

    calls = 0
    cur_d = abs(ev(var) - target_value)
    calls += 1
    best_d, best_var = cur_d, var.copy()
    trace = []
    for _ in range(50):  # outer loop
        improved = False
        for i in range(N_VAR):
            for delta in (step, -step):
                q = var.copy()
                q[i] = q[i] + delta
                d = abs(ev(q) - target_value)
                calls += 1
                if d < cur_d:
                    var, cur_d, improved = q, d, True
                    if d < best_d:
                        best_d, best_var = d, q.copy()
            if calls > budget:
                break
        trace.append((calls, best_d))
        if not improved or calls > budget:
            break
    return np_build(best_var, template), ev(best_var), trace


def unit_tests():
    print('== unit tests ==', flush=True)
    rs = np.random.RandomState(0)
    for _ in range(20):
        v = torch.tensor(rs.normal(0, 1, N_ELEM).astype(np.float32))
        p = project_simplex_torch(v)
        assert p.min() >= -1e-6 and abs(float(p.sum()) - 1.0) < 1e-5
    print('  simplex projection constraints 20/20 PASS', flush=True)
    v = torch.tensor([0.5] * N_ELEM + [99.0], requires_grad=True)
    x = compose_full_x(v, torch.tensor(np.zeros(31, np.float32)))
    assert abs(float(x[N_ELEM + 1].detach()) - CHARGE_HI) < 1e-6
    print('  charge projection (upper clip) PASS', flush=True)
    x0 = featurize_global([json.load(open(LNQM_TRAIN))[0]])[0]
    model = make_model(31, 'cpu')
    X = torch.tensor(x0[None, :], requires_grad=True)
    model(X).sum().backward()
    assert float(X.grad.abs().sum()) > 0
    print('  surrogate input gradient nonzero PASS', flush=True)


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_data = []
    for p in [LNQM_TRAIN, LNQM_TEST]:
        all_data.extend(json.load(open(p)))
    ext = json.load(open(LNQM_EXT))
    ext_by_index = {r['index']: r for r in ext}
    part = json.load(open(PART_LNQM))
    tr = [all_data[i] for i in part['train_idx']]
    va = [all_data[i] for i in part['val_idx']]
    te = [all_data[i] for i in part['test_idx']]
    Xtr, Xva, Xte = featurize_global(tr), featurize_global(va), featurize_global(te)
    Y = build_xy(all_data, ext_by_index)
    tr_idx = np.array(part['train_idx']); va_idx = np.array(part['val_idx']); te_idx = np.array(part['test_idx'])
    print(f'loaded | {time.time()-t0:.0f}s', flush=True)

    proxies = {}
    for key in PROXY_TARGETS:
        mask, y = Y[key]
        m_tr, m_va = mask[tr_idx], mask[va_idx]
        _, state, stats = fit_mlp(Xtr[m_tr], y[tr_idx][m_tr], Xva[m_va], y[va_idx][m_va],
                                  None, device=device)
        torch.save({'state': state, 'stats': stats, 'target': key,
                    'y_std': float(y[mask].std()), 'y_mean': float(y[mask].mean())},
                   OUTDIR / f'proxy_{key}.pt')
        proxies[key] = (state, stats)
        print(f'proxy {key} trained, y std {y[mask].std():.4f} | {time.time()-t0:.0f}s', flush=True)

    unit_tests()

    # ---- diagnostic: two targets with different levers (homo via charge; pol_sum via composition?) ----
    for key in ['homo_up', 'pol_sum']:
        state, stats = proxies[key]
        xm, xs, ym, ys = stats
        model = make_model(31, device)
        model.load_state_dict(state)
        model.eval()
        xm_t = torch.tensor(xm, device=device); xs_t = torch.tensor(xs, device=device)
        ym_t = torch.tensor(float(ym), device=device); ys_t = torch.tensor(float(ys), device=device)

        def predict_grad(X):
            return model((X - xm_t) / xs_t).squeeze(1) * ys_t + ym_t

        def predict_np(X):
            with torch.no_grad():
                return predict_grad(torch.tensor(X, device=device)).cpu().numpy()

        mask, y = Y[key]
        yv_all = y[mask]
        # target: take the value of a real test structure (non-trivial, non-median)
        te_first = te_idx[:50][mask[te_idx[:50]]][0]
        target_value = float(y[te_first])
        x0 = featurize_global([all_data[te_idx[1]]])[0]  # start point = a different structure
        pred0 = float(predict_np(x0[None, :])[0])
        std_y = float(yv_all.std())
        print(f'== diagnostic {key}: target {target_value:.4f} | start pred {pred0:.4f} '
              f'(|Δ|/std = {abs(pred0-target_value)/std_y:.3f}) ==', flush=True)
        _, bp, _ = optimize(predict_grad, x0, target_value, budget=800, device=device)
        print(f'  gradient Δ/std {abs(bp-target_value)/std_y:.4f}', flush=True)
        _, bp, _ = random_search(predict_np, x0, target_value, budget=800)
        print(f'  random   Δ/std {abs(bp-target_value)/std_y:.4f}', flush=True)
        _, bp, _ = ga_search(predict_np, x0, target_value, budget=800)
        print(f'  ga       Δ/std {abs(bp-target_value)/std_y:.4f}', flush=True)
        _, bp, _ = greedy_search(predict_np, x0, target_value, budget=800)
        print(f'  greedy   Δ/std {abs(bp-target_value)/std_y:.4f}', flush=True)
    print(f'DONE E04 | {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
