"""Physics-inspired descriptor baselines (SOAP + KRR) on LnQM.

Fixed geometric descriptors + kernel
regression (mirrors Cho et al.'s representation family).

- descriptor: dscribe SOAP (per-atom, mean-pooled), rcut=5.0, nmax=6, lmax=4
- model: sklearn KernelRidge (laplacian), alpha chosen on val
- target: HOMO alpha (eV); same scaffold split as all graph-model experiments

usage:
  python soap_krr.py --subset 200          # quick self-test
  python soap_krr.py --cache full          # full run (parallel)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase.data import chemical_symbols
from ase import Atoms
from dscribe.descriptors import SOAP
from joblib import Parallel, delayed
from sklearn.kernel_ridge import KernelRidge

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
OUT = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
RCUT, NMAX, LMAX = 5.0, 6, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--cache", default=None, help="tag for descriptor cache name")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--series", default="lnqm")
    args = ap.parse_args()

    gpath = (OUT / "graphs_lnqm.pt" if args.series == "lnqm"
             else OUT / f"graphs_{args.series}.pt")
    d = torch.load(gpath, weights_only=False)
    zs, pos, y = d["z"], d["pos"], d["y_homo"]
    tr, va, te = d["train_rows"], d["val_rows"], d["test_rows"]
    if args.subset:
        g = torch.Generator().manual_seed(42)
        tr = tr[torch.randperm(len(tr), generator=g)[:args.subset]]
        va = va[: max(100, args.subset // 8)]
        te = te[: max(100, args.subset // 8)]

    symbols = sorted({chemical_symbols[int(z)] for z in torch.unique(
        torch.cat([zs[i] for i in range(len(zs))]))})
    print(f"species ({len(symbols)}): {symbols}", flush=True)

    soap = SOAP(species=symbols, r_cut=RCUT, n_max=NMAX, l_max=LMAX, periodic=False,
                sparse=False)

    cache = OUT / f"X_soap_{args.cache or 'tmp'}_{args.series}.npy"
    part = OUT / f"X_soap_{args.cache or 'tmp'}_{args.series}_partial.npy"
    if args.cache and cache.exists():
        X = np.load(cache)
        print(f"loaded descriptor cache {cache} shape {X.shape}", flush=True)
    else:
        t0 = time.time()

        def one(i):
            at = Atoms(numbers=zs[i].numpy(), positions=pos[i].numpy())
            return soap.create(at).mean(axis=0)

        need = np.concatenate([tr.numpy(), va.numpy(), te.numpy()])
        # chunked with a cumulative partial cache: an interrupted run resumes
        # from the last completed chunk instead of losing everything.
        done0 = 0
        if part.exists():
            Xacc = np.load(part)
            done0 = Xacc.shape[0]
            print(f"resume from partial {part.name}: {done0}/{len(need)} done",
                  flush=True)
        else:
            Xacc = np.zeros((0, 0), dtype=np.float32)
        CHUNK = 2000
        for start in range(done0, len(need), CHUNK):
            idx = need[start:start + CHUNK]
            Xs = Parallel(n_jobs=args.jobs, verbose=1)(
                delayed(one)(int(i)) for i in idx)
            Xc = np.asarray(Xs, dtype=np.float32)
            Xacc = Xc if Xacc.size == 0 else np.concatenate([Xacc, Xc])
            np.save(part, Xacc)
            print(f"chunk done: {Xacc.shape[0]}/{len(need)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        X = Xacc
        if args.cache:
            np.save(cache, X)
            if part.exists():
                part.unlink()
        print(f"descriptors done: {X.shape} in {time.time()-t0:.0f}s", flush=True)

    ntr, nva, nte = len(tr), len(va), len(te)
    Xtr, Xva, Xte = X[:ntr], X[ntr:ntr + nva], X[ntr + nva:]
    ytr = y[tr].numpy(); yva = y[va].numpy(); yte = y[te].numpy()
    ym, ys = ytr.mean(), ytr.std()
    ytr_n, yva_n = (ytr - ym) / ys, (yva - ym) / ys

    best = (1e9, None)
    alphas = [args.alpha] if args.alpha else [1e-3, 1e-2, 1e-1, 1.0]
    for a in alphas:
        krr = KernelRidge(kernel="laplacian", alpha=a)
        krr.fit(Xtr, ytr_n)
        vm = np.abs(krr.predict(Xva) - yva_n).mean() * ys
        print(f"  alpha={a}: val MAE {vm:.4f} eV", flush=True)
        if vm < best[0]:
            best = (vm, a)
    vmae, a = best
    krr = KernelRidge(kernel="laplacian", alpha=a)
    krr.fit(Xtr, ytr_n)
    pred = krr.predict(Xte) * ys + ym
    tmae = float(np.abs(pred - yte).mean())
    res = {"model": "SOAP+KRR", "subset": args.subset or len(tr),
           "alpha": a, "best_val_mae_eV": round(float(vmae), 4),
           "test_mae_eV": round(tmae, 4), "anchor_composition_rf": 0.418,
           "descriptor_dim": int(X.shape[1])}
    tag = f"{args.series}_{'smoke' if args.subset else 'full'}"
    (OUT / f"result_soap_{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez(OUT / f"preds_soap_{tag}.npz", pred=pred, true=yte)
    print(f"== SOAP+KRR ({tag}) == val {vmae:.4f}  TEST {tmae:.4f} eV "
          f"(anchor RF 0.418) | dim {X.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
