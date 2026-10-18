"""SpookyNet (pretrained parameters.pth) frozen features
+ Ridge regression for HOMO alpha on LnQM.

Pipeline per molecule:
  ASE Atoms (Angstrom) + charge/magmom (from h5)
  -> SpookyNetCalculator -> atomic features f (N, 64) from results["features"]
  -> [mean(f), max(f), Ln-mean(f)]  (192-dim)
Downstream: Ridge (alpha selected on val). Same scaffold split as all graph-model runs.

usage:
  python spookynet_features.py --subset 200     # quick self-test
  python spookynet_features.py --cache full     # full run
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import Atoms

SPOOKY = os.environ.get("SPOOKYNET_DIR", "")
sys.path.insert(0, SPOOKY)
from spookynet import SpookyNetCalculator  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
OUT = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
CKPT = os.path.join(SPOOKY, "parameters.pth")
H5 = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--cache", default=None)
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

    chg = d["charge"].numpy() if "charge" in d else np.zeros(len(zs), dtype=int)
    unpaired = (d["unpaired"].numpy() if "unpaired" in d
                else np.zeros(len(zs), dtype=int))

    need = np.concatenate([tr.numpy(), va.numpy(), te.numpy()])
    cache = OUT / f"X_spooky_{args.cache or 'tmp'}_{args.series}.npy"
    if args.cache and cache.exists():
        X = np.load(cache)
        print(f"loaded feature cache {cache} shape {X.shape}", flush=True)
    else:
        try:
            calc = SpookyNetCalculator(load_from=CKPT, charge=0, magmom=0,
                                       use_gpu=True)
            print("SpookyNet on GPU", flush=True)
        except Exception as e:
            print(f"gpu init failed ({e}); cpu mode", flush=True)
            calc = SpookyNetCalculator(load_from=CKPT, charge=0, magmom=0,
                                       use_gpu=False)
        t0 = time.time()
        feats = []
        for k, i in enumerate(need.tolist()):
            zi = zs[i]
            calc.parameters.charge = int(chg[i])
            calc.parameters.magmom = int(unpaired[i])
            atoms = Atoms(numbers=zi.numpy(), positions=pos[i].numpy())
            atoms.calc = calc
            atoms.get_potential_energy()          # triggers calculate
            f = calc.results["features"]          # (N, 64) numpy
            is_ln = (zi.numpy() >= 57) & (zi.numpy() <= 71)
            feats.append(np.concatenate([
                f.mean(0), f.max(0),
                f[is_ln].mean(0) if is_ln.any() else np.zeros(f.shape[1])]))
            if (k + 1) % 500 == 0:
                print(f"  {k+1}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
        X = np.asarray(feats, dtype=np.float32)
        if args.cache:
            np.save(cache, X)
        print(f"features done: {X.shape} in {time.time()-t0:.0f}s", flush=True)

    ntr, nva, nte = len(tr), len(va), len(te)
    Xtr, Xva, Xte = X[:ntr], X[ntr:ntr + nva], X[ntr + nva:]
    ytr, yva, yte = y[tr].numpy(), y[va].numpy(), y[te].numpy()

    best = (1e9, None)
    for a in [1e-2, 1e-1, 1.0, 10.0]:
        rg = Ridge(alpha=a).fit(Xtr, ytr)
        vm = float(np.abs(rg.predict(Xva) - yva).mean())
        print(f"  alpha={a}: val MAE {vm:.4f} eV", flush=True)
        if vm < best[0]:
            best = (vm, a)
    vmae, a = best
    rg = Ridge(alpha=a).fit(Xtr, ytr)
    pred = rg.predict(Xte)
    tmae = float(np.abs(pred - yte).mean())
    res = {"model": "SpookyNet-feat+Ridge", "subset": args.subset or ntr,
           "alpha": a, "best_val_mae_eV": round(vmae, 4),
           "test_mae_eV": round(tmae, 4), "feat_dim": int(X.shape[1]),
           "anchor_composition_rf": 0.418}
    tag = f"{args.series}_{'smoke' if args.subset else 'full'}"
    (OUT / f"result_spooky_{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez(OUT / f"preds_spooky_{tag}.npz", pred=pred, true=yte)
    print(f"== SpookyNet+Ridge ({tag}) == val {vmae:.4f}  TEST {tmae:.4f} eV "
          f"(anchor RF 0.418)", flush=True)


if __name__ == "__main__":
    main()
