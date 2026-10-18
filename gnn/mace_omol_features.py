"""MACE-OMol-0 (charge+spin embedding foundation model) frozen features
+ Ridge regression for HOMO alpha on LnQM.

- model: mace_omol local weights (extra-large-1024 by default; 4M variant option)
- charge/spin injected via atoms.info (MACE convention) from h5 fields
- atomic features captured at model.interactions[-1] via forward hook
- pooled [mean, max, Ln-mean] -> Ridge (alpha on val), same split as all graph-model runs

usage:
  python mace_omol_features.py --subset 100          # quick self-test
  python mace_omol_features.py --cache full          # full run
"""
import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import Atoms

from mace.calculators import mace_omol
from sklearn.linear_model import Ridge

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
OUT = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
H5 = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--variant", default="1024", choices=["1024", "4M"])
    ap.add_argument("--series", default="lnqm",
                    choices=["lnqm", "3d", "4d", "5d", "5f"])
    args = ap.parse_args()

    gpath = (OUT / "graphs_lnqm.pt" if args.series == "lnqm"
             else OUT / f"graphs_{args.series}.pt")
    d = torch.load(gpath, weights_only=False)
    zs, pos, y = d["z"], d["pos"], d["y_eV"]
    tr, va, te = d["train_rows"], d["val_rows"], d["test_rows"]
    if args.subset:
        g = torch.Generator().manual_seed(42)
        tr = tr[torch.randperm(len(tr), generator=g)[:min(args.subset, len(tr))]]
    chg = d["charge"].numpy() if "charge" in d else np.zeros(len(zs), dtype=int)
    unpaired = (d["unpaired"].numpy() if "unpaired" in d
                else np.zeros(len(zs), dtype=int))

    need = np.concatenate([tr.numpy(), va.numpy(), te.numpy()])
    tagv = f"{args.variant}_{args.series}"
    cache = OUT / f"X_maceomol{args.variant}_{args.cache or 'tmp'}_{args.series}.npy"
    if args.cache and cache.exists():
        X = np.load(cache)
        print(f"loaded feature cache {cache} shape {X.shape}", flush=True)
    else:
        ckpt = str(OUT / f"models/MACE-omol-0-extra-large-{args.variant}.model")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_omol(model=ckpt, device=dev, default_dtype="float64")
        if hasattr(calc, "models"):
            model = calc.models[0]
        else:
            model = torch.load(ckpt, map_location=dev, weights_only=False).to(dev).eval()
        print(f"mace_omol {tagv} loaded on {dev}", flush=True)

        buf = []
        # first interaction block: healthy scale (absmean ~0.2) and strongest
        # signal; deep blocks explode (1e12+) under OMol's large-norm weights.
        handle = model.interactions[0].register_forward_hook(
            lambda m, i, o: buf.append((o[0] if isinstance(o, tuple) else o).detach()))

        t0 = time.time()
        feats = []
        for k, i in enumerate(need.tolist()):
            at = Atoms(numbers=zs[i].numpy(), positions=pos[i].numpy())
            at.info["charge"] = int(chg[i])
            at.info["spin"] = int(unpaired[i])
            try:
                batch = calc._atoms_to_batch(at).to(dev)
            except Exception as e:
                if k < 3:
                    print(f"  [warn] atoms_to_batch failed @{i}: {e}", flush=True)
                feats.append(np.zeros(3 * 1024, dtype=np.float32))
                continue
            buf.clear()
            with torch.no_grad():
                model(batch, compute_force=False, compute_virials=False,
                      compute_stress=False)
            if not buf:
                if k < 3:
                    print(f"  [warn] empty feature buffer @{i}", flush=True)
                feats.append(np.zeros(3 * 1024, dtype=np.float32))
                continue
            f = buf[0].cpu().numpy()
            if f.ndim == 3:          # (N, channels, L) -> take L=0 scalar part
                f = f[:, :, 0]
            zi = zs[i].numpy()
            is_ln = (zi >= 57) & (zi <= 71)
            feats.append(np.concatenate([
                f.mean(0), f.max(0),
                f[is_ln].mean(0) if is_ln.any() else np.zeros(f.shape[1])]))
            if (k + 1) % 500 == 0:
                print(f"  {k+1}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
            if k == 0:
                print(f"  [selfcheck] first-mol feat absmean {np.abs(f).mean():.3e}",
                      flush=True)
        handle.remove()
        del model, calc
        torch.cuda.empty_cache()
        X = np.asarray(feats, dtype=np.float32)
        if args.cache:
            np.save(cache, X)
        print(f"features done: {X.shape} in {time.time()-t0:.0f}s", flush=True)

    ntr, nva, nte = len(tr), len(va), len(te)
    Xtr, Xva, Xte = X[:ntr], X[ntr:ntr + nva], X[ntr + nva:]
    ytr, yva, yte = y[tr].numpy(), y[va].numpy(), y[te].numpy()
    # OMol deep-layer features come at huge scale (~1e12); standardize w/ train stats
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    print(f"feature scale after standardize: std min/max "
          f"{Xtr.std(0).min():.3f}/{Xtr.std(0).max():.3f}", flush=True)

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
    res = {"model": f"OMol-0-{args.variant}-feat+Ridge", "series": args.series,
           "subset": args.subset or ntr,
           "alpha": a, "best_val_mae_eV": round(vmae, 4),
           "test_mae_eV": round(tmae, 4), "feat_dim": int(X.shape[1]),
           "anchor_composition_rf": 0.418}
    tag = f"{args.series}_{'smoke' if args.subset else 'full'}"
    (OUT / f"result_maceomol_{args.variant}_{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez(OUT / f"preds_maceomol_{args.variant}_{tag}.npz", pred=pred, true=yte)
    print(f"== MACE-OMol-0({tagv})+Ridge ({tag}) == val {vmae:.4f}  TEST {tmae:.4f} eV "
          f"(anchor RF 0.418)", flush=True)


if __name__ == "__main__":
    main()
