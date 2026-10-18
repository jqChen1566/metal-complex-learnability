"""Unified composition-RF baselines for all targets x series.

Features : element-count vector (dim 120) + formal charge + unpaired
           electrons (total charge/spin are dominant for frontier orbitals
           in this project; a counts-only baseline is unfairly weak).
Model    : RandomForestRegressor(500 trees, seed 42, n_jobs=8).
Split    : the same scaffold partition as every graph-model run (train/test rows).
Targets  : homo/lumo/gap/dipole/pol, whatever the graph carries.
Output   : result_rfcomp_<series>_<target>.json (test MAE + n's).
Note     : provides a reproducible baseline in place of the previously
           quoted 0.418 anchor; homo numbers are cross-checked against it.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
GRAPH_DIR = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
SERIES = ["lnqm", "3d", "4d", "5d", "5f"]
TARGETS = ["homo", "lumo", "gap", "dipole", "pol"]


def feats(zs, chg=None, unp=None):
    X = np.zeros((len(zs), 122), dtype=np.float32)
    for i, z in enumerate(zs):
        for zi in z.tolist():
            if 0 <= zi < 120:
                X[i, zi] += 1
    if chg is not None:
        X[:, 120] = chg
    if unp is not None:
        X[:, 121] = unp
    return X


def main():
    for s in SERIES:
        g = GRAPH_DIR / ("graphs_lnqm.pt" if s == "lnqm" else f"graphs_{s}.pt")
        if not g.exists():
            print(f"{s}: no graph, skip")
            continue
        d = torch.load(g, weights_only=False)
        chg = d["charge"].numpy() if "charge" in d else None
        unp = d["unpaired"].numpy() if "unpaired" in d else None
        X = feats(d["z"], chg, unp)
        tr = d["train_rows"].numpy()
        te = d["test_rows"].numpy()
        for t in TARGETS:
            key = f"y_{t}"
            if key not in d:
                continue
            y = d[key].numpy()
            tr2, te2 = tr, te
            if not np.isfinite(y[tr]).all() or not np.isfinite(y[te]).all():
                mtr, mte = np.isfinite(y[tr]), np.isfinite(y[te])
                tr2, te2 = tr[mtr], te[mte]
            m = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=8)
            m.fit(X[tr2], y[tr2])
            p = m.predict(X[te2])
            mae = float(np.abs(p - y[te2]).mean())
            res = {"model": "RF-composition", "series": s, "target": t,
                   "n_train": int(len(tr2)), "n_test": int(len(te2)),
                   "test_mae": round(mae, 4),
                   "features": "element counts (120) + charge + unpaired",
                   "rf": "500 trees, seed 42"}
            (GRAPH_DIR / f"result_rfcomp_{s}_{t}.json").write_text(
                json.dumps(res, indent=2))
            print(f"{s:5s} {t:7s}: test MAE {mae:9.4f}  "
                  f"(n_tr {len(tr2)}, n_te {len(te2)})", flush=True)


if __name__ == "__main__":
    main()
