# Atomic additivity baseline for polarizability: estimate the mean atomic
# polarizability per element from the training set and predict test systems by
# summing over their atoms. The baseline measures how much of the
# polarizability learnability is explained by local additivity.
import os
from pathlib import Path

import h5py
import numpy as np
import torch

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
GRAPH_DIR = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
H5 = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))

with h5py.File(H5, "r") as f:
    pol = f["data/polarizabilities"][:]
    numbers = f["data/numbers"][:].reshape(-1)
    nsl = f["slices/numbers"][:].reshape(-1)

d = torch.load(GRAPH_DIR / "graphs_lnqm.pt", weights_only=False)
y_pol = d["y_pol"].numpy()
tr = d["train_rows"].numpy()
te = d["test_rows"].numpy()

per_elem = {}
for i in tr:
    a, b = int(nsl[i]), int(nsl[i + 1])
    for z, p in zip(numbers[a:b], pol[a:b]):
        per_elem.setdefault(int(z), []).append(float(p))
mean_p = {z: float(np.mean(v)) for z, v in per_elem.items()}

pred = np.zeros(len(te))
for k, i in enumerate(te):
    a, b = int(nsl[i]), int(nsl[i + 1])
    pred[k] = sum(mean_p.get(int(z), 0.0) for z in numbers[a:b])
mae = float(np.abs(pred - y_pol[te]).mean())
print("atom-additivity MAE (a.u.):", round(mae, 4), "| n_train:", len(tr), "| n_test:", len(te))
