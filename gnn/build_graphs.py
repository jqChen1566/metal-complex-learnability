"""Graph construction: build a PyG-ready graph dataset from LnQM (h5) for the geometry-model
benchmark (XYZ+BO GNN, 3D-GCN, 3D-GAT, SchNet, DimeNet++, GIN/GCN,
SpookyNet features).

- nodes: element index z (embedding at train time)
- edges: built on-the-fly during training via radius_graph(r=5.0 A) from pos
- target: HOMO alpha in eV (primary); h5 stores Hartree
- split: scaffold partition with the DOUBLE-LAYER mapping
  (scaffold idx -> all_data row -> h5 index); sanity check: 790 Gd in train.

Output: graphs_lnqm.pt  (dict of tensors) + split masks + summary.
"""
import json
import os
import numpy as np
import h5py
import torch
from pathlib import Path

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
OUT = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
H5 = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))
OUT.mkdir(parents=True, exist_ok=True)
EV = 27.211386245988
BOHR2ANG = 0.529177210903

# ---- load structures + targets
with h5py.File(H5, "r") as f:
    numbers = f["data/numbers"][:]
    coord = f["data/coord"][:]              # Bohr
    slices = f["slices/numbers"][:].reshape(-1)
    homo_up = f["data/homo_spin_up"][:].ravel()
    homo_dn = f["data/homo_spin_down"][:].ravel()
    lumo_up = f["data/lumo_spin_up"][:].ravel()
    lumo_dn = f["data/lumo_spin_down"][:].ravel()
    dip = f["data/dipole_tot"][:]  # (n, 3) vector
    unpaired = f["data/unpaired_e"][:].ravel().astype(int)
    chg_raw = f["data/charge"][:].ravel()
n = len(slices) - 1
charge = np.array([int(c) - (1 << 64) if int(c) > (1 << 40) else int(c)
                   for c in chg_raw])
# unified single-value calibers: homo=max(up,dn), lumo=min(up,dn), gap=lumo-homo (Hartree)
homo_s = np.maximum(homo_up, homo_dn)
lumo_s = np.minimum(lumo_up, lumo_dn)
gap_s = lumo_s - homo_s
dipole_mag = np.linalg.norm(dip, axis=1)  # Debye

# ---- scaffold partition with double-layer mapping
part = json.load(open(DATA / "partitions/lnqm_scaffold_12444.json"))
_all = []
for p in (DATA / "lnqm_properties_train.json", DATA / "lnqm_properties_test.json"):
    _all.extend(json.load(open(p)))
ad = np.array([r["index"] for r in _all], dtype=int)
train_rows = ad[np.array(part["train_idx"], dtype=int)]
val_rows = ad[np.array(part["val_idx"], dtype=int)]
test_rows = ad[np.array(part["test_idx"], dtype=int)]

assert int((unpaired[train_rows] == 7).sum()) == 790, "double-layer mapping broken!"

# ---- pack per-molecule arrays
zs, pos, y = [], [], []
for i in range(n):
    a, b = slices[i], slices[i + 1]
    zs.append(torch.tensor(numbers[a:b].reshape(-1), dtype=torch.long))
    pos.append(torch.tensor(coord[a:b] * BOHR2ANG, dtype=torch.float32))
    y.append((homo_up[i] - homo_up.mean()) * EV)  # center for sanity; raw kept too
y_raw = torch.tensor(homo_up * EV, dtype=torch.float32)

torch.save({
    "z": zs,                    # list of (Ni,) long
    "pos": pos,                 # list of (Ni,3) float32, Angstrom
    "y_eV": y_raw,              # (n,) raw HOMO alpha in eV (legacy key for graph-model runs)
    "y_homo": torch.tensor(homo_s * EV, dtype=torch.float32),
    "y_lumo": torch.tensor(lumo_s * EV, dtype=torch.float32),
    "y_gap": torch.tensor(gap_s * EV, dtype=torch.float32),
    "y_dipole": torch.tensor(dipole_mag, dtype=torch.float32),
    "charge": torch.tensor(charge, dtype=torch.long),
    "unpaired": torch.tensor(unpaired, dtype=torch.long),
    "train_rows": torch.tensor(train_rows),
    "val_rows": torch.tensor(val_rows),
    "test_rows": torch.tensor(test_rows),
}, OUT / "graphs_lnqm.pt")

print(f"packed {n} molecules | natoms {min(len(p) for p in pos)}-{max(len(p) for p in pos)}")
print(f"split: train {len(train_rows)} | val {len(val_rows)} | test {len(test_rows)}")
print(f"target y_eV: mean {y_raw.mean():.3f} std {y_raw.std():.3f} | Gd check 790 OK")
print(f"-> {OUT / 'graphs_lnqm.pt'}")
