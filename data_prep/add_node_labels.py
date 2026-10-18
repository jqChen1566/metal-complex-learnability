# Add node-level labels to the lnqm graphs: Hirshfeld atomic charges and spins.
# Atom order matches z and pos; assertions below check the row count and the
# per-system atom count.
import os
from pathlib import Path

import h5py
import torch

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
GRAPH_DIR = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
H5 = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))

with h5py.File(H5, "r") as f:
    hq = f["data/hirshfeld_charges"][:].reshape(-1)
    hs = f["data/hirshfeld_spins"][:].reshape(-1)
    nsl = f["slices/numbers"][:].reshape(-1)

gpath = GRAPH_DIR / "graphs_lnqm.pt"
d = torch.load(gpath, weights_only=False)
n = len(d["z"])
assert n == len(nsl) - 1, f"row count mismatch: graph {n}, h5 {len(nsl) - 1}"

yq, ys = [], []
for i in range(n):
    a, b = int(nsl[i]), int(nsl[i + 1])
    assert b - a == len(d["z"][i]), f"system {i}: atom count mismatch"
    yq.append(torch.tensor(hq[a:b], dtype=torch.float32))
    ys.append(torch.tensor(hs[a:b], dtype=torch.float32))

d["y_hirsh_q"] = yq
d["y_hirsh_s"] = ys
tmp = gpath.with_suffix(".tmp_nodelab.pt")
torch.save(d, tmp)
chk = torch.load(tmp, weights_only=False)
assert len(chk["y_hirsh_q"]) == n
assert len(chk["y_hirsh_q"][0]) == len(chk["z"][0])
os.replace(tmp, gpath)
print("node labels written:", n, "systems")
print("charge range: %.3f to %.3f" % (min(float(t.min()) for t in yq),
                                      max(float(t.max()) for t in yq)))
print("spin range: %.3f to %.3f" % (min(float(t.min()) for t in ys),
                                    max(float(t.max()) for t in ys)))
