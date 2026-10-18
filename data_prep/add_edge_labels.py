# Add edge-level labels to the lnqm graphs: Mayer bond orders.
# h5 data/mayer_bo is split per system by slices/mayer_bo; each row is
# (atom i, atom j, bond order), with i < j and 0-based atom indices within the
# system. The recorded lower bound is |BO| >= 0.1 (global minimum |BO| =
# 0.100000), i.e. the record set holds atom pairs whose |BO| reaches 0.1;
# unrecorded pairs have |BO| < 0.1 with unknown exact values.
# The graph file stores per-system atom pairs and bond-order values
# (bo_pairs / bo_vals); training builds per-edge labels and masks by lookup
# from graph edges, so labels do not depend on the edge generation order.
# Also exports the edge-label pack edge_labels_lnqm.pt for downstream injection.
import os
from pathlib import Path

import hashlib
import h5py
import numpy as np
import torch

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
GRAPH_DIR = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
H5 = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))
BO_MIN = 0.1
BOHR2ANG = 0.529177210903   # graph pos is converted from h5 coord (Bohr) to Angstrom

with h5py.File(H5, "r") as f:
    bo = f["data/mayer_bo"][:]
    sl = f["slices/mayer_bo"][:].reshape(-1)
    nsl = f["slices/numbers"][:].reshape(-1)
    hnum = f["data/numbers"][:].reshape(-1)
    hcoord = f["data/coord"][:]

gpath = GRAPH_DIR / "graphs_lnqm.pt"
d = torch.load(gpath, weights_only=False)
n = len(d["z"])
assert n == len(nsl) - 1, f"system count mismatch: graph {n}, h5 {len(nsl) - 1}"
assert n == len(sl) - 1, f"system count mismatch: graph {n}, mayer_bo slices {len(sl) - 1}"
assert abs(bo[:, 2]).min() >= BO_MIN - 1e-12, "bond-order lower bound does not match expectation"
assert (bo[:, 0] < bo[:, 1]).all(), "found records with i >= j"

pairs, vals = [], []
for i in range(n):
    a, b = int(nsl[i]), int(nsl[i + 1])
    za = d["z"][i].numpy()
    assert b - a == len(za), f"system {i}: atom count mismatch"
    assert (za == hnum[a:b]).all(), f"system {i}: element sequence does not match h5"
    dp = np.abs(d["pos"][i].numpy().astype(np.float64)
                - hcoord[a:b] * BOHR2ANG).max()
    assert dp < 1e-4, f"system {i}: coordinates do not match h5 (max diff {dp})"
    s, t = int(sl[i]), int(sl[i + 1])
    rec = bo[s:t]
    pairs.append(torch.tensor(rec[:, :2], dtype=torch.int32))
    vals.append(torch.tensor(rec[:, 2], dtype=torch.float32))

d["bo_pairs"] = pairs
d["bo_vals"] = vals
tmp = gpath.with_suffix(".tmp_edgelab.pt")
torch.save(d, tmp)
chk = torch.load(tmp, weights_only=False)
assert len(chk["bo_pairs"]) == n and len(chk["bo_vals"]) == n
assert len(chk["bo_pairs"][0]) == len(chk["bo_vals"][0])
os.replace(tmp, gpath)

pack = GRAPH_DIR / "edge_labels_lnqm.pt"


def fp(t):
    # per-system structural fingerprint (byte digest of the element-sequence or
    # coordinate tensor); the injection side uses it to verify graph/label match.
    h = hashlib.md5()
    h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


torch.save({"bo_pairs": pairs, "bo_vals": vals,
            "z_fp": [fp(d["z"][i]) for i in range(n)],
            "pos_fp": [fp(d["pos"][i]) for i in range(n)],
            "source": "lnqm.h5 data/mayer_bo, |BO| >= 0.1"}, pack)

rec_n = np.array([len(t) for t in pairs])
print("edge labels written:", n, "systems")
print("total records:", int(rec_n.sum()))
print("records per system min/median/max:", int(rec_n.min()),
      int(np.median(rec_n)), int(rec_n.max()))
allv = torch.cat(vals)
print("bond-order range: %.4f to %.4f" % (float(allv.min()), float(allv.max())))
print("negative fraction: %.4f" % float((allv < 0).float().mean()))
ln_pairs = 0
ln_vals = []
for i in range(n):
    z = d["z"][i]
    m = (z >= 57) & (z <= 71)
    if not bool(m.any()):
        continue
    p = pairs[i]
    sel = m[p[:, 0].long()] | m[p[:, 1].long()]
    ln_pairs += int(sel.sum())
    ln_vals.append(vals[i][sel])
ln_vals = torch.cat(ln_vals)
print("atom pairs containing a lanthanide:", ln_pairs)
print("bond-order median/mean among them: %.4f / %.4f" % (float(ln_vals.median()),
                                                          float(ln_vals.mean())))
print("exported:", pack)
