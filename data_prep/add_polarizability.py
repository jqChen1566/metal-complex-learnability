"""Add y_pol (per-system polarizability, a.u.) to an existing graph file.

y_pol[i] = sum of atomic polarizabilities of system i, from h5
data/polarizabilities via slices/polarizabilities.  Graph row order follows
h5 row order (verified), and the sum equals lnqm_properties_ext.json pol_sum
(diff = 0 on spot checks).  Rows are NOT filtered: y_pol aligns with y_homo.

Atomic replace: save temp -> verify load -> os.replace, so a training job
launching concurrently either sees the old or the new file, never a partial.

Usage: python add_polarizability.py <graph.pt> <h5>
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


def main():
    gpath = Path(sys.argv[1])
    h5path = sys.argv[2]
    with h5py.File(h5path, "r") as f:
        pol = f["data/polarizabilities"][:]
        sl = f["slices/polarizabilities"][:].reshape(-1)
    nsys = len(sl) - 1
    y_pol = np.array([pol[int(sl[i]):int(sl[i + 1])].sum() for i in range(nsys)],
                     dtype=np.float32)

    d = torch.load(gpath, weights_only=False)
    n = len(d["z"])
    if nsys != n:
        raise SystemExit(f"row mismatch: h5 systems {nsys} vs graph {n}")

    d["y_pol"] = torch.tensor(y_pol, dtype=torch.float32)
    tmp = gpath.with_suffix(".tmp_addpol.pt")
    torch.save(d, tmp)
    chk = torch.load(tmp, weights_only=False)
    assert "y_pol" in chk and "y_homo" in chk and len(chk["z"]) == n
    assert abs(float(chk["y_pol"][0]) - float(y_pol[0])) < 1e-6
    os.replace(tmp, gpath)
    print(f"OK: y_pol added to {gpath} (n={n}); "
          f"range {y_pol.min():.2f}..{y_pol.max():.2f} a.u.; "
          f"mean {y_pol.mean():.2f}")


if __name__ == "__main__":
    main()
