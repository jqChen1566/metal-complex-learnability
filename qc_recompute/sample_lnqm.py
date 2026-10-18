"""Recompute sampling: pick 100 LnQM complexes for independent ORCA re-check.

Design:
  S1 (30): two complete 15-lanthanide scaffold series (same non-metal
           composition/count/charge, metal La..Lu) -- direct cross-metal
           reproducibility series, also reused for multi-reference anchors.
  S2 (70): stratified random, element-balanced (4-5 per lanthanide) with
           greedy charge/unpaired diversity within each element.

Outputs (in RECOMP):
  selected_100.csv      -- uid, h5 index, metal, charge, multiplicity,
                           natoms, h5 reference values (HOMO/LUMO up/dn)
  structures/<idx>.xyz  -- coordinates in Angstrom (h5 stores Bohr)
"""
import csv
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
H5 = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))
RECOMP = Path(os.environ.get("METAL_ML_RECOMP", Path(os.environ.get("METAL_ML_RESULTS", "results")) / "recompute"))
STRUCT = RECOMP / "structures"
RECOMP.mkdir(parents=True, exist_ok=True)
STRUCT.mkdir(parents=True, exist_ok=True)

BOHR2ANG = 0.529177210903
METAL_Z = list(range(57, 72))
SEED = 42

# ---------------------------------------------------------------- load h5
with h5py.File(H5, "r") as f:
    uid_raw = f["data/uid"][:]
    uid = [x.decode() if isinstance(x, bytes) else str(x) for x in uid_raw]
    charge_raw = f["data/charge"][:].ravel()
    # h5 stores charge as uint64: negative values wrap around (uint64 max = 2**64 - 1)
    charge = np.array(
        [int(c) - (1 << 64) if int(c) > (1 << 40) else int(c) for c in charge_raw],
        dtype=np.int64,
    )
    unpaired = f["data/unpaired_e"][:].ravel()
    numbers = f["data/numbers"][:]
    coord = f["data/coord"][:]           # Bohr
    slices = f["slices/numbers"][:].reshape(-1)  # n+1 offsets (top-level 'slices' group)
    homo_up = f["data/homo_spin_up"][:].ravel()
    homo_dn = f["data/homo_spin_down"][:].ravel()
    lumo_up = f["data/lumo_spin_up"][:].ravel()
    lumo_dn = f["data/lumo_spin_down"][:].ravel()

n = len(uid)
assert slices.shape[0] == n + 1, "slices length mismatch"

def mol_z(i):
    return np.array(numbers[slices[i]:slices[i + 1]], dtype=int).reshape(-1)

def mol_coord(i):
    return np.array(coord[slices[i]:slices[i + 1]], dtype=float)

def metal_of(i):
    ms = [int(z) for z in mol_z(i) if int(z) in set(METAL_Z)]
    assert len(ms) == 1, f"complex {i}: {len(ms)} metal atoms"
    return ms[0]

def scaffold_key(i):
    zs = mol_z(i)
    nonmetal = sorted(int(z) for z in zs if int(z) not in set(METAL_Z))
    counts = tuple(sorted(set((z, nonmetal.count(z)) for z in nonmetal)))
    return (counts, len(zs), int(round(charge[i])))

# ------------------------------------------------- build scaffold groups
scaf = defaultdict(list)
for i in range(n):
    scaf[scaffold_key(i)].append(i)

# ------------------------------------------------- S1: full 15-Ln series
rng = np.random.default_rng(SEED)
full_series = []
for key, idxs in scaf.items():
    metals = sorted(metal_of(i) for i in idxs)
    if metals == METAL_Z:  # one complex per lanthanide, complete series
        full_series.append((key, idxs))
full_series.sort(key=lambda kv: len(kv[1]))
print(f"complete 15-Ln series available: {len(full_series)}")

picked = []
series_tag = {}
# choose 2 series with small-ish molecule size for compute feasibility
cand = sorted(full_series, key=lambda kv: np.mean([mol_z(i).size for i in kv[1]]))
for si, (key, idxs) in enumerate(cand[:2]):
    by_metal = {metal_of(i): i for i in idxs}
    for mz in METAL_Z:
        i = by_metal[mz]
        picked.append(i)
        series_tag[i] = f"series{si + 1}"

# ------------------------------------------------- S2: stratified random
chosen_set = set(picked)
per_element = {mz: [] for mz in METAL_Z}
for i in range(n):
    if i in chosen_set:
        continue
    per_element[metal_of(i)].append(i)

# target: 70 total across 15 elements -> rotating quota 4 or 5
quotas = [5 if k < 70 - 4 * 15 else 4 for k in range(15)]
rng.shuffle(quotas)

for mz, quota in zip(METAL_Z, quotas):
    cands = per_element[mz]
    rng.shuffle(cands)
    # greedy: prefer new (charge, unpaired) combos for diversity
    seen = set()
    first, rest = [], []
    for i in cands:
        combo = (int(round(charge[i])), int(unpaired[i]))
        if combo not in seen:
            seen.add(combo)
            first.append(i)
        else:
            rest.append(i)
    sel = first[:quota]
    if len(sel) < quota:
        sel += rest[: quota - len(sel)]
    picked.extend(sel)

assert len(picked) == 100, f"picked {len(picked)}"

# ------------------------------------------------- outputs
rows = []
for i in picked:
    zs = mol_z(i)
    rows.append({
        "uid": uid[i],
        "h5_index": i,
        "metal": metal_of(i),
        "charge": int(round(charge[i])),
        "mult": int(unpaired[i]) + 1,
        "unpaired": int(unpaired[i]),
        "natoms": int(zs.size),
        "homo_up_h5": float(homo_up[i]),
        "homo_dn_h5": float(homo_dn[i]),
        "lumo_up_h5": float(lumo_up[i]),
        "lumo_dn_h5": float(lumo_dn[i]),
        "series": series_tag.get(i, "random"),
    })

with open(RECOMP / "selected_100.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

for r in rows:
    i = r["h5_index"]
    xyz_ang = mol_coord(i) * BOHR2ANG
    with open(STRUCT / f"{i:05d}.xyz", "w") as fh:
        fh.write(f"{xyz_ang.shape[0]}\nuid={r['uid']} charge={r['charge']} mult={r['mult']}\n")
        for (x, y, z), za in zip(xyz_ang, mol_z(i)):
            fh.write(f"{za:3d} {x:16.8f} {y:16.8f} {z:16.8f}\n")

# quick summary
chg = np.array([r["charge"] for r in rows])
print("charge distribution:", {int(c): int((chg == c).sum()) for c in sorted(set(chg))})
print("natoms range:", min(r["natoms"] for r in rows), "-", max(r["natoms"] for r in rows))
print("series picked:", sum(1 for r in rows if r["series"].startswith("series")))
print(f"wrote {RECOMP / 'selected_100.csv'} and {len(rows)} xyz files")
