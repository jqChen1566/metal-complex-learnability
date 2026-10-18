"""Build graph datasets for the non-4f series (3d / 4d / 5d / 5f).

- 3d/4d/5d: coordinates parsed from data/tmqm_raw2024/tmQM_X{1,2,3}.xyz.gz and
  paired by CSD code with data/tmqm_properties{,_4d,_5d}.json. The scaffold
  partition idx (partitions/tmqm*_scaffold_*.json) indexes the properties list
  directly (verified by range assertion).
- 5f: data/acqm/acqm_full.extxyz (ASE, Angstrom) + data/acqm_properties.json;
  pairing asserted order-wise on atomic_numbers.
- target: homo (Hartree in source) -> eV (x EV_PER_HA); acqm takes
  max(homo_up, homo_dn) per the unified caliber.
Output: graphs_<series>.pt  (same dict format as graphs_lnqm.pt)
"""
import gzip
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from ase.io import read

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
OUT = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
EV_PER_HA = 27.211386245988

SERIES = {
    "3d": dict(props="tmqm_properties.json", part="tmqm_scaffold_33086.json"),
    "4d": dict(props="tmqm_properties_4d.json", part="tmqm4d_scaffold_37450.json"),
    "5d": dict(props="tmqm_properties_5d.json", part="tmqm5d_scaffold_29292.json"),
}
ELEMENT_Z = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16,
             'Cl': 17, 'Br': 35, 'I': 53}


def parse_tmqm_coords():
    """csd -> list of (z, x, y, z) rows from the three xyz.gz parts."""
    coords = {}
    for fname in ["tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz"]:
        with gzip.open(DATA / "tmqm_raw2024" / fname, "rt") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    natoms = int(line)
                except ValueError:
                    continue
                comment = f.readline().strip()
                m = re.search(r"CSD_code\s*=\s*(\S+)", comment)
                csd = m.group(1) if m else None
                rows = []
                for _ in range(natoms):
                    parts = f.readline().split()
                    if len(parts) < 4:
                        continue
                    z = ELEMENT_Z.get(parts[0], 0)
                    if z == 0:  # metal or other heavy element: use periodic map below
                        z = _z_of(parts[0])
                    rows.append((z, float(parts[1]), float(parts[2]), float(parts[3])))
                if csd:
                    coords[csd] = rows
    return coords


def _z_of(sym):
    table = {'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27,
             'Ni': 28, 'Cu': 29, 'Zn': 30, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
             'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
             'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
             'Au': 79, 'Hg': 80}
    return table.get(sym, 0)


def build_tmqm(name, cfg):
    structs = json.load(open(DATA / cfg["props"]))
    coords = parse_tmqm_coords()
    part = json.load(open(DATA / "partitions" / cfg["part"]))
    zs, pos, y = [], [], []
    missing = 0
    for d in structs:
        c = coords.get(d["csd"])
        if c is None:
            missing += 1
            zs.append(torch.zeros(1, dtype=torch.long))
            pos.append(torch.zeros(1, 3))
            y.append(0.0)
            continue
        zs.append(torch.tensor([r[0] for r in c], dtype=torch.long))
        pos.append(torch.tensor([[r[1], r[2], r[3]] for r in c], dtype=torch.float32))
        y.append(d["homo"] * EV_PER_HA)
    tr = np.array(part["train_idx"]); va = np.array(part["val_idx"]); te = np.array(part["test_idx"])
    assert tr.max() < len(structs) and te.max() < len(structs), "partition domain mismatch"
    chg = torch.tensor([int(round(d["total_charge"])) for d in structs], dtype=torch.long)
    unp = torch.tensor([int(d.get("unpaired_e", 0)) for d in structs], dtype=torch.long)
    torch.save({"z": zs, "pos": pos, "y_eV": torch.tensor(y, dtype=torch.float32),
                "y_homo": torch.tensor([d["homo"] * EV_PER_HA for d in structs], dtype=torch.float32),
                "y_lumo": torch.tensor([d["lumo"] * EV_PER_HA for d in structs], dtype=torch.float32),
                "y_gap": torch.tensor([d["gap"] * EV_PER_HA for d in structs], dtype=torch.float32),
                "y_dipole": torch.tensor([d["dipole_mag"] for d in structs], dtype=torch.float32),
                "charge": chg, "unpaired": unp,
                "train_rows": torch.tensor(tr), "val_rows": torch.tensor(va),
                "test_rows": torch.tensor(te)}, OUT / f"graphs_{name}.pt")
    ytr_mean = float(np.mean([y[i] for i in tr.tolist()]))
    print(f"[{name}] n={len(structs)} missing_coords={missing} | "
          f"train {len(tr)} val {len(va)} test {len(te)} | y_train mean {ytr_mean:.3f}", flush=True)


def build_acqm():
    props = json.load(open(DATA / "acqm_properties.json"))
    atoms_list = read(str(DATA / "acqm" / "acqm_full.extxyz"), index=":")
    assert len(atoms_list) == len(props), "acqm length mismatch"
    zs, pos, y = [], [], []
    for at, d in zip(atoms_list, props):
        znum = np.array(at.numbers)
        assert np.array_equal(znum, np.array(d["atomic_numbers"])), "acqm order mismatch"
        zs.append(torch.tensor(znum, dtype=torch.long))
        pos.append(torch.tensor(at.positions, dtype=torch.float32))
        y.append(max(d["homo_up"], d["homo_dn"]) * EV_PER_HA)
    part = json.load(open(DATA / "partitions" / "acqm_scaffold_2394.json"))
    tr = np.array(part["train_idx"]); va = np.array(part["val_idx"]); te = np.array(part["test_idx"])
    chg = torch.tensor([int(round(d["total_charge"])) for d in props], dtype=torch.long)
    unp = torch.tensor([int(d.get("unpaired_e", 0)) for d in props], dtype=torch.long)
    lumo_a = torch.tensor([min(d["lumo_up"], d["lumo_dn"]) * EV_PER_HA for d in props], dtype=torch.float32)
    homo_a = torch.tensor([max(d["homo_up"], d["homo_dn"]) * EV_PER_HA for d in props], dtype=torch.float32)
    torch.save({"z": zs, "pos": pos, "y_eV": torch.tensor(y, dtype=torch.float32),
                "y_homo": homo_a, "y_lumo": lumo_a, "y_gap": lumo_a - homo_a,
                "y_dipole": torch.tensor([d["dipole_mag"] for d in props], dtype=torch.float32),
                "charge": chg, "unpaired": unp,
                "train_rows": torch.tensor(tr), "val_rows": torch.tensor(va),
                "test_rows": torch.tensor(te)}, OUT / "graphs_5f.pt")
    print(f"[5f] n={len(props)} | train {len(tr)} val {len(va)} test {len(te)}", flush=True)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or list(SERIES) + ["5f"]
    for name in which:
        if name == "5f":
            build_acqm()
        else:
            build_tmqm(name, SERIES[name])
    print("done")
