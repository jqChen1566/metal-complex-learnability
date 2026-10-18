"""Add y_pol (polarizability) to the tmQM graphs (3d/4d/5d).

Row alignment: graph row order == properties-json order; every entry carries a
CSD code; tmQM_y.csv (108,542 rows) maps CSD -> Polarizability (GFN2-xTB level,
a.u.).  Unit note: different level of theory than the wB97M-V frontier
orbitals -- label separately in reports.

Atomic replace: temp save -> verify -> os.replace (safe with running jobs).
Usage: python add_tmqm_polarizability.py
"""
import csv
import json
import os
from pathlib import Path

import torch

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
GRAPH_DIR = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
CSV = Path(os.environ.get("TMQM_CSV", DATA / "tmqm_y.csv"))

SERIES = {"3d": ("tmqm_properties.json", "graphs_3d.pt"),
          "4d": ("tmqm_properties_4d.json", "graphs_4d.pt"),
          "5d": ("tmqm_properties_5d.json", "graphs_5d.pt")}


def load_pol_map():
    m, dup = {}, 0
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            key = row.get("CSD_code")
            try:
                v = float(row["Polarizability"])
            except (ValueError, TypeError, KeyError):
                continue
            if key in m:
                dup += 1
            m[key] = v
    return m, dup


def main():
    polmap, dup = load_pol_map()
    print(f"y.csv polarizability entries: {len(polmap)} (dup keys: {dup})")
    for s, (pj, gp) in SERIES.items():
        structs = json.load(open(DATA / pj))
        vals, miss = [], 0
        for d in structs:
            v = polmap.get(d["csd"])
            if v is None:
                miss += 1
                v = float("nan")
            vals.append(v)
        y_pol = torch.tensor(vals, dtype=torch.float32)
        g = GRAPH_DIR / gp
        dg = torch.load(g, weights_only=False)
        if len(dg["z"]) != len(structs) or len(y_pol) != len(structs):
            raise SystemExit(f"{s}: length mismatch graph {len(dg['z'])} "
                             f"json {len(structs)}")
        dg["y_pol"] = y_pol
        tmp = g.with_suffix(".tmp_addpol.pt")
        torch.save(dg, tmp)
        chk = torch.load(tmp, weights_only=False)
        assert "y_pol" in chk and len(chk["z"]) == len(structs)
        os.replace(tmp, g)
        fin = torch.isfinite(y_pol)
        print(f"{s}: n={len(y_pol)} matched={int(fin.sum())} missing={miss} | "
              f"range {float(torch.nan_to_num(y_pol).min()):.1f}.."
              f"{float(torch.nan_to_num(y_pol).max()):.1f}")


if __name__ == "__main__":
    main()
