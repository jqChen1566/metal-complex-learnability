"""Recompute analysis: parse ORCA outputs and compare against h5 reference values.

For each protocol dir (out_A, out_B) reads each <idx>.out and extracts:
  - final single point energy (Eh)
  - SCF convergence flag + cycles
  - HOMO/LUMO alpha/beta orbital energies (Eh)
  - HF exchange / <S**2> if printed
Compares vs selected_100.csv h5 values (Hartree), reports dE in eV.
Outputs comparison_A.csv, comparison_B.csv and a printed summary.
"""
import csv
import os
import re
import sys
from pathlib import Path

RECOMP = Path(os.environ.get("METAL_ML_RECOMP", Path(os.environ.get("METAL_ML_RESULTS", "results")) / "recompute"))
EV = 27.211386245988

sel = {r["h5_index"]: r for r in csv.DictReader(open(RECOMP / "selected_100.csv"))}

NUM = r"(-?\d+\.\d+)"

def parse_out(path):
    res = {"energy": None, "scf_ok": None, "scf_cycles": None, "s2": None,
           "homo_up": None, "homo_dn": None, "lumo_up": None, "lumo_dn": None}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return res
    m = re.findall(r"FINAL SINGLE POINT ENERGY\s+" + NUM, text)
    if m:
        res["energy"] = float(m[-1])
    if "SCF NOT CONVERGED" in text or "The SCF did not converge" in text:
        res["scf_ok"] = False
    elif "SCF CONVERGED AFTER" in text or "The SCF was converged" in text:
        res["scf_ok"] = True
        mc = re.findall(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", text)
        if mc:
            res["scf_cycles"] = int(mc[-1])
    ms = re.findall(r"<S\*\*2>\s*=\s*" + NUM, text)
    if ms:
        res["s2"] = float(ms[-1])
    # orbital tables: split on SPIN UP / SPIN DOWN blocks
    blocks = re.split(r"SPIN (UP|DOWN) ORBITALS", text)
    tables = []
    if len(blocks) > 1:
        # blocks = [pre, 'UP', up_text, 'DOWN', down_text, ...]
        for i in range(1, len(blocks) - 1, 2):
            label = blocks[i]
            tables.append((label, blocks[i + 1]))
    else:
        tables.append(("UP", text))  # closed shell or merged table
    for label, tb in tables:
        rows = re.findall(r"^\s*\d+\s+([0-9.]+)\s+" + NUM + r"\s+" + NUM + r"\s*$",
                          tb, re.M)
        if not rows:
            continue
        occs = [(float(o), float(eh)) for o, eh, _ in rows]
        homo = lumo = None
        for occ, eh in occs:
            if occ > 0.5:
                homo = eh
            elif homo is not None and lumo is None:
                lumo = eh
        if label == "UP":
            res["homo_up"], res["lumo_up"] = homo, lumo
        else:
            res["homo_dn"], res["lumo_dn"] = homo, lumo
    return res


def run_phase(tag):
    outdir = RECOMP / f"out_{tag}"
    rows, n_ok, n_bad, n_missing = [], 0, 0, 0
    for idx, r in sorted(sel.items()):
        out = outdir / f"{int(idx):05d}.out"  # inputs/outputs use zero-padded names
        if not out.exists() or out.stat().st_size < 1000:
            n_missing += 1
            continue
        p = parse_out(out)
        if p["scf_ok"] is True:
            n_ok += 1
        elif p["scf_ok"] is False:
            n_bad += 1
        d = {"h5_index": idx, "uid": r["uid"], "metal": r["metal"],
             "charge": r["charge"], "mult": r["mult"], "scf_ok": p["scf_ok"],
             "scf_cycles": p["scf_cycles"], "energy_Eh": p["energy"], "s2": p["s2"]}
        for key in ("homo_up", "homo_dn", "lumo_up", "lumo_dn"):
            ref = float(r[f"{key}_h5"])
            got = p[key]
            d[f"{key}_ref_eV"] = ref * EV
            d[f"{key}_new_eV"] = got * EV if got is not None else None
            d[f"d_{key}_eV"] = (got - ref) * EV if got is not None else None
        rows.append(d)
    outcsv = RECOMP / f"comparison_{tag}.csv"
    if rows:
        with open(outcsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[{tag}] parsed={len(rows)} scf_converged={n_ok} not_converged={n_bad} missing={n_missing} -> {outcsv.name}")
    ds = [abs(r["d_homo_up_eV"]) for r in rows if r["d_homo_up_eV"] is not None]
    if ds:
        ds.sort()
        n = len(ds)
        print(f"[{tag}] |dHOMO_up| eV: median={ds[n//2]:.4f} mean={sum(ds)/n:.4f} "
              f"max={ds[-1]:.4f} | within 0.01 eV: {sum(1 for x in ds if x <= 0.01)}/{n}")
    return rows


if __name__ == "__main__":
    tags = sys.argv[1:] or ["A", "B"]
    for t in tags:
        run_phase(t)
