"""Multi-reference: parse the 12-system anchor outputs (stage2 CASSCF+NEVPT2).

Per system extracts:
  - termination, wall time
  - final CASSCF energy + last-iteration DE + final ||g|| (convergence quality)
  - active space (nel, norb) and active window indices (from NEVPT2 block)
  - active-orbital occupations N(occ) (final line) -> 4f occupation purity
  - active-window orbital energies (CASSCF natural orbitals)
  - NEVPT2 total correction and the 8 class breakdown
  - metal(atom0)-ligand Mayer bond orders

Writes multiref_summary.csv + multiref_classes.csv next to the outputs, and prints a
human-readable table.
"""
import csv
import os
import re
from pathlib import Path

RECOMP = Path(os.environ.get("METAL_ML_RECOMP", Path(os.environ.get("METAL_ML_RESULTS", "results")) / "recompute"))
MULTIREF = Path(os.environ.get("MULTIREF_DIR", RECOMP / "multi_reference"))
SKIP_MARK = "_atom"  # SCF-only copies (e.g. stage2_casscf_17205_atom64.out)

CLASSES = ["V0_ijab", "Vm1_iab", "Vm2_ab", "V1_ija", "V2_ij",
           "V0_ia", "Vm1_a", "V1_i"]


def parse(path: Path) -> dict:
    txt = path.read_text(errors="replace")
    r = {"file": path.name}

    r["terminated"] = int("TERMINATED NORMALLY" in txt)

    m = re.search(r"TOTAL RUN TIME: (\d+) days (\d+) hours (\d+) minutes (\d+) seconds",
                  txt)
    if m:
        d, h, mi, s = map(int, m.groups())
        r["wall_min"] = round(d * 1440 + h * 60 + mi + s / 60, 1)

    # --- CASSCF energies along iterations (E(CAS) lines) ---
    e_seq = [float(x) for x in re.findall(r"E\(CAS\)=\s*(-?\d+\.\d+)", txt)]
    if e_seq:
        r["e_casscf"] = e_seq[-1]
        r["e_iter"] = len(e_seq)
        r["de_last"] = abs(e_seq[-1] - e_seq[-2]) if len(e_seq) > 1 else float("nan")
        tail = e_seq[-20:]
        r["de_tail_max"] = max((abs(a - b) for a, b in zip(tail, tail[1:])),
                               default=float("nan"))
    m = re.search(r"Final CASSCF energy\s*:\s*(-?\d+\.\d+)", txt)
    if m:
        r["e_casscf_final"] = float(m.group(1))
    g_all = re.findall(r"\|\|g\|\|\s*=\s*([\d.]+e[+-]\d+)\s+Max\(G\)=", txt)
    if g_all:
        r["grad_norm"] = float(g_all[-1])

    # --- active space definition (NEVPT2 block) ---
    m = re.search(r"Number of active electrons\s*\.*\s*(\d+)", txt)
    if m:
        r["n_act_el"] = int(m.group(1))
    m = re.search(r"Number of active orbitals\s*\.*\s*(\d+)", txt)
    if m:
        r["n_act_orb"] = int(m.group(1))
    m = re.search(r"Active\s+(\d+)\s*-\s*(\d+)\s*\(\s*(\d+) orbitals\)", txt)
    if m:
        r["act_lo"], r["act_hi"] = int(m.group(1)), int(m.group(2))

    # --- final active-orbital occupations ---
    occ_lines = re.findall(r"N\(occ\)=\s*([\d\.\s]+)", txt)
    if occ_lines:
        occ = [float(x) for x in occ_lines[-1].split()]
        r["occ"] = occ
        r["occ_sum"] = round(sum(occ), 4)

    # --- active-window orbital energies (NO blocks in ORBITAL ENERGIES) ---
    orb = re.findall(r"^\s*(\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
                     txt, re.M)
    if orb and "act_lo" in r:
        lo, hi = r["act_lo"], r["act_hi"]
        win = [(int(no), float(occ), float(e)) for no, occ, e, _ in orb
               if lo <= int(no) <= hi]
        # orb list may repeat (per iteration); keep last full window
        if win:
            r["act_orb_occ"] = [w[1] for w in win[-7:]]
            r["act_orb_e"] = [w[2] for w in win[-7:]]

    # --- NEVPT2 ---
    m = re.search(r"Total Energy Correction\s*:\s*dE\s*=\s*(-?\d+\.\d+)", txt)
    if m:
        r["nevpt2_de"] = float(m.group(1))
    m = re.search(r"Total Energy \(E0\+dE\)\s*:\s*E\s*=\s*(-?\d+\.\d+)", txt)
    if m:
        r["e_total"] = float(m.group(1))
    cls = {}
    for name in CLASSES:
        m = re.search(rf"Class {name}\s*:\s*dE\s*=\s*(-?\d+\.\d+)", txt)
        if m:
            cls[name] = float(m.group(1))
    r["classes"] = cls

    # --- Mayer bond orders, metal = atom 0 ---
    bonds = {}
    for m in re.finditer(r"B\(\s*0-([A-Z][a-z]?)\s*,\s*(\d+)-([A-Z][a-z]?)\s*\)\s*:\s*(\d+\.\d+)",
                         txt):
        mz, idx, lz, val = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
        bonds.setdefault(f"{lz}{idx}", []).append(val)
    if bonds:
        r["mayer_metal_ligand"] = {k: v[-1] for k, v in bonds.items()}

    return r


def main():
    files = sorted(p for p in MULTIREF.glob("stage2_casscf_*.out")
                   if SKIP_MARK not in p.stem)
    rows, all_classes = [], {}
    for p in files:
        r = parse(p)
        all_classes[r["file"]] = r.pop("classes", {})
        rows.append(r)

    if not rows:
        print("no stage2 outputs found")
        return

    # CSV 1: scalar table
    scalar_keys = ["file", "terminated", "wall_min", "e_casscf_final", "e_casscf",
                   "e_iter", "de_last", "de_tail_max", "grad_norm",
                   "n_act_el", "n_act_orb", "act_lo", "act_hi", "occ_sum",
                   "nevpt2_de", "e_total"]
    with open(MULTIREF / "multiref_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=scalar_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # CSV 2: occupation vectors
    with open(MULTIREF / "multiref_occupations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "occ_active"] + [f"E_act_{i}" for i in range(7)])
        for r in rows:
            w.writerow([r["file"],
                        " ".join(f"{x:.4f}" for x in r.get("occ", [])),
                        *[f"{x:.4f}" for x in r.get("act_orb_e", [])]])

    # CSV 3: NEVPT2 class breakdown
    with open(MULTIREF / "multiref_classes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file"] + CLASSES)
        for fname, cls in all_classes.items():
            w.writerow([fname] + [f"{cls.get(c, float('nan')):.6f}"
                                  for c in CLASSES])

    # console table
    hdr = (f"{'file':28s} {'term':4s} {'wall_m':>7s} {'E_CASSCF':>16s} "
           f"{'dE_NEVPT2':>10s} {'n_el':>4s} {'occ_sum':>7s} {'grad':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['file']:28s} {r.get('terminated', 0):4d} "
              f"{r.get('wall_min', float('nan')):7.1f} "
              f"{r.get('e_casscf_final', float('nan')):16.8f} "
              f"{r.get('nevpt2_de', float('nan')):10.4f} "
              f"{r.get('n_act_el', -1):4d} {r.get('occ_sum', float('nan')):7.3f} "
              f"{r.get('grad_norm', float('nan')):9.2e}")

    n_done = sum(r.get("terminated", 0) for r in rows)
    print(f"\n{n_done}/{len(rows)} terminated normally; "
          f"CSVs -> multiref_summary.csv / multiref_occupations.csv / multiref_classes.csv")


if __name__ == "__main__":
    main()
