"""Recompute input generation: ORCA single-point inputs for the 100 sampled complexes.

Two SCF protocols per complex:
  inputs_A/<idx>.inp : A_repro  -- default convergence, reproduces published protocol
  inputs_B/<idx>.inp : B_strict -- TightSCF + SlowConv (hardened convergence)

Inputs run with %pal nprocs 1 (no MPI) and embedded coordinates
(self-contained inputs).
"""
import csv
import os
from pathlib import Path

RECOMP = Path(os.environ.get("METAL_ML_RECOMP", Path(os.environ.get("METAL_ML_RESULTS", "results")) / "recompute"))
SEL = RECOMP / "selected_100.csv"
STRUCT = RECOMP / "structures"
OUTA = RECOMP / "inputs_A"
OUTB = RECOMP / "inputs_B"

HEAD_A = """! wB97M-V def2-SVPD RIJCOSX def2/J HIRSHFELD
%pal nprocs 1 end
%maxcore 3000
%scf
  MaxIter 500
end
* xyz {charge} {mult}
"""
HEAD_B = """! wB97M-V def2-SVPD RIJCOSX def2/J HIRSHFELD TightSCF SlowConv
%pal nprocs 1 end
%maxcore 3000
%scf
  MaxIter 800
end
* xyz {charge} {mult}
"""

for d in (OUTA, OUTB):
    d.mkdir(parents=True, exist_ok=True)

with open(SEL) as fh:
    rows = list(csv.DictReader(fh))

for r in rows:
    idx = int(r["h5_index"])
    tag = f"{idx:05d}"
    xyz_lines = (STRUCT / f"{tag}.xyz").read_text().splitlines()
    coords = "\n".join(xyz_lines[2:])
    for outdir, head in ((OUTA, HEAD_A), (OUTB, HEAD_B)):
        text = head.format(charge=r["charge"], mult=r["mult"]) + coords + "\n*\n"
        (outdir / f"{tag}.inp").write_text(text)

print(f"wrote {len(rows)} inputs x 2 protocols -> {OUTA.name}/, {OUTB.name}/")
