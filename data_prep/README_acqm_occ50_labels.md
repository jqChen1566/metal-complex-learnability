# Corrected AcQM frontier-orbital labels

`acqm_occ50_labels.npz` contains the corrected HOMO/LUMO labels for the 2,394
AcQM structures used in the 5f series.

The original h5 frontier orbitals were parsed with the rule "HOMO = highest
orbital with occupation > 0; LUMO = lowest orbital with occupation == 0". That
rule is invalid for ORCA `SmearTemp` calculations, whose orbital occupations are
fractional (0.0001–0.9999). The corrected labels use the 0.5 occupation
boundary: HOMO is the highest orbital with occupation >= 0.5, and LUMO is the
lowest orbital with occupation < 0.5.

Arrays (all length 2,394, aligned with the row order of
`acqm_properties_occ50.json`):

- `index`: 0..2393
- `homo_up`, `homo_dn`, `lumo_up`, `lumo_dn`: Hartree
- `smear`: 1 for structures with fractional occupations, 0 otherwise

147 structures change relative to the original h5 labels; the remaining 2,247
are unchanged. For 200 structures with ORCA Fermi-smearing output, the corrected
gap agrees with the gap printed by ORCA to within 5.2e-4 eV.

`code/align_acqm.py` applies this table and writes
`acqm_properties_occ50.json`. Do not use the raw h5 HOMO/LUMO values for the 5f
series.
