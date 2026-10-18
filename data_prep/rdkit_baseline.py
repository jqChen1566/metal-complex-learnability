# RDKit 2D topological baseline (Morgan fingerprint + physicochemical
# descriptors + random forest), tmQM 3d/4d/5d series, target HOMO energy (eV).
# Intermediate rung of the information ladder: composition -> 2D topology
# (this script) -> 3D geometry.
#
# Parsing strategy: MolFromSmiles with default full sanitization fails with
# AtomValenceException on 18 molecules containing hypervalent As/P (e.g.
# five-coordinate [As+2]). We therefore use:
#   mol = MolFromSmiles(smi, sanitize=False)
#   mol.UpdatePropertyCache(strict=False)
#   Chem.SanitizeMol(mol, sanitizeOps=SANITIZE_ALL ^ SANITIZE_PROPERTIES)
# i.e. partial sanitization that skips the valence check (SANITIZE_PROPERTIES).
# Full check (108116 rows across the three series): with this strategy no
# non-empty SMILES fails to parse, and for the 100715 non-empty molecules the
# 15 physicochemical descriptors and the 2048-bit Morgan fingerprint are
# bitwise identical to default full sanitization (0 differences).
# Empty SMILES (7383 rows, no 2D structure, 3158/2503/1722 per series) count as
# parse failures and are excluded from training and testing.
#
# Features (2063 dims): Morgan fingerprint (radius 2, 2048 bits) + 15
# physicochemical descriptors:
#   MolWt / ExactMolWt / TPSA / NumHDonors / NumHAcceptors /
#   NumRotatableBonds / RingCount / NumAromaticRings / NumAliphaticRings /
#   NumSaturatedRings / NumHeteroatoms / HeavyAtomCount /
#   NumValenceElectrons / FractionCSP3 / NumRadicalElectrons
#
# Model: RandomForestRegressor (500 trees, seed 42, n_jobs=8), same
# specification as the composition baseline result_rfcomp_*.
# Split and target: train_rows / test_rows and y_homo (eV) from
# graphs_{series}.pt, so the split and units match the composition baseline
# and the geometric models.
# Training budget: the main configuration trains on all train_rows (same as
# result_rfcomp_*); a matched_budget block additionally reproduces the training
# subset used by the geometric models (schnet/dimenet/ccschnet; torch.randperm
# generator seed 42, first 12444 rows) and retrains both the 2D model and the
# composition baseline under the same budget for the equal-budget
# information-ladder comparison.
# Output: result_rdkit_{series}_homo.json
import json
import os
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog("rdApp.*")

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
GRAPH_DIR = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
FP_RADIUS, FP_BITS = 2, 2048
SUBSET_N = 12444  # training row budget of the geometric models

SERIES = {
    "3d": DATA / "tmqm_properties.json",
    "4d": DATA / "tmqm_properties_4d.json",
    "5d": DATA / "tmqm_properties_5d.json",
}

DESC_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("ExactMolWt", Descriptors.ExactMolWt),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
]

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def comp_feats(d, n):
    # composition-baseline features: element counts (120) + charge + unpaired
    # electrons, identical to the composition baseline
    X = np.zeros((n, 122), dtype=np.float32)
    for i, z in enumerate(d["z"]):
        for zi in z.tolist():
            if 0 <= zi < 120:
                X[i, zi] += 1
    X[:, 120] = d["charge"].numpy()
    X[:, 121] = d["unpaired"].numpy()
    return X


def parse_smiles(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    return mol


def featurize(smiles):
    # returns the feature matrix, the failed row indices (empty or unparseable)
    # and the number of empty SMILES among them
    X = np.zeros((len(smiles), FP_BITS + len(DESC_FUNCS)), dtype=np.float32)
    failed, n_empty = [], 0
    for i, smi in enumerate(smiles):
        if smi.strip() == "":
            n_empty += 1
            failed.append(i)
            continue
        mol = parse_smiles(smi)
        if mol is None:
            failed.append(i)
            continue
        fp = np.zeros((FP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(mol), fp)
        X[i, :FP_BITS] = fp
        X[i, FP_BITS:] = [f(mol) for _, f in DESC_FUNCS]
    assert np.isfinite(X).all()
    return X, failed, n_empty


def main():
    for name, pj in SERIES.items():
        recs = json.loads(pj.read_text())
        smiles = [r["smiles"] for r in recs]
        d = torch.load(GRAPH_DIR / f"graphs_{name}.pt", weights_only=False)
        y = d["y_homo"].numpy()
        assert len(smiles) == len(y), f"{name}: json {len(smiles)} != graph {len(y)}"
        assert np.isfinite(y).all(), f"{name}: y_homo contains non-finite values"
        tr = d["train_rows"].numpy()
        te = d["test_rows"].numpy()

        X, failed, n_empty = featurize(smiles)
        bad = np.zeros(len(smiles), dtype=bool)
        bad[failed] = True
        tr_use = tr[~bad[tr]]
        te_use = te[~bad[te]]
        n_bad_train = int(bad[tr].sum())
        n_bad_test = int(bad[te].sum())

        m = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=8)
        m.fit(X[tr_use], y[tr_use])
        p = m.predict(X[te_use])
        mae = float(np.abs(p - y[te_use]).mean())

        # matched comparison: refit and re-evaluate the composition baseline on
        # exactly the same row subset, distinct from result_rfcomp_* (full test
        # set including empty-SMILES rows)
        Xc = comp_feats(d, len(smiles))
        mc = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=8)
        mc.fit(Xc[tr_use], y[tr_use])
        mae_c = float(np.abs(mc.predict(Xc[te_use]) - y[te_use]).mean())

        # matched budget: reproduce the geometric-model training subset and
        # retrain both the 2D model and the composition baseline under it
        g = torch.Generator().manual_seed(42)
        tr_sub = tr[torch.randperm(len(tr), generator=g)[:min(SUBSET_N, len(tr))]]
        tr_sub_use = tr_sub[~bad[tr_sub]]
        m2 = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=8)
        m2.fit(X[tr_sub_use], y[tr_sub_use])
        mae2 = float(np.abs(m2.predict(X[te_use]) - y[te_use]).mean())
        mc2 = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=8)
        mc2.fit(Xc[tr_sub_use], y[tr_sub_use])
        mae2c = float(np.abs(mc2.predict(Xc[te_use]) - y[te_use]).mean())

        res = {
            "model": "RF-rdkit2d",
            "series": name,
            "target": "homo",
            "n_train": int(len(tr_use)),
            "n_test": int(len(te_use)),
            "test_mae_eV": round(mae, 4),
            "features": "Morgan FP (radius 2, 2048 bits) + 15 physchem descriptors",
            "descriptor_list": [n for n, _ in DESC_FUNCS],
            "parse_strategy": "MolFromSmiles(sanitize=False) + "
                              "UpdatePropertyCache(strict=False) + "
                              "SanitizeMol(SANITIZE_ALL ^ SANITIZE_PROPERTIES)",
            "n_parse_failed": len(failed),
            "n_parse_failed_empty_smiles": n_empty,
            "n_parse_failed_in_train": n_bad_train,
            "n_parse_failed_in_test": n_bad_test,
            "ref_rfcomp_matched_test_mae_eV": round(mae_c, 4),
            "ref_rfcomp_note": "RF-composition refit and evaluated on the same "
                               "parseable row subset as this run",
            "matched_budget": {
                "note": "training subset identical to the geometric-model runs "
                        "(torch.randperm generator seed 42, first 12444 train rows)",
                "n_train": int(len(tr_sub_use)),
                "test_mae_eV": round(mae2, 4),
                "ref_rfcomp_matched_test_mae_eV": round(mae2c, 4),
            },
            "rf": "500 trees, seed 42, n_jobs=8",
        }
        (GRAPH_DIR / f"result_rdkit_{name}_homo.json").write_text(json.dumps(res, indent=2))
        print(f"{name}: test MAE {mae:.4f} eV  (n_tr {len(tr_use)}, n_te {len(te_use)}, "
              f"parse failed {len(failed)} [empty {n_empty}, in_train {n_bad_train}, "
              f"in_test {n_bad_test}])  ref rfcomp matched {mae_c:.4f} eV; "
              f"matched budget (n_tr {len(tr_sub_use)}): 2D {mae2:.4f} eV, "
              f"rfcomp {mae2c:.4f} eV", flush=True)


if __name__ == "__main__":
    main()
