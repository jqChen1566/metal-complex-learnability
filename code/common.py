"""Shared constants and helpers for the analysis pipeline.

Every script imports paths, units and feature functions from this module.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("METAL_ML_ROOT", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("METAL_ML_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("METAL_ML_RESULTS", ROOT / "results"))
PART_DIR = Path(os.environ.get("METAL_ML_PARTITIONS", DATA / "partitions"))

PART_LNQM = PART_DIR / "lnqm_scaffold_12444.json"
PART_TMQM3D = PART_DIR / "tmqm_scaffold_33086.json"

H5_LNQM = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))
LNQM_TRAIN = DATA / "lnqm_properties_train.json"
LNQM_VAL = DATA / "lnqm_properties_val.json"
LNQM_TEST = DATA / "lnqm_properties_test.json"
LNQM_EXT = DATA / "lnqm_properties_ext.json"

EV_PER_HA = 27.2114
AU_TO_DEBYE = 2.5417

# 31-dimensional global composition features.
_AN = np.array([1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53,
                57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])
_EL_MAP = {int(z): i for i, z in enumerate(_AN)}


def featurize_global(structs):
    """Return the 31-dimensional composition feature matrix."""
    X = np.zeros((len(structs), 31), dtype=np.float32)
    for i, d in enumerate(structs):
        z = np.array(d["atomic_numbers"])
        n = len(z)
        for zi in z:
            if zi in _EL_MAP:
                X[i, _EL_MAP[zi]] += 1.0
        X[i, :26] /= max(n, 1)
        X[i, 26] = n
        X[i, 27] = float(d["total_charge"])
        X[i, 28] = float(d.get("unpaired_e", 0))
        X[i, 29] = float(np.mean(z))
        X[i, 30] = float(np.std(z)) if n > 1 else 0.0
    return X


# Metal ranges used by the scaffold key.
METAL_RANGES = {"ln": (57, 71), "tm3d": (21, 30), "tm4d": (39, 48),
                "tm5d": (72, 80), "an": (89, 103)}


def scaffold_key(d, metal_range=(57, 71)):
    """Return the ligand-framework scaffold key for one complex."""
    lo, hi = metal_range
    non_m = tuple(sorted(int(z) for z in d["atomic_numbers"] if not (lo <= z <= hi)))
    return (non_m, len(non_m), d["total_charge"])
