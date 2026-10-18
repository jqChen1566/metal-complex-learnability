"""MLP CPU vs GPU benchmark (single target homo_up). Diagnostic only."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import LNQM_TRAIN, LNQM_TEST, LNQM_EXT, PART_LNQM, featurize_global
from train_composition_models import build_xy, train_mlp


def main():
    all_data = []
    for path in [LNQM_TRAIN, LNQM_TEST]:
        all_data.extend(json.load(open(path)))
    ext = json.load(open(LNQM_EXT))
    ext_by_index = {r["index"]: r for r in ext}
    part = json.load(open(PART_LNQM))
    tr_idx = np.array(part["train_idx"])
    va_idx = np.array(part["val_idx"])
    te_idx = np.array(part["test_idx"])
    tr = [all_data[i] for i in tr_idx]
    va = [all_data[i] for i in va_idx]
    te = [all_data[i] for i in te_idx]
    Xtr, Xva, Xte = featurize_global(tr), featurize_global(va), featurize_global(te)
    Y = build_xy(all_data, ext_by_index)
    mask, y = Y["homo_up"]
    m_tr, m_va, m_te = mask[tr_idx], mask[va_idx], mask[te_idx]

    print(f"device check: cuda available = {__import__('torch').cuda.is_available()}", flush=True)
    for dev, bs in [("cpu", 256), ("cuda", 4096)]:
        t = time.time()
        pred = train_mlp(Xtr[m_tr], y[tr_idx][m_tr], Xva[m_va], y[va_idx][m_va],
                         Xte[m_te], device=dev, batch=bs)
        mae = float(np.abs(pred - y[te_idx][m_te]).mean())
        print(f"{dev:5s} batch {bs:5d}: {time.time()-t:6.1f}s | test MAE {mae:.6f} Ha "
              f"({mae*27.2114:.4f} eV)", flush=True)


if __name__ == "__main__":
    main()
