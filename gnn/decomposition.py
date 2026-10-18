"""Information-decomposition analysis of the geometric gain (no new QC).

Uses existing test-set predictions from the graph-model runs (all share the same scaffold
test split, same order):
  - SchNet (radial)          : preds_schnet_full_seed43/44
  - DimeNet++ (radial+angle) : preds_dimenet_full_seed42
  - zw_gat (topology+attn)   : preds_zw_gat_full_seed42

Questions:
  1. Where does DimeNet's gain over SchNet concentrate?
     (per lanthanide Z / charge / unpaired / natoms / cn_mean / angular
      distortion of the metal coordination sphere)
  2. Is the topology model blind to lanthanide contraction? (zw_gat MAE vs Z)

Outputs: decomposition CSVs + printed summary.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
OUT = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
H5 = Path(os.environ.get("LNQM_H5", DATA / "lnqm.h5"))
METAL = np.arange(57, 72)


def coordination_metrics(pos_l, zi_l, r=3.5):
    """Angular & radial distortion of the metal coordination sphere."""
    pos = pos_l.numpy()
    zi = zi_l.numpy()
    m = np.where((zi >= 57) & (zi <= 71))[0]
    if len(m) != 1:
        return np.nan, np.nan
    c = pos[m[0]]
    d = np.linalg.norm(pos - c, axis=1)
    nb = np.where((d > 0.1) & (d < r))[0]
    if len(nb) < 3:
        return np.nan, np.nan
    dists = d[nb]
    radial = dists.std() / dists.mean()
    v = (pos[nb] - c) / dists[:, None]
    cosang = np.clip(v @ v.T, -1, 1)
    iu = np.triu_indices(len(nb), k=1)
    angs = np.degrees(np.arccos(cosang[iu]))
    angular = angs.std() if len(angs) > 0 else np.nan
    return float(radial), float(angular)


def main():
    d = torch.load(OUT / "graphs_lnqm.pt", weights_only=False)
    zs, pos, y = d["z"], d["pos"], d["y_eV"]
    te = d["test_rows"].numpy()

    with h5py.File(H5, "r") as f:
        chg_raw = f["data/charge"][:].ravel()
        unpaired = f["data/unpaired_e"][:].ravel().astype(int)
    chg = np.array([int(c) - (1 << 64) if int(c) > (1 << 40) else int(c)
                    for c in chg_raw])
    ext = json.load(open(DATA / "lnqm_properties_ext.json"))
    cn_mean = np.array([r.get("cn_mean", np.nan) for r in ext])

    models = {}
    for name, path in [
        ("schnet43", "preds_schnet_full_seed43.npz"),
        ("schnet44", "preds_schnet_full_seed44.npz"),
        ("dimenet", "preds_dimenet_full_seed42.npz"),
        ("zwgat", "preds_zw_gat_full_seed42.npz"),
    ]:
        f = OUT / path
        if f.exists():
            z = np.load(f)
            models[name] = (z["pred"], z["true"])
    print("models loaded:", list(models.keys()))
    true = models["schnet43"][1]
    for k, (p, t) in models.items():
        assert np.allclose(t, true), f"{k} test order mismatch"

    # per-structure data
    meta = {"z": [], "charge": [], "unpaired": [], "natoms": [],
            "cn": [], "radial": [], "angular": []}
    for i in te.tolist():
        zi = zs[i]
        is_ln = (zi.numpy() >= 57) & (zi.numpy() <= 71)
        meta["z"].append(int(zi.numpy()[is_ln][0]) if is_ln.any() else -1)
        meta["charge"].append(int(chg[i]))
        meta["unpaired"].append(int(unpaired[i]))
        meta["natoms"].append(int(zi.numel()))
        meta["cn"].append(float(cn_mean[i]))
        r, a = coordination_metrics(pos[i], zi)
        meta["radial"].append(r)
        meta["angular"].append(a)
    meta = {k: np.array(v) for k, v in meta.items()}

    err = {k: np.abs(p - t) for k, (p, t) in models.items()}
    schnet_err = 0.5 * (err["schnet43"] + err["schnet44"])
    gain_dn = schnet_err - err["dimenet"]           # DimeNet gain over SchNet
    gain_sm = None

    rows = []
    def group_stat(key, bins):
        out = []
        for lo, hi, label in bins:
            m = (meta[key] >= lo) & (meta[key] < hi) & np.isfinite(gain_dn)
            if m.sum() < 5:
                continue
            out.append((label, int(m.sum()),
                        float(schnet_err[m].mean()),
                        float(err["dimenet"][m].mean()),
                        float(gain_dn[m].mean())))
        return out

    print("\n== DimeNet gain over SchNet by group ==")
    print(f"{'group':>16} {'n':>5} {'SchNet':>8} {'DimeNet':>8} {'gain':>8}")
    groups = {
        "metal": group_stat("z", [(z, z + 1, f"Z={z}") for z in METAL]),
        "charge": group_stat("charge", [(-2, -0.5, "-1"), (-0.5, 0.5, "0"),
                                        (0.5, 1.5, "+1"), (1.5, 2.5, "+2"),
                                        (2.5, 3.5, "+3")]),
        "unpaired": group_stat("unpaired", [(k, k + 1, f"u={k}") for k in range(8)]),
        "natoms-q": None, "cn-q": None, "angular-q": None, "radial-q": None,
    }
    for key in ("natoms", "cn", "angular", "radial"):
        vals = meta[key]
        m = np.isfinite(vals)
        qs = np.nanpercentile(vals[m], [0, 25, 50, 75, 100])
        bins = [(qs[i] - 1e-9, qs[i + 1] + 1e-9, f"Q{i+1}")
                for i in range(4)]
        groups[key + "-q"] = group_stat(key, bins)

    for gname, g in groups.items():
        if g is None:
            continue
        print(f"\n[{gname}]")
        for label, n, se, de, gn in g:
            print(f"{label:>16} {n:5d} {se:8.4f} {de:8.4f} {gn:+8.4f}")
            rows.append({"group": gname, "bin": label, "n": n,
                         "schnet_mae": round(se, 4), "dimenet_mae": round(de, 4),
                         "gain": round(gn, 4)})

    # zw_gat vs Z (blindness check)
    print("\n== zw_gat MAE by lanthanide (blindness check) ==")
    for z in METAL:
        m = meta["z"] == z
        if m.sum() >= 5:
            print(f"Z={z}: n={int(m.sum())} zwgat={err['zwgat'][m].mean():.3f} "
                  f"dimenet={err['dimenet'][m].mean():.3f}")
            rows.append({"group": "zwgat_by_Z", "bin": f"Z={z}",
                         "n": int(m.sum()),
                         "schnet_mae": round(float(schnet_err[m].mean()), 4),
                         "dimenet_mae": round(float(err["dimenet"][m].mean()), 4),
                         "gain": round(float(err['zwgat'][m].mean()), 4)})

    import csv as csvmod
    with open(OUT / "decomposition.csv", "w", newline="") as fh:
        w = csvmod.DictWriter(fh, fieldnames=["group", "bin", "n", "schnet_mae",
                                              "dimenet_mae", "gain"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {OUT / 'decomposition.csv'}")


if __name__ == "__main__":
    main()
