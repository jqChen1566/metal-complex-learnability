# Main figure set: test MAE heatmaps (model x series x property) for the LnQM / AcQM / tmQM
# multi-property ML paper.  Rows: RF-composition / SchNet / DimeNet++ / CC-Schnet.
# Columns: 3d / 4d / 5d / 4f (LnQM) / 5f (AcQM).  One panel per property (HOMO, LUMO,
# HOMO-LUMO gap, dipole moment), each in two versions: absolute test MAE, and the ratio
# (RF-composition MAE / model MAE) expressing the improvement of the geometric models
# over the composition baseline.
#
# Inputs : data_raw/*.json  (raw result files; values used as stored)
# Outputs: fig_<prop>_mae.png / .pdf, fig_<prop>_ratio.png / .pdf,
#          main_figure_data.csv, pol_data_supplementary.csv

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"

SERIES = ["3d", "4d", "5d", "lnqm", "5f"]
SERIES_LABELS = ["3d", "4d", "5d", "4f", "5f"]
MODELS = ["rfcomp", "schnet", "dimenet", "ccschnet"]
MODEL_LABELS = ["RF-composition", "SchNet", "DimeNet++", "CC-Schnet"]

PROPS = {
    "homo": {"title": "HOMO energy", "unit": "eV"},
    "lumo": {"title": "LUMO energy", "unit": "eV"},
    "gap": {"title": "HOMO–LUMO gap", "unit": "eV"},
    "dipole": {"title": "Dipole moment", "unit": "Debye"},
}

# Single-target HOMO result files (test_mae_eV field).
HOMO_SINGLE_FILES = {
    "schnet": {"3d": "result_schnet_3d_full.json",
               "4d": "result_schnet_4d_full.json",
               "5d": "result_schnet_5d_full_s43.json",
               "5f": "result_schnet_5f_full.json",
               "lnqm": "result_schnet_lnqm_full.json"},
    "dimenet": {"3d": "result_dimenet_3d_full.json",
                "4d": "result_dimenet_4d_full.json",
                "5d": "result_dimenet_5d_full.json",
                "5f": "result_dimenet_5f_full.json",
                "lnqm": "result_dimenet_full.json"},
    "ccschnet": {"3d": "result_ccschnet_3d_full.json",
                 "4d": "result_ccschnet_4d_full.json",
                 "5d": "result_ccschnet_5d_full.json",
                 "5f": "result_ccschnet_5f_full.json",
                 "lnqm": "result_ccschnet_lnqm_full.json"},
}

HOMO_SINGLE_REMARKS = {
    ("schnet", "5d"): "seed 43",
    ("schnet", "lnqm"): "300-epoch single-target run, seed 42",
}

RF_COMP_REMARK = ("RF baseline: element counts (120) + total charge + unpaired electrons; "
                  "500 trees, seed 42; trained on the full series train split")
MULTI_REMARK = "multi-task joint training (4 targets); train subset 12444; seed 42"

# Polarizability result files (supplementary table).
POL_FILES = [
    ("rfcomp", "3d", "result_rfcomp_3d_pol.json"),
    ("rfcomp", "4d", "result_rfcomp_4d_pol.json"),
    ("rfcomp", "5d", "result_rfcomp_5d_pol.json"),
    ("rfcomp", "lnqm", "result_rfcomp_lnqm_pol.json"),
    ("schnet", "lnqm", "result_schnet_lnqm_pol_full.json"),
]
POL_REMARKS = {
    ("rfcomp", "lnqm"): "composition baseline",
    ("schnet", "lnqm"): "single-target polarizability; seed 42",
}


def load(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def cell(prop, series, model):
    """Resolve one grid cell to (value or None, source file, seed, remark)."""
    if model == "rfcomp":
        f = f"result_rfcomp_{series}_{prop}.json"
        d = load(f)
        return d["test_mae"], f, 42, RF_COMP_REMARK
    if prop == "homo":
        f = HOMO_SINGLE_FILES[model][series]
        d = load(f)
        remark = HOMO_SINGLE_REMARKS.get((model, series), "single-target HOMO; seed 42")
        return d["test_mae_eV"], f, d.get("seed", 42), remark
    f = f"result_{model}_{series}_multi_full.json"
    if not (RAW / f).exists():
        return None, None, None, "not determined"
    d = load(f)
    return d["test_mae"][prop], f, d.get("seed", 42), MULTI_REMARK


def n_test(series, prop):
    return load(f"result_rfcomp_{series}_{prop}.json")["n_test"]


def build_grids():
    """Return values[prop] and ratios[prop] as 4x5 arrays (NaN for missing cells)."""
    rf = {}
    for prop in PROPS:
        for series in SERIES:
            rf[(prop, series)] = cell(prop, series, "rfcomp")[0]
    values, ratios = {}, {}
    for prop in PROPS:
        m = np.full((len(MODELS), len(SERIES)), np.nan)
        r = np.full((len(MODELS), len(SERIES)), np.nan)
        for i, model in enumerate(MODELS):
            for j, series in enumerate(SERIES):
                v = cell(prop, series, model)[0]
                if v is None:
                    continue
                m[i, j] = v
                r[i, j] = rf[(prop, series)] / v
        values[prop] = m
        ratios[prop] = r
    return values, ratios


def text_color(rgba):
    r, g, b = rgba[:3]
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def draw_panel(matrix, norm, cmap, fmt, title, cbar_label, out_stem,
               cell_marks=None, figsize=(7.4, 5.6)):
    nrow, ncol = matrix.shape
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.205, right=0.87, top=0.80, bottom=0.16)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#f0f0f0")
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap_obj, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(ncol), labels=SERIES_LABELS)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(nrow), labels=MODEL_LABELS)
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, pad=30, fontsize=12.5)

    for i in range(nrow):
        for j in range(ncol):
            v = matrix[i, j]
            mark = (cell_marks or {}).get((i, j), "")
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#808080")
            else:
                ax.text(j, i, fmt(v) + mark, ha="center", va="center", fontsize=10,
                        color=text_color(cmap_obj(norm(v))))

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(cbar_label, fontsize=10.5)
    cbar.outline.set_visible(False)

    fig.savefig(HERE / f"{out_stem}.png", dpi=300)
    fig.savefig(HERE / f"{out_stem}.pdf")
    plt.close(fig)


def draw_mae(prop, values):
    meta = PROPS[prop]
    matrix = values[prop]
    norm = Normalize(vmin=0.0, vmax=float(np.nanmax(matrix)))
    title = f"{meta['title']} — test-set MAE"
    cbar_label = f"Test MAE ({meta['unit']})"
    marks = None
    if prop == "homo":
        marks = {(MODELS.index("schnet"), SERIES.index("5d")): "*"}
    draw_panel(matrix, norm, "viridis", lambda v: f"{v:.3f}", title, cbar_label,
               f"fig_{prop}_mae", cell_marks=marks)


def draw_ratio(prop, ratios):
    meta = PROPS[prop]
    matrix = ratios[prop]
    finite = matrix[np.isfinite(matrix)]
    vmin, vmax = float(finite.min()), float(finite.max())
    norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
    title = f"{meta['title']} — improvement over composition RF"
    cbar_label = "MAE ratio (RF-composition / model)"
    draw_panel(matrix, norm, "coolwarm", lambda v: f"{v:.2f}", title, cbar_label,
               f"fig_{prop}_ratio")


def write_main_csv():
    with (HERE / "main_figure_data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["property", "series", "series_label", "model", "model_label",
                    "test_mae", "unit", "n_test", "seed", "remark"])
        for prop in PROPS:
            for series in SERIES:
                for model in MODELS:
                    v, _, seed, remark = cell(prop, series, model)
                    w.writerow([prop, series, SERIES_LABELS[SERIES.index(series)],
                                model, MODEL_LABELS[MODELS.index(model)],
                                "" if v is None else v, PROPS[prop]["unit"],
                                n_test(series, prop), "" if seed is None else seed,
                                remark])


def pol_value(spec):
    model, series, fname = spec
    d = load(fname)
    if "test_mae_eV" in d:
        return d["test_mae_eV"], d.get("unit", "a.u.")
    return d["test_mae"], "a.u."


def write_pol_csv():
    with (HERE / "pol_data_supplementary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["property", "series", "series_label", "model", "model_label",
                    "test_mae", "unit", "n_test", "remark"])
        for series in SERIES:
            for model in MODELS:
                spec = next((s for s in POL_FILES if s[0] == model and s[1] == series), None)
                base = ["pol", series, SERIES_LABELS[SERIES.index(series)], model,
                        MODEL_LABELS[MODELS.index(model)]]
                if spec is not None:
                    v, unit = pol_value(spec)
                    remark = POL_REMARKS.get((model, series), "single-target polarizability")
                    w.writerow(base + [v, unit, n_test(series, "homo"), remark])
                elif series == "5f":
                    w.writerow(base + ["", "a.u.", "",
                                       "the AcQM (5f) dataset provides no polarizability target"])
                else:
                    w.writerow(base + ["", "a.u.", "", "not determined"])


def main():
    values, ratios = build_grids()
    for prop in PROPS:
        draw_mae(prop, values)
        draw_ratio(prop, ratios)
    write_main_csv()
    write_pol_csv()
    print("figures and CSV written to", HERE)


if __name__ == "__main__":
    main()
