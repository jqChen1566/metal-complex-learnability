# Fig. 5: improvement over the composition baseline.
# Same 2 x 2 panel layout as Fig. 4. Cell value is the ratio of the
# RF-composition MAE to the model MAE for the same series and property; values
# above one mean the model beats the composition baseline. The ratio is
# dimensionless, so all panels share one diverging colour scale centred at 1.
#
# Input : data_raw/main_figure_data.csv (snapshot of the main-figure table)
# Output: fig5_matrix_ratio.png / .pdf

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
PNG = HERE / "fig5_matrix_ratio.png"
PDF = HERE / "fig5_matrix_ratio.pdf"
SVG = HERE / "fig5_matrix_ratio.svg"

MODELS = ["rfcomp", "schnet", "dimenet", "ccschnet"]
MODEL_LABELS = ["RF-composition", "SchNet", "DimeNet++", "CC-Schnet"]
SERIES = ["3d", "4d", "5d", "lnqm", "5f"]
SERIES_LABELS = ["3d", "4d", "5d", "4f", "5f"]

PANELS = [
    ("homo", "(a) HOMO energy"),
    ("lumo", "(b) LUMO energy"),
    ("gap", "(c) HOMO–LUMO gap"),
    ("dipole", "(d) Dipole moment"),
]

MISSING_COLOR = "#f0f0f0"
MISSING_TEXT = "#808080"


def load_grids():
    grids = {prop: np.full((len(MODELS), len(SERIES)), np.nan)
             for prop, _ in PANELS}
    with open(RAW / "main_figure_data.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            prop = row["property"]
            if prop not in grids:
                continue
            i = MODELS.index(row["model"])
            j = SERIES.index(row["series"])
            raw = row["test_mae"].strip()
            grids[prop][i, j] = float(raw) if raw else np.nan
    return grids


def check_missing(grids):
    expected = {("lumo", "ccschnet", "4d"), ("gap", "ccschnet", "4d"),
                ("dipole", "ccschnet", "4d")}
    found = set()
    for prop, matrix in grids.items():
        for i in range(len(MODELS)):
            for j in range(len(SERIES)):
                if np.isnan(matrix[i, j]):
                    found.add((prop, MODELS[i], SERIES[j]))
    if found != expected:
        raise ValueError(f"unexpected missing cells: {found ^ expected}")


def text_color(rgba):
    r, g, b = rgba[:3]
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def build_ratios(grids):
    ratios = {}
    for prop, _ in PANELS:
        matrix = grids[prop]
        ratio = np.full_like(matrix, np.nan)
        for j in range(len(SERIES)):
            base = matrix[MODELS.index("rfcomp"), j]
            if np.isnan(base):
                raise ValueError(f"missing RF baseline for {prop}/{SERIES[j]}")
            for i in range(len(MODELS)):
                if not np.isnan(matrix[i, j]):
                    ratio[i, j] = base / matrix[i, j]
        ratios[prop] = ratio
    return ratios


def draw_panel(ax, matrix, title, norm, cmap, show_rows):
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(SERIES)), SERIES_LABELS, fontsize=9)
    ax.set_yticks(np.arange(len(MODELS)), MODEL_LABELS, fontsize=9)
    if not show_rows:
        ax.tick_params(labelleft=False)
    ax.set_xticks(np.arange(-0.5, len(SERIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, fontsize=10, pad=6)

    for i in range(len(MODELS)):
        for j in range(len(SERIES)):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8,
                        color=MISSING_TEXT)
            else:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color(cmap(norm(value))))


def main():
    grids = load_grids()
    check_missing(grids)
    ratios = build_ratios(grids)

    finite = np.concatenate([r[np.isfinite(r)].ravel() for r in ratios.values()])
    vmin = float(np.floor(finite.min() * 20.0) / 20.0)
    vmax = float(np.ceil(finite.max() * 20.0) / 20.0)
    if not vmin < 1.0 < vmax:
        raise ValueError(f"ratio range does not bracket 1: {vmin}-{vmax}")
    norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(MISSING_COLOR)

    fig = plt.figure(figsize=(7.0, 5.6))
    outer = fig.add_gridspec(2, 2, left=0.15, right=0.875, top=0.925,
                             bottom=0.075, wspace=0.12, hspace=0.36)
    for index, (prop, title) in enumerate(PANELS):
        r, c = divmod(index, 2)
        ax = fig.add_subplot(outer[r, c])
        draw_panel(ax, ratios[prop], title, norm, cmap, show_rows=(c == 0))

    cax = fig.add_axes([0.915, 0.075, 0.016, 0.850])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    ticks = [t for t in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.4)
             if vmin <= t <= vmax]
    cbar = fig.colorbar(sm, cax=cax, ticks=ticks)
    cbar.set_label("MAE ratio (RF-composition / model)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)

    print(f"[Fig.5] ratio range {vmin:.2f}-{vmax:.2f}, vcenter=1")
    for prop, _ in PANELS:
        print(f"  {prop}")
        for i, model in enumerate(MODEL_LABELS):
            cells = []
            for j in range(len(SERIES)):
                v = ratios[prop][i, j]
                cells.append("   —" if np.isnan(v) else f"{v:7.3f}")
            print(f"    {model:15s}" + "".join(cells))


if __name__ == "__main__":
    main()
