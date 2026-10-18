# Fig. 4: model x series x property test-error matrix.
# Four panels (2 x 2), one per property. Rows: model families; columns: five
# metal series. Cell colour is the test MAE; each panel has its own colour
# scale because the units differ (eV for orbital energies and gaps, Debye for
# dipole moments). Missing cells are drawn as light grey dashes.
#
# Input : data_raw/main_figure_data.csv (snapshot of the main-figure table)
# Output: fig4_matrix_mae.png / .pdf

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
PNG = HERE / "fig4_matrix_mae.png"
PDF = HERE / "fig4_matrix_mae.pdf"
SVG = HERE / "fig4_matrix_mae.svg"

MODELS = ["rfcomp", "schnet", "dimenet", "ccschnet"]
MODEL_LABELS = ["RF-composition", "SchNet", "DimeNet++", "CC-Schnet"]
SERIES = ["3d", "4d", "5d", "lnqm", "5f"]
SERIES_LABELS = ["3d", "4d", "5d", "4f", "5f"]

PANELS = [
    ("homo", "(a) HOMO energy", "eV"),
    ("lumo", "(b) LUMO energy", "eV"),
    ("gap", "(c) HOMO–LUMO gap", "eV"),
    ("dipole", "(d) Dipole moment", "Debye"),
]

MISSING_COLOR = "#f0f0f0"
MISSING_TEXT = "#808080"


def load_grids():
    grids = {prop: np.full((len(MODELS), len(SERIES)), np.nan)
             for prop, _, _ in PANELS}
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


def draw_panel(fig, ax, cax, matrix, title, unit, show_rows):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(MISSING_COLOR)
    norm = Normalize(vmin=0.0, vmax=float(np.nanmax(matrix)))
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm,
                   aspect="auto")

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
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        fontsize=7.5, color=text_color(cmap(norm(value))))

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"Test MAE ({unit})", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)


def main():
    grids = load_grids()
    check_missing(grids)

    fig = plt.figure(figsize=(7.0, 5.8))
    outer = fig.add_gridspec(2, 2, left=0.15, right=0.90, top=0.925,
                             bottom=0.065, wspace=0.30, hspace=0.34)
    for index, (prop, title, unit) in enumerate(PANELS):
        r, c = divmod(index, 2)
        sub = outer[r, c].subgridspec(1, 2, width_ratios=[1, 0.048], wspace=0.03)
        ax = fig.add_subplot(sub[0, 0])
        cax = fig.add_subplot(sub[0, 1])
        draw_panel(fig, ax, cax, grids[prop], title, unit, show_rows=(c == 0))

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)

    print("[Fig.4] test MAE grid (rows = models, columns = series)")
    for prop, title, unit in PANELS:
        print(f"  {prop} ({unit})")
        for i, model in enumerate(MODEL_LABELS):
            cells = []
            for j, series in enumerate(SERIES_LABELS):
                v = grids[prop][i, j]
                cells.append("   —" if np.isnan(v) else f"{v:7.3f}")
            print(f"    {model:15s}" + "".join(cells))
        print(f"    vmax={np.nanmax(grids[prop]):.4f}")


if __name__ == "__main__":
    main()
