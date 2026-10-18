# Fig. 7: zero-shot transfer matrix among the five series.
# Rows: source series; columns: target series. Cell value: zero-shot test MAE
# on the HOMO target (eV), source-standardized features, no adaptation.
# Colour scale is logarithmic; self-transfer entries sit on the diagonal.
#
# Input : data_raw/transfer_zeroshot.csv
# Output: fig7_transfer_matrix.png / .pdf

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
PNG = HERE / "fig7_transfer_matrix.png"
PDF = HERE / "fig7_transfer_matrix.pdf"
SVG = HERE / "fig7_transfer_matrix.svg"

SERIES = ["3d", "4d", "5d", "4f", "5f"]


def text_color(rgba):
    r, g, b = rgba[:3]
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def main():
    with open(RAW / "transfer_zeroshot.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    values = {}
    for row in rows:
        values[(row["src"], row["tgt"])] = float(row["zeroshot_mae_eV"])
    missing = [(s, t) for s in SERIES for t in SERIES if (s, t) not in values]
    if missing:
        raise ValueError(f"missing transfer matrix cells: {missing}")

    matrix = np.array([[values[(s, t)] for t in SERIES] for s in SERIES])
    if not np.all(matrix > 0):
        raise ValueError("zero-shot MAEs must be positive for the log colour scale")

    fig, ax = plt.subplots(figsize=(4.9, 4.35))
    fig.subplots_adjust(left=0.155, right=0.90, top=0.885, bottom=0.135)

    norm = LogNorm(vmin=float(matrix.min()), vmax=float(matrix.max()))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")

    n = len(SERIES)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=9.5, color=text_color(cmap(norm(matrix[i, j]))),
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(np.arange(n), SERIES, fontsize=9.5)
    ax.set_yticks(np.arange(n), SERIES, fontsize=9.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Target series", fontsize=9.5)
    ax.set_ylabel("Source series", fontsize=9.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_ticks([0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
    cbar.set_ticklabels(["0.3", "0.5", "1", "2", "3", "5"])
    cbar.set_label("Zero-shot MAE (eV, log scale)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title("Zero-shot transfer (HOMO energy)", fontsize=10.5, pad=6)

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)

    print("[Fig.7] zero-shot MAE (eV), rows = source, columns = target")
    print("        " + "".join(f"{t:>10s}" for t in SERIES))
    for i, s in enumerate(SERIES):
        print(f"  {s:>4s}" + "".join(f"{matrix[i, j]:10.3f}" for j in range(n)))
    print(f"  min={matrix.min():.3f} (row/col of minimum), "
          f"max={matrix.max():.3f}, diagonal={[round(matrix[i, i], 3) for i in range(n)]}")


if __name__ == "__main__":
    main()
