# Fig. 3: determinants of learnability across the fifteen targets.
# Rows: charge Gini importance, permutation-importance share, and the percentage
# change of the test MAE under charge ablation. Colours are row-normalized
# (min-max within each row); the raw value is printed in every cell.
# Targets are ordered by the charge Gini importance, descending.
#
# Input : data_raw/decision_matrix.csv
# Output: fig3_determinants.png / .pdf

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
PNG = HERE / "fig3_determinants.png"
PDF = HERE / "fig3_determinants.pdf"
SVG = HERE / "fig3_determinants.svg"

LABELS = {
    "homo_up": "HOMO↑", "homo_dn": "HOMO↓",
    "lumo_up": "LUMO↑", "lumo_dn": "LUMO↓",
    "gap": "gap", "dipole_mag": "dipole", "exc": "exc", "ecnl": "ecnl",
    "pol_sum": r"pol$_{\mathrm{sum}}$", "mayer_pop_sum": "Mayer",
    "cn_mean": r"cn$_{\mathrm{mean}}$", "cn_max": r"cn$_{\mathrm{max}}$",
    "cn_std": r"cn$_{\mathrm{std}}$", "rot_mean": r"rot$_{\mathrm{mean}}$",
    "hirshfeld_abs": "Hirshfeld",
}

ROWS = [
    ("gini", "Charge Gini"),
    ("perm_share", "Permutation\nimportance"),
    ("delta_pct", "Charge ablation\n(% MAE change)"),
]
ORBITAL = {"homo_up", "homo_dn", "lumo_up", "lumo_dn"}


def fmt_gini(value):
    return f"{value:.3f}"


def fmt_ablation(value):
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}%"
    if magnitude >= 1:
        return f"{value:.1f}%"
    return f"{value:.2f}%"


def text_color(rgba):
    r, g, b = rgba[:3]
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def main():
    with open(RAW / "decision_matrix.csv", newline="", encoding="utf-8") as fh:
        records = {row["target"]: row for row in csv.DictReader(fh)}
    missing = sorted(set(LABELS) - set(records))
    if missing:
        raise ValueError(f"missing targets in decision_matrix.csv: {missing}")

    order = sorted(records, key=lambda k: float(records[k]["gini"]), reverse=True)
    raw = np.array([[float(records[k][col]) for k in order] for col, _ in ROWS])
    if not np.all(np.isfinite(raw)):
        raise ValueError("non-finite value in decision_matrix.csv")

    lo = raw.min(axis=1, keepdims=True)
    hi = raw.max(axis=1, keepdims=True)
    span = hi - lo
    if np.any(span <= 0):
        raise ValueError("a determinant row has zero range; row normalization undefined")
    scaled = (raw - lo) / span

    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    fig.subplots_adjust(left=0.175, right=0.895, top=0.875, bottom=0.355)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(scaled, cmap=cmap, norm=norm, aspect="auto")

    nrow, ncol = raw.shape
    for i in range(nrow):
        for j in range(ncol):
            value = raw[i, j]
            if i == 2:
                label = fmt_ablation(value)
            else:
                label = fmt_gini(value)
            ax.text(j, i, label, ha="center", va="center", fontsize=6.8,
                    color=text_color(cmap(norm(scaled[i, j]))))

    ax.set_xticks(np.arange(ncol), [LABELS[k] for k in order], rotation=45,
                  ha="right", fontsize=8)
    ax.set_yticks(np.arange(nrow), [label for _, label in ROWS], fontsize=8)
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)

    # Separate the charge-dominated orbital family (first four columns after sorting).
    n_orbital = sum(k in ORBITAL for k in order)
    ax.axvline(n_orbital - 0.5, color="#666666", linewidth=0.9,
               linestyle=(0, (3, 2.5)))

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.012)
    cbar.set_label("Within-row min–max scaling", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)
    ax.set_title("Determinants of learnability", fontsize=10.5, pad=6)

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)

    print("[Fig.3] rows by target, ordered by charge Gini")
    print("  order:", ", ".join(order))
    for target in order:
        row = records[target]
        print(f"  {target:15s} gini={float(row['gini']):.4f} "
              f"perm={float(row['perm_share']):.4f} "
              f"ablation={float(row['delta_pct']):+.2f}%")


if __name__ == "__main__":
    main()
