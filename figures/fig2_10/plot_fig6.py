# Fig. 6: charge dominance across the five series.
# (a) Charge Gini of the HOMO model: mean over 20 seeds, error bars span the
#     +/-2 SD interval stored in gini_ci.csv.
# (b) Charge Gini of the gap and dipole models from the single-seed baseline run.
#
# Inputs : data_raw/gini_ci.csv, data_raw/five_series_baseline.csv
# Outputs: fig6_charge_gini.png / .pdf

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
PNG = HERE / "fig6_charge_gini.png"
PDF = HERE / "fig6_charge_gini.pdf"
SVG = HERE / "fig6_charge_gini.svg"

SERIES = ["3d", "4d", "5d", "4f", "5f"]
C_D = "#4C72B0"
C_4F = "#DD8452"
C_5F = "#8172B3"
SERIES_COLORS = {"3d": C_D, "4d": C_D, "5d": C_D, "4f": C_4F, "5f": C_5F}
C_GAP = "#4C72B0"
C_DIPOLE = "#DD8452"


def read_gini_ci():
    out = {}
    with open(RAW / "gini_ci.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["series"]] = {
                "mean": float(row["gini_mean"]),
                "lo": float(row["ci_lo"]),
                "hi": float(row["ci_hi"]),
            }
    return out


def read_baseline():
    out = {}
    with open(RAW / "five_series_baseline.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["series"], row["target"])] = float(row["gini_charge"])
    return out


def main():
    gini = read_gini_ci()
    baseline = read_baseline()
    for series in SERIES:
        if series not in gini:
            raise ValueError(f"missing Gini CI entry for series {series}")
        for target in ("gap", "dipole"):
            if (series, target) not in baseline:
                raise ValueError(f"missing baseline row for {series}/{target}")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.0, 1.3]})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.155,
                        wspace=0.30)

    # Panel (a): HOMO charge Gini with the +/-2 SD interval.
    x = np.arange(len(SERIES))
    means = np.array([gini[s]["mean"] for s in SERIES])
    yerr = np.array([[gini[s]["mean"] - gini[s]["lo"] for s in SERIES],
                     [gini[s]["hi"] - gini[s]["mean"] for s in SERIES]])
    ax_a.bar(x, means, width=0.62, color=[SERIES_COLORS[s] for s in SERIES],
             zorder=3, edgecolor="white", linewidth=0.6)
    ax_a.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#222222",
                  elinewidth=1.0, capsize=3.5, capthick=1.0, zorder=4)
    for xi, value in zip(x, means):
        ax_a.text(xi, value + 0.020, f"{value:.3f}", ha="center", va="bottom",
                  fontsize=7.5, color="#222222")
    ax_a.set_xticks(x, SERIES, fontsize=9)
    ax_a.set_ylim(0, 1.10)
    ax_a.set_yticks(np.arange(0, 1.1, 0.2))
    ax_a.set_xlabel("Series", fontsize=9)
    ax_a.set_ylabel("Charge Gini importance", fontsize=9)
    ax_a.tick_params(axis="y", labelsize=8)
    ax_a.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_title("(a) HOMO energy", fontsize=10, pad=6)

    # Panel (b): gap and dipole charge Gini.
    xb = np.arange(len(SERIES))
    width = 0.36
    gap = [baseline[(s, "gap")] for s in SERIES]
    dipole = [baseline[(s, "dipole")] for s in SERIES]
    ax_b.bar(xb - width / 2, gap, width, color=C_GAP, zorder=3,
             edgecolor="white", linewidth=0.6, label="gap")
    ax_b.bar(xb + width / 2, dipole, width, color=C_DIPOLE, zorder=3,
             edgecolor="white", linewidth=0.6, label="dipole")
    for xi, value in zip(xb - width / 2, gap):
        ax_b.text(xi, value + 0.003, f"{value:.3f}", ha="center", va="bottom",
                  fontsize=6.5, color="#222222", rotation=90)
    for xi, value in zip(xb + width / 2, dipole):
        ax_b.text(xi, value + 0.003, f"{value:.3f}", ha="center", va="bottom",
                  fontsize=6.5, color="#222222", rotation=90)
    ax_b.set_xticks(xb, SERIES, fontsize=9)
    ax_b.set_ylim(0, 0.215)
    ax_b.set_yticks(np.arange(0, 0.21, 0.05))
    ax_b.set_xlabel("Series", fontsize=9)
    ax_b.set_ylabel("Charge Gini importance", fontsize=9)
    ax_b.tick_params(axis="y", labelsize=8)
    ax_b.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95,
                edgecolor="#cccccc")
    ax_b.set_title("(b) Gap and dipole moment", fontsize=10, pad=6)

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)

    print("[Fig.6a] HOMO charge Gini, mean and +/-2 SD interval")
    for s in SERIES:
        d = gini[s]
        print(f"  {s:3s} mean={d['mean']:.4f} lo={d['lo']:.4f} hi={d['hi']:.4f}")
    print("[Fig.6b] gap / dipole charge Gini (baseline run)")
    for s in SERIES:
        print(f"  {s:3s} gap={baseline[(s, 'gap')]:.4f} "
              f"dipole={baseline[(s, 'dipole')]:.4f}")


if __name__ == "__main__":
    main()
