# Fig. 9: design benchmark on the composition surrogate.
# Fraction of runs reaching the requested target within 0.1 sigma (hit rate)
# for five targets and four methods under an equal budget of 800 surrogate
# evaluations. Bar colours follow Fig. 10.
#
# Input : data_raw/design_benchmark.csv
# Output: fig9_design_bench.png / .pdf

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["hatch.linewidth"] = 0.6

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
PNG = HERE / "fig9_design_bench.png"
PDF = HERE / "fig9_design_bench.pdf"
SVG = HERE / "fig9_design_bench.svg"

TARGETS = ["cn_mean", "dipole_mag", "gap", "homo_up", "pol_sum"]
TARGET_LABELS = {
    "cn_mean": "CN mean",
    "dipole_mag": "dipole",
    "gap": "gap",
    "homo_up": "HOMO",
    "pol_sum": "polarizability",
}
METHODS = ["gradient", "random", "ga", "greedy"]
METHOD_LABELS = {
    "gradient": "projected gradient",
    "random": "random",
    "ga": "genetic algorithm",
    "greedy": "greedy",
}
METHOD_COLORS = {
    "gradient": "#1F77B4",
    "random": "#FF7F0E",
    "ga": "#2CA02C",
    "greedy": "#D62728",
}
# Hatch patterns provide a non-colour channel for the four methods.
METHOD_HATCHES = {
    "gradient": "",
    "random": "///",
    "ga": r"\\",
    "greedy": "...",
}


def load_hit_rates():
    with open(RAW / "design_benchmark.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for row in rows:
        out[(row["target"], row["method"])] = float(row["hit_rate"])
    return out


def main():
    hit = load_hit_rates()
    missing = [(t, m) for t in TARGETS for m in METHODS if (t, m) not in hit]
    if missing:
        raise ValueError(f"missing benchmark cells: {missing}")
    bad = {k: v for k, v in hit.items() if not 0.0 <= v <= 1.0}
    if bad:
        raise ValueError(f"hit rates outside [0, 1]: {bad}")

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.16)

    x = np.arange(len(TARGETS))
    width = 0.20
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, method in zip(offsets, METHODS):
        values = [hit[(t, method)] for t in TARGETS]
        ax.bar(x + offset, values, width, color=METHOD_COLORS[method],
               edgecolor="white", linewidth=0.6, hatch=METHOD_HATCHES[method],
               label=METHOD_LABELS[method], zorder=3)

    ax.set_xticks(x, [TARGET_LABELS[t] for t in TARGETS], fontsize=9)
    ax.set_ylabel("Hit rate (|Δ|/σ < 0.1)", fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.05, 0.2))
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
              fontsize=8.5, frameon=False, handlelength=1.2,
              handletextpad=0.5, columnspacing=1.6)
    ax.set_title("Design benchmark on the composition surrogate",
                 fontsize=10.5, pad=28)

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)

    print("[Fig.9] hit rate (|delta|/sigma < 0.1)")
    print("  " + " " * 14 + "".join(f"{METHOD_LABELS[m]:>21s}" for m in METHODS))
    for target in TARGETS:
        print(f"  {target:12s}" + "".join(
            f"{hit[(target, m)]:21.2f}" for m in METHODS))


if __name__ == "__main__":
    main()
