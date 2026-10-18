# Fig. 10: independently audited design performance.
# Re-evaluation residual (in units of the target spread, sigma) of the proposals
# returned by four design methods, with the paired starting structure shown as
# a per-target baseline. Ten runs per target and method.
#
# Input : data_raw/rebacktest_compare.csv
# Output: fig10_rebacktest.png / .pdf

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
PNG = HERE / "fig10_rebacktest.png"
PDF = HERE / "fig10_rebacktest.pdf"
SVG = HERE / "fig10_rebacktest.svg"

METHODS = ["gradient", "random", "ga", "greedy"]
METHOD_LABELS = {"gradient": "gradient", "random": "random", "ga": "GA",
                 "greedy": "greedy"}
METHOD_COLORS = {"gradient": "#1F77B4", "random": "#FF7F0E",
                 "ga": "#2CA02C", "greedy": "#D62728"}
OFFSETS = {"gradient": -0.27, "random": -0.09, "ga": 0.09, "greedy": 0.27}

# Same target order and labels as Fig. 9 so the two design figures read side by side.
TARGETS = ["cn_mean", "dipole_mag", "gap", "homo_up", "pol_sum"]
TARGET_LABELS = {
    "cn_mean": "CN mean",
    "dipole_mag": "dipole",
    "gap": "gap",
    "homo_up": "HOMO",
    "pol_sum": "polarizability",
}
START_COLOR = "#333333"


def load_rows():
    with open(RAW / "rebacktest_compare.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for row in rows:
        key = (row["target"], row["method"])
        out[key] = {
            "runs": int(row["n_runs"]),
            "proxy": float(row["proxy_dstd"]),
            "rf": float(row["rf_dstd"]),
            "start": float(row["rf_start_dstd"]),
        }
    return out


def main():
    data = load_rows()
    missing = [(t, m) for t in TARGETS for m in METHODS if (t, m) not in data]
    if missing:
        raise ValueError(f"missing re-backtest cells: {missing}")

    starts = {}
    for target in TARGETS:
        values = {data[(target, m)]["start"] for m in METHODS}
        if len(values) != 1:
            raise ValueError(f"inconsistent starting baseline for {target}: {values}")
        starts[target] = values.pop()

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.145)

    x = np.arange(len(TARGETS))
    for i, target in enumerate(TARGETS):
        # Paired starting-structure baseline for this target.
        ax.plot([i - 0.38, i + 0.38], [starts[target], starts[target]],
                color=START_COLOR, linewidth=1.2, zorder=2)
        ax.plot([i], [starts[target]], marker="D", markersize=5.0,
                markerfacecolor="white", markeredgecolor=START_COLOR,
                markeredgewidth=1.2, linestyle="none", zorder=4)
        for method in METHODS:
            value = data[(target, method)]["rf"]
            ax.scatter(i + OFFSETS[method], value, s=58,
                       color=METHOD_COLORS[method], edgecolor="white",
                       linewidth=0.8, zorder=3)

    ax.set_xticks(x, [TARGET_LABELS[t] for t in TARGETS], fontsize=9)
    ax.set_ylabel("Re-evaluation residual (target σ)", fontsize=9.5)
    top = max(max(data[(t, m)]["rf"] for m in METHODS) for t in TARGETS)
    top = max(top, max(starts.values())) * 1.12
    ax.set_ylim(0, top)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=METHOD_COLORS[m], markeredgecolor="white",
                          label=METHOD_LABELS[m]) for m in METHODS]
    handles.append(plt.Line2D([], [], marker="D", linestyle="none", markersize=5.5,
                              markerfacecolor="white", markeredgecolor=START_COLOR,
                              markeredgewidth=1.2, label="starting structure"))
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=8,
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              handletextpad=0.4, columnspacing=1.0)
    ax.set_title("Independently audited design performance", fontsize=10.5, pad=7)

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)

    print("[Fig.10] re-backtest residual in target sigma")
    print("  target        start   method    proxy      rf     delta   pct")
    for target in TARGETS:
        for method in METHODS:
            d = data[(target, method)]
            delta = d["rf"] - starts[target]
            pct = delta / starts[target] * 100.0
            print(f"  {target:12s} {starts[target]:6.3f}  {method:8s} "
                  f"{d['proxy']:9.2e} {d['rf']:7.3f} {delta:+7.3f} {pct:+7.1f}%")


if __name__ == "__main__":
    main()
