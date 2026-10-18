# Fig. 2: fifteen-property landscape of the 4f series.
# Grouped bars: test MAE of the composition RF and of the MLP, each expressed as
# a percentage of the target's p1-p99 span (same-unit ratio: native MAE divided
# by the native span). Targets are ordered by the RF relative error, descending.
#
# Inputs : data_raw/full_targets_test_rf.csv
#          data_raw/full_targets_test_mlp.csv
#          data_raw/target_stats_test.txt
# Outputs: fig2_relative_error.png / .pdf

import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
PNG = HERE / "fig2_relative_error.png"
PDF = HERE / "fig2_relative_error.pdf"
SVG = HERE / "fig2_relative_error.svg"

TARGET_ORDER = [
    "homo_up", "homo_dn", "lumo_up", "lumo_dn", "gap", "dipole_mag", "exc",
    "ecnl", "pol_sum", "mayer_pop_sum", "cn_mean", "cn_max", "cn_std",
    "rot_mean", "hirshfeld_abs",
]
LABELS = {
    "homo_up": "HOMO↑", "homo_dn": "HOMO↓",
    "lumo_up": "LUMO↑", "lumo_dn": "LUMO↓",
    "gap": "gap", "dipole_mag": "dipole", "exc": "exc", "ecnl": "ecnl",
    "pol_sum": r"pol$_{\mathrm{sum}}$", "mayer_pop_sum": "Mayer",
    "cn_mean": r"cn$_{\mathrm{mean}}$", "cn_max": r"cn$_{\mathrm{max}}$",
    "cn_std": r"cn$_{\mathrm{std}}$", "rot_mean": r"rot$_{\mathrm{mean}}$",
    "hirshfeld_abs": "Hirshfeld",
}

C_RF = "#9E9E9E"
C_MLP = "#4C72B0"


def read_mae(fname, col):
    out = {}
    with open(RAW / fname, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            val = row[col].strip()
            if val:
                out[row["target"]] = float(val)
    return out


def read_stats():
    out = {}
    for line in (RAW / "target_stats_test.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\S+)\s+n=\s*(\d+).*?p1=\s*(\S+)\s+p99=\s*(\S+)", line)
        if m:
            out[m.group(1)] = (float(m.group(3)), float(m.group(4)))
    return out


def relative_error(mae, stats):
    out = {}
    for key in TARGET_ORDER:
        if key not in mae:
            raise ValueError(f"missing test MAE for target {key}")
        if key not in stats:
            raise ValueError(f"missing p1/p99 statistics for target {key}")
        lo, hi = stats[key]
        span = hi - lo
        if span <= 0:
            raise ValueError(f"non-positive p1-p99 span for target {key}: {span}")
        out[key] = mae[key] / span * 100.0
    return out


def main():
    rf = relative_error(read_mae("full_targets_test_rf.csv", "rf_test"), read_stats())
    mlp = relative_error(read_mae("full_targets_test_mlp.csv", "mlp_test"), read_stats())
    order = sorted(TARGET_ORDER, key=lambda k: rf[k], reverse=True)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.335)

    x = np.arange(len(order))
    width = 0.38
    ax.bar(x - width / 2, [rf[k] for k in order], width, color=C_RF, zorder=3,
           edgecolor="white", linewidth=0.6, label="RF (composition baseline)")
    ax.bar(x + width / 2, [mlp[k] for k in order], width, color=C_MLP, zorder=3,
           edgecolor="white", linewidth=0.6, label="MLP")

    ax.set_xticks(x, [LABELS[k] for k in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Test MAE (% of p1–p99 span)", fontsize=9)
    ax.set_ylim(0, 19.5)
    ax.set_yticks(np.arange(0, 20, 5))
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95,
              edgecolor="#cccccc")
    ax.set_title("Fifteen-property landscape (4f series)", fontsize=10.5, pad=6)

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)

    print("[Fig.2] relative test error (% of p1-p99 span), sorted by RF")
    for key in order:
        print(f"  {key:15s} RF={rf[key]:7.3f}  MLP={mlp[key]:7.3f}")
    for key in ["homo_up", "dipole_mag", "exc", "ecnl"]:
        print(f"  check {key:12s} RF={rf[key]:.2f} MLP={mlp[key]:.2f}")


if __name__ == "__main__":
    main()
