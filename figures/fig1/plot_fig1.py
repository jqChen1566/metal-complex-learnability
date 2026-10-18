# Fig. 1: narrative overview (Terrain / Extent / Use triptych plus a data-base band).
# Canvas 7.5 x 3.1 inches (full text width); font sizes are final layout sizes.
#
# Data snapshot in data_raw/:
#   full_targets_test_rf.csv  fifteen-target RF test MAE (native units)
#   target_stats_test.txt     p1/p99 statistics of the fifteen targets on the test
#                             set (n=1545), computed on the train+test concatenation
#                             using the test row indices
#   transfer_zeroshot.csv     5x5 zero-shot transfer MAE (eV)
#   five_series_baseline.csv  five-series n_total (the 4f row is the protocol count)
#
# Panel a: RF test MAE normalized to the p1-p99 span of the test set for each
#   target (comparable across units, robust to outliers).
# Panel b: values from zeroshot_mae_eV in transfer_zeroshot.csv; ranges include
#   direction pairs. The d-internal range 0.28-0.59 eV includes self-transfer;
#   the cross-transfer range is 0.41-0.59 eV.
#   Data volumes in the bottom band: 3d/4d/5d/5f use n_total from
#   five_series_baseline.csv (consistent with the split totals); 4f uses the LnQM
#   dataset size 17269 (the split total is 15542; the difference of 1727 structures
#   are unlabeled or not extracted).
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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
FIG_PNG = HERE / "fig1_overview.png"
FIG_PDF = HERE / "fig1_overview.pdf"
FIG_SVG = HERE / "fig1_overview.svg"

C_CHARGE = "#C44E52"   # charge-dominated (orbital-energy family)
C_OTHER = "#4C72B0"    # not charge-dominated
C_D = "#4C72B0"        # panel b: d-block nodes
C_4F = "#DD8452"       # panel b: 4f node
C_5F = "#8172B3"       # panel b: 5f node
C_DASH = "#C44E52"     # panel b: broken d<->f links
C_HL = "#55A868"       # panel b: highlighted 5f->4f exception
C_GOLD = "#D9A030"     # panel c: audit emphasis

TARGET_ORDER = ["homo_up", "homo_dn", "lumo_up", "lumo_dn", "gap", "dipole_mag",
                "exc", "ecnl", "pol_sum", "mayer_pop_sum", "cn_mean", "cn_max",
                "cn_std", "rot_mean", "hirshfeld_abs"]
TARGET_LABELS = {"homo_up": "HOMO↑", "homo_dn": "HOMO↓",
                 "lumo_up": "LUMO↑", "lumo_dn": "LUMO↓",
                 "gap": "gap", "dipole_mag": "dipole", "exc": "exc",
                 "ecnl": "ecnl", "pol_sum": r"pol$_{\mathrm{sum}}$", "mayer_pop_sum": "Mayer",
                 "cn_mean": r"cn$_{\mathrm{mean}}$", "cn_max": r"cn$_{\mathrm{max}}$",
                 "cn_std": r"cn$_{\mathrm{std}}$",
                 "rot_mean": r"rot$_{\mathrm{mean}}$", "hirshfeld_abs": "Hirshfeld"}
CHARGE_GROUP = {"homo_up", "homo_dn", "lumo_up", "lumo_dn"}

SERIES_SIZE = {"3d": 41374, "4d": 37450, "5d": 29292, "4f": 17269, "5f": 2394}


def read_target_mae():
    out = {}
    with open(RAW / "full_targets_test_rf.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["target"]] = float(row["rf_test"])
    return out


def read_target_stats():
    out = {}
    for line in (RAW / "target_stats_test.txt").read_text(encoding="utf-8").splitlines():
        out[line.split()[0]] = {k: float(v)
                                for k, v in re.findall(r"(\w+)=\s*([-\d.eE+]+)", line)}
    return out


def read_transfer():
    out = {}
    with open(RAW / "transfer_zeroshot.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["src"], row["tgt"])] = float(row["zeroshot_mae_eV"])
    return out


def transfer_range(transfer, srcs, tgts):
    vals = [v for (s, t), v in transfer.items() if s in srcs and t in tgts and s != t]
    return min(vals), max(vals)


def fmt_k(n):
    return f"{n / 1000:.1f}k"


def draw_panel_a(ax, mae, stats):
    rel = [mae[k] / (stats[k]["p99"] - stats[k]["p1"]) * 100 for k in TARGET_ORDER]
    colors = [C_CHARGE if k in CHARGE_GROUP else C_OTHER for k in TARGET_ORDER]
    x = np.arange(len(TARGET_ORDER))
    ax.bar(x, rel, width=0.62, color=colors, zorder=3)
    for xi, v in zip(x, rel):
        ax.text(xi, v + 0.45, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5,
                color="#333333", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[k] for k in TARGET_ORDER], rotation=90,
                       ha="center", fontsize=6.5)
    ax.set_ylabel("RF test MAE\n(% of p1–p99 span)", fontsize=7.5)
    ax.set_ylim(0, 28)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.axvline(3.5, color="#999999", linewidth=0.8, linestyle=(0, (3, 2.5)),
               zorder=2, ymax=9.3 / 28.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_CHARGE),
               plt.Rectangle((0, 0), 1, 1, color=C_OTHER)]
    ax.legend(handles,
              ["charge-dominated\norbital energies (Gini 0.96–0.97)",
               "not charge-dominated\nall others (Gini ≤ 0.18)"],
              loc="upper right", fontsize=6.5, frameon=True, framealpha=0.95,
              edgecolor="#cccccc", handlelength=0.9, borderpad=0.3,
              labelspacing=0.4, borderaxespad=0.4, handletextpad=0.4)
    ax.set_title("(a) Property axis: learnability\nof fifteen targets",
                 fontsize=6.5, fontweight="bold", pad=4)


def draw_node(ax, x, y, name, n, color, k=0.030, name_fs=8, label_mode="inside"):
    s = k * n
    ax.scatter([x], [y], s=s, color=color, zorder=3, edgecolor="white", linewidth=0.8)
    if label_mode == "inside":
        ax.text(x, y, name, ha="center", va="center", fontsize=name_fs,
                fontweight="bold", color="white", zorder=4)
    elif label_mode == "above":
        ax.text(x, y + 0.055, name, ha="center", va="bottom", fontsize=name_fs,
                fontweight="bold", color=color, zorder=4)
    elif label_mode == "right":
        ax.text(x + 0.038, y, name, ha="left", va="center", fontsize=name_fs,
                fontweight="bold", color=color, zorder=4)


def draw_panel_b(ax, transfer):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.995, "zero-shot transfer MAE (eV)", ha="center", va="top",
            fontsize=6.8, color="#444444", style="italic")
    ax.text(0.99, 0.935, "node area proportional to dataset size", ha="right",
            va="top", fontsize=6.3, color="#888888")

    pos = {"3d": (0.12, 0.68), "4d": (0.44, 0.77), "5d": (0.76, 0.67),
           "4f": (0.29, 0.21), "5f": (0.58, 0.20)}
    draw_node(ax, *pos["3d"], "3d", SERIES_SIZE["3d"], C_D)
    draw_node(ax, *pos["4d"], "4d", SERIES_SIZE["4d"], C_D)
    draw_node(ax, *pos["5d"], "5d", SERIES_SIZE["5d"], C_D)
    draw_node(ax, *pos["4f"], "4f", SERIES_SIZE["4f"], C_4F)
    draw_node(ax, *pos["5f"], "5f", SERIES_SIZE["5f"], C_5F,
              name_fs=7, label_mode="right")

    ax.text(0.12, 0.815, fmt_k(SERIES_SIZE["3d"]), ha="center", va="bottom",
            fontsize=6.5, color="#555555")
    ax.text(0.44, 0.632, fmt_k(SERIES_SIZE["4d"]), ha="center", va="top",
            fontsize=6.5, color="#555555")
    ax.text(0.76, 0.805, fmt_k(SERIES_SIZE["5d"]), ha="center", va="bottom",
            fontsize=6.5, color="#555555")
    ax.text(0.29, 0.115, fmt_k(SERIES_SIZE["4f"]), ha="center", va="top",
            fontsize=6.5, color="#555555")
    ax.text(0.62, 0.145, f"{SERIES_SIZE['5f']:,}", ha="left", va="center",
            fontsize=6.5, color="#555555")

    def arrow(p0, p1, color, style, lw, rad=0.0, zorder=2):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle=style, color=color, linewidth=lw, zorder=zorder,
            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
            mutation_scale=9))

    # d-internal transfer: solid double-headed arrows (0.28-0.59 eV includes
    # self-transfer; the cross-transfer range recomputed from the csv is 0.41-0.59)
    arrow((0.235, 0.735), (0.315, 0.755), "#555555", "<->", 1.0)
    arrow((0.565, 0.745), (0.645, 0.725), "#555555", "<->", 1.0)
    arrow((0.200, 0.565), (0.680, 0.555), "#555555", "<->", 1.0, rad=0.16)
    ax.text(0.44, 0.430, "d-internal:", ha="center", va="center", fontsize=6.5,
            color="#333333")
    ax.text(0.44, 0.370, "0.28–0.59 eV", ha="center", va="center", fontsize=6.5,
            color="#333333")

    # d <-> f: broken dashed double-headed arrows
    arrow((0.165, 0.565), (0.260, 0.290), C_DASH, "<->", 1.2)
    ax.text(0.125, 0.390, "2.7–3.6 eV", ha="center", va="center", fontsize=7,
            color=C_DASH, rotation=75)
    arrow((0.745, 0.555), (0.635, 0.235), C_DASH, "<->", 1.2)
    ax.text(0.800, 0.400, "3.3–4.9 eV", ha="center", va="center", fontsize=7,
            color=C_DASH, rotation=-75)

    # f <-> f: single arrows in each direction (the asymmetry is the point)
    arrow((0.545, 0.235), (0.375, 0.235), C_HL, "->", 2.2, zorder=5)
    ax.text(0.475, 0.245, "5f→4f: 0.51", ha="center", va="bottom", fontsize=7,
            color=C_HL, fontweight="bold")
    arrow((0.350, 0.145), (0.520, 0.145), "#999999", "->", 1.0)
    ax.text(0.450, 0.128, "4f→5f: 2.2", ha="center", va="top", fontsize=6.0,
            color="#777777")

    ax.text(0.5, 0.02, "LoEO: 4f ≈ interpolation · 5f ≈ true extrapolation",
            ha="center", va="bottom", fontsize=6.3, color="#444444")
    ax.set_title("(b) Element axis: zero-shot\ntransfer across five series",
                 fontsize=6.5, fontweight="bold", pad=4)


def node_box(ax, x0, y0, x1, y1, title, lines, bold=False, title_fs=7.5,
             title_offset=0.042, line_start=0.052, line_step=0.044):
    if bold:
        box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                             boxstyle="round,pad=0.010,rounding_size=0.020",
                             linewidth=1.8, edgecolor=C_GOLD, facecolor="#FFF8E8",
                             zorder=2)
    else:
        box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                             boxstyle="round,pad=0.010,rounding_size=0.020",
                             linewidth=1.0, edgecolor="#8a9bb0", facecolor="#F4F7FA",
                             zorder=2)
    ax.add_patch(box)
    y_title = y1 - title_offset
    ax.text((x0 + x1) / 2, y_title, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="#222222", zorder=3)
    for i, line in enumerate(lines):
        ax.text((x0 + x1) / 2, y_title - line_start - i * line_step, line,
                ha="center", va="center", fontsize=6.0, color="#333333",
                zorder=3)


def draw_panel_c(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    node_box(ax, 0.10, 0.805, 0.98, 0.970, "Train differentiable proxy",
             ["MLP on composition + charge"])
    node_box(ax, 0.10, 0.590, 0.98, 0.730, "Gradient inverse design",
             ["composition simplex + charge"])
    node_box(ax, 0.10, 0.305, 0.98, 0.490, "Independent re-evaluation",
             ["held-out RF re-evaluation",
              "surrogate–auditor discrepancy",
              "quantified"], bold=True, title_offset=0.030)
    node_box(ax, 0.10, 0.012, 0.98, 0.245, "Optimizer-dependent feasibility",
             ["low-dim levers (HOMO/LUMO, gap,",
              "dipole): reachable by all methods",
              "fine structure (polarizability,",
              r"cn$_{\mathrm{mean}}$): gradient-only"], title_fs=6.0,
             line_start=0.048, line_step=0.040)

    def v_arrow(y0, y1):
        ax.add_patch(FancyArrowPatch((0.54, y0), (0.54, y1), arrowstyle="-|>",
                                     color="#555555", linewidth=1.2, zorder=1,
                                     mutation_scale=11, shrinkA=0, shrinkB=0))

    v_arrow(0.805, 0.736)
    v_arrow(0.580, 0.512)
    v_arrow(0.305, 0.253)

    # audit feedback: Box3 -> Box1 (routed along the left side)
    ax.plot([0.10, 0.02], [0.3875, 0.3875], color=C_GOLD, linewidth=1.0,
            linestyle=(0, (3.5, 2.0)), zorder=1)
    ax.plot([0.02, 0.02], [0.3875, 0.8875], color=C_GOLD, linewidth=1.0,
            linestyle=(0, (3.5, 2.0)), zorder=1)
    ax.add_patch(FancyArrowPatch((0.02, 0.8875), (0.10, 0.8875), arrowstyle="-|>",
                                 color=C_GOLD, linewidth=1.0, zorder=1,
                                 mutation_scale=10, shrinkA=0, shrinkB=0,
                                 linestyle=(0, (3.5, 2.0))))
    ax.text(0.056, 0.6375, "gap feedback", rotation=90, ha="center", va="center",
            fontsize=6, color=C_GOLD)

    # audit stamp (placed in the gap between Box2 and Box3)
    ax.text(0.86, 0.541, "AUDITED", ha="center", va="center", fontsize=6.0,
            fontweight="bold", color=C_GOLD, zorder=5,
            bbox=dict(boxstyle="round,pad=0.06", facecolor="white",
                      edgecolor=C_GOLD, linewidth=1.2))

    ax.set_title("(c) Design axis: differentiable\ninverse design with independent audit",
                 fontsize=6.5, fontweight="bold", pad=4)


def draw_data_band(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    series = ["3d", "4d", "5d", "4f", "5f"]
    nmax = max(SERIES_SIZE.values())
    w = 0.158
    for i, name in enumerate(series):
        x0 = 0.012 + i * (w + 0.005)
        xc = x0 + w / 2
        ax.add_patch(Rectangle((x0, 0.14), w, 0.74, fill=False,
                               edgecolor="#bbbbbb", linewidth=0.8))
        ax.text(xc - 0.008, 0.62, name, ha="right", va="center", fontsize=8,
                fontweight="bold", color="#333333")
        ax.text(xc + 0.008, 0.62, f"{SERIES_SIZE[name]:,}", ha="left",
                va="center", fontsize=7, color="#444444")
        bar_w = (w - 0.016) * SERIES_SIZE[name] / nmax
        ax.add_patch(Rectangle((x0 + 0.008, 0.21), bar_w, 0.14,
                               color=C_D, alpha=0.75))
    x0 = 0.012 + 5 * (w + 0.005)
    xc = x0 + w / 2
    ax.add_patch(Rectangle((x0, 0.14), w, 0.74, fill=False,
                           edgecolor="#bbbbbb", linewidth=0.8))
    ax.text(xc - 0.008, 0.62, "total", ha="right", va="center", fontsize=8,
            fontweight="bold", color="#333333")
    ax.text(xc + 0.008, 0.62, f"{sum(SERIES_SIZE.values()):,}", ha="left",
            va="center", fontsize=7, color="#444444")


def main():
    mae = read_target_mae()
    stats = read_target_stats()
    transfer = read_transfer()

    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[4.6, 0.75],
                          left=0.075, right=0.985, top=0.895, bottom=0.035,
                          wspace=0.15, hspace=0.55,
                          width_ratios=[1.40, 1.76, 1.16])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :])

    draw_panel_a(ax_a, mae, stats)
    draw_panel_b(ax_b, transfer)
    draw_panel_c(ax_c)
    draw_data_band(ax_d)

    fig.savefig(FIG_PNG, dpi=600)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG, format="svg")
    plt.close(fig)
    print("written", FIG_PNG)
    print("written", FIG_PDF)

    print("[panel a] value check (rf test MAE / p1-p99 span)")
    for k in TARGET_ORDER:
        span = stats[k]["p99"] - stats[k]["p1"]
        print(f"  {k:15s} rf={mae[k]:.6f} span={span:.6f} "
              f"rel={mae[k] / span * 100:.2f}%")

    print("[panel b] directional transfer pairs (eV)")
    for a, b in [("3d", "4f"), ("4d", "4f"), ("5d", "4f"),
                 ("3d", "5f"), ("4d", "5f"), ("5d", "5f")]:
        print(f"  {a}->{b}: {transfer[(a, b)]:.3f} | {b}->{a}: {transfer[(b, a)]:.3f}")

    lo, hi = transfer_range(transfer, {"3d", "4d", "5d"}, {"3d", "4d", "5d"})
    print(f"d-internal (cross) range: {lo:.3f}-{hi:.3f} eV")
    lo, hi = transfer_range(transfer, {"3d", "4d", "5d"}, {"4f"})
    print(f"d->4f range: {lo:.3f}-{hi:.3f} eV")
    lo, hi = transfer_range(transfer, {"4f"}, {"3d", "4d", "5d"})
    print(f"4f->d range: {lo:.3f}-{hi:.3f} eV")
    lo, hi = transfer_range(transfer, {"3d", "4d", "5d"}, {"5f"})
    print(f"d->5f range: {lo:.3f}-{hi:.3f} eV")
    lo, hi = transfer_range(transfer, {"5f"}, {"3d", "4d", "5d"})
    print(f"5f->d range: {lo:.3f}-{hi:.3f} eV")


if __name__ == "__main__":
    main()
