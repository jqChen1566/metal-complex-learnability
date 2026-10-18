# Fig. 8: fine-tuning curves on the target series.
# Test MAE on the target versus target-set size (25, 100, 500 complexes),
# averaged over ten draws; error bars are +/-1 SD across draws. Each curve is
# fine-tuning from one source series, compared with training from scratch.
# transfer_curve.csv stores MAE in Ha; values are converted to eV here.
#
# Input : data_raw/transfer_curve.csv
# Output: fig8_finetune_curves.png / .pdf

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
PNG = HERE / "fig8_finetune_curves.png"
PDF = HERE / "fig8_finetune_curves.pdf"
SVG = HERE / "fig8_finetune_curves.svg"

EV_PER_HA = 27.2114
BUDGETS = [25, 100, 500]

C_D = "#4C72B0"
C_4F = "#DD8452"
C_SCRATCH = "#444444"

PANELS = [
    ("4f", [("3d", C_D), ("4f", C_4F)]),
    ("5f", [("5d", C_D), ("4f", C_4F)]),
]


def load_rows():
    with open(RAW / "transfer_curve.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        out.append({
            "src": row["src"],
            "tgt": row["tgt"],
            "n": int(row["n"]),
            "draw": int(row["draw"]),
            "scratch": float(row["scratch_mae"]) * EV_PER_HA,
            "finetune": float(row["finetune_mae"]) * EV_PER_HA,
        })
    return out


def stats(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def finetune_stats(rows, src, tgt, n):
    values = [r["finetune"] for r in rows
              if r["src"] == src and r["tgt"] == tgt and r["n"] == n]
    if len(values) != 10:
        raise ValueError(f"expected 10 draws for {src}->{tgt} n={n}, got {len(values)}")
    return stats(values)


def scratch_stats(rows, tgt, n):
    values = {}
    for r in rows:
        if r["tgt"] == tgt and r["n"] == n:
            key = r["draw"]
            if key in values and abs(values[key] - r["scratch"]) > 1e-12:
                raise ValueError(
                    f"scratch MAE inconsistent across sources for {tgt} n={n} "
                    f"draw={key}")
            values[key] = r["scratch"]
    if len(values) != 10:
        raise ValueError(f"expected 10 scratch draws for {tgt} n={n}, got {len(values)}")
    return stats(list(values.values()))


def main():
    rows = load_rows()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.155,
                        wspace=0.28)
    x = np.arange(len(BUDGETS))

    for panel_index, (ax, (tgt, sources)) in enumerate(zip(axes, PANELS)):
        top = 0.0
        for src, color in sources:
            means, sds = zip(*(finetune_stats(rows, src, tgt, n) for n in BUDGETS))
            ax.errorbar(x, means, yerr=sds, color=color, marker="o", markersize=4.5,
                        linewidth=1.4, elinewidth=0.9, capsize=3,
                        label=f"from {src}")
            top = max(top, max(m + s for m, s in zip(means, sds)))

        scratch_means, scratch_sds = zip(*(scratch_stats(rows, tgt, n)
                                           for n in BUDGETS))
        ax.errorbar(x, scratch_means, yerr=scratch_sds, color=C_SCRATCH,
                    marker="s", markersize=4.5, markerfacecolor="white",
                    linewidth=1.3, linestyle="--", elinewidth=0.9, capsize=3,
                    label="from scratch")
        top = max(top, max(m + s for m, s in zip(scratch_means, scratch_sds)))

        ax.set_xticks(x, [str(n) for n in BUDGETS], fontsize=9)
        ax.set_xlim(-0.25, len(BUDGETS) - 0.75)
        ax.set_ylim(0, top * 1.12)
        ax.set_xlabel("Target-set size", fontsize=9.5)
        ax.set_ylabel("Test MAE on target (eV)", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95,
                  edgecolor="#cccccc")
        ax.set_title(f"({chr(97 + panel_index)}) Target: {tgt} series",
                     fontsize=10, pad=6)

        print(f"[Fig.8] target {tgt} (eV)")
        for src, _ in sources:
            vals = [finetune_stats(rows, src, tgt, n) for n in BUDGETS]
            print(f"  from {src}: " + "  ".join(
                f"n={n}: {m:.3f} +/- {s:.3f}" for n, (m, s) in zip(BUDGETS, vals)))
        print("  from scratch: " + "  ".join(
            f"n={n}: {m:.3f} +/- {s:.3f}"
            for n, (m, s) in zip(BUDGETS, zip(scratch_means, scratch_sds))))

    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG, format="svg")
    plt.close(fig)
    print("written", PNG)
    print("written", PDF)


if __name__ == "__main__":
    main()
