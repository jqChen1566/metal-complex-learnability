# TOC graphic for the JCIM submission (emoji variant).
# Five equal-width rounded cards (3d, 4d, 5d, 4f, 5f) in periodic order.
# Each card carries the series name (11 pt bold white) and dataset size
# (7.5 pt white). Below each card one emoji (7.8 pt, Segoe UI Emoji) sits
# above one black 6.0 pt role line; the footer is black as well.
# Canvas 3.25 x 1.375 in (1950 x 825 px at 600 dpi); no title text.
#
# Outputs: toc_graphic.png / .pdf / .svg (the TIFF is made by make_tiff.py)

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent

mpl.rcParams["font.family"] = ["Arial", "Segoe UI Emoji"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

# name, data size, fill colour, emoji, role line
SERIES = [
    ("3d", 41374, "#4C72B0", "🔋", "weakest charge"),
    ("4d", 37450, "#4C72B0", "⭐", "most learnable"),
    ("5d", 29292, "#4C72B0", "🔁", "best 5f source"),
    ("4f", 17269, "#E07B39", "⚡", "Gd break"),
    ("5f", 2394, "#765898", "🚀", "extrapolative"),
]
TOTAL = 127779
FOOTER = "127,779 complexes · 15 properties · 5 metal series"

X0, X1 = 0.1625, 3.0875      # 90% of the 3.25 in width
GAP = 0.06                   # in
BOX_BOTTOM, BOX_HEIGHT = 0.64, 0.44
NAME_Y = 0.93                # 11 pt line, inside the card
SIZE_Y = 0.78                # 7.5 pt line, inside the card
EMOJI_Y = 0.545              # emoji line, below the card
TEXT_Y = 0.4285              # role line, below the emoji
FOOTER_Y = 0.24
BLACK = "#000000"


def main():
    fig = plt.figure(figsize=(3.25, 1.375))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 3.25)
    ax.set_ylim(0.0, 1.375)
    ax.set_aspect("equal")
    ax.axis("off")

    n = len(SERIES)
    width = (X1 - X0 - (n - 1) * GAP) / n
    if sum(size for _, size, _, _, _ in SERIES) != TOTAL:
        raise ValueError("series sizes do not sum to the footer total")

    for i, (name, size, color, emoji, role) in enumerate(SERIES):
        left = X0 + i * (width + GAP)
        center = left + width / 2
        box = FancyBboxPatch(
            (left, BOX_BOTTOM), width, BOX_HEIGHT,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=0, facecolor=color, edgecolor="none", zorder=2)
        ax.add_patch(box)
        ax.text(center, NAME_Y, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=3)
        ax.text(center, SIZE_Y, f"{size:,}", ha="center", va="center",
                fontsize=7.5, color="white", zorder=3)
        ax.text(center, EMOJI_Y, emoji, ha="center", va="center",
                fontsize=7.8, fontfamily="Segoe UI Emoji", color=BLACK,
                zorder=3)
        ax.text(center, TEXT_Y, role, ha="center", va="center",
                fontsize=6.0, color=BLACK, zorder=3)

    ax.text(3.25 / 2, FOOTER_Y, FOOTER, ha="center", va="center",
            fontsize=7.0, color=BLACK)

    fig.savefig(HERE / "toc_graphic.png", dpi=600)
    fig.savefig(HERE / "toc_graphic.pdf")
    fig.savefig(HERE / "toc_graphic.svg", format="svg")
    plt.close(fig)
    print("written toc_graphic.png / .pdf / .svg")
    print(f"card {width:.3f} x {BOX_HEIGHT:.3f} in, gap {GAP:.3f} in")
    for name, size, color, emoji, role in SERIES:
        print(f"  {name}: {size:,} | U+{ord(emoji):04X} {role} | {color}")


if __name__ == "__main__":
    main()
