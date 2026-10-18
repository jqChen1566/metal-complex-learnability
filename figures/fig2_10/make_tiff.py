# Convert the manuscript figure PNGs to RGB TIFF for ACS submission.
# Output: one .tif next to each source .png (600 dpi, LZW, alpha flattened,
# sRGB ICC profile embedded when available).
#
# Scope: fig*.png in this directory and in the Fig. 1 directory, plus
# toc_graphic.png. The legacy main-figure panels are not part of the
# manuscript figure set and are intentionally left out.

import sys
from pathlib import Path

from PIL import Image, ImageCms

HERE = Path(__file__).resolve().parent
FIG1_DIRS = sorted(p for p in HERE.parent.glob("Fig1_*") if p.is_dir())
DIRS = [HERE] + FIG1_DIRS
EXTRA = [HERE / "toc_graphic.png"]


def srgb_profile():
    try:
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        return profile.tobytes()
    except Exception as exc:  # pragma: no cover - depends on Pillow build
        print(f"warning: sRGB ICC profile unavailable ({exc}); "
              "saving without an embedded profile", file=sys.stderr)
        return None


def main():
    icc = srgb_profile()
    sources = []
    for directory in DIRS:
        sources.extend(sorted(directory.glob("fig*.png")))
    sources.extend(p for p in EXTRA if p.exists())
    if not sources:
        raise SystemExit("no source PNGs found")

    for png in sources:
        image = Image.open(png).convert("RGB")
        target = png.with_suffix(".tif")
        kwargs = {"format": "TIFF", "dpi": (600, 600), "compression": "tiff_lzw"}
        if icc is not None:
            kwargs["icc_profile"] = icc
        image.save(target, **kwargs)
        print(f"{png.name} -> {target.name}  {image.size[0]}x{image.size[1]} px, "
              f"RGB, 600 dpi" + (", sRGB" if icc is not None else ""))


if __name__ == "__main__":
    main()
