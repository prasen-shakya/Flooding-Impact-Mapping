"""Turns the annotation tool's PNG into label.tif on the same grid as image.tif.

Red = flood (1), blue = dry (0), unpainted = unknown (-1).

    python ann_to_label.py annotatedImg_X.png path/to/image_N
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


def convert(png_path, folder):
    folder = Path(folder)
    ann = np.array(Image.open(png_path).convert("RGBA")).astype(int)
    r, b, a = ann[..., 0], ann[..., 2], ann[..., 3]

    label = np.full(r.shape, -1, np.int16)  # int16 like the existing label.tif files
    painted = a > 0
    label[painted & (r > b)] = 1
    label[painted & (b > r)] = 0

    with rasterio.open(folder / "image.tif") as src:
        assert (src.height, src.width) == label.shape, f"annotation {label.shape} != image {src.shape}"
        profile = src.profile
    profile.update(count=1, dtype="int16", nodata=None, photometric=None)
    with rasterio.open(folder / "label.tif", "w", **profile) as dst:
        dst.write(label, 1)
    return label


if __name__ == "__main__":
    label = convert(sys.argv[1], sys.argv[2])
    n = label.size
    print(f"flood {(label == 1).sum() / n:.1%}  dry {(label == 0).sum() / n:.1%}  unknown {(label == -1).sum() / n:.1%}")
