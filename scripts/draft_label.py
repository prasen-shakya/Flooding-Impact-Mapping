"""Makes a rough draft annotation for the annotation tool, to be corrected by hand.

Flood = dark, smooth, low-lying pixels. Dry = bright or high pixels. The rest is left blank (unknown).
Writes <name>_draft.png (red = flood, blue = dry, transparent = blank) and a preview over the photo.

    python draft_label.py path/to/area_folder
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage


def draft(rgb, z):
    gray = rgb.mean(axis=2)
    roughness = ndimage.generic_filter(gray, np.std, size=5)  # forest canopy is rough, water is smooth
    low = z < np.percentile(z, 50)

    flood = (gray < np.percentile(gray, 30)) & (roughness < 6) & low
    dry = ((gray > np.percentile(gray, 60)) | (z > np.percentile(z, 80))) & ~flood

    # drop specks, fill small holes
    flood = ndimage.binary_opening(flood, iterations=2)
    flood = ndimage.binary_closing(flood, iterations=3)
    labels, n = ndimage.label(flood)
    sizes = ndimage.sum(flood, labels, range(1, n + 1))
    flood = np.isin(labels, np.nonzero(sizes >= 300)[0] + 1)  # isolated blobs under ~0.1 ha are mostly shadows/tree lines
    dry = ndimage.binary_opening(dry, iterations=2) & ~flood
    return flood, dry


def main(folder):
    folder = Path(folder)
    with rasterio.open(folder / "image.tif") as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0).astype(float)
    with rasterio.open(folder / "dem.tif") as src:
        z = src.read(1)
    flood, dry = draft(rgb, z)

    ann = np.zeros(flood.shape + (4,), np.uint8)
    ann[flood] = [255, 0, 0, 255]  # the tool paints flood as 'red' and dry as 'blue'
    ann[dry] = [0, 0, 255, 255]
    Image.fromarray(ann).save(folder / f"{folder.name}_draft.png")

    preview = rgb.copy()
    preview[flood] = preview[flood] * 0.4 + np.array([255, 0, 0]) * 0.6
    preview[dry] = preview[dry] * 0.6 + np.array([0, 0, 255]) * 0.4
    Image.fromarray(preview.astype(np.uint8)).save(folder / f"{folder.name}_draft_preview.png")

    n = flood.size
    print(f"flood {flood.sum() / n:.1%}  dry {dry.sum() / n:.1%}  blank {1 - (flood | dry).mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
