"""Cuts each image into 256x256 tiles in memory. Nothing is saved to disk.

- X: (N, 256, 256, 4) float32. Channels are R, G, B (0-1) and the DEM, centred and scaled per image.
- y: (N, 256, 256) int8. 0 = dry, 1 = flood, -1 = unknown.

The split is by whole image, so no tile from a test image ends up in training.
"""
from pathlib import Path

import numpy as np
import rasterio

# Relative to this file, so it works no matter which folder the notebook runs from
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TILE = 256

# True: DEM = (DEM - image mean) / image std, so a small dip on flat ground counts as much as a big one on hilly ground.
# The held-out Ida image is nearly flat (std 0.4 m; Houston images are 1-4 m) and the metre-scaled model saw no low ground there.
# False: DEM = DEM - image mean, in metres. Weights trained before 2026-11-10 (0.834, 0.825) need False.
DEM_STANDARDIZE = True

SPLITS = {
    "train": [1, 2, 3, 8, 9, 13, 14, 15, 16, 17, 19],  # areas we labeled: 13, 14, 19 = Florence; 15-17 = Harvey (west Houston)
    "val": [12],
    "test": [4, 5, 6, 7, 10, 11],
    # Held-out hard case from Hurricane Ida (flat coastal Louisiana). Never train, validate or pick a cutoff on it.
    "ida_hard_case": [18],
}


def load_image(i):
    """Reads RGB, DEM and label. Pixels with no image data (alpha 0) get label -1."""
    folder = DATA_DIR / f"image_{i}"
    with rasterio.open(folder / "image.tif") as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        alpha = src.read(4)
    with rasterio.open(folder / "dem.tif") as src:
        dem = src.read(1)
    with rasterio.open(folder / "label.tif") as src:
        label = src.read(1).astype(np.int8)  # signed, so -1 fits

    label[alpha == 0] = -1
    return rgb, dem, label


def make_input(rgb, dem):
    """Scales RGB to 0-1, centres (and by default scales) the elevation, stacks into (H, W, 4)."""
    rgb = rgb / 255
    dem = dem - dem.mean()
    if DEM_STANDARDIZE:
        # clipped: a levee or pit can sit 10 std from the mean, and inputs that large made the training loss go NaN
        dem = np.clip(dem / dem.std(), -5, 5)
    return np.dstack([rgb, dem]).astype(np.float32)


def cut_tiles(x, label, stride, max_unknown):
    """Slides a TILE x TILE window over the image. The last row/col is snapped to the edge."""
    H, W = label.shape
    rows = list(range(0, H - TILE, stride)) + [H - TILE]
    cols = list(range(0, W - TILE, stride)) + [W - TILE]

    x_tiles, y_tiles = [], []
    for r in rows:
        for c in cols:
            x_tile = x[r:r + TILE, c:c + TILE]
            y_tile = label[r:r + TILE, c:c + TILE]

            unknown = (y_tile == -1).mean()
            if unknown > max_unknown:
                continue

            x_tiles.append(x_tile)
            y_tiles.append(y_tile)
    return x_tiles, y_tiles


def get_tiles(images, stride=TILE, max_unknown=1.0):
    """Runs load -> scale -> cut for each image and stacks all tiles into two arrays."""
    X, y = [], []
    for i in images:
        rgb, dem, label = load_image(i)
        x = make_input(rgb, dem)
        x_tiles, y_tiles = cut_tiles(x, label, stride, max_unknown)
        X += x_tiles
        y += y_tiles
    return np.stack(X), np.stack(y)


def load_dataset():
    """Returns {"train": (X, y), "val": (X, y), "test": (X, y)}.

    Training tiles overlap by half and skip tiles that are mostly unknown.
    Val and test keep every tile.
    """
    return {
        "train": get_tiles(SPLITS["train"], stride=128, max_unknown=0.5),
        "val": get_tiles(SPLITS["val"]),
        "test": get_tiles(SPLITS["test"]),
    }