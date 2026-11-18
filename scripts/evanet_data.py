"""Loads EvaNet's 3 regions as extra training tiles, in the same format as dataset.py.

The regions come from the EvaNet authors (https://github.com/MTSami/EvaNet, dataset on UAB Box).
They are NOAA aerial imagery from Hurricane Matthew 2016 in North Carolina, about 2 m per pixel.
Expected location: data/evanet_regions/Dataset/Region_{1,2,3}_Features7Channel.npy and _labels.npy

- X: (N, 256, 256, 4) float32. Channels are R, G, B (0-1) and the DEM in metres minus the region mean.
- y: (N, 256, 256) int8. 0 = dry, 1 = flood, -1 = unknown.

That is the same as dataset.get_tiles(), so the tiles can be concatenated with the project's own.
Only use them for training. The project's test scenes are in Texas, and these are not.

Feature channels in the files: 0-2 flood-time RGB, 3 elevation, 4-6 pre-flood RGB. The project's
scenes have no pre-flood image, so channels 4-6 are dropped.
"""
import numpy as np

from dataset import DATA_DIR, cut_tiles

REGIONS_DIR = DATA_DIR / "evanet_regions" / "Dataset"
REGIONS = [1, 2, 3]

# The elevations look like US survey feet: Greenville NC spans -2 to 76, which is only plausible as
# feet (-0.6 to 23 m). Neither the paper nor the repo says so. Set to 1.0 to keep the raw values.
DEM_TO_METRES = 0.3048


def load_region(r):
    """Reads RGB (uint8), DEM in metres (float32) and label (int8, 1 / 0 / -1) for one region."""
    features = np.load(REGIONS_DIR / f"Region_{r}_Features7Channel.npy", allow_pickle=False, mmap_mode="r")
    label = np.load(REGIONS_DIR / f"Region_{r}_labels.npy", allow_pickle=False)

    rgb = np.asarray(features[..., :3]).round().clip(0, 255).astype(np.uint8)
    dem = np.asarray(features[..., 3], dtype=np.float32) * DEM_TO_METRES
    label = label.astype(np.int8)  # files mix float64 and int64; values are already 1 / 0 / -1

    assert set(np.unique(label)) <= {-1, 0, 1}, f"Region_{r}: unexpected label values"
    return rgb, dem, label


def make_input(rgb, dem):
    """Same as dataset.make_input: RGB to 0-1, DEM minus its mean, stacked into (H, W, 4)."""
    return np.dstack([rgb / 255, dem - dem.mean()]).astype(np.float32)


def get_region_tiles(regions=REGIONS, stride=128, max_unknown=1.0):
    """Loads each region, cuts it into 256 x 256 tiles and stacks them, like dataset.get_tiles()."""
    X, y = [], []
    for r in regions:
        rgb, dem, label = load_region(r)
        x_tiles, y_tiles = cut_tiles(make_input(rgb, dem), label, stride, max_unknown)
        X += x_tiles
        y += y_tiles
    return np.stack(X), np.stack(y)
