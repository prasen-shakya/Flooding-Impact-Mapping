"""Loads FloodNet (drone photos from Hurricane Harvey) as an extra test set for the Houston model.

- load_floodnet_image: one full-size photo. rgb (H, W, 3) uint8, label (H, W) int8.
- get_floodnet_tiles: every photo shrunk to the Houston pixel size (1.8 m) and padded to a tile.
  X: (N, 256, 256, 3) float32, RGB 0-1. There is no DEM, so use it with the RGB-only model.
  y: (N, 256, 256) int8. 0 = dry, 1 = flood, -1 = unknown (also used for the padding).
- iter_full_res_tiles: tiles of any size cut from the original photos, for training a separate
  model at drone resolution. Yields one tile at a time instead of returning an array.

FloodNet's 10 classes are merged into those three values by REMAP.
"""
import re
from pathlib import Path

import numpy as np
from PIL import Image

# Relative to this file, so it works no matter which folder the notebook runs from
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FLOODNET_DIR = DATA_DIR / "floodnet" / "FloodNet-Supervised_v1.0"

CLASSES = ["Background", "Building-flooded", "Building-non-flooded", "Road-flooded",
           "Road-non-flooded", "Water", "Tree", "Vehicle", "Pool", "Grass"]

# Indexed by FloodNet class id. Any standing water (pools, rivers, ponds) counts as flood.
REMAP = np.array([
    -1,  # Background (a mix of lawns, fields and some unlabeled rivers, so ignored)
    1,   # Building-flooded
    0,   # Building-non-flooded
    1,   # Road-flooded
    0,   # Road-non-flooded
    1,   # Water
    0,   # Tree
    0,   # Vehicle
    1,   # Pool
    0,   # Grass
], dtype=np.int8)


TILE = 256
TARGET_M_PER_PX = 1.8   # Houston scenes
SPLITS = ["train", "val", "test"]   # FloodNet's own splits. All of them are test data for us.
CACHE = DATA_DIR / "floodnet" / "small_1.8m.npz"

# Sensor width (mm) of the three cameras in FloodNet, found in the photos' EXIF "Model" tag
SENSOR_WIDTH_MM = {
    "FC220": 6.17,        # DJI Mavic Pro (1,911 photos)
    "FC330": 6.17,        # DJI Phantom 4 (80 photos)
    "NIKON 1 S2": 13.2,   # 352 photos
}
DEFAULT_ALTITUDE_M = 60   # median flight height; the Nikon photos don't record one


def list_names(split):
    """Image names (without extension) in "train", "val" or "test", sorted."""
    folder = FLOODNET_DIR / split / f"{split}-org-img"
    return sorted(p.stem for p in folder.glob("*.jpg"))


def load_floodnet_image(split, name):
    """Reads one photo and its mask, and remaps the mask to 0 / 1 / -1."""
    rgb = np.array(Image.open(FLOODNET_DIR / split / f"{split}-org-img" / f"{name}.jpg").convert("RGB"))
    classes = np.array(Image.open(FLOODNET_DIR / split / f"{split}-label-img" / f"{name}_lab.png"))
    label = REMAP[classes]
    return rgb, label


def meters_per_pixel(split, name):
    """Ground size of one pixel, from flight height x sensor width / (focal length x image width)."""
    path = FLOODNET_DIR / split / f"{split}-org-img" / f"{name}.jpg"
    img = Image.open(path)
    exif = img.getexif()
    model = exif.get(272, "").strip("\x00 ")
    focal_mm = float(exif.get_ifd(0x8769)[37386])

    # DJI stores height above the takeoff point in XMP text near the start of the file
    with open(path, "rb") as f:
        match = re.search(rb'RelativeAltitude="([+-]?[\d.]+)"', f.read(200_000))
    altitude = float(match.group(1)) if match else DEFAULT_ALTITUDE_M

    return altitude * SENSOR_WIDTH_MM[model] / (focal_mm * img.width)


def shrink(rgb, label, m_per_px):
    """Shrinks a photo and its label to TARGET_M_PER_PX.

    RGB is averaged (box filter); labels use nearest so classes don't blend.
    """
    scale = m_per_px / TARGET_M_PER_PX
    h, w = label.shape
    size = (min(TILE, round(w * scale)), min(TILE, round(h * scale)))
    small_rgb = np.array(Image.fromarray(rgb).resize(size, Image.BOX))
    small_label = np.array(Image.fromarray((label + 1).astype(np.uint8)).resize(size, Image.NEAREST))
    small_label = small_label.astype(np.int8) - 1   # shifted by 1 above because PIL has no int8 mode
    return small_rgb, small_label


def pad_to_tile(small_rgb, small_label):
    """Pads to TILE x TILE and scales RGB to 0-1.

    The image padding mirrors the photo so the model doesn't see a hard black edge;
    the label padding is -1 so it never counts in the loss or the scores.
    """
    h, w = small_label.shape
    pad = ((0, TILE - h), (0, TILE - w))
    x = np.pad(small_rgb, pad + ((0, 0),), mode="symmetric") / 255
    y = np.pad(small_label, pad, constant_values=-1)
    return x.astype(np.float32), y


def build_cache():
    """Reads every photo (about 10 minutes) and saves the shrunken versions to CACHE.

    Only the small photos are stored, not the padding. They come in a few sizes, so each one
    sits in the top-left of a shared (N, max_h, max_w) array and "sizes" records its real size.
    """
    rgbs, labels, split_of = [], [], []
    for split in SPLITS:
        for name in list_names(split):
            rgb, label = load_floodnet_image(split, name)
            small_rgb, small_label = shrink(rgb, label, meters_per_pixel(split, name))
            rgbs.append(small_rgb)
            labels.append(small_label)
            split_of.append(split)

    sizes = np.array([l.shape for l in labels])
    max_h, max_w = sizes.max(axis=0)
    rgb_arr = np.zeros((len(rgbs), max_h, max_w, 3), np.uint8)
    label_arr = np.full((len(rgbs), max_h, max_w), -1, np.int8)
    for i, (r, l) in enumerate(zip(rgbs, labels)):
        h, w = l.shape
        rgb_arr[i, :h, :w] = r
        label_arr[i, :h, :w] = l
    np.savez_compressed(CACHE, rgb=rgb_arr, label=label_arr, sizes=sizes, split=np.array(split_of))


def load_small_photos(splits=SPLITS):
    """The shrunken photos and labels from CACHE, as two lists. Builds CACHE on the first run."""
    if not CACHE.exists():
        build_cache()
    # Read each array once; indexing the npz file directly would decompress it on every access
    with np.load(CACHE) as data:
        rgb, label, sizes, split_of = data["rgb"], data["label"], data["sizes"], data["split"]

    rgbs, labels = [], []
    for i in np.flatnonzero(np.isin(split_of, splits)):
        h, w = sizes[i]
        rgbs.append(rgb[i, :h, :w])
        labels.append(label[i, :h, :w])
    return rgbs, labels


def get_floodnet_tiles(splits=SPLITS):
    """One 256 x 256 tile per photo, in the top-left corner. For testing the Houston model."""
    X, y = [], []
    for small_rgb, small_label in zip(*load_small_photos(splits)):
        x_tile, y_tile = pad_to_tile(small_rgb, small_label)
        X.append(x_tile)
        y.append(y_tile)
    return np.stack(X), np.stack(y)


def random_flip_rotate(rgb, label, rng):
    """Random 90-degree rotation and horizontal flip, applied the same way to photo and label."""
    k = rng.integers(4)
    rgb, label = np.rot90(rgb, k), np.rot90(label, k)
    if rng.random() < 0.5:
        rgb, label = rgb[:, ::-1], label[:, ::-1]
    return rgb, label


def get_floodnet_mosaics(splits=("train",), seed=None, flip_rotate=True):
    """Packs many shrunken photos into each 256 x 256 tile, for training at the Houston scale.

    Same output format as get_floodnet_tiles, but ~85% of each tile is real photo instead of ~2%.
    Photos go in random order, left to right in rows (like words on a page):
    - A row is as tall as its tallest photo; shorter photos are mirror-padded down to fill it,
      with -1 labels, so there are no empty holes.
    - Photos that hang over the right or bottom edge are cut off there.
    Each seed gives a different arrangement, so call it with a new seed every epoch
    (e.g. seed=epoch) to show the model new combinations.
    """
    rng = np.random.default_rng(seed)
    photos = list(zip(*load_small_photos(splits)))
    order = rng.permutation(len(photos))

    X, y = [], []
    i = 0
    while i < len(order):
        x_tile = np.zeros((TILE, TILE, 3), np.uint8)
        y_tile = np.full((TILE, TILE), -1, np.int8)
        top = 0
        while top < TILE and i < len(order):
            # Pick the photos for this row until it reaches the right edge
            row = []
            width = 0
            while width < TILE and i < len(order):
                rgb, label = photos[order[i]]
                if flip_rotate:
                    rgb, label = random_flip_rotate(rgb, label, rng)
                row.append((rgb, label))
                width += label.shape[1]
                i += 1

            row_h = max(label.shape[0] for _, label in row)
            left = 0
            for rgb, label in row:
                h, w = label.shape
                pad = ((0, row_h - h), (0, 0))
                rgb = np.pad(rgb, pad + ((0, 0),), mode="symmetric")
                label = np.pad(label, pad, constant_values=-1)
                fit_h, fit_w = min(row_h, TILE - top), min(w, TILE - left)
                x_tile[top:top + fit_h, left:left + fit_w] = rgb[:fit_h, :fit_w]
                y_tile[top:top + fit_h, left:left + fit_w] = label[:fit_h, :fit_w]
                left += w
            top += row_h

        if top < TILE:
            # The last tile ran out of photos; mirror what's there into the empty bottom
            x_tile = np.pad(x_tile[:top], ((0, TILE - top), (0, 0), (0, 0)), mode="symmetric")
        X.append((x_tile / 255).astype(np.float32))
        y.append(y_tile)
    return np.stack(X), np.stack(y)


def cut_full_res_tiles(x, label, tile, stride, max_unknown):
    """Same sliding window as dataset.cut_tiles, with the tile size as a parameter.

    A photo smaller than the tile is padded first, the same way as pad_to_tile.
    """
    H, W = label.shape
    if H < tile or W < tile:
        pad = ((0, max(0, tile - H)), (0, max(0, tile - W)))
        x = np.pad(x, pad + ((0, 0),), mode="symmetric")
        label = np.pad(label, pad, constant_values=-1)
        H, W = label.shape
    rows = list(range(0, H - tile, stride)) + [H - tile]
    cols = list(range(0, W - tile, stride)) + [W - tile]
    for r in rows:
        for c in cols:
            y_tile = label[r:r + tile, c:c + tile]
            if (y_tile == -1).mean() > max_unknown:
                continue
            yield x[r:r + tile, c:c + tile], y_tile


def iter_full_res_tiles(splits=SPLITS, tile=512, stride=None, max_unknown=1.0, downscale=1,
                        shuffle=False, seed=None):
    """Yields (x, y) tiles cut from the original photos, one photo at a time.

    For training a separate model on FloodNet at drone resolution, with any input size.
    x: (tile, tile, 3) float32, RGB 0-1. y: (tile, tile) int8, 0 = dry, 1 = flood, -1 = unknown.

    All 2,343 photos at full size would need about 84 GB of memory, so unlike get_floodnet_tiles
    this doesn't return one big array. Loop over it, or wrap it in tf.data.Dataset.from_generator.
    One pass reads every photo from disk again (several minutes).

    downscale: shrink each photo by this factor first (2 -> ~4 cm/px, 4 -> ~8 cm/px), so each
        tile covers more ground. 1 keeps the original size.
    shuffle: shuffles the photo order; tiles from one photo still come out together.
    """
    stride = stride or tile
    photos = [(split, name) for split in splits for name in list_names(split)]
    if shuffle:
        np.random.default_rng(seed).shuffle(photos)

    for split, name in photos:
        rgb, label = load_floodnet_image(split, name)
        if downscale != 1:
            size = (label.shape[1] // downscale, label.shape[0] // downscale)
            rgb = np.array(Image.fromarray(rgb).resize(size, Image.BOX))
            label = np.array(Image.fromarray((label + 1).astype(np.uint8)).resize(size, Image.NEAREST))
            label = label.astype(np.int8) - 1   # shifted by 1 above because PIL has no int8 mode
        x = (rgb / 255).astype(np.float32)
        yield from cut_full_res_tiles(x, label, tile, stride, max_unknown)
