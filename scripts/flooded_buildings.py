"""Labels every building in our images as flooded or not flooded.

For each image:
1. Load the building footprints (from data/buildings/) that fall inside the image.
2. Draw each footprint onto the image's pixel grid, then grow it by BUFFER_PX pixels.
   The buffer matters: a roof is above the water, so the flood shows up around a house, not on it.
3. Look at the flood map under that grown footprint. The flood map is either
   - the hand-drawn label (label.tif), or
   - a model's prediction (pred_<model>.tif from scripts/predict.py), cut at --cutoff.
   flood_frac = flood pixels / known pixels. Flooded if flood_frac >= FLOOD_THRESHOLD.
   If too few pixels are known (no image data), the building is "unknown".

Writes data/buildings/flooded_buildings_<source>.csv (one row per building, per image),
where <source> is "label" or the model name.

Run from anywhere:
    python scripts/flooded_buildings.py                                  # hand-drawn labels
    python scripts/flooded_buildings.py --model unet_dem --cutoff 0.4    # model predictions
Missing building files are downloaded automatically (same as python scripts/buildings.py).
Model predictions need:  python scripts/predict.py outputs/unet_dem.weights.h5
"""
import argparse
import gzip
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform, transform_bounds
from scipy.ndimage import binary_dilation

from buildings import BUILDINGS_DIR, NUM_IMAGES, ensure_files, image_quadkeys
from dataset import DATA_DIR
from predict import pred_path

BUFFER_PX = 2            # ~4 m at our ~2 m/pixel resolution
FLOOD_THRESHOLD = 0.25   # at least a quarter of the pixels around the house are water
MIN_KNOWN = 0.5          # need image data for at least half of those pixels


def out_csv(source="label"):
    return BUILDINGS_DIR / f"flooded_buildings_{source}.csv"


def flood_map(i, model=None, cutoff=0.5):
    """(H, W) int8: 1 = flood, 0 = dry, -1 = unknown. From label.tif, or from a model's prediction."""
    with rasterio.open(DATA_DIR / f"image_{i}" / "image.tif") as src:
        alpha = src.read(4)
    if model is None:
        with rasterio.open(DATA_DIR / f"image_{i}" / "label.tif") as src:
            flood = src.read(1).astype(np.int8)
    else:
        with rasterio.open(pred_path(i, model)) as src:
            prob = src.read(1)
        flood = (prob > cutoff).astype(np.int8)
        flood[prob < 0] = -1
    flood[alpha == 0] = -1
    return flood


def load_buildings(i):
    """Buildings whose box overlaps image i. Returns (rings in lon/lat, rings in image coords, crs, transform, shape)."""
    with rasterio.open(DATA_DIR / f"image_{i}" / "image.tif") as src:
        crs, tf, shape = src.crs, src.transform, (src.height, src.width)
        lon_w, lat_s, lon_e, lat_n = transform_bounds(crs, "EPSG:4326", *src.bounds)

    quadkeys = sorted(image_quadkeys(i))
    ensure_files(quadkeys)  # downloads any building file this image needs that isn't here yet

    rings = []
    # Sorted, so building numbers are the same on every run: the label and model CSVs are matched on them
    for qk in quadkeys:
        for path in sorted(BUILDINGS_DIR.glob(f"*_{qk}.csv.gz")):
            with gzip.open(path, "rt") as f:
                for line in f:
                    geometry = json.loads(line)["geometry"]
                    if geometry["type"] != "Polygon":
                        continue
                    ring = np.array(geometry["coordinates"][0])
                    lons, lats = ring[:, 0], ring[:, 1]
                    if lons.max() < lon_w or lons.min() > lon_e:
                        continue
                    if lats.max() < lat_s or lats.min() > lat_n:
                        continue
                    rings.append(ring)
    return rings, crs, tf, shape


def label_image(i, model=None, cutoff=0.5):
    rings, crs, tf, shape = load_buildings(i)
    label = flood_map(i, model, cutoff)

    H, W = shape
    struct = np.ones((3, 3), bool)
    rows = []
    for b, ring in enumerate(rings):
        xs, ys = transform("EPSG:4326", crs, ring[:, 0].tolist(), ring[:, 1].tolist())
        geom = {"type": "Polygon", "coordinates": [list(zip(xs, ys))]}

        # Pixel box of the building, padded by the buffer, clipped to the image
        cols, rws = ~tf * (np.array(xs), np.array(ys))
        r0 = max(int(np.floor(rws.min())) - BUFFER_PX - 1, 0)
        r1 = min(int(np.ceil(rws.max())) + BUFFER_PX + 1, H)
        c0 = max(int(np.floor(cols.min())) - BUFFER_PX - 1, 0)
        c1 = min(int(np.ceil(cols.max())) + BUFFER_PX + 1, W)
        if r1 <= r0 or c1 <= c0:
            continue  # building is outside the image

        # Draw the footprint just in that small box, then grow it
        win_tf = tf * tf.translation(c0, r0)
        mask = rasterize([geom], out_shape=(r1 - r0, c1 - c0), transform=win_tf,
                         all_touched=True, dtype=np.uint8).astype(bool)
        if not mask.any():
            continue
        footprint_px = int(mask.sum())
        mask = binary_dilation(mask, struct, iterations=BUFFER_PX)

        lab = label[r0:r1, c0:c1][mask]
        known = lab != -1
        known_frac = known.mean()
        flood_frac = (lab[known] == 1).mean() if known.any() else np.nan

        if known_frac < MIN_KNOWN:
            status = "unknown"
        elif flood_frac >= FLOOD_THRESHOLD:
            status = "flooded"
        else:
            status = "not_flooded"

        rows.append({
            "image": i,
            "building": b,
            "center_lon": ring[:, 0].mean(),
            "center_lat": ring[:, 1].mean(),
            "footprint_px": footprint_px,
            "known_frac": round(float(known_frac), 3),
            "flood_frac": round(float(flood_frac), 3),
            "status": status,
            "geometry": json.dumps(ring.tolist()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="use pred_<model>.tif instead of label.tif, e.g. unet_dem")
    parser.add_argument("--cutoff", type=float, default=0.5, help="flood if probability > cutoff (model only)")
    args = parser.parse_args()

    tables = []
    for i in range(1, NUM_IMAGES + 1):
        if args.model and not pred_path(i, args.model).exists():
            print(f"image_{i}: no {pred_path(i, args.model).name}, skipping (run scripts/predict.py)")
            continue
        df = label_image(i, args.model, args.cutoff)
        counts = df.status.value_counts().to_dict() if len(df) else {}
        print(f"image_{i}: {len(df)} buildings {counts}")
        tables.append(df)
    out = pd.concat(tables, ignore_index=True)
    path = out_csv(args.model or "label")
    out.to_csv(path, index=False)
    print("\nTotal:", out.status.value_counts().to_dict())
    print("Wrote", path)


if __name__ == "__main__":
    main()
