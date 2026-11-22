"""Downloads the building footprint files that cover our images.

Finds every zoom-9 quadkey that touches each image in data/, looks the quadkeys up in
data/buildings/dataset-links.csv, and downloads the matching files into data/buildings/.

Run from anywhere:  python scripts/buildings.py [--dry-run]
"""
import argparse
import math

import pandas as pd
import rasterio
import requests
from rasterio.warp import transform_bounds

from dataset import DATA_DIR

ZOOM = 9
BUILDINGS_DIR = DATA_DIR / "buildings"
LINKS_CSV = BUILDINGS_DIR / "dataset-links.csv"
NUM_IMAGES = 25


def lonlat_to_xy(lon, lat):
    """Column and row of the zoom-9 world grid square that holds this point."""
    n = 2 ** ZOOM
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def xy_to_quadkey(x, y):
    """Builds the quadkey one digit at a time, from the biggest square to the smallest."""
    quadkey = ""
    for level in range(ZOOM, 0, -1):
        bit = 1 << (level - 1)
        digit = 0
        if x & bit:
            digit += 1
        if y & bit:
            digit += 2
        quadkey += str(digit)
    return quadkey


def get_quadkey(lon, lat):
    return xy_to_quadkey(*lonlat_to_xy(lon, lat))


def image_quadkeys(i):
    """Every quadkey square the image touches (uses the whole bounding box, not just tile centers)."""
    with rasterio.open(DATA_DIR / f"image_{i}" / "image.tif") as src:
        # Some images are in meters (UTM), so convert the bounds to lon/lat first
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    x0, y0 = lonlat_to_xy(west, north)  # top-left square
    x1, y1 = lonlat_to_xy(east, south)  # bottom-right square
    keys = set()
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            keys.add(xy_to_quadkey(x, y))
    return keys


def download(url, path, tries=10):
    """Downloads into a .part file, then renames it. The server sometimes stalls mid-file, so on a
    timeout it retries and picks up where it stopped (HTTP Range) instead of starting over."""
    tmp = path.with_suffix(".part")
    for attempt in range(1, tries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=(10, 60), headers=headers) as r:
                if r.status_code == 416:  # nothing left to fetch: the .part file is already complete
                    break
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"  # 200 = server ignored Range, start over
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            break
        except requests.exceptions.RequestException as e:
            if attempt == tries:
                raise
            print(f"   stalled at {tmp.stat().st_size / 1e6 if tmp.exists() else 0:.1f} MB ({type(e).__name__}), "
                  f"retrying {attempt}/{tries - 1}")
    tmp.rename(path)


def ensure_files(quadkeys, dry_run=False):
    """Downloads the building files for these quadkeys that aren't in data/buildings/ yet."""
    # Read as text so leading zeros in the quadkey survive
    links = pd.read_csv(LINKS_CSV, dtype={"QuadKey": str})
    rows = links[links.QuadKey.isin(quadkeys)]

    missing = set(quadkeys) - set(rows.QuadKey)
    if missing:
        print("No download link for:", sorted(missing))

    for r in rows.itertuples():
        # Location is in the name because a quadkey can be listed under more than one region
        path = BUILDINGS_DIR / f"{r.Location}_{r.QuadKey}.csv.gz"
        if path.exists():
            continue
        if dry_run:
            print("would download", path.name, r.Size)
        else:
            print("downloading", path.name, r.Size)
            download(r.Url, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="list the files, don't download")
    args = parser.parse_args()

    needed = set()
    for i in range(1, NUM_IMAGES + 1):
        keys = image_quadkeys(i)
        print(f"image_{i}: {sorted(keys)}")
        needed |= keys
    ensure_files(needed, args.dry_run)


if __name__ == "__main__":
    main()
