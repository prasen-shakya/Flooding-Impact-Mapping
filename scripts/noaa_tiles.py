"""Downloads NOAA storm-imagery map tiles for a box and stitches them into one GeoTIFF (EPSG:3857, about 1 m).

Tiles the flight does not cover are left black (0, 0, 0).

    python noaa_tiles.py <flight> <xmin> <ymin> <xmax> <ymax> <out.tif>     e.g. flight = 20210902a
"""
import io
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_origin

Z, TILE = 17, 256
R = 20037508.342789244  # half the width of the web-mercator world, metres
N = 2 ** Z
RES = 2 * R / (N * TILE)


def fetch(flight, tx, ty):
    """One tile as (256, 256, 3) uint8, or None where the flight has no imagery."""
    url = f"https://stormscdn.ngs.noaa.gov/{flight}-rgb/{Z}/{tx}/{ty}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                return None
            img = Image.open(io.BytesIO(r.content))
            if img.size != (TILE, TILE):  # the server answers a 1x1 image for an empty tile
                return None
            rgba = np.array(img.convert("RGBA"))
            return np.where(rgba[..., 3:] > 0, rgba[..., :3], 0)
        except requests.RequestException:
            if attempt == 2:
                raise


def main(flight, xmin, ymin, xmax, ymax, out):
    tx0, tx1 = math.floor((xmin + R) / (2 * R) * N), math.floor((xmax + R) / (2 * R) * N)
    ty0, ty1 = math.floor((R - ymax) / (2 * R) * N), math.floor((R - ymin) / (2 * R) * N)
    tiles = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    assert 0 < len(tiles) < 5000, f"{len(tiles)} tiles: check the extent numbers"

    mosaic = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), np.uint8)
    got = 0
    with ThreadPoolExecutor(8) as pool:
        for (tx, ty), img in zip(tiles, pool.map(lambda t: fetch(flight, *t), tiles)):
            if img is not None:
                mosaic[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = img
                got += 1

    transform = from_origin(tx0 * TILE * RES - R, R - ty0 * TILE * RES, RES, RES)
    with rasterio.open(out, "w", driver="GTiff", height=mosaic.shape[0], width=mosaic.shape[1], count=3,
                       dtype="uint8", crs="EPSG:3857", transform=transform, compress="deflate") as dst:
        dst.write(mosaic.transpose(2, 0, 1))
    print(f"{got}/{len(tiles)} tiles have imagery")
    return got


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]), float(a[3]), float(a[4]), float(a[5]), a[6])
