"""Downloads the USGS 3DEP 1 m elevation for a big mosaic, on the mosaic's own pixel grid.

Writes <name>_dem.tif next to the image: float32 metres, same size/grid, -9999 where the mosaic has no imagery.
Only blocks that contain imagery are downloaded. Safe to re-run: finished blocks are skipped.

    python dem_for_mosaic.py path/to/mosaic.tif [more.tif ...]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.windows import Window

BLOCK = 2048  # 2048x2048 float32 = 17 MB, under Earth Engine's 32 MB per-download limit
NODATA = -9999.0


def fetch(dem, crs, transform, win):
    """One block of elevation from Earth Engine, exactly on the mosaic grid."""
    t = rasterio.windows.transform(win, transform)
    params = {"format": "GEO_TIFF", "crs": crs,
              "crs_transform": [t.a, 0, t.c, 0, t.e, t.f],
              "dimensions": [int(win.width), int(win.height)]}
    for attempt in range(4):
        try:
            r = requests.get(dem.getDownloadURL(params), timeout=300)
            r.raise_for_status()
            with MemoryFile(r.content) as mem, mem.open() as src:
                z = src.read(1)
            assert z.shape == (win.height, win.width), (z.shape, win)
            return np.where(np.isfinite(z), z, NODATA).astype("float32")
        except Exception as e:  # network hiccups and Earth Engine rate limits
            if attempt == 3:
                raise
            print(f"  retry {win.col_off},{win.row_off}: {e}")
            time.sleep(10 * (attempt + 1))


def main(path):
    path = Path(path)
    out = path.with_name(path.stem + "_dem.tif")
    dem = ee.ImageCollection("USGS/3DEP/1m").select("elevation").mosaic().toFloat()

    with rasterio.open(path) as img:
        crs = img.crs.to_string()  # the request must use the mosaic's own CRS (these are NAD83, not WGS84)
        profile = img.profile
        profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate", predictor=3,
                       tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES", photometric=None)
        if not out.exists():
            rasterio.open(out, "w", **profile).close()

        with rasterio.open(out, "r+") as dst:
            todo = []
            for r in range(0, img.height, BLOCK):
                for c in range(0, img.width, BLOCK):
                    win = Window(c, r, min(BLOCK, img.width - c), min(BLOCK, img.height - r))
                    has_image = img.read(1, window=win, out_shape=(64, 64)).any()
                    done = (dst.read(1, window=win, out_shape=(64, 64)) != NODATA).any()
                    if has_image and not done:
                        todo.append(win)
            print(f"{path.name}: {len(todo)} blocks to download")

            with ThreadPoolExecutor(4) as pool:
                for n, (win, z) in enumerate(zip(todo, pool.map(lambda w: fetch(dem, crs, img.transform, w), todo)), 1):
                    dst.write(z, 1, window=win)
                    if n % 10 == 0 or n == len(todo):
                        print(f"  {n}/{len(todo)}", flush=True)
    print("saved", out)


if __name__ == "__main__":
    ee.Initialize(project="flood-capstone-514209")
    for p in sys.argv[1:]:
        main(p)
