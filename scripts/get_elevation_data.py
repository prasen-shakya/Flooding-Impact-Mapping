import math
import os
import tempfile

import ee
import rasterio
import requests
from rasterio.merge import merge
from rasterio.warp import transform_bounds

ee.Authenticate()
ee.Initialize(project='flood-capstone-514209')  # replace with your GCP project id

MAX_BYTES_PER_REQUEST = 25 * 1024 * 1024  # stay under the ~32 MB getDownloadUrl limit


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def get_lat_lon(input_filepath):
    """Return [lat1, lon1, lat2, lon2] = top-left, bottom-right in WGS84,
    regardless of the input raster's CRS."""
    with rasterio.open(input_filepath) as ds:
        west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    return [north, west, south, east]


def load_3dep_1m(region):
    col = ee.ImageCollection('USGS/3DEP/1m').filterBounds(region)
    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError("No USGS 3DEP 1m coverage for this area "
                           "(fall back to 'USGS/3DEP/10m').")
    print(f"Found {n} 1m 3DEP tile(s) intersecting the area.")
    return col.select('elevation').mosaic().toFloat()


def download(image, region, scale, crs, path):
    url = image.getDownloadURL({
        'region': region,
        'scale': scale,
        'crs': crs,
        'format': 'GEO_TIFF',   # plain GeoTIFF, not a zip
    })
    r = requests.get(url, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"Download failed ({r.status_code}): {r.text[:500]}")
    with open(path, 'wb') as f:
        f.write(r.content)


def export_usgs_3dep_data_locally(lat_lon, outfile_path, scale=1):
    lat1, lon1, lat2, lon2 = lat_lon
    west, east = sorted([lon1, lon2])
    south, north = sorted([lat1, lat2])

    region = ee.Geometry.Rectangle([west, south, east, north])
    dem = load_3dep_1m(region)

    # Output in local UTM (meters) so a 1 m pixel is really 1 m x 1 m.
    crs = utm_epsg((west + east) / 2, (south + north) / 2)

    # Estimate size and split into tiles if needed (float32 = 4 bytes/pixel).
    width_m = (east - west) * 111320 * math.cos(math.radians((south + north) / 2))
    height_m = (north - south) * 110540
    est_bytes = (width_m / scale) * (height_m / scale) * 4
    n = max(1, math.ceil(math.sqrt(est_bytes / MAX_BYTES_PER_REQUEST)))
    print(f"Area ~{width_m:.0f} x {height_m:.0f} m -> {n}x{n} tile(s), CRS {crs}")

    if n == 1:
        download(dem, region, scale, crs, outfile_path)
    else:
        dx, dy = (east - west) / n, (north - south) / n
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(n):
                for j in range(n):
                    tile = ee.Geometry.Rectangle([west + i * dx, south + j * dy,
                                                  west + (i + 1) * dx, south + (j + 1) * dy])
                    p = os.path.join(tmp, f"tile_{i}_{j}.tif")
                    print(f"  downloading tile {len(paths) + 1}/{n * n}")
                    download(dem, tile, scale, crs, p)
                    paths.append(p)

            srcs = [rasterio.open(p) for p in paths]
            mosaic, transform = merge(srcs)
            profile = srcs[0].profile
            profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                           transform=transform, compress='deflate')
            for s in srcs:
                s.close()
            with rasterio.open(outfile_path, 'w', **profile) as dst:
                dst.write(mosaic)

    print(f"File saved locally at {outfile_path}")


if __name__ == "__main__":
    input_filepath = "image.tif"
    lat_lon = get_lat_lon(input_filepath)  # [lat_top, lon_left, lat_bottom, lon_right]

    outfile_path = 'dem_1m.tif'
    scale = 1  # meters
    export_usgs_3dep_data_locally(lat_lon, outfile_path, scale)