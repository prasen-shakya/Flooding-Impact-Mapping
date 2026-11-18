#!/bin/bash
# Cuts one Harvey area for labeling, straight from NOAA's map tiles (no big download):
# image.tif (2 m), dem.tif (1 m USGS averaged onto the same grid), <name>_rgb.png, <name>_elevation.png
#
#   new_area.sh <folder> <xmin> <ymin> <xmax> <ymax> [flight ...]
#
# The four numbers are the QGIS "Extents" readout (EPSG:3857). Flights are tried in order until one covers the area.
# An area that runs past the edge of a flight: name that flight and set MIN_COVER lower, e.g. MIN_COVER=50.
# The empty part gets alpha 0, which dataset.py turns into label -1 (unknown).
# Run inside WSL with the ~/flood environment.
# shortcut: UTM zone 15N and the Harvey flight list are hard-coded; change EPSG and FLIGHTS for another storm.
set -e
DIR=$1; XMIN=$2; YMIN=$3; XMAX=$4; YMAX=$5; shift 5
FLIGHTS=${*:-20170830 20170831a 20170831b 20170829a 20170829b 20170901a 20170901b 20170901c 20170902a 20170902b 20170902c 20170903a 20170828a 20170828b 20170827}
EPSG=EPSG:32615
PY=~/flood/bin/python
HERE=$(dirname "$(readlink -f "$0")")
NAME=$(basename "$DIR")
mkdir -p "$DIR"; cd "$DIR"

for F in $FLIGHTS; do
  $PY "$HERE/noaa_tiles.py" $F $XMIN $YMIN $XMAX $YMAX tiles.tif
  gdalwarp -q -overwrite -te $XMIN $YMIN $XMAX $YMAX -te_srs EPSG:3857 -t_srs $EPSG -tr 2 2 -r average \
           -dstalpha -co COMPRESS=DEFLATE tiles.tif image.tif
  rm -f tiles.tif
  COVER=$($PY -c "import rasterio; x = rasterio.open('image.tif').read([1, 2, 3]); print(int(100 * (x.max(0) > 0).mean()))")
  echo "flight $F covers $COVER%"
  if [ "$COVER" -ge "${MIN_COVER:-98}" ]; then echo "$F" > flight.txt; break; fi
done
[ -f flight.txt ] || { echo "no flight covers this area"; exit 1; }
rm -f image.tif.aux.xml

# 1 m elevation with the professor's script (reads image.tif, writes dem_1m.tif), then onto the photo's grid
cp "$HERE/get_elevation_data.py" .
$PY get_elevation_data.py | tail -1
$PY - <<EOF
import subprocess
import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds

with rasterio.open("image.tif", "r+") as a:
    empty = a.read([1, 2, 3]).max(0) == 0  # no imagery here (past the edge of the flight)
    a.write(np.where(empty, 0, 255).astype("uint8"), 4)
    print(f"no imagery in {empty.mean():.1%} of the area")
    b, w, h, crs = a.bounds, a.width, a.height, a.crs
    Image.fromarray(a.read([1, 2, 3]).transpose(1, 2, 0)).save("${NAME}_rgb.png")
subprocess.run(["gdalwarp", "-q", "-overwrite", "-t_srs", "$EPSG", "-r", "average",
                "-te", str(b.left), str(b.bottom), str(b.right), str(b.top), "-ts", str(w), str(h),
                "dem_1m.tif", "dem.tif"], check=True)
with rasterio.open("image.tif") as a, rasterio.open("dem.tif") as d:
    assert a.transform == d.transform and a.shape == d.shape and a.crs == d.crs, "DEM grid does not match the photo"
    z = d.read(1).astype(float)
assert np.isfinite(z).all(), "DEM has gaps"
lo = np.percentile(z, 0.5)
hi = max(np.percentile(z, 99.5), lo + 15)  # at least 15 m of range, so small bumps do not become walls in 3D
Image.fromarray((np.clip((z - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), "L").save("${NAME}_elevation.png")
l, s, r, t = transform_bounds(crs, "EPSG:4326", *b)
print(f"{w} x {h} px | elevation {z.min():.1f} to {z.max():.1f} m | centre {(s + t) / 2:.4f}, {(l + r) / 2:.4f}")
print(f"bounds lon {l:.4f}..{r:.4f} lat {s:.4f}..{t:.4f}")
EOF
