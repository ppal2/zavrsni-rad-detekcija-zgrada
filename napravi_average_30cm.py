"""
napravi_average_30cm.py - smanjenje ortofota sa 10 na 30 cm metodom average, iz pune slike od 10 cm

pokrece se u python konzoli qgis-a (plugins -> python console -> show editor -> open -> run),
jer gdal iz qgis-a cita ecw datoteke.

novi raster ima potpuno istu mrezu piksela kao campus_30cm.tif (isti obuhvat, isti broj
redova i kolona), pa se oba mogu porediti piksel po piksel, a ista maska rucnih oznaka
vazi za oba.
"""
from osgeo import gdal
import numpy as np
import sys
from pathlib import Path

# podesavanja.py je u istom folderu kao ovaj program. qgis konzola taj folder
# ne dodaje uvek sama u putanju za import, pa se ovde dodaje rucno.
if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from podesavanja import DATA   # putanje su u podesavanja.py

SRC_10CM = str(DATA / "Campus Novi Sad.ecw")
REF_30CM = str(DATA / "campus_30cm.tif")           # postojeci snimak (bilinear iz nivoa od 20 cm)
OUT_30CM = str(DATA / "campus_30cm_average.tif")   # novi snimak (average iz 10 cm)

gdal.UseExceptions()
ref = gdal.Open(REF_30CM)
gt = ref.GetGeoTransform()
xmin, ymax = gt[0], gt[3]
xmax = xmin + ref.RasterXSize * gt[1]
ymin = ymax + ref.RasterYSize * gt[5]

# -ovr NONE: gdal ne sme da koristi ugradjene nivoe ecw datoteke, nego punu sliku od 10 cm
# -r average: svaki piksel od 30 cm je prosek piksela od 10 cm koje pokriva (3 x 3 = 9)
gdal.Warp(OUT_30CM, SRC_10CM, options=(
    f"-of GTiff -te {xmin} {ymin} {xmax} {ymax} -ts {ref.RasterXSize} {ref.RasterYSize} "
    "-r average -ovr NONE -co COMPRESS=DEFLATE -co TILED=YES"))

# gdal.Warp racuna velicinu piksela iz obuhvata, pa ona izadje 0.29999999999999744 umesto 0.3.
# ocena.py trazi potpuno istu mrezu piksela, zato se prepisuje tacna transformacija iz campus_30cm.tif.
out = gdal.Open(OUT_30CM, gdal.GA_Update)
out.SetGeoTransform(gt)
out = None
print("saved:", OUT_30CM)

# poredjenje sa postojecim snimkom, kanal po kanal
new = gdal.Open(OUT_30CM)
print("size:", new.RasterXSize, "x", new.RasterYSize,
      "(should be", ref.RasterXSize, "x", ref.RasterYSize, ")")
for b in (1, 2, 3):
    a = ref.GetRasterBand(b).ReadAsArray().astype(np.int16)
    c = new.GetRasterBand(b).ReadAsArray().astype(np.int16)
    d = np.abs(a - c)
    print(f"band {b}: mean |difference| = {d.mean():.2f}, max = {d.max()}, "
          f"identical = {100 * (d == 0).mean():.1f} %, within 5 = {100 * (d <= 5).mean():.1f} %")