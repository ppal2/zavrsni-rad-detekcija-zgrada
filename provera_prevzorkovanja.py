"""
provera_prevzorkovanja.py - kojim nacinom prevzorkovanja je napravljen campus_30cm.tif? (verzija 3)

pokrece se u python konzoli qgis-a (plugins -> python console -> "show editor"
-> otvoriti ovu datoteku -> run). koristi se qgis jer njegov gdal moze da cita
ecw datoteke.

verzija 2 je pokazala da pikseli od 30 cm nisu ni srednji piksel ni obican
prosek 3 x 3 piksela snimka od 10 cm. dva verovatna razloga:
  1. ecw datoteka ima ugradjene nivoe manje rezolucije (20 cm, 40 cm, ...).
     kada gdal smanjuje sliku, polazi od najblizeg takvog nivoa, a ne od pune
     slike od 10 cm, pa rezultat nije ni srednji piksel ni prosek;
  2. datoteka od 30 cm je mozda sacuvana sa kompresijom sa gubicima (jpeg).

zato ova verzija ne pogadja formulu, nego ponovo radi prevzorkovanje malih
probnih blokova svakim nacinom (sa ugradjenim nivoima i bez njih) i proverava
koji rezultat je isti kao campus_30cm.tif.
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

# ---- ove dve putanje treba podesiti ---------------------------------------
SRC_10CM = str(DATA / "Campus Novi Sad.ecw")        # originalna datoteka od 10 cm
DST_30CM = str(DATA / "campus_30cm.tif")
N = 300                                              # velicina bloka u pikselima od 30 cm
PLACES = [(0.20, 0.20), (0.20, 0.75), (0.80, 0.60)]  # centar grada, liman, petrovaradin
METHODS = ["near", "bilinear", "cubic", "cubicspline", "lanczos", "average", "mode"]
# ---------------------------------------------------------------------------

gdal.UseExceptions()
src, dst = gdal.Open(SRC_10CM), gdal.Open(DST_30CM)
gs, gd = src.GetGeoTransform(), dst.GetGeoTransform()
k = int(round(gd[1] / gs[1]))                        # odnos velicina piksela (30 / 10 = 3)

# nacin cuvanja datoteke od 30 cm (kompresija) i ugradjeni nivoi ecw datoteke
print("30 cm file storage:", dst.GetMetadata("IMAGE_STRUCTURE"))
b1 = src.GetRasterBand(1)
print("ECW built-in levels:", [f"{b1.GetOverview(i).XSize}x{b1.GetOverview(i).YSize}"
                               for i in range(b1.GetOverviewCount())])

score = {}                                           # nacin -> [isti pikseli, zbir |razlika|, broj piksela]
for fx, fy in PLACES:
    # probni blok: polozaj u datoteci od 30 cm i isti prostor u datoteci od 10 cm
    c30, r30 = int(dst.RasterXSize * fx), int(dst.RasterYSize * fy)
    ref = dst.GetRasterBand(1).ReadAsArray(c30, r30, N, N).astype(float)
    xmin, ymax = gd[0] + c30 * gd[1], gd[3] + r30 * gd[5]
    xmax, ymin = xmin + N * gd[1], ymax + N * gd[5]
    c10, r10 = int(round((xmin - gs[0]) / gs[1])), int(round((ymax - gs[3]) / gs[5]))

    candidates = {}
    for m in METHODS:
        for ovr in ("AUTO", "NONE"):                 # auto = gdal sme da koristi ugradjene nivoe ecw datoteke
            candidates[f"gdalwarp -r {m} (levels {ovr})"] = gdal.Warp(
                "", src, options=f"-of MEM -te {xmin} {ymin} {xmax} {ymax} -ts {N} {N} -r {m} -ovr {ovr}")
        tm = "nearest" if m == "near" else m         # gdal_translate isti nacin zove drugacije
        candidates[f"gdal_translate / Save As -r {tm}"] = gdal.Translate(
            "", src, options=f"-of MEM -srcwin {c10} {r10} {N * k} {N * k} -outsize {N} {N} -r {tm}")

    # poredjenje svakog kandidata sa pravom datotekom od 30 cm (prvi kanal)
    for name, ds in candidates.items():
        arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
        s = score.setdefault(name, [0, 0.0, 0])
        s[0] += int((arr == ref).sum()); s[1] += float(np.abs(arr - ref).sum()); s[2] += arr.size

# ispis od najmanje srednje razlike ka najvecoj
print(f"\n{'candidate':45s} {'identical':>10s} {'mean diff':>10s}")
for name, (same, diff, n) in sorted(score.items(), key=lambda kv: kv[1][1] / kv[1][2]):
    print(f"{name:45s} {same / n:10.1%} {diff / n:10.2f}")
print("\nThe first line is the method that best reproduces campus_30cm.tif.")
print("'identical' near 100 % = certain. If no line is near 100 %, look at 'storage' above:")
print("COMPRESSION=JPEG means the file was saved lossy, then the smallest 'mean diff' wins.")
