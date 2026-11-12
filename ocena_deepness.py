"""
ocena_deepness.py - ocena rezultata iz qgis dodatka deepness na kampusu

meri rezultat iz deepness-a na kampusu i poredi ga sa istim modelom pokrenutim
iz pythona (dotreniravanje.py). onnx model je dotreniran samo na petrovaradinu
i centru, pa je kampus jedino posteno podrucje za proveru.

ispisuje tri reda:
  deepness prema oznakama   - koliko je dobar rezultat iz qgis-a
  python   prema oznakama   - broj iz dotreniravanje.py (0,826)
  deepness prema pythonu    - koliko su dva predvidjanja slicna
                              (iou 1,0 bi znacio da su ista)

pokretanje:  python ocena_deepness.py
(mora biti u istom folderu kao ocena.py, jer koristi njenu funkciju pixel_scores)
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from ocena import pixel_scores   # iste mere kao u ostatku rada
from podesavanja import DATA   # putanje su u podesavanja.py

# ----------------------------------------------------------------------------
# podesavanja
# ----------------------------------------------------------------------------
DEEPNESS_SHP = DATA / "deepness_zgrada.shp"          # rezultat deepness-a, sacuvan u folder data
MASK = DATA / "isecci_kampus" / "maska_30cm.tif"
AREAS = DATA / "isecci_kampus" / "oblasti_30cm.tif"
PYTHON_PRED = DATA / "dotreniran_inria_kampus.tif"   # iz dotreniravanje.py
AREA_CODE = 2                                        # 2 = kampus
# ----------------------------------------------------------------------------


def main():
    with rasterio.open(MASK) as s:
        mask = s.read(1)
        transform, crs, shape = s.transform, s.crs, s.shape
    with rasterio.open(AREAS) as s:
        areas = s.read(1)

    # poligoni iz deepness-a -> raster na istoj mrezi od 30 cm (pravilo sredine
    # piksela, isto kao kad je pravi_isecke.py pravio masku)
    gdf = gpd.read_file(DEEPNESS_SHP)
    if gdf.crs is not None and gdf.crs.to_epsg() != crs.to_epsg():
        gdf = gdf.to_crs(crs.to_wkt())
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    deep = rasterize(((g, 1) for g in gdf.geometry), out_shape=shape,
                     transform=transform, fill=0, dtype="uint8") == 1
    print(f"Deepness: {len(gdf)} polygons")

    # predvidjanje iz pythona pokriva samo okolinu kampusa - upisuje se u punu mrezu
    py = np.full(shape, 255, dtype=np.uint8)
    with rasterio.open(PYTHON_PRED) as s:
        win = from_bounds(*s.bounds, transform=transform).round_offsets().round_lengths()
        r0, c0 = int(win.row_off), int(win.col_off)
        py[r0:r0 + s.height, c0:c0 + s.width] = s.read(1)

    # racuna se samo unutar kampusa, i samo gde python predvidjanje postoji
    inside = (areas == AREA_CODE) & (py != 255)
    ref = (mask == 1) & inside
    deep_in = deep & inside
    py_in = (py == 1) & inside

    rows = [("Deepness vs labels", pixel_scores(ref, deep_in)),
            ("Python   vs labels", pixel_scores(ref, py_in)),
            ("Deepness vs Python", pixel_scores(py_in, deep_in))]
    print(f"\n{'':<20}{'IoU':>7}{'prec':>7}{'recall':>8}{'F1':>7}")
    for name, s in rows:
        print(f"{name:<20}{s['IoU']:>7.3f}{s['precision']:>7.3f}{s['recall']:>8.3f}{s['F1']:>7.3f}")
    print("\n(campus only; in the last row the Python prediction plays the role of the reference)")


if __name__ == "__main__":
    main()
