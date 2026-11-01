"""
za_proveru.py - spisak mesta na kojima su rucne oznake mozda pogresne

pravi spisak kandidata za proveru rucnih oznaka, da bi se pregledali jedan po
jedan u qgis-u, umesto da se karte gresaka pregledaju odokativno.

dve vrste kandidata, obe unutar oznacenih podrucja:
  lazno_pozitivno  oba modela predvidjaju zgradu, a u maski je pozadina
                   -> mozda zgrada koja nije nacrtana
  propusteno       nijedan model ne predvidja zgradu, a u maski je zgrada
                   -> mozda poligon nacrtan tamo gde nema zgrade

uski pojasevi uz ivice zgrada (nagib zgrade, senka, malo drugacija ivica) nisu
greske u oznakama, pa se prvo uklanjaju morfoloskim otvaranjem (nestaje sve sto
je uze od oko 1,5 m). zatim se zadrzavaju samo grupe od najmanje min_area_m2.

izlaz: za_proveru.gpkg sa dva sloja, najvece grupe prve. svaka grupa ima prazno
polje "odluka", u koje se pri pregledu upisuje odluka (npr. "dodata zgrada" ili
"nije zgrada").

pravilo pri pregledu: oznaka se menja samo ako ortofoto jasno pokazuje da je
oznaka pogresna, a nikad samo zato sto model tako kaze.

pokretanje:  python za_proveru.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

# ----------------------------------------------------------------------------
# podesavanja
# ----------------------------------------------------------------------------
DATA = Path(r"C:\thesis\data")
MASK = DATA / "isecci_kampus" / "maska_30cm.tif"
AREAS = DATA / "isecci_kampus" / "oblasti_30cm.tif"
PRED_A = DATA / "campus_buildings_whu_overlap.tif"    # predvidjanje modela whu
PRED_B = DATA / "campus_buildings_inria.tif"          # predvidjanje modela inria
AREA_NAMES = {1: "petrovaradin", 2: "kampus", 3: "centar"}

MIN_AREA_M2 = 20      # manje grupe se ne prikazuju (20 m2 = mala supa)
OPENING_PX = 2        # uklanja pojaseve uze od oko 2 * 2 + 1 = 5 piksela (1,5 m)

OUT = DATA / "za_proveru.gpkg"
# ----------------------------------------------------------------------------


def read(path):
    """cita prvi kanal rastera, transformaciju i koordinatni sistem."""
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def candidates(binary, transform, crs, areas, px_area, name):
    """od logicke maske kandidata pravi poligone sa povrsinom i podrucjem."""
    # 1) uklanjanje uskih pojaseva (morfolosko otvaranje, element 3 x 3)
    if OPENING_PX:
        binary = ndimage.binary_opening(binary, structure=np.ones((3, 3), bool),
                                        iterations=OPENING_PX)
    # 2) brisanje malih grupa
    labels, n = ndimage.label(binary)
    if n == 0:
        print(f"  {name}: no candidates")
        return None
    sizes = np.bincount(labels.ravel())
    keep = sizes * px_area >= MIN_AREA_M2
    keep[0] = False
    binary = keep[labels]

    # 3) grupe piksela -> poligoni
    geoms = [shape(g) for g, v in shapes(binary.astype(np.uint8), mask=binary,
                                         transform=transform, connectivity=4) if v == 1]
    if not geoms:
        print(f"  {name}: no candidates")
        return None
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs.to_wkt())
    gdf["povrsina_m2"] = gdf.geometry.area.round(1)

    # u kom podrucju je kandidat (gleda se tacka unutar poligona)
    inv = ~transform
    codes = []
    for p in gdf.geometry.representative_point():
        col, row = inv * (p.x, p.y)
        codes.append(int(areas[int(row), int(col)]))
    gdf["oblast"] = [AREA_NAMES.get(c, "?") for c in codes]

    # prazno polje za odluku, redni broj i redosled od najvece grupe
    gdf["odluka"] = ""
    gdf = gdf.sort_values("povrsina_m2", ascending=False).reset_index(drop=True)
    gdf.insert(0, "br", range(1, len(gdf) + 1))
    print(f"  {name}: {len(gdf)} candidates  " +
          ", ".join(f"{k} {v}" for k, v in gdf["oblast"].value_counts().items()))
    return gdf


def main():
    mask, tr, crs = read(MASK)
    areas, tr_a, _ = read(AREAS)
    a, tr1, _ = read(PRED_A)
    b, tr2, _ = read(PRED_B)
    for arr, t in ((areas, tr_a), (a, tr1), (b, tr2)):
        if arr.shape != mask.shape or t != tr:
            raise ValueError("rasters are not on the same grid")

    px_area = abs(tr.a * tr.e)           # povrsina jednog piksela u m2
    inside = (areas > 0) & (a != 255) & (b != 255)
    ref = mask == 1

    fp = inside & ~ref & (a == 1) & (b == 1)   # oba modela: zgrada, maska: pozadina
    fn = inside & ref & (a == 0) & (b == 0)    # oba modela: pozadina, maska: zgrada

    if OUT.exists():
        OUT.unlink()
    for name, arr in (("lazno_pozitivno", fp), ("propusteno", fn)):
        gdf = candidates(arr, tr, crs, areas, px_area, name)
        if gdf is not None:
            gdf.to_file(OUT, layer=name, driver="GPKG")

    print(f"\nSaved {OUT}")
    print("In QGIS: open the layer's attribute table, select a row and press "
          "'Zoom map to selected rows' (or Ctrl+J).")


if __name__ == "__main__":
    main()
