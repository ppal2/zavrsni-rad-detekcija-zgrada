"""
pravi_isecke.py - rucno nacrtane zgrade -> maska na mrezi ortofotoa od 30 cm

pretvara poligone zgrada u masku na istoj mrezi piksela kao ortofoto od 30 cm,
a zatim sece isecke slike i maske 512 x 512 za svako podrucje.

koraci:
  1. cita ortofoto od 30 cm (prvo samo mrezu: velicinu, transformaciju, koordinatni sistem);
  2. cita slojeve "zgrade" i "oblasti" iz geopackage datoteke;
  3. rasterizuje zgrade: piksel cija je sredina unutar poligona zgrade dobija 1,
     ostali 0; mreza je ista kao kod ortofotoa, pa svaki piksel maske lezi
     tacno na svom pikselu ortofotoa;
  4. rasterizuje podrucja: 1 = trening, 2 = validacija, 3 = test, 0 = izvan
     (imena poticu iz prvog plana rada; to su petrovaradin, kampus i centar);
  5. pomera prozor 512 x 512 preko snimka i zadrzava samo prozore koji su
     ceo unutar jednog podrucja (i bez piksela bez podataka); prozor koji izlazi
     iz podrucja imao bi neoznacene zgrade upisane kao 0;
  6. cuva svaki prozor kao isecak slike i isecak maske sa istim imenom, u
     <podrucje>/image i <podrucje>/label (isti raspored kao inria_tiles);
  7. ispisuje pregled po podrucjima.

cuva i dva rastera za ceo ortofoto, koji su u radu korisceni za ocenu i dotreniravanje:
  maska_30cm.tif    maska zgrada (0/1)
  oblasti_30cm.tif  kodovi podrucja (0/1/2/3)

isecci pokrivaju samo deo podrucja (oko 37-63 %), pa na kraju nisu korisceni:
ocena se radi na celim podrucjima (ocena.py), a dotreniravanje na slucajnim
prozorima (dotreniravanje.py).

instalacija:  pip install rasterio geopandas
pokretanje:   python pravi_isecke.py
"""

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from podesavanja import DATA   # putanje su u podesavanja.py

# ----------------------------------------------------------------------------
# podesavanja
# ----------------------------------------------------------------------------
ORTHO = DATA / "campus_30cm.tif"                       # ortofoto od 30 cm (isti kao u predict_overlap.py)
GPKG = DATA / "kampus_oznake.gpkg"                     # geopackage sa slojevima zgrade i oblasti
OUT = DATA / "isecci_kampus"                           # izlazni folder (pravi se ako ne postoji)

TILE = 512  # velicina isecka u pikselima (512 x 0,3 m = 153,6 m)

# korak izmedju susednih isecaka: 256 = preklapanje 50 % (vise primera sa malog
# podrucja), 512 = bez preklapanja (svaki piksel jednom)
STEP = {"trening": 256, "validacija": 512, "test": 512}

# vrednost za "zgradu" u isecima maske; mora biti 1, kao kod skupova whu i inria,
# jer bce gubitak trazi vrednosti izmedju 0 i 1
MASK_VALUE = 1

# vrednosti polja "skup" i kod koji svako podrucje dobija u rasteru podrucja
SETS = {"trening": 1, "validacija": 2, "test": 3}
# ----------------------------------------------------------------------------


def read_layers(gpkg, raster_crs):
    """cita zgrade i podrucja i prebacuje ih u koordinatni sistem ortofotoa."""
    bld = gpd.read_file(gpkg, layer="zgrade")
    areas = gpd.read_file(gpkg, layer="oblasti")

    for name, gdf in (("zgrade", bld), ("oblasti", areas)):
        if gdf.crs is None:
            raise ValueError(f"Layer '{name}' has no CRS - set it to EPSG:32634 in QGIS.")
    if bld.crs.to_epsg() != raster_crs.to_epsg():
        print(f"  reprojecting zgrade  {bld.crs.to_epsg()} -> {raster_crs.to_epsg()}")
        bld = bld.to_crs(raster_crs.to_wkt())
    if areas.crs.to_epsg() != raster_crs.to_epsg():
        print(f"  reprojecting oblasti {areas.crs.to_epsg()} -> {raster_crs.to_epsg()}")
        areas = areas.to_crs(raster_crs.to_wkt())

    # svako podrucje mora imati jedno od ocekivanih imena, inace bi bilo preskoceno bez upozorenja
    bad = areas[~areas["skup"].isin(list(SETS))]
    if len(bad):
        raise ValueError(
            "These areas have an unknown 'skup' value (must be trening/validacija/test):\n"
            + bad[["skup"]].to_string()
        )

    # izbacuju se prazni poligoni
    bld = bld[bld.geometry.notna() & ~bld.geometry.is_empty]
    return bld, areas


def save_full_raster(path, array, profile):
    """cuva jednokanalni uint8 raster na mrezi ortofotoa."""
    p = profile.copy()
    p.update(driver="GTiff", count=1, dtype="uint8", nodata=None, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array, 1)


def main():
    with rasterio.open(ORTHO) as src:
        H, W = src.height, src.width
        res_x, res_y = src.res
        print(f"Orthophoto: {W} x {H} px, pixel {res_x:.3f} x {res_y:.3f} m, "
              f"EPSG:{src.crs.to_epsg()}, {src.count} bands")
        if abs(res_x - 0.3) > 1e-6 or abs(res_y - 0.3) > 1e-6:
            print("  WARNING: pixel size is not 0.3 m - is this the right file?")

        bld, areas = read_layers(GPKG, src.crs)
        print(f"Buildings: {len(bld)} polygons, areas: {len(areas)} polygons")

        # --- 3. maska zgrada na mrezi ortofotoa ------------------------------
        # piksel dobija 1 ako mu je sredina unutar poligona (all_touched je iskljuceno);
        # rupe napravljene alatom add ring ostaju 0
        mask = rasterize(
            ((geom, 1) for geom in bld.geometry),
            out_shape=(H, W),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )

        # --- 4. kodovi podrucja na istoj mrezi -------------------------------
        area_codes = rasterize(
            ((geom, SETS[name]) for geom, name in zip(areas.geometry, areas["skup"])),
            out_shape=(H, W),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )

        # pikseli sa stvarnim podacima (0 = nema podataka / izvan snimka)
        valid = src.dataset_mask() > 0

        OUT.mkdir(parents=True, exist_ok=True)
        save_full_raster(OUT / "maska_30cm.tif", mask, src.profile)
        save_full_raster(OUT / "oblasti_30cm.tif", area_codes, src.profile)

        # profili za isecke (sa koordinatama, pa se mogu otvoriti u qgis-u)
        img_profile = src.profile.copy()
        img_profile.update(driver="GTiff", width=TILE, height=TILE, count=3,
                           dtype="uint8", nodata=None, compress=None)
        for key in ("transform", "blockxsize", "blockysize", "tiled", "interleave"):
            img_profile.pop(key, None)  # transformacija se zadaje posebno za svaki isecak
        msk_profile = img_profile.copy()
        msk_profile.update(count=1)

        covered = np.zeros((H, W), dtype=bool)  # koristi se za svako podrucje, za procenat pokrivenosti
        summary = []

        # --- 5. i 6. isecci za svako podrucje --------------------------------
        for name, code in SETS.items():
            step = STEP[name]
            # brisu se isecci iz prethodnog pokretanja, da se ne pomesaju sa novim
            if (OUT / name).exists():
                shutil.rmtree(OUT / name)
            img_dir = OUT / name / "image"   # isti nazivi foldera kao u inria_tiles
            msk_dir = OUT / name / "label"
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)

            inside = (area_codes == code) & valid
            if not inside.any():
                print(f"  {name}: no pixels - is there an area with skup = '{name}'?")
                continue

            # gledaju se samo prozori oko ovog podrucja (brze nego ceo snimak)
            rows, cols = np.nonzero(inside)
            r0, r1 = rows.min(), rows.max()
            c0, c1 = cols.min(), cols.max()

            covered[:] = False
            n_tiles = n_empty = 0
            bld_pixels = 0
            for r in range(r0, r1 - TILE + 2, step):
                for c in range(c0, c1 - TILE + 2, step):
                    if not inside[r:r + TILE, c:c + TILE].all():
                        continue  # prozor izlazi iz podrucja ili dodiruje piksele bez podataka

                    win = Window(c, r, TILE, TILE)
                    img = src.read(indexes=[1, 2, 3], window=win)
                    m = mask[r:r + TILE, c:c + TILE]

                    stem = f"{name}_r{r:05d}_c{c:05d}"
                    wt = src.window_transform(win)
                    with rasterio.open(img_dir / f"{stem}.tif", "w", transform=wt, **img_profile) as dst:
                        dst.write(img)
                    with rasterio.open(msk_dir / f"{stem}.tif", "w", transform=wt, **msk_profile) as dst:
                        dst.write((m * MASK_VALUE).astype("uint8"), 1)

                    covered[r:r + TILE, c:c + TILE] = True
                    n_tiles += 1
                    bld_pixels += int(m.sum())
                    n_empty += int(m.sum() == 0)

            area_km2 = inside.sum() * res_x * res_y / 1e6
            summary.append({
                "skup": name,
                "km2": area_km2,
                "bld_share_area": mask[inside].mean() * 100,
                "tiles": n_tiles,
                "step": step,
                "bld_share_tiles": (bld_pixels / (n_tiles * TILE * TILE) * 100) if n_tiles else 0.0,
                "empty": n_empty,
                "coverage": covered[inside].mean() * 100,
            })

    # --- 7. pregled ----------------------------------------------------------
    print()
    print(f"{'set':<11}{'km2':>7}{'bld % area':>12}{'tiles':>7}{'step':>6}"
          f"{'bld % tiles':>13}{'empty':>7}{'area covered %':>16}")
    for s in summary:
        print(f"{s['skup']:<11}{s['km2']:>7.3f}{s['bld_share_area']:>12.1f}{s['tiles']:>7}"
              f"{s['step']:>6}{s['bld_share_tiles']:>13.1f}{s['empty']:>7}{s['coverage']:>16.1f}")
    print()
    print(f"Tiles written to {OUT}")
    print("Check: open one image tile and its mask in QGIS - the mask must sit exactly on the roofs.")


if __name__ == "__main__":
    main()
