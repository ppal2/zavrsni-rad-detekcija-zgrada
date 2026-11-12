"""
brojanje_zgrada.py - brojevi za tabelu 4.2, sa konacnim oznakama (posle pregleda oznaka)

za svako podrucje (sloj "oblasti") ispisuje:
  - povrsinu u km2,
  - koliko poligona zgrada mu pripada - zgrada pripada podrucju u kome lezi
    njen centroid (isto pravilo kao ranije brojanje centroida u qgis-u),
  - udeo zgrada: deo povrsine podrucja koji pokrivaju zgrade, u %.
broji i zgrade ciji je centroid izvan sva tri podrucja.

pokretanje (u conda okruzenju "thesis", iz foldera c:\\thesis):
    python brojanje_zgrada.py
"""
import geopandas as gpd

from podesavanja import DATA   # putanje su u podesavanja.py

GPKG = DATA / "kampus_oznake.gpkg"
# vrednosti polja "skup" -> imena podrucja u radu
NAMES = {"trening": "Petrovaradin", "validacija": "kampus", "test": "centar"}

buildings = gpd.read_file(GPKG, layer="zgrade")
areas = gpd.read_file(GPKG, layer="oblasti")
buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty]
print(f"building polygons in the layer: {len(buildings)}")

# 1) brojanje po centroidu: svaki poligon se pretvara u tacku (centroid), pa se
#    trazi podrucje koje tu tacku sadrzi
centroids = buildings.copy()
centroids["geometry"] = buildings.geometry.centroid
joined = gpd.sjoin(centroids, areas[["skup", "geometry"]], how="left", predicate="within")

print(f"\n{'area':14s}{'km2':>8s}{'polygons':>10s}{'share %':>10s}")
for _, a in areas.iterrows():
    name = NAMES.get(a["skup"], a["skup"])
    n = int((joined["skup"] == a["skup"]).sum())
    # 2) udeo zgrada: sve zgrade se iseku granicom podrucja i saberu se povrsine delova
    inside = buildings.geometry.intersection(a.geometry).area.sum()
    print(f"{name:14s}{a.geometry.area / 1e6:8.3f}{n:10d}{100 * inside / a.geometry.area:10.1f}")

# zgrade bez podrucja dobijaju praznu vrednost polja "skup" posle spajanja
outside = int(joined["skup"].isna().sum())
print(f"\nbuildings with the centroid outside all three areas: {outside}")
