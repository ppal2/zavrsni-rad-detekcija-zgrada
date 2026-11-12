"""
podesavanja.py - jedno mesto za sve putanje

svi programi citaju putanje iz ove datoteke. na drugom racunaru treba promeniti
samo THESIS_DIR (i DATA, ako su podaci u drugom folderu).

ocekivani raspored foldera:
  THESIS_DIR/              modeli (.pth, .onnx) i slike
  THESIS_DIR/data/         ortofoto, oznake (kampus_oznake.gpkg), skupovi whu i inria,
                           predvidjanja i rezultati (.tif, .csv)
"""

from pathlib import Path

THESIS_DIR = Path(r"C:\thesis")   # folder sa modelima
DATA = THESIS_DIR / "data"        # folder sa podacima
