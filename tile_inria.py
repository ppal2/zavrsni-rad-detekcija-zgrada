"""
tile_inria.py - priprema skupa inria za treniranje

sece snimke skupa inria (5000 x 5000 piksela) na isecke 512 x 512 i cuva ih
u istom rasporedu foldera kao skup whu (train/val, image/label), tako da
program train.py radi sa oba skupa, uz promenu samo promenljive data_root.

sta program radi:
  1. koristi samo dva grada: bec (vienna) i zapadni tirol (tyrol-w), jer po
     gradnji vise lice na novi sad od americkih gradova u skupu;
  2. prvih pet snimaka svakog grada (brojevi 1-5) ide u validacioni skup,
     kao kod autora skupa, a ostali u skup za treniranje;
  3. 5000 nije deljivo sa 512, pa se poslednji red i poslednja kolona isecaka
     pomere unazad da se zavrse tacno na ivici snimka (malo se preklapaju sa
     susednim isecima, ali nema crnih ivica);
  4. oznake inria imaju 255 za zgradu i 0 za pozadinu - ovde se pisu kao 0/1,
     isto kao oznake skupa whu;
  5. od isecaka bez ijedne zgrade zadrzava se slucajnih 20 %, a ostali se
     preskacu.

pokretanje:
    python tile_inria.py
"""

import os

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

from podesavanja import DATA   # putanje su u podesavanja.py

# ---------------------------------------------------------------------------
# podesavanja
# ---------------------------------------------------------------------------

INRIA_ROOT = DATA / "inria" / "train"             # folder sa podfolderima images i gt
OUTPUT_ROOT = DATA / "inria_tiles"               # ovde se upisuju isecci

# gradovi koji se koriste (u skupu postoje: austin, chicago, kitsap, tyrol-w, vienna)
REGIONS = ["vienna", "tyrol-w"]

TILE_SIZE = 512

# prvih pet snimaka svakog grada ide u validacioni skup, kao kod autora skupa inria
VAL_IMAGE_NUMBERS = {1, 2, 3, 4, 5}

# udeo isecaka bez zgrada koji se zadrzava (ostalih 80 % se preskace)
KEEP_EMPTY_TILE_RATIO = 0.2


def get_tile_offsets(image_size, tile_size):
    """
    vraca pocetne polozaje isecaka duz jedne ose.

    za snimak od 5000 piksela i isecke od 512 to su 0, 512, 1024, ... 4096
    i na kraju 4488 (a ne 4608), da bi poslednji isecak zavrsio tacno na 5000.
    """
    offsets = list(range(0, image_size - tile_size + 1, tile_size))

    if not offsets:
        raise ValueError(f"Image smaller than tile size ({image_size} < {tile_size})")

    # ako poslednji isecak ne stize do ivice, dodaje se jos jedan, pomeren unazad
    if offsets[-1] + tile_size < image_size:
        offsets.append(image_size - tile_size)

    return offsets


def parse_region(filename):
    """
    deli ime datoteke na grad i broj snimka, npr. 'vienna12.tif' -> ('vienna', 12).

    ime grada moze imati crticu ('tyrol-w'), pa se cifre skidaju sa kraja imena.
    za datoteke koje nisu snimci (npr. .aux.xml koje pravi qgis) vraca dva prazna rezultata.
    """
    stem = os.path.splitext(filename)[0]

    digits = ""
    while stem and stem[-1].isdigit():
        digits = stem[-1] + digits
        stem = stem[:-1]

    if not digits or not stem:
        return None, None

    return stem, int(digits)


def main():
    image_dir = os.path.join(INRIA_ROOT, "images")
    label_dir = os.path.join(INRIA_ROOT, "gt")

    all_files = sorted(os.listdir(image_dir))

    selected = []
    found_regions = set()

    for f in all_files:
        # preskacu se datoteke koje nisu snimci (npr. .aux.xml)
        if not f.lower().endswith(".tif"):
            continue

        region, number = parse_region(f)

        if region is None:
            print(f"Skipping unrecognised filename: {f}")
            continue

        found_regions.add(region)

        if region in REGIONS:
            selected.append((f, region, number))

    if not selected:
        raise SystemExit(
            f"No files matched {REGIONS}. Regions present: {sorted(found_regions)}"
        )

    print(f"Selected {len(selected)} source images from {REGIONS}")

    # folderi u istom rasporedu kao skup whu
    for split in ("train", "val"):
        for kind in ("image", "label"):
            os.makedirs(os.path.join(OUTPUT_ROOT, split, kind), exist_ok=True)

    rng = np.random.default_rng(seed=42)   # stalno seme = uvek isti izbor praznih isecaka

    counts = {"train": 0, "val": 0}
    skipped_empty = 0

    for filename, region, number in tqdm(selected, desc="Tiling"):

        split = "val" if number in VAL_IMAGE_NUMBERS else "train"

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)

        if not os.path.exists(label_path):
            print(f"No label for {filename}, skipping")
            continue

        with rasterio.open(image_path) as img_src, \
             rasterio.open(label_path) as lbl_src:

            row_offsets = get_tile_offsets(img_src.height, TILE_SIZE)
            col_offsets = get_tile_offsets(img_src.width, TILE_SIZE)

            for r_i, row_off in enumerate(row_offsets):
                for c_i, col_off in enumerate(col_offsets):

                    window = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)

                    img_tile = img_src.read(window=window)
                    lbl_tile = lbl_src.read(1, window=window)

                    # oznaka inria: 255 = zgrada, 0 = pozadina; ovde se pise 0/1 kao kod skupa whu
                    lbl_binary = (lbl_tile > 127).astype(np.uint8)

                    # isecak bez zgrada se zadrzava samo sa verovatnocom 20 %
                    if lbl_binary.max() == 0:
                        if rng.random() > KEEP_EMPTY_TILE_RATIO:
                            skipped_empty += 1
                            continue

                    base = f"{region}{number}_{r_i:02d}_{c_i:02d}.tif"

                    # isecci se cuvaju kao obicni tiff (bez koordinata), jer sluze samo za
                    # treniranje; uzimaju se samo prva tri kanala (rgb)
                    Image.fromarray(
                        img_tile[:3].transpose(1, 2, 0)
                    ).save(os.path.join(OUTPUT_ROOT, split, "image", base))

                    Image.fromarray(lbl_binary).save(
                        os.path.join(OUTPUT_ROOT, split, "label", base)
                    )

                    counts[split] += 1

    print(f"\nWrote {counts['train']} training tiles, {counts['val']} validation tiles")
    print(f"Skipped {skipped_empty} empty tiles")
    print(f"\nNow set DATA_ROOT in train.py to:\n  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
