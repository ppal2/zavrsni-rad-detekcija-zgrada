"""
predict.py - prva verzija predvidjanja, prozori bez preklapanja

primenjuje sacuvani model na ceo ortofoto campus_30cm.tif i pise jednu masku
zgrada kao geotiff sa koordinatama (moze se odmah otvoriti u qgis-u).

ortofoto se deli na prozore 512 x 512 jedan do drugog, bez preklapanja.
na rezultatu se zato vide savovi na granicama prozora (na svakih 512 piksela),
jer je predvidjanje na ivici prozora losije. ovaj problem resava
predict_overlap.py (preklapanje 50 % i hann tezine), koji je koriscen u radu.

izlaz: 0 = pozadina, 1 = zgrada, 255 = nema podataka

pokretanje:
    python predict.py
"""

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.windows import Window
from tqdm import tqdm

# ---------------------------------------------------------------------------
# podesavanja
# ---------------------------------------------------------------------------

INPUT_RASTER = r"C:\thesis\data\campus_30cm.tif"       # ortofoto od 30 cm
OUTPUT_RASTER = r"C:\thesis\data\campus_buildings.tif"
MODEL_PATH = r"C:\thesis\best_model.pth"               # model iz train.py

ENCODER = "resnet34"       # mora biti isti kao pri treniranju
TILE_SIZE = 512
THRESHOLD = 0.5            # piksel je zgrada ako je verovatnoca veca od 0,5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ista normalizacija kao pri treniranju (vrednosti imagenet-a)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    print(f"Device: {DEVICE}")

    # pravi se ista mreza kao pri treniranju i u nju se ucitavaju naucene tezine.
    # tezine sa imagenet-a se ne preuzimaju (encoder_weights), jer se ucitavaju nase tezine.
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()   # rezim predvidjanja (batch normalization koristi sacuvane vrednosti)

    with rasterio.open(INPUT_RASTER) as src:
        print(f"Input: {src.width} x {src.height}, CRS {src.crs}")

        # izlaz ima jedan kanal tipa uint8: 0 = pozadina, 1 = zgrada, 255 = nema podataka;
        # mreza piksela i koordinatni sistem su isti kao kod ortofotoa
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            compress="deflate",
            tiled=True,
        )

        with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:

            n_rows = (src.height + TILE_SIZE - 1) // TILE_SIZE
            n_cols = (src.width + TILE_SIZE - 1) // TILE_SIZE
            total = n_rows * n_cols

            with torch.no_grad():   # gradijenti nisu potrebni pri predvidjanju
                pbar = tqdm(total=total, desc="Predicting")

                # prozori jedan do drugog, bez preklapanja
                for row_off in range(0, src.height, TILE_SIZE):
                    for col_off in range(0, src.width, TILE_SIZE):

                        window = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)

                        # prozor koji izlazi preko ivice snimka dopunjava se nulama
                        # (boundless), pa je svaki ulaz tacno 512 x 512
                        patch = src.read(
                            window=window, boundless=True, fill_value=0
                        )

                        # piksel u kome su sva tri kanala 0 nema podatke (van snimka)
                        valid = patch.sum(axis=0) > 0

                        # (kanali, redovi, kolone) -> (redovi, kolone, kanali) za normalizaciju,
                        # pa nazad za pytorch
                        img = patch.transpose(1, 2, 0).astype(np.float32)
                        img = img / 255.0
                        img = (img - MEAN) / STD
                        img = torch.from_numpy(img).permute(2, 0, 1)

                        # mreza ocekuje grupu prozora, pa se dodaje osa grupe:
                        # (kanali, redovi, kolone) -> (1, kanali, redovi, kolone)
                        img = img.unsqueeze(0).to(DEVICE)

                        logits = model(img)

                        # mreza daje logite; sigmoida ih pretvara u verovatnocu zgrade (0-1)
                        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

                        mask = (probs > THRESHOLD).astype(np.uint8)

                        # pikseli bez podataka dobijaju 255
                        mask[~valid] = 255

                        # prozor na desnoj i donjoj ivici je dopunjen do 512, a u izlaz
                        # se upisuje samo deo koji je unutar snimka
                        h = min(TILE_SIZE, src.height - row_off)
                        w = min(TILE_SIZE, src.width - col_off)

                        dst.write(
                            mask[:h, :w],
                            1,
                            window=Window(col_off, row_off, w, h),
                        )

                        pbar.update(1)

                pbar.close()

    print(f"\nWrote {OUTPUT_RASTER}")


if __name__ == "__main__":
    main()
