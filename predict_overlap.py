"""
predict_overlap.py - predvidjanje na celom ortofotou, prozori sa preklapanjem od 50 %

isti posao kao predict.py, ali bez savova na granicama prozora. koriscen je za
oba polazna modela (u radu, odeljak 4.6):
  - model whu:   model_path = best_model_whu.pth,   izlaz campus_buildings_whu_overlap.tif
  - model inria: model_path = best_model_inria.pth, izlaz campus_buildings_inria.tif
(ovde su upisana podesavanja za model whu.)

kako radi:
  1. prozor 512 x 512 se pomera za 256 piksela, pa skoro svaki piksel pokriju
     cetiri prozora;
  2. verovatnoca zgrade iz svakog prozora mnozi se hann tezinama, koje su 1 u
     sredini prozora i skoro 0 na ivici, i dodaje u zbirni raster;
     u drugi raster se dodaju same tezine;
  3. na kraju se zbir verovatnoca deli zbirom tezina (tezinski prosek);
  4. piksel je zgrada ako je prosek veci od 0,5.
ivica prozora, gde je predvidjanje najlosije (dopunjavanje nulama), tako skoro
ne utice na rezultat.

izlaz: 0 = pozadina, 1 = zgrada, 255 = nema podataka
(po zelji i raster verovatnoca, ako je zadat prob_raster)

pokretanje:
    python predict_overlap.py
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

INPUT_RASTER = r"C:\thesis\data\campus_30cm.tif"
OUTPUT_RASTER = r"C:\thesis\data\campus_buildings_whu_overlap.tif"   # za inria: campus_buildings_inria.tif
PROB_RASTER = None   # putanja za raster verovatnoca (float32); ovde se ne pravi
MODEL_PATH = r"C:\thesis\best_model_whu.pth"                         # za inria: best_model_inria.pth

ENCODER = "resnet34"
TILE_SIZE = 512
OVERLAP = 0.5        # 0.5 = prozor se pomera za pola prozora (256 piksela)
THRESHOLD = 0.5      # prag verovatnoce za zgradu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ista normalizacija kao pri treniranju (vrednosti imagenet-a)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_weight_window(size):
    """
    dvodimenzionalne hann tezine: 1 u sredini prozora, skoro 0 na ivicama.

    jednodimenzionalni hann prozor se pomnozi sam sa sobom (spoljasnji proizvod).
    mali broj 1e-6 sprecava deljenje nulom na ivici snimka, gde piksel pokriva
    samo jedan prozor sa tezinom skoro nula.
    """
    hann = np.hanning(size).astype(np.float32)
    window_2d = np.outer(hann, hann)
    return window_2d + 1e-6


def main():
    print(f"Device: {DEVICE}")

    # ista mreza kao pri treniranju, sa nasim naucenim tezinama
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    weight = make_weight_window(TILE_SIZE)
    step = int(TILE_SIZE * (1 - OVERLAP))      # 256 piksela

    with rasterio.open(INPUT_RASTER) as src:
        print(f"Input: {src.width} x {src.height}, CRS {src.crs}")

        # dva zbirna rastera velicine celog snimka:
        # prob_sum - zbir verovatnoca pomnozenih tezinama, weight_sum - zbir tezina
        prob_sum = np.zeros((src.height, src.width), dtype=np.float32)
        weight_sum = np.zeros((src.height, src.width), dtype=np.float32)

        row_offsets = list(range(0, src.height, step))
        col_offsets = list(range(0, src.width, step))
        total = len(row_offsets) * len(col_offsets)

        with torch.no_grad():
            pbar = tqdm(total=total, desc="Predicting (overlapping)")

            for row_off in row_offsets:
                for col_off in col_offsets:

                    # prozor koji izlazi preko ivice snimka dopunjava se nulama
                    window = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
                    patch = src.read(window=window, boundless=True, fill_value=0)

                    img = patch.transpose(1, 2, 0).astype(np.float32)
                    img = img / 255.0
                    img = (img - MEAN) / STD
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                    img = img.to(DEVICE)

                    # mesovita preciznost (amp), kao pri treniranju
                    with torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                        logits = model(img)

                    probs = torch.sigmoid(logits.float())[0, 0].cpu().numpy()

                    # deo prozora koji je unutar snimka (na desnoj i donjoj ivici je manji)
                    h = min(TILE_SIZE, src.height - row_off)
                    w = min(TILE_SIZE, src.width - col_off)

                    prob_sum[row_off:row_off + h, col_off:col_off + w] += (
                        probs[:h, :w] * weight[:h, :w]
                    )
                    weight_sum[row_off:row_off + h, col_off:col_off + w] += (
                        weight[:h, :w]
                    )

                    pbar.update(1)

            pbar.close()

        # tezinski prosek; weight_sum nikad nije nula, jer svaki piksel pokriva
        # bar jedan prozor, a tezine su vece od nule
        avg_prob = prob_sum / weight_sum

        # pikseli u kojima su sva tri kanala 0 nemaju podatke
        full = src.read()
        valid = full.sum(axis=0) > 0
        del full, prob_sum, weight_sum

        mask = (avg_prob > THRESHOLD).astype(np.uint8)
        mask[~valid] = 255

        profile = src.profile.copy()
        profile.update(
            count=1, dtype="uint8", nodata=255,
            compress="deflate", tiled=True,
        )

        with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
            dst.write(mask, 1)

        print(f"Wrote {OUTPUT_RASTER}")

        # raster verovatnoca omogucava da se u qgis-u probaju razliciti pragovi
        # bez ponovnog pokretanja modela
        if PROB_RASTER:
            prob_profile = src.profile.copy()
            prob_profile.update(
                count=1, dtype="float32", nodata=-1,
                compress="deflate", tiled=True,
            )
            avg_prob[~valid] = -1

            with rasterio.open(PROB_RASTER, "w", **prob_profile) as dst:
                dst.write(avg_prob.astype(np.float32), 1)

            print(f"Wrote {PROB_RASTER}")

    print("\nCompare against campus_buildings.tif in QGIS - the tile seams "
          "should be gone.")


if __name__ == "__main__":
    main()
