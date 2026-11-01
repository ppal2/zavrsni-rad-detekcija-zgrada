"""
dotreniravanje.py - dotreniravanje po postupku "izostavi jedno podrucje"

za svaki polazni model (whu, inria) i svako podrucje (centar, kampus, petrovaradin):
  1. uzme se polazni model;
  2. oceni se na tom (testnom) podrucju                    -> rezultat "pre";
  3. dotrenira se samo na druga dva podrucja;
  4. dotrenirani model se ponovo oceni na testnom podrucju -> rezultat "posle".
tako svako podrucje dobija rezultat pre i posle od modela koji nikad nisu videli
njegove oznake. 2 modela x 3 podrucja = 6 dotreniravanja, sva sa istim podesavanjima.

kako se prave primeri za ucenje (namerno drugacije nego u train.py):
  umesto stalnih isecaka, svaki primer je slucajan prozor 512 x 512 cija je
  sredina negde unutar podrucja za treniranje. pikseli izvan podrucja za
  treniranje se ne racunaju u gubitku, pa model nikad ne uci da je
  neoznacena zgrada pozadina. tako se koriste svi oznaceni pikseli, i oni
  blizu ivica podrucja koje je pravi_isecke.py morao da izostavi.
  uvecanje skupa je isto kao u train.py (odrazavanja i rotacije za 90 stepeni),
  gubitak isti (dice + bce, ali samo na oznacenim pikselima), normalizacija ista.

podesavanja su izabrana unapred (nema posebnog podrucja za izbor): stopa ucenja
1e-5 (deset puta manja nego u train.py), 10 epoha po 200 slucajnih prozora.

ocena je ista kao u ocena.py (iste funkcije se uvoze), na celom testnom
podrucju, sa istim predvidjanjem sa preklapanjem i hann tezinama kao u
predict_overlap.py. polazni i dotrenirani model prolaze kroz isti kod, pa je
razlika pre/posle posledica samo dotreniravanja.

datoteka mora biti u istom folderu kao ocena.py.
pokretanje:  python dotreniravanje.py
"""

import csv
import random
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

# ----------------------------------------------------------------------------
# podesavanja
# ----------------------------------------------------------------------------
DATA = Path(r"C:\thesis\data")
MODELS_DIR = Path(r"C:\thesis")

ORTHO = DATA / "campus_30cm.tif"
MASK = DATA / "isecci_kampus" / "maska_30cm.tif"
AREAS = DATA / "isecci_kampus" / "oblasti_30cm.tif"

BASE_MODELS = {
    "WHU": MODELS_DIR / "best_model_whu.pth",
    "Inria": MODELS_DIR / "best_model_inria.pth",
}

# kod svakog podrucja u oblasti_30cm.tif (iz vrednosti polja skup:
# trening = 1, validacija = 2, test = 3, kada je pokrenut pravi_isecke.py)
AREA_CODES = {"petrovaradin": 1, "kampus": 2, "centar": 3}

# koja se podrucja izostavljaju, jedno po pokretanju
TEST_AREAS = ["centar", "kampus", "petrovaradin"]

TILE = 512
BATCH_SIZE = 2
EPOCHS = 10
CROPS_PER_EPOCH = 200        # slucajnih prozora 512 x 512 po epohi
LEARNING_RATE = 1e-5         # deset puta manja stopa ucenja nego u train.py
SEED = 42

PRED_STEP = 256              # isto kao predict_overlap.py (preklapanje 50 %)
THRESHOLD = 0.5

OUT_CSV = DATA / "rezultati_dotreniravanje.csv"
SAVE_PREDICTIONS = True      # predvidjanje svakog testnog podrucja kao geotiff (za slike)
# ----------------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# delovi koji rade samo sa numpy nizovima (podaci) - ovde pytorch nije potreban
# ============================================================================

def normalise(crop_chw_uint8):
    """uint8 (3, redovi, kolone) -> float32 (3, redovi, kolone), isto kao u train.py."""
    x = crop_chw_uint8.astype(np.float32) / 255.0
    return (x - MEAN[:, None, None]) / STD[:, None, None]


def padded_crop(image, r, c, size):
    """
    prozor size x size koji pocinje u redu r i koloni c. delovi izvan snimka se
    popunjavaju nulama, kao opcija boundless u predict_overlap.py.
    """
    _, H, W = image.shape
    out = np.zeros((image.shape[0], size, size), dtype=image.dtype)
    r0, c0 = max(r, 0), max(c, 0)
    r1, c1 = min(r + size, H), min(c + size, W)
    if r1 > r0 and c1 > c0:
        out[:, r0 - r:r1 - r, c0 - c:c1 - c] = image[:, r0:r1, c0:c1]
    return out


def random_training_example(rng, image, mask, train_ok, centres):
    """
    jedan slucajan prozor cija je sredina unutar podrucja za treniranje, sa uvecanjem.
    vraca sliku (3, t, t) uint8, oznaku (t, t) 0/1 i masku vaznih piksela (t, t).
    """
    _, H, W = image.shape
    k = rng.integers(len(centres[0]))
    r = int(np.clip(centres[0][k] - TILE // 2, 0, H - TILE))
    c = int(np.clip(centres[1][k] - TILE // 2, 0, W - TILE))
    img = image[:, r:r + TILE, c:c + TILE]
    lbl = mask[r:r + TILE, c:c + TILE]
    val = train_ok[r:r + TILE, c:c + TILE]

    # isto uvecanje kao u train.py: odrazavanja i rotacija za 0/90/180/270 stepeni;
    # slika, oznaka i maska vaznih piksela menjaju se na isti nacin
    if rng.random() > 0.5:                       # levo-desno
        img, lbl, val = img[:, :, ::-1], lbl[:, ::-1], val[:, ::-1]
    if rng.random() > 0.5:                       # gore-dole
        img, lbl, val = img[:, ::-1, :], lbl[::-1, :], val[::-1, :]
    k = int(rng.integers(4))
    img = np.rot90(img, k, axes=(1, 2))
    lbl = np.rot90(lbl, k)
    val = np.rot90(val, k)
    return (np.ascontiguousarray(img), np.ascontiguousarray(lbl),
            np.ascontiguousarray(val))


def hann_weight(size):
    """dvodimenzionalne hann tezine, iste kao u predict_overlap.py."""
    hann = np.hanning(size).astype(np.float32)
    return np.outer(hann, hann) + 1e-6


def predict_region(infer, image, r0, r1, c0, c1):
    """
    predvidjanje sa preklapanjem za redove r0:r1 i kolone c0:c1, sa hann
    tezinama, kao u predict_overlap.py. infer prima normalizovan niz (3, t, t)
    i vraca verovatnocu zgrade (t, t).
    """
    weight = hann_weight(TILE)
    h, w = r1 - r0, c1 - c0
    prob_sum = np.zeros((h + TILE, w + TILE), dtype=np.float32)
    weight_sum = np.zeros_like(prob_sum)
    for r in range(r0, r1, PRED_STEP):
        for c in range(c0, c1, PRED_STEP):
            patch = padded_crop(image, r, c, TILE)
            probs = infer(normalise(patch))
            prob_sum[r - r0:r - r0 + TILE, c - c0:c - c0 + TILE] += probs * weight
            weight_sum[r - r0:r - r0 + TILE, c - c0:c - c0 + TILE] += weight
    return prob_sum[:h, :w] / weight_sum[:h, :w]


def test_region(areas, code, H, W):
    """
    pravougaonik oko testnog podrucja, prosiren za pola prozora sa svake strane,
    da svaki ocenjeni piksel bude daleko od ivice tog pravougaonika.
    """
    rows, cols = np.nonzero(areas == code)
    m = TILE // 2
    return (max(rows.min() - m, 0), min(rows.max() + 1 + m, H),
            max(cols.min() - m, 0), min(cols.max() + 1 + m, W))


# ============================================================================
# delovi sa pytorch-om (model, gubitak, treniranje)
# ============================================================================

def build_torch_parts():
    """pytorch se uvozi tek ovde, pa se gornje funkcije mogu koristiti i bez njega."""
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_seed(seed):
        """isto seme za sva pokretanja, da bi rezultati mogli da se ponove."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def load_model(path):
        """ista mreza kao u train.py, sa tezinama iz datoteke."""
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=3, classes=1)
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device)

    def masked_loss(logits, target, valid):
        """
        dice + bce kao u train.py, ali samo na vaznim (oznacenim) pikselima.
        dice se racuna za celu grupu i jednak je nuli kada u grupi nema piksela
        zgrade - isto kao dice gubitak iz biblioteke smp (binarni rezim).
        """
        v = valid.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * v).sum() / v.sum().clamp(min=1.0)
        p = torch.sigmoid(logits) * v
        t = target * v
        if t.sum() > 0:
            dice = 1 - 2 * (p * t).sum() / (p.sum() + t.sum()).clamp(min=1e-7)
        else:
            dice = torch.zeros((), device=logits.device)
        return bce + dice

    def make_infer(model):
        """vraca funkciju koja za jedan prozor daje verovatnocu zgrade."""
        model.eval()

        def infer(x_chw):
            with torch.no_grad():
                x = torch.from_numpy(x_chw).unsqueeze(0).to(device)
                with torch.amp.autocast(device, enabled=(device == "cuda")):
                    logits = model(x)
                return torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        return infer

    def fine_tune(model, image, mask, train_ok, rng):
        """dotreniravanje: epochs epoha po crops_per_epoch slucajnih prozora."""
        # moguce sredine prozora: svaki osmi piksel unutar podrucja za treniranje
        cr, cc = np.nonzero(train_ok[::8, ::8])
        centres = (cr * 8, cc * 8)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
        steps = CROPS_PER_EPOCH // BATCH_SIZE

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total = 0.0
            for _ in range(steps):
                batch = [random_training_example(rng, image, mask, train_ok, centres)
                         for _ in range(BATCH_SIZE)]
                x = torch.from_numpy(np.stack([normalise(b[0]) for b in batch])).to(device)
                y = torch.from_numpy(np.stack([b[1] for b in batch])).float().unsqueeze(1).to(device)
                v = torch.from_numpy(np.stack([b[2] for b in batch])).unsqueeze(1).to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device, enabled=(device == "cuda")):
                    logits = model(x)
                loss = masked_loss(logits.float(), y, v)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()
            print(f"      epoch {epoch:2d}/{EPOCHS}: loss {total / steps:.4f}")
        return model

    def free():
        """oslobadja memoriju graficke karte izmedju pokretanja."""
        if device == "cuda":
            torch.cuda.empty_cache()

    def save(model, path):
        torch.save(model.state_dict(), path)

    return device, set_seed, load_model, make_infer, fine_tune, save, free


# ============================================================================

def main():
    from ocena import HAVE_SCIPY, object_scores, pixel_scores   # iste mere kao u ocena.py
    import ocena
    device, set_seed, load_model, make_infer, fine_tune, save, free = build_torch_parts()
    print(f"Device: {device}")

    with rasterio.open(ORTHO) as src:
        image = src.read([1, 2, 3])
        img_transform, img_crs, profile = src.transform, src.crs, src.profile
    with rasterio.open(MASK) as s:
        mask = s.read(1)
    with rasterio.open(AREAS) as s:
        areas = s.read(1)
    H, W = mask.shape
    if image.shape[1:] != (H, W):
        raise ValueError("orthophoto and mask are not on the same grid")
    has_data = image.any(axis=0)              # isto pravilo kao u predict_overlap.py (sva tri kanala 0 = nema podataka)

    rows_out = []
    for model_name, model_path in BASE_MODELS.items():
        for test_name in TEST_AREAS:
            code = AREA_CODES[test_name]
            train_codes = [c for n, c in AREA_CODES.items() if n != test_name]
            train_names = [n for n in AREA_CODES if n != test_name]
            print(f"\n=== {model_name}: fine-tune on {' + '.join(train_names)}, "
                  f"test on {test_name} ===")

            set_seed(SEED)
            rng = np.random.default_rng(SEED)
            train_ok = np.isin(areas, train_codes) & has_data   # pikseli za treniranje

            r0, r1, c0, c1 = test_region(areas, code, H, W)
            inside = (areas[r0:r1, c0:c1] == code) & has_data[r0:r1, c0:c1]
            ref = (mask[r0:r1, c0:c1] == 1) & inside

            model = load_model(model_path)
            for phase in ("pre", "posle"):
                if phase == "posle":
                    t0 = time.time()
                    model = fine_tune(model, image, mask, train_ok, rng)
                    minutes = (time.time() - t0) / 60
                    print(f"      fine-tuning took {minutes:.1f} min")
                    save(model, MODELS_DIR / f"dotreniran_{model_name.lower()}_bez_{test_name}.pth")

                prob = predict_region(make_infer(model), image, r0, r1, c0, c1)
                pred = (prob > THRESHOLD) & inside

                row = {"model": model_name, "test_oblast": test_name, "faza": phase,
                       "trening_oblasti": "+".join(train_names) if phase == "posle" else ""}
                row.update(pixel_scores(ref, pred))
                if HAVE_SCIPY:
                    row.update(object_scores(ref, pred, ocena.OBJECT_IOU, ocena.MIN_OBJECT_PX))
                if phase == "posle":
                    row["minuta"] = round(minutes, 1)
                rows_out.append(row)
                print(f"   {phase:<6} IoU {row['IoU']:.3f}  precision {row['precision']:.3f}"
                      f"  recall {row['recall']:.3f}  F1 {row['F1']:.3f}")

                # predvidjanje dotreniranog modela na testnom podrucju, kao geotiff za slike
                if SAVE_PREDICTIONS and phase == "posle":
                    out = np.full(pred.shape, 255, dtype=np.uint8)
                    out[has_data[r0:r1, c0:c1]] = 0
                    out[(prob > THRESHOLD) & has_data[r0:r1, c0:c1]] = 1
                    p = profile.copy()
                    p.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="lzw",
                             width=c1 - c0, height=r1 - r0,
                             transform=window_transform(Window(c0, r0, c1 - c0, r1 - r0), img_transform))
                    for key in ("blockxsize", "blockysize", "tiled", "interleave", "photometric"):
                        p.pop(key, None)
                    path = DATA / f"dotreniran_{model_name.lower()}_{test_name}.tif"
                    with rasterio.open(path, "w", **p) as dst:
                        dst.write(out, 1)

            del model
            free()

    # ---- pregled rezultata ---------------------------------------------------
    print("\nmodel  test area      phase     IoU   prec  recall     F1")
    for r in rows_out:
        print(f"{r['model']:<7}{r['test_oblast']:<14}{r['faza']:<7}{r['IoU']:>7.3f}"
              f"{r['precision']:>7.3f}{r['recall']:>8.3f}{r['F1']:>7.3f}")

    keys = []
    for r in rows_out:
        keys += [k for k in r if k not in keys]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
