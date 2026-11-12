"""
train.py - treniranje polaznog modela (u-net sa enkoderom resnet34)

isti program je koriscen za oba polazna modela; razlikuje se samo skup podataka:
  - model whu:   DATA_ROOT = DATA / "whu",         MODEL_OUT = best_model_whu.pth
  - model inria: DATA_ROOT = DATA / "inria_tiles"  (napravio tile_inria.py),
                 MODEL_OUT = best_model_inria.pth
(ovde su upisana podesavanja za model inria; putanje su u podesavanja.py.)

seed: SEED je dodat posle izrade rada. polazni modeli koji su korisceni u radu
trenirani su bez seed-a, pa se njihovi rezultati ne mogu ponoviti potpuno isto.
sa seed-om novo treniranje svaki put pravi iste nasumicne izbore (mesanje
isecaka, uvecanje skupa, pocetne tezine dekodera). na grafickoj kartici neki
proracuni ni tada nisu potpuno deterministicki, pa rezultat moze malo da se razlikuje.

podesavanja: 20 epoha, grupa od 2 isecka 512 x 512, adam sa stopom ucenja 1e-4,
gubitak dice + bce, mesovita preciznost (amp), nasumicna odrazavanja i rotacije.
posle svake epohe racuna se iou na validacionom skupu, a cuva se model sa
najvecim iou (ne poslednji, jer kasnije epohe mogu biti pretrenirane).

pokretanje:
    python train.py
"""

import glob
import os
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from podesavanja import DATA, THESIS_DIR   # putanje su u podesavanja.py

# ---------------------------------------------------------------------------
# podesavanja
# ---------------------------------------------------------------------------

DATA_ROOT = DATA / "inria_tiles"                 # za model whu: DATA / "whu"
MODEL_OUT = THESIS_DIR / "best_model_inria.pth"  # za model whu: best_model_whu.pth
SEED = 42                  # pocetna vrednost generatora slucajnih brojeva (isto kao u dotreniravanje.py)
ENCODER = "resnet34"       # enkoder (kicma) mreze u-net
BATCH_SIZE = 2             # mala grupa, jer graficka karta gtx 1050 ti ima samo 4 gb memorije
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 0            # 0 je najsigurnije na windows-u

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# srednja vrednost i standardna devijacija kanala skupa imagenet. enkoder je
# unapred treniran na imagenet-u, pa ulaz mora biti normalizovan na isti nacin.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed):
    """
    postavlja isti pocetak za sve generatore slucajnih brojeva (python, numpy,
    pytorch), pa svako novo treniranje pravi iste nasumicne izbore.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn bira iste (deterministicke) algoritme pri svakom pokretanju
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# skup podataka
# ---------------------------------------------------------------------------

class WHUBuildingDataset(Dataset):
    """
    cita isecke iz foldera root/split/image i root/split/label.

    ime klase potice iz prve verzije (samo whu), ali klasa radi i za skup inria,
    jer tile_inria.py pravi isti raspored foldera i iste 0/1 oznake.
    dataset kaze pytorch-u koliko ima primera (__len__) i kako se ucitava
    primer broj i (__getitem__); dataloader zatim pravi grupe i mesa redosled.
    """

    def __init__(self, root, split, augment=False):
        self.image_paths = sorted(glob.glob(os.path.join(root, split, "image", "*.tif")))
        self.label_paths = sorted(glob.glob(os.path.join(root, split, "label", "*.tif")))

        assert len(self.image_paths) == len(self.label_paths), \
            f"{len(self.image_paths)} images but {len(self.label_paths)} labels"
        assert len(self.image_paths) > 0, f"No files found under {root}\\{split}"

        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]), dtype=np.float32)
        label = np.array(Image.open(self.label_paths[idx]))

        # oznake su 0 = pozadina i 1 = zgrada; funkcija gubitka trazi float
        label = label.astype(np.float32)

        # piksele iz opsega 0-255 prvo delimo sa 255, pa normalizujemo vrednostima imagenet-a
        image = image / 255.0
        image = (image - MEAN) / STD

        if self.augment:
            # uvecanje skupa: nasumicno odrazavanje (levo-desno i gore-dole) i rotacija
            # za 0, 90, 180 ili 270 stepeni. snimak odozgo nema "gore", pa je to i dalje
            # ispravan primer; slika i oznaka se menjaju na isti nacin.
            if np.random.rand() > 0.5:
                image, label = np.fliplr(image), np.fliplr(label)
            if np.random.rand() > 0.5:
                image, label = np.flipud(image), np.flipud(label)
            k = np.random.randint(4)
            image, label = np.rot90(image, k), np.rot90(label, k)

            # posle odrazavanja niz ima negativne korake, koje torch.from_numpy ne prima;
            # .copy() pravi obican niz
            image, label = image.copy(), label.copy()

        # pytorch ocekuje redosled (kanali, redovi, kolone), a slika je (redovi, kolone, kanali)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # model daje jedan kanal, pa i oznaka dobija osu kanala: (1, redovi, kolone)
        label = torch.from_numpy(label).unsqueeze(0)

        return image, label


# ---------------------------------------------------------------------------
# mera tacnosti
# ---------------------------------------------------------------------------

def compute_iou(logits, targets, threshold=0.5):
    """
    iou (odnos preseka i unije) za jednu grupu prozora.

    iou = presek predvidjenih i stvarnih zgrada / njihova unija; 1.0 je savrseno
    poklapanje, 0.0 nema preklapanja. ako u grupi nema ni predvidjenih ni stvarnih
    zgrada, unija je nula i takva grupa dobija 1.0.
    validacioni iou je zatim prosek ovih vrednosti po grupama.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    if union == 0:
        return 1.0

    return (intersection / union).item()


# ---------------------------------------------------------------------------
# treniranje
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = WHUBuildingDataset(DATA_ROOT, "train", augment=True)
    val_ds = WHUBuildingDataset(DATA_ROOT, "val", augment=False)

    print(f"Train tiles: {len(train_ds)}, validation tiles: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # encoder_weights="imagenet" je transfer ucenje: enkoder pocinje od tezina
    # naucenih na imagenet-u, a ne od slucajnih vrednosti
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    # dva gubitka se sabiraju, jer svaki pokriva slabost drugog:
    # bce radi piksel po piksel, ali vecina piksela je pozadina;
    # dice meri preklapanje celih oblasti, pa resava taj problem, ali sam daje nemirnije ucenje
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # mesovita preciznost (amp): deo racunanja sa 16 bita, pa treba manje memorije i brze je
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):

        # --- prolaz kroz skup za treniranje ---
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # gradijenti se inace sabiraju, pa se brisu pre svakog koraka
            optimizer.zero_grad()

            with torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                logits = model(images)
                loss = dice_loss(logits, labels) + bce_loss(logits, labels)

            scaler.scale(loss).backward()   # racuna gradijente
            scaler.step(optimizer)          # menja tezine
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- validacija: model se ne menja, samo se meri ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        # no_grad: gradijenti nisu potrebni, pa se stedi memorija i vreme
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                with torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                    logits = model(images)
                    loss = dice_loss(logits, labels) + bce_loss(logits, labels)

                val_loss += loss.item()
                val_iou += compute_iou(logits.float(), labels)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch {epoch}: train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val IoU {val_iou:.4f}")

        # cuva se najbolji model, a ne poslednji
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  -> new best (IoU {best_iou:.4f}), saved to {MODEL_OUT.name}")

    print(f"\nFinished. Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
