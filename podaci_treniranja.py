"""
podaci_treniranja.py - brojevi za odeljak 4.4 i tabelu 6.2 (treniranje dva polazna modela)

nista se zaista ne trenira i nista se ne cuva: .pth datoteke se samo citaju.
traje oko 5 minuta.

sta radi:
  1. broji isecke u folderima whu i inria.                    -> tacni brojevi
  2. cita datume datoteka best_model_whu.pth i best_model_inria.pth.
  3. pokrece celu validaciju sa svakim sacuvanim modelom, tacno kao train.py.
     tako se ponovo proveravaju vrednosti 0,85 i 0,73 iz rada i meri se koliko
     traje jedan prolaz validacije.
  4. meri koliko traje jedan korak treniranja na graficku kartici, sa tacno
     istim podesavanjima kao train.py (u-net/resnet34, grupa od 2 isecka, amp,
     adam, isecci se citaju sa diska, okretanja i rotacije). koristi se novi
     model koji se posle odbacuje.
  5. iz 3 i 4: trajanje jedne epohe i 20 epoha.
     iz 2 i 5: u kojoj epohi je sacuvan najbolji model.

zasto datumi datoteke otkrivaju najbolju epohu:
  train.py pocinje sa best_iou = 0, pa se posle prve epohe model uvek cuva ->
  datoteka nastaje na kraju prve epohe. svako kasnije poboljsanje prepisuje istu
  datoteku. windows pri prepisivanju zadrzava datum nastanka datoteke i menja samo
  datum izmene (kasnije preimenovanje, npr. u best_model_whu.pth, zadrzava oba
  datuma). zato je
        izmena - nastanak = vreme od kraja prve epohe do kraja najbolje epohe
        najbolja epoha    = 1 + (izmena - nastanak) / (trajanje jedne epohe)
  ovo vazi samo ako je datoteka preimenovana, a ne kopirana (kopija dobija nov
  datum nastanka - program to prepoznaje), i ako racunar nije bio u stanju
  spavanja tokom treniranja.

pokretanje (u conda okruzenju "thesis", iz foldera c:\\thesis):
    python podaci_treniranja.py
"""
import glob
import math
import os
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# podesavanja - iste putanje kao u train.py, tile_inria.py i dotreniravanje.py
# ---------------------------------------------------------------------------
DATASETS = {
    #  ime:    (folder sa iseccima,                 sacuvani model)
    "WHU":   (r"C:\thesis\data\whu",              r"C:\thesis\best_model_whu.pth"),
    "Inria": (r"C:\thesis\data\inria_tiles",      r"C:\thesis\best_model_inria.pth"),
}
EPOCHS = 20            # kao u train.py
BATCH_SIZE = 2         # kao u train.py
WARMUP_STEPS = 20      # prvi koraci su sporiji (pokretanje graficke kartice), ne mere se
TIMED_STEPS = 150      # koraci treniranja koji se mere (oko 1 minut)

# srednje vrednosti i standardne devijacije imagenet-a, kao u train.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tiles(root, split):
    """spisak isecaka slike u jednom delu skupa (train, val ili test)."""
    return sorted(glob.glob(os.path.join(root, split, "image", "*.tif")))


def fmt_minutes(minutes):
    """minuti -> tekst oblika "1 h 05 min" ili "45 min"."""
    h, m = divmod(int(round(minutes)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


# ---------------------------------------------------------------------------
# 1 + 2: broj isecaka i datumi datoteka (pytorch nije potreban)
# ---------------------------------------------------------------------------
def count_and_dates():
    info = {}
    print("1) TILES")
    for name, (root, model_path) in DATASETS.items():
        counts = {s: len(tiles(root, s)) for s in ("train", "val", "test")}
        labels = {s: len(glob.glob(os.path.join(root, s, "label", "*.tif"))) for s in counts}
        line = ", ".join(f"{s} {counts[s]}" for s in counts if counts[s])
        print(f"   {name:6s} {line or 'NO TILES FOUND under ' + root}")
        # svaki isecak slike mora imati svoj isecak maske
        for s in counts:
            if counts[s] != labels[s]:
                print(f"   WARNING: {name} {s}: {counts[s]} images but {labels[s]} labels")
        info[name] = {"counts": counts}

    print("\n2) DATES OF THE SAVED MODELS")
    for name, (root, model_path) in DATASETS.items():
        if not os.path.exists(model_path):
            print(f"   {name:6s} {model_path} not found")
            continue
        st = os.stat(model_path)
        # na windows-u je datum nastanka st_birthtime (python 3.12 i noviji) ili st_ctime
        created = getattr(st, "st_birthtime", st.st_ctime)
        modified = st.st_mtime
        info[name].update(created=created, modified=modified)
        c = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
        m = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
        print(f"   {name:6s} created {c}   modified {m}   "
              f"difference {fmt_minutes((modified - created) / 60)}")
    return info


# ---------------------------------------------------------------------------
# 3 + 4: validacija sacuvanih modela i brzina jednog koraka treniranja
# ---------------------------------------------------------------------------
def measure(info):
    import torch
    import segmentation_models_pytorch as smp
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print("WARNING: no GPU found - the times below will NOT be the times of your real training")

    class Tiles(Dataset):                       # isto kao klasa skupa podataka u train.py
        def __init__(self, root, split, augment=False):
            self.images = tiles(root, split)
            self.labels = sorted(glob.glob(os.path.join(root, split, "label", "*.tif")))
            self.augment = augment

        def __len__(self):
            return len(self.images)

        def __getitem__(self, i):
            image = np.array(Image.open(self.images[i]), dtype=np.float32)
            label = np.array(Image.open(self.labels[i])).astype(np.float32)
            image = (image / 255.0 - MEAN) / STD
            if self.augment:
                # ista slucajna okretanja i rotacije kao u train.py
                if np.random.rand() > 0.5:
                    image, label = np.fliplr(image), np.fliplr(label)
                if np.random.rand() > 0.5:
                    image, label = np.flipud(image), np.flipud(label)
                k = np.random.randint(4)
                image, label = np.rot90(image, k).copy(), np.rot90(label, k).copy()
            return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(label).unsqueeze(0)

    def compute_iou(logits, targets):           # isto kao u train.py (po grupi isecaka)
        preds = (torch.sigmoid(logits) > 0.5).float()
        inter = (preds * targets).sum()
        union = preds.sum() + targets.sum() - inter
        return 1.0 if union == 0 else (inter / union).item()

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()            # ceka da graficka kartica zaista zavrsi posao

    amp = device == "cuda"
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # --- 3) cela validacija sa svakim sacuvanim modelom -----------------------
    print("\n3) VALIDATION WITH THE SAVED MODELS (same metric as train.py)")
    for name, (root, model_path) in DATASETS.items():
        if not os.path.exists(model_path) or not info[name]["counts"]["val"]:
            print(f"   {name}: skipped (model or validation tiles missing)")
            continue
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        loader = DataLoader(Tiles(root, "val"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        iou_sum = 0.0
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"   {name} val"):
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device, enabled=amp):
                    logits = model(images)
                    dice_loss(logits, labels) + bce_loss(logits, labels)   # i train.py racuna gubitak, pa se i ovde racuna
                iou_sum += compute_iou(logits.float(), labels)
        sync(); val_minutes = (time.perf_counter() - t0) / 60
        info[name].update(val_iou=iou_sum / len(loader), val_minutes=val_minutes)
        print(f"   {name:6s} val IoU {iou_sum / len(loader):.4f}   one validation pass {val_minutes:.1f} min")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- 4) brzina jednog koraka treniranja (novi model, posle se odbacuje) ---
    print("\n4) SPEED OF ONE TRAINING STEP (as in train.py, nothing is saved)")
    # uzima se prvi skup koji ima isecke za treniranje
    root = next((r for n, (r, _) in DATASETS.items() if info[n]["counts"]["train"]), None)
    if root is None:
        print("   no training tiles found")
        return None
    try:
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    except Exception:                           # bez interneta / bez sacuvanih tezina: brzina je ista
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device, enabled=amp)
    loader = DataLoader(Tiles(root, "train", augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    it = iter(loader)
    for step in tqdm(range(WARMUP_STEPS + TIMED_STEPS), desc="   training steps"):
        if step == WARMUP_STEPS:
            sync(); t0 = time.perf_counter()    # merenje pocinje posle koraka za zagrevanje
        images, labels = next(it)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device, enabled=amp):
            logits = model(images)
            loss = dice_loss(logits, labels) + bce_loss(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss.item()                             # i train.py ovo radi u svakom koraku
    sync()
    sec_per_step = (time.perf_counter() - t0) / TIMED_STEPS
    print(f"   {sec_per_step:.3f} s per training step (batch of {BATCH_SIZE} tiles)")
    return sec_per_step


# ---------------------------------------------------------------------------
# 5: pregled rezultata
# ---------------------------------------------------------------------------
def summary(info, sec_per_step):
    print("\n" + "=" * 70)
    print("SUMMARY - paste this to Claude")
    print("=" * 70)
    for name, d in info.items():
        c = d["counts"]
        print(f"{name}: train {c['train']} tiles, val {c['val']}" + (f", test {c['test']}" if c["test"] else ""))
        if "val_iou" in d:
            print(f"   val IoU of the saved model: {d['val_iou']:.4f}")
        if sec_per_step is None or "val_minutes" not in d or not c["train"]:
            continue
        # jedna epoha = svi koraci treniranja + jedan prolaz validacije
        steps = math.ceil(c["train"] / BATCH_SIZE)
        epoch_min = steps * sec_per_step / 60 + d["val_minutes"]
        print(f"   one epoch = {steps} steps x {sec_per_step:.3f} s + validation {d['val_minutes']:.1f} min"
              f" = {epoch_min:.1f} min")
        print(f"   {EPOCHS} epochs = about {fmt_minutes(EPOCHS * epoch_min)}")
        if "created" not in d:
            continue
        diff_min = (d["modified"] - d["created"]) / 60
        if diff_min < -1:
            # datum nastanka je posle datuma izmene -> datoteka je kopirana posle treniranja
            print("   best epoch: CANNOT be estimated - the file was created after it was last modified,"
                  " so it was copied after training (a copy gets a new creation date)")
            continue
        k = 1 + diff_min / epoch_min
        note = "" if 0.5 <= k <= EPOCHS + 0.5 else "   <- not possible, the dates are not reliable"
        print(f"   best epoch from the file dates: 1 + {diff_min:.0f} / {epoch_min:.1f} = {k:.1f}"
              f" -> epoch {round(k)} (+/- 1, the GPU speed today is not exactly the speed back then){note}")


if __name__ == "__main__":
    info = count_and_dates()
    sec_per_step = measure(info)
    summary(info, sec_per_step)
