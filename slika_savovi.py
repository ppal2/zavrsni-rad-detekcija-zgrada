"""
slika_savovi.py - slika 4.4: predvidjanje bez preklapanja (savovi) i sa preklapanjem 50 % i hann tezinama

traje oko 4 minuta. samo cita modele i rastere. pravi:
    c:\\thesis\\slika_savovi_whu.png, c:\\thesis\\slika_savovi_inria.png   po jedna slika za svaki model
    c:\\thesis\\slika_savovi_podaci.npz                               isecci od kojih su slike nacrtane

slike se crtaju u drugom, posebnom python procesu koji ne ucitava pytorch (na
windows-u pytorch i matplotlib mogu doneti dve kopije biblioteke openmp, sto
zaustavlja program greskom "omp: error #15"). da bi se slike samo ponovo
nacrtale iz sacuvanih isecaka, bez novog predvidjanja:
    python slika_savovi.py --draw

sta radi, za svaki polazni model (whu i inria):
  1. predvidja ceo ortofoto od 30 cm dva puta, istim modelom:
       a) prozori 512 x 512 jedan do drugog, bez preklapanja   = kao predict.py
       b) prozori 512 x 512, korak 256, hann tezine             = kao predict_overlap.py
     oba nacina koriste istu normalizaciju, amp i prag 0,5, pa je jedina
     razlika nacin na koji se prozori spajaju.
  2. poredi oba rezultata sa rucno nacrtanom maskom u tri podrucja, istim
     merama po pikselima i po objektima kao ocena.py:
       - na celim podrucjima,
       - samo u uskom pojasu uz granice prozora (+-12 px = +-3,6 m),
         jer se tu pojavljuju savovi.
  3. gleda svaki ugao prozora (tacku u kojoj se srecu 4 prozora) i broji koliko
     piksela menja klasu izmedju a) i b) duz dve granice kroz taj ugao. ugao sa
     najvise promena je mesto gde su savovi najvidljiviji.
zatim crta po jednu sliku za svaki model, na uglu sa najvidljivijim savovima:
  gornji red:  verovatnoca zgrade (crno = 0, belo = 1)
  donji red:   ortofoto sa maskom (verovatnoca > 0,5) u crvenoj boji
  levo:        bez preklapanja        desno: sa preklapanjem
zute crtice na ivicama panela pokazuju gde su granice prozora.

pokretanje (u conda okruzenju "thesis", iz foldera c:\\thesis; ocena.py mora
biti u istom folderu):
    python slika_savovi.py
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# podesavanja
# ---------------------------------------------------------------------------
DATA = Path(r"C:\thesis\data")
ORTHO = DATA / "campus_30cm.tif"
MASK = DATA / "isecci_kampus" / "maska_30cm.tif"
AREAS = DATA / "isecci_kampus" / "oblasti_30cm.tif"
MODELS = {
    "WHU": Path(r"C:\thesis\best_model_whu.pth"),
    "Inria": Path(r"C:\thesis\best_model_inria.pth"),
}
OUT_DIR = Path(r"C:\thesis")
CROPS_NPZ = OUT_DIR / "slika_savovi_podaci.npz"

TILE = 512
STEP = 256            # preklapanje 50 %, kao u predict_overlap.py
THRESHOLD = 0.5
BATCH = 4             # prozori po jednom pozivu graficke kartice (samo zbog brzine; rezultat je isti kao jedan po jedan)
PIXEL_M = 0.3         # velicina piksela campus_30cm.tif u metrima
CROP = 700            # velicina isecka za sliku u pikselima (700 x 0,3 m = 210 m)
BAND = 12             # pojas sa svake strane granice prozora, u pikselima (12 x 0,3 m = 3,6 m)

FORCE_CENTRE = None   # npr. (5120, 3584) - red i kolona sredine isecka u pikselima (najbolje umnosci broja 512)

# srednje vrednosti i standardne devijacije imagenet-a, kao u train.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# predvidjanje
# ---------------------------------------------------------------------------
def make_infer(model_path):
    """ucitava model i vraca funkciju: uint8 prozori (b, 3, 512, 512) -> verovatnoce (b, 512, 512)."""
    import torch
    import segmentation_models_pytorch as smp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()      # eval: batch normalizacija koristi sacuvane statistike,
                                 # pa prozor daje isti rezultat u grupi od 4 kao i sam

    def infer(patches):
        x = (patches.astype(np.float32) / 255.0 - MEAN) / STD      # isto kao u train.py
        with torch.no_grad():
            xt = torch.from_numpy(x).to(device)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                logits = model(xt)
            return torch.sigmoid(logits.float())[:, 0].cpu().numpy()
    return infer


def window(image, r, c):
    """prozor 512 x 512 od reda r i kolone c; deo izvan slike je 0 (kao citanje van granica u rasterio-u)."""
    H, W = image.shape[1:]
    out = np.zeros((3, TILE, TILE), np.uint8)
    h, w = min(TILE, H - r), min(TILE, W - c)
    out[:, :h, :w] = image[:, r:r + h, c:c + w]
    return out


def predict_windows(infer, image, step, desc):
    """pusta model preko svih prozora sa datim korakom; za svaki vraca (verovatnoca, r, c, h, w)."""
    H, W = image.shape[1:]
    offsets = [(r, c) for r in range(0, H, step) for c in range(0, W, step)]
    for i in tqdm(range(0, len(offsets), BATCH), desc=desc):
        chunk = offsets[i:i + BATCH]
        probs = infer(np.stack([window(image, r, c) for r, c in chunk]))
        for (r, c), p in zip(chunk, probs):
            # prozori na desnoj i donjoj ivici su delom izvan slike - zadrzava se samo deo unutar nje
            h, w = min(TILE, H - r), min(TILE, W - c)
            yield p[:h, :w], r, c, h, w


def predict_no_overlap(infer, image):
    """a) prozori jedan do drugog: svaki piksel dolazi iz tacno jednog prozora (predict.py)."""
    prob = np.zeros(image.shape[1:], np.float32)
    for p, r, c, h, w in predict_windows(infer, image, TILE, "   without overlap"):
        prob[r:r + h, c:c + w] = p
    return prob


def predict_overlap(infer, image):
    """b) korak 256 + hann tezine: tezinski prosek do 4 prozora po pikselu (predict_overlap.py)."""
    hann = np.hanning(TILE).astype(np.float32)
    weight = np.outer(hann, hann) + 1e-6       # 1e-6 da tezina na ivici prozora ne bude tacno 0
    prob_sum = np.zeros(image.shape[1:], np.float32)
    weight_sum = np.zeros(image.shape[1:], np.float32)
    for p, r, c, h, w in predict_windows(infer, image, STEP, "   with overlap   "):
        prob_sum[r:r + h, c:c + w] += p * weight[:h, :w]
        weight_sum[r:r + h, c:c + w] += weight[:h, :w]
    prob_sum /= weight_sum
    return prob_sum


# ---------------------------------------------------------------------------
# ocena i izbor mesta za sliku
# ---------------------------------------------------------------------------
def border_strip(H, W):
    """logicka maska piksela na najvise band px od granice dva prozora bez preklapanja (redovi/kolone k*512, k >= 1)."""
    def near(n):
        i = np.arange(n)
        k = np.maximum(np.rint(i / TILE), 1)          # najbliza granica; k = 0 bi bila ivica slike
        return np.abs(i - k * TILE) <= BAND
    return near(H)[:, None] | near(W)[None, :]


def seam_score(diff, r, c):
    """pikseli koji menjaju klasu izmedju a) i b) duz dve granice kroz ugao (r, c)."""
    half = CROP // 2
    d = diff[r - half:r + half, c - half:c + half]
    return int(d[half - BAND:half + BAND, :].sum() + d[:, half - BAND:half + BAND].sum())


def best_corner(p_no, p_ov, has_data, min_data=0.98, min_built=0.10):
    """ugao prozora sa najvise promena klase; isecak mora biti pokriven snimkom i imati zgrade."""
    H, W = p_no.shape
    half = CROP // 2
    diff = (p_no > THRESHOLD) != (p_ov > THRESHOLD)
    best = None
    for r in range(TILE, H - half, TILE):              # uglovi prozora su na umnoscima broja 512
        for c in range(TILE, W - half, TILE):
            rs, cs = slice(r - half, r + half), slice(c - half, c + half)
            if has_data[rs, cs].mean() < min_data:     # isecak mora biti (skoro) ceo pokriven ortofotoom
                continue
            if (p_ov[rs, cs] > THRESHOLD).mean() < min_built:   # i mora sadrzati zgrade
                continue
            s = seam_score(diff, r, c)
            if best is None or s > best[0]:
                best = (s, r, c)
    if best is None and min_built > 0:                 # nijedan ugao nije prosao: ponovo, sa blazim uslovima
        return best_corner(p_no, p_ov, has_data, 0.5, 0)
    return best


def evaluate(name, p_no, p_ov, ref, inside, strip, pixel_scores, object_scores, have_scipy):
    """mere za oba nacina spajanja: na celim podrucjima i u pojasu uz granice prozora."""
    rows = []
    for method, prob in (("without overlap", p_no), ("with overlap", p_ov)):
        pred = prob > THRESHOLD
        row = {"model": name, "method": method}
        whole = pixel_scores(ref & inside, pred & inside)
        row.update({k: whole[k] for k in ("IoU", "precision", "recall")})
        row["IoU strip"] = pixel_scores(ref & inside & strip, pred & inside & strip)["IoU"]
        if have_scipy:
            o = object_scores(ref & inside, pred & inside, 0.5, 20)     # isto pravilo kao u ocena.py
            row.update(obj_pred=o["obj_pred"], found=o["found"], obj_F1=o["obj_F1"])
        rows.append(row)
    # procenat oznacenih piksela koji menjaju klasu kada se doda preklapanje
    changed = ((p_no > THRESHOLD) != (p_ov > THRESHOLD)) & inside
    rows[0]["changed %"] = 100 * changed.sum() / inside.sum()
    return rows


# ---------------------------------------------------------------------------
# slika
# ---------------------------------------------------------------------------
def draw(crop, out_png):
    """crta sliku 2 x 2 (verovatnoca i maska, bez i sa preklapanjem); natpisi su cirilicom, kao u radu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    half = CROP // 2
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6), layout="constrained")
    titles = [("(а) без преклапања", "(б) преклапање 50 % и Hann тежине"),
              ("(в) без преклапања", "(г) преклапање 50 % и Hann тежине")]
    for j, key in enumerate(("p_no", "p_ov")):
        im = axes[0, j].imshow(crop[key], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].imshow(crop["ortho"], interpolation="nearest")
        # providni crveni sloj preko ortofotoa tamo gde je verovatnoca > 0,5
        overlay = np.zeros(crop[key].shape + (4,), np.float32)
        overlay[crop[key] > THRESHOLD] = (0.92, 0.05, 0.05, 0.45)
        axes[1, j].imshow(overlay, interpolation="nearest")
        axes[0, j].set_title(titles[0][j], fontsize=9, loc="left", pad=12)    # pad: mesto za crtice iznad
        axes[1, j].set_title(titles[1][j], fontsize=9, loc="left", pad=12)
    axes[0, 0].set_ylabel("вероватноћа зграде")
    axes[1, 0].set_ylabel("маска (вероватноћа > 0,5)")

    for ax in axes.flat:                                  # zute crtice = granice prozora
        ax.set_xticks([half - 0.5]); ax.set_yticks([half - 0.5])
        ax.tick_params(length=8, width=2.5, color="#ffc400", direction="out", top=True, right=True,
                       labelbottom=False, labelleft=False)

    ax = axes[1, 0]                                       # razmernik 50 m
    px = 50 / PIXEL_M
    x0, y0 = 20, CROP - 28
    ax.add_patch(Rectangle((x0 - 10, y0 - 34), px + 70, 52, fc="white", ec="none", alpha=0.9))
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * px / 4, y0), px / 4, 9, fc="black" if i % 2 == 0 else "white",
                               ec="black", lw=0.8))
    ax.text(x0, y0 - 5, "0", ha="center", va="bottom", fontsize=8)
    ax.text(x0 + px, y0 - 5, "50 m", ha="center", va="bottom", fontsize=8)

    cb = fig.colorbar(im, ax=axes[0, :], orientation="vertical", fraction=0.035, pad=0.02)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", "0,5", "1"])
    fig.text(0.5, -0.01, f"Модел {crop['model']}. Жуте ознаке на ивицама: границе прозора 512 × 512 "
             "(на сваких 153,6 m).", ha="center", va="top", fontsize=8.5, color="#333333")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    import rasterio
    from ocena import HAVE_SCIPY, object_scores, pixel_scores      # iste mere kao u ocena.py

    with rasterio.open(ORTHO) as src:
        image = src.read([1, 2, 3])
        transform = src.transform
    with rasterio.open(MASK) as s:
        ref = s.read(1) == 1
    with rasterio.open(AREAS) as s:
        areas = s.read(1)
    H, W = ref.shape
    if image.shape[1:] != (H, W):
        raise ValueError("orthophoto and mask are not on the same grid")
    has_data = image.any(axis=0)                     # isto pravilo za piksele bez podataka kao u predict_overlap.py
    inside = (areas > 0) & has_data
    strip = border_strip(H, W)
    print(f"Orthophoto {W} x {H} px; labelled pixels {inside.sum()}, "
          f"of them {100 * (inside & strip).sum() / inside.sum():.1f} % in the strip along window borders")

    # gde granice prozora predvidjanja bez preklapanja prolaze kroz svako oznaceno
    # podrucje (za poredjenje sa pravim linijama koje se vide na karti gresaka u qgis-u)
    print("Window borders of the no-overlap prediction inside each area (every 153.6 m):")
    for code, area_name in {1: "petrovaradin", 2: "kampus", 3: "centar"}.items():
        rows, cols = np.nonzero(areas == code)
        if not len(rows):
            continue
        xs = [transform * (k * TILE, 0) for k in range(1, W // TILE + 1)
              if cols.min() <= k * TILE <= cols.max()]
        ys = [transform * (0, k * TILE) for k in range(1, H // TILE + 1)
              if rows.min() <= k * TILE <= rows.max()]
        print(f"   {area_name:13s} vertical at x = {', '.join(f'{p[0]:.1f}' for p in xs) or '-'}")
        print(f"   {'':13s} horizontal at y = {', '.join(f'{p[1]:.1f}' for p in ys) or '-'}")

    results, crops = [], {}
    for name, path in MODELS.items():
        print(f"\n=== {name} ===")
        infer = make_infer(path)
        p_no = predict_no_overlap(infer, image)
        p_ov = predict_overlap(infer, image)
        results += evaluate(name, p_no, p_ov, ref, inside, strip, pixel_scores, object_scores, HAVE_SCIPY)

        # mesto za sliku: zadato rucno ili ugao sa najvidljivijim savovima
        if FORCE_CENTRE:
            r, c = FORCE_CENTRE
            score = seam_score((p_no > THRESHOLD) != (p_ov > THRESHOLD), r, c)
        else:
            score, r, c = best_corner(p_no, p_ov, has_data)
        half = CROP // 2
        rs, cs = slice(r - half, r + half), slice(c - half, c + half)
        crops[f"{name}_ortho"] = image[:, rs, cs].transpose(1, 2, 0).copy()
        crops[f"{name}_p_no"] = p_no[rs, cs].copy()
        crops[f"{name}_p_ov"] = p_ov[rs, cs].copy()
        x, y = transform * (c, r)
        print(f"   most visible seams at row {r}, col {c} = x {x:.1f}, y {y:.1f} (EPSG:32634): "
              f"{score} pixels change class along the borders")
        del infer, p_no, p_ov                        # oslobadja memoriju pre sledeceg modela

    # ---- rezultati -------------------------------------------------------------
    print("\nRESULTS - all three areas together, threshold 0.5 (paste this to Claude)")
    head = f"{'model':<7}{'method':<17}{'IoU':>7}{'prec':>7}{'recall':>8}{'IoU strip':>11}"
    if HAVE_SCIPY:
        head += f"{'pred obj':>10}{'found':>7}{'obj F1':>8}"
    print(head)
    for row in results:
        line = (f"{row['model']:<7}{row['method']:<17}{row['IoU']:>7.3f}{row['precision']:>7.3f}"
                f"{row['recall']:>8.3f}{row['IoU strip']:>11.3f}")
        if HAVE_SCIPY:
            line += f"{row['obj_pred']:>10}{row['found']:>7}{row['obj_F1']:>8.3f}"
        print(line)
    for row in results:
        if "changed %" in row:
            print(f"{row['model']}: {row['changed %']:.2f} % of the labelled pixels change class with overlap")
    print("(the 'with overlap' rows should match ocena.py)")

    # isecci se cuvaju, a slike crta novi python proces (vidi opis na vrhu)
    np.savez_compressed(CROPS_NPZ, **crops)
    print(f"\nCut-outs saved to {CROPS_NPZ}; drawing the figures in a separate process ...")
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")   # samo za crtanje, svi brojevi su vec izracunati
    subprocess.run([sys.executable, os.path.abspath(__file__), "--draw"], env=env, check=True)


def draw_all():
    """crta slike iz sacuvanih isecaka (poziva se sa --draw)."""
    data = np.load(CROPS_NPZ)
    for name in MODELS:
        if f"{name}_ortho" not in data:
            continue
        crop = {"model": name, "ortho": data[f"{name}_ortho"],
                "p_no": data[f"{name}_p_no"], "p_ov": data[f"{name}_p_ov"]}
        out = OUT_DIR / f"slika_savovi_{name.lower()}.png"
        draw(crop, out)
        print(f"   {out}")


if __name__ == "__main__":
    if "--draw" in sys.argv:
        draw_all()
    else:
        main()
