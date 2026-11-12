"""
ocena.py - ocena tacnosti polaznih modela na tri rucno oznacena podrucja

poredi predvidjanje modela sa rucno nacrtanom maskom, posebno za svako
podrucje i za sva tri zajedno. radi na celim rasterima od 30 cm (bez isecaka),
pa se racuna svaki oznaceni piksel.

ulazi (svi na istoj mrezi piksela kao campus_30cm.tif):
  maska_30cm.tif    iz pravi_isecke.py      0 = pozadina, 1 = zgrada
  oblasti_30cm.tif  iz pravi_isecke.py      kod podrucja, 0 = nije oznaceno
  po jedno predvidjanje za svaki model, iz predict_overlap.py
                    0 = pozadina, 1 = zgrada, 255 = nema podataka

1) mere po pikselima - tp, fp i fn se prebroje na celom podrucju odjednom:
     tp = zgrada u oba            fp = predvidjena zgrada, a u stvari pozadina
     fn = stvarna zgrada, propustena   (tacno negativni nisu potrebni)
     iou         = tp / (tp + fp + fn)
     preciznost  = tp / (tp + fp)       koliki deo predvidjenog je tacan
     odziv       = tp / (tp + fn)       koliki deo stvarnih zgrada je pronadjen
     f1          = 2 tp / (2 tp + fp + fn)   (isti broj kao dice)

2) mere po objektima (treba biblioteka scipy) - svaka povezana grupa piksela
   zgrade je jedan objekat (zgrade koje se dodiruju su jedan objekat, kao u
   maski). referentni objekat je pronadjen ako ga jedan predvidjeni objekat
   preklapa sa iou >= 0,5 (pravilo skupa spacenet); svaki predvidjeni objekat
   moze se iskoristiti samo jednom.
     pronadjeni  = upareni objekti
     propusteni  = referentni objekti bez para    (inspire "nedostatak")
     visak       = predvidjeni objekti bez para   (inspire "visak")

3) karte gresaka za slike: 1 = tp, 2 = fp, 3 = fn, 0 = tn, 255 = izvan podrucja.

pokretanje:  python ocena.py
"""

import csv
from pathlib import Path

import numpy as np
import rasterio

from podesavanja import DATA   # putanje su u podesavanja.py

# ----------------------------------------------------------------------------
# podesavanja
# ----------------------------------------------------------------------------

MASK = DATA / "isecci_kampus" / "maska_30cm.tif"
AREAS = DATA / "isecci_kampus" / "oblasti_30cm.tif"

# ime modela -> raster predvidjanja. oba moraju biti iz predict_overlap.py (isto
# preklapanje, iste hann tezine, isti prag), da bi poredjenje bilo posteno.
PREDICTIONS = {
    "WHU": DATA / "campus_buildings_whu_overlap.tif",   # predict_overlap.py sa best_model_whu.pth
    "Inria": DATA / "campus_buildings_inria.tif",       # predict_overlap.py sa best_model_inria.pth
}

# kod podrucja u oblasti_30cm.tif -> ime u rezultatima. kodovi poticu od vrednosti
# polja "skup" kada je pokrenut pravi_isecke.py: trening = 1, validacija = 2, test = 3.
AREA_NAMES = {1: "petrovaradin", 2: "kampus", 3: "centar"}

OBJECT_IOU = 0.5     # prag za uparivanje objekata
MIN_OBJECT_PX = 20   # grupe manje od 20 piksela (1,8 m2) se ne racunaju kao objekti,
                     # ni u referenci ni u predvidjanju
SAVE_ERROR_MAPS = True

OUT_CSV = DATA / "rezultati_ocena.csv"
# ----------------------------------------------------------------------------


def read_band(path):
    """cita prvi kanal rastera, zajedno sa transformacijom, koordinatnim sistemom i profilom."""
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs, src.profile


def pixel_scores(ref, pred):
    """mere po pikselima iz dve logicke maske (referenca i predvidjanje)."""
    tp = int(np.count_nonzero(ref & pred))
    fp = int(np.count_nonzero(~ref & pred))
    fn = int(np.count_nonzero(ref & ~pred))
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else float("nan")
    return {"IoU": iou, "precision": precision, "recall": recall, "F1": f1,
            "TP": tp, "FP": fp, "FN": fn}


def label_objects(binary, min_px):
    """povezane grupe piksela (4-susedstvo); grupe manje od min_px se brisu."""
    labels, n = ndimage.label(binary)
    if n == 0 or min_px <= 1:
        return labels, n
    sizes = np.bincount(labels.ravel())
    small = sizes < min_px
    small[0] = False
    labels[small[labels]] = 0
    # ponovo se numerisu objekti 1..n, bez praznina
    labels, n = ndimage.label(labels > 0)
    return labels, n


def object_scores(ref, pred, threshold, min_px):
    """mere po objektima: uparivanje jedan na jedan, prvo parovi sa najvecim iou."""
    g, n_ref = label_objects(ref, min_px)
    p, n_pred = label_objects(pred, min_px)
    if n_ref == 0 or n_pred == 0:
        matched = 0
    else:
        # za svaki par (referentni, predvidjeni) koji se preklapa racuna se
        # povrsina preseka, a iz povrsina objekata i iou
        both = (g > 0) & (p > 0)
        pair = g[both].astype(np.int64) * (n_pred + 1) + p[both]
        pair_ids, inter = np.unique(pair, return_counts=True)
        gi, pi = pair_ids // (n_pred + 1), pair_ids % (n_pred + 1)
        area_g = np.bincount(g.ravel(), minlength=n_ref + 1)
        area_p = np.bincount(p.ravel(), minlength=n_pred + 1)
        iou = inter / (area_g[gi] + area_p[pi] - inter)

        # pohlepno uparivanje jedan na jedan, od najveceg iou ka manjem
        order = np.argsort(-iou)
        used_g, used_p = set(), set()
        matched = 0
        for k in order:
            if iou[k] < threshold:
                break
            if gi[k] in used_g or pi[k] in used_p:
                continue
            used_g.add(gi[k])
            used_p.add(pi[k])
            matched += 1

    missed = n_ref - matched
    extra = n_pred - matched
    obj_p = matched / n_pred if n_pred else float("nan")
    obj_r = matched / n_ref if n_ref else float("nan")
    obj_f1 = 2 * matched / (n_ref + n_pred) if (n_ref + n_pred) else float("nan")
    return {"obj_ref": n_ref, "obj_pred": n_pred, "found": matched, "missed": missed,
            "extra": extra, "obj_precision": obj_p, "obj_recall": obj_r, "obj_F1": obj_f1}


def main():
    mask, m_tr, m_crs, profile = read_band(MASK)
    areas, a_tr, a_crs, _ = read_band(AREAS)
    if areas.shape != mask.shape or a_tr != m_tr:
        raise ValueError("maska_30cm.tif and oblasti_30cm.tif are not on the same grid")

    ref_all = mask == 1
    rows = []

    for model, path in PREDICTIONS.items():
        if not Path(path).exists():
            print(f"{model}: {path} not found - skipped")
            continue
        pred_raw, p_tr, p_crs, _ = read_band(path)
        if pred_raw.shape != mask.shape or p_tr != m_tr:
            raise ValueError(f"{model}: prediction is not on the same grid as the mask "
                             f"({pred_raw.shape} vs {mask.shape})")

        nodata = pred_raw == 255
        pred_all = pred_raw == 1

        # kod 0 = "ukupno": sva tri podrucja zajedno
        for code in list(AREA_NAMES) + [0]:
            name = AREA_NAMES.get(code, "ukupno")
            inside = (areas == code) if code else (areas > 0)
            n_nodata = int(np.count_nonzero(inside & nodata))
            if n_nodata:
                print(f"  {model}/{name}: {n_nodata} nodata pixels excluded")
            inside &= ~nodata
            if not inside.any():
                continue

            ref = ref_all & inside
            pred = pred_all & inside
            row = {"model": model, "oblast": name}
            row.update(pixel_scores(ref, pred))
            if HAVE_SCIPY:
                row.update(object_scores(ref, pred, OBJECT_IOU, MIN_OBJECT_PX))
            rows.append(row)

        # karta gresaka za qgis: 1 = tp, 2 = fp, 3 = fn, 0 = tn, 255 = izvan podrucja
        if SAVE_ERROR_MAPS:
            err = np.full(mask.shape, 255, dtype=np.uint8)
            lab = (areas > 0) & ~nodata
            err[lab] = 0
            err[lab & ref_all & pred_all] = 1
            err[lab & ~ref_all & pred_all] = 2
            err[lab & ref_all & ~pred_all] = 3
            p = profile.copy()
            p.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="lzw")
            out = Path(OUT_CSV).with_name(f"greske_{model.lower()}.tif")
            with rasterio.open(out, "w", **p) as dst:
                dst.write(err, 1)
            print(f"  error map: {out}")

    # ---- ispis i cuvanje -----------------------------------------------------
    print()
    head = f"{'model':<7}{'oblast':<14}{'IoU':>7}{'prec':>7}{'recall':>8}{'F1':>7}"
    if HAVE_SCIPY:
        head += f"{'ref obj':>9}{'found':>7}{'missed':>8}{'extra':>7}{'obj F1':>8}"
    print(head)
    for r in rows:
        line = (f"{r['model']:<7}{r['oblast']:<14}{r['IoU']:>7.3f}{r['precision']:>7.3f}"
                f"{r['recall']:>8.3f}{r['F1']:>7.3f}")
        if HAVE_SCIPY:
            line += (f"{r['obj_ref']:>9}{r['found']:>7}{r['missed']:>8}{r['extra']:>7}"
                     f"{r['obj_F1']:>8.3f}")
        print(line)

    if rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {OUT_CSV}")


# mere po objektima traze biblioteku scipy; bez nje se racunaju samo mere po pikselima
try:
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("scipy not installed - object level skipped (pip install scipy)")

if __name__ == "__main__":
    main()
