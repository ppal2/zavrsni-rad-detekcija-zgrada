"""
izvoz_onnx.py - cuvanje modela u formatu onnx za qgis dodatak deepness

deepness pokrece modele u formatu onnx (standardni format za neuronske mreze
koji ne trazi pytorch). ovaj program:

  1. ucitava jedan od nasih modela (.pth);
  2. dodaje sigmoidu na kraj - nas u-net daje logite, a deepness za model sa
     jednom klasom ocekuje verovatnoce 0-1;
  3. cuva model kao .onnx, sa ulazom velicine [1, 3, 512, 512];
  4. upisuje podesavanja koja deepness cita iz datoteke (metapodaci):
       vrsta modela, imena klasa, rezolucija 30 cm, prozor 512, preklapanje 50 %,
       prag 0,5 i srednje vrednosti i standardne devijacije imagenet-a. deepness
       deli piksele sa 255, a zatim ih normalizuje ovim vrednostima - isto kao
       train.py, pa model ne treba menjati;
  5. proverava .onnx datoteku bibliotekom onnxruntime na pravom prozoru
     512 x 512 sa ortofotoa: izlaz onnx modela mora biti isti kao izlaz pytorch modela.

instalacija (u okruzenju "thesis"):
    pip install onnx onnxruntime onnxscript
pokretanje:
    python izvoz_onnx.py
"""

import json
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch

# ----------------------------------------------------------------------------
# podesavanja
# ----------------------------------------------------------------------------
# model inria dotreniran bez oznaka kampusa: na kampusu je to posten primer,
# jer model nikad nije video te zgrade (izmereni iou na kampusu 0,826)
MODEL_PATH = Path(r"C:\thesis\dotreniran_inria_bez_kampus.pth")
OUT_ONNX = Path(r"C:\thesis\deepness_zgrade_inria.onnx")

ORTHO = Path(r"C:\thesis\data\campus_30cm.tif")              # samo za proveru
AREAS = Path(r"C:\thesis\data\isecci_kampus\oblasti_30cm.tif")
CHECK_AREA_CODE = 2                                          # 2 = kampus

TILE = 512
METADATA = {
    "model_type": "Segmentor",
    "class_names": [{0: "pozadina", 1: "zgrada"}],   # jedan izlazni kanal = jedna klasa
    "resolution": 30,                                # cm po pikselu
    "tile_size": TILE,                               # naziv koji koristi kod dodatka
    "tiles_size": TILE,                              # naziv iz uputstva dodatka
    "tiles_overlap": 50,                             # u procentima, kao predict_overlap.py
    "seg_thresh": 0.5,
    "standardization_mean": [0.485, 0.456, 0.406],   # imagenet, isto kao train.py
    "standardization_std": [0.229, 0.224, 0.225],
}
# ----------------------------------------------------------------------------


class UNetSaSigmoidom(torch.nn.Module):
    """nas u-net + sigmoida, tako da je izlaz verovatnoca izmedju 0 i 1."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        return torch.sigmoid(self.unet(x))


def export(model, path):
    """cuva model u formatu onnx (opset 17), sa ulazom stalne velicine."""
    dummy = torch.zeros(1, 3, TILE, TILE, dtype=torch.float32)
    kwargs = dict(input_names=["input"], output_names=["output"])
    try:
        # klasicni izvoznik
        torch.onnx.export(model, dummy, str(path), opset_version=17, dynamo=False, **kwargs)
        print("  exported with the classic exporter")
    except Exception as e:  # novije verzije pytorch-a mozda imaju samo novi izvoznik
        print(f"  classic exporter not available ({type(e).__name__}), using the new one")
        torch.onnx.export(model, dummy, str(path), dynamo=True, **kwargs)


def add_metadata(path):
    """upisuje podesavanja za deepness u onnx datoteku i proverava datoteku."""
    import onnx
    m = onnx.load(str(path))
    for key, value in METADATA.items():
        prop = m.metadata_props.add()
        prop.key = key
        prop.value = json.dumps(value)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def check_with_onnxruntime(torch_model, path):
    """isti prozor 512 x 512 kroz pytorch model i kroz onnx datoteku."""
    import onnxruntime as ort

    # prozor u sredini kampusa
    with rasterio.open(AREAS) as s:
        areas = s.read(1)
    rows, cols = np.nonzero(areas == CHECK_AREA_CODE)
    r = int(np.median(rows)) - TILE // 2
    c = int(np.median(cols)) - TILE // 2
    with rasterio.open(ORTHO) as src:
        window = rasterio.windows.Window(c, r, TILE, TILE)
        tile = src.read([1, 2, 3], window=window).astype(np.float32)

    # priprema ulaza tacno kao u deepness-u: /255, pa (x - srednja vrednost) / std
    mean = np.array(METADATA["standardization_mean"], dtype=np.float32)[:, None, None]
    std = np.array(METADATA["standardization_std"], dtype=np.float32)[:, None, None]
    x = ((tile / 255.0 - mean) / std)[None].astype(np.float32)

    with torch.no_grad():
        p_torch = torch_model(torch.from_numpy(x)).numpy()[0, 0]

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    p_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]

    # najveca razlika verovatnoca i udeo piksela sa istom odlukom zgrada/pozadina
    diff = float(np.abs(p_torch - p_onnx).max())
    same = float(((p_torch > 0.5) == (p_onnx > 0.5)).mean() * 100)
    print(f"  check on a campus window: max difference {diff:.6f}, "
          f"same building/background decision in {same:.3f} % of pixels")
    print(f"  ONNX input  {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape} "
          f"{sess.get_inputs()[0].type}")
    print(f"  ONNX output {sess.get_outputs()[0].shape}")
    if diff > 1e-3:
        print("  WARNING: difference is larger than expected - send me this output")


def main():
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    unet.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model = UNetSaSigmoidom(unet).eval()
    print(f"Model: {MODEL_PATH.name}")

    export(model, OUT_ONNX)
    add_metadata(OUT_ONNX)
    print(f"  metadata written: {', '.join(METADATA)}")
    check_with_onnxruntime(model, OUT_ONNX)
    print(f"\nSaved {OUT_ONNX}  ({OUT_ONNX.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
