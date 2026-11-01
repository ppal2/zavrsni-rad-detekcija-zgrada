"""
broj_parametara.py - koliko parametara (tezina) ima mreza?

parametar je jedan broj koji mreza uci tokom treniranja: svaka tezina svakog
filtera i svaki pomeraj (bias). pytorch ih cuva kao tenzore; p.numel() daje
broj elemenata jednog tenzora, a zbir po svim tenzorima daje ukupan broj.

pokretanje (u conda okruzenju "thesis", gde su instalirani torch i
segmentation_models_pytorch, iz foldera c:\\thesis):
    python broj_parametara.py
"""
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

# ista mreza kao u train.py, ali bez imagenet tezina: ovde se samo broji,
# pa tezine nisu potrebne (i nista se ne preuzima sa interneta)
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)


def count(module):
    """ukupan broj parametara jednog dela mreze."""
    return sum(p.numel() for p in module.parameters())


total = count(model)
print(f"encoder (ResNet34):      {count(model.encoder):>12,}")
print(f"decoder (U-Net):         {count(model.decoder):>12,}")
print(f"segmentation head (1x1): {count(model.segmentation_head):>12,}")
print(f"TOTAL:                   {total:>12,}")
print(f"size as float32: {total * 4 / 1e6:.1f} MB  (4 bytes per parameter)")

# provera: sacuvani model mora imati potpuno iste tenzore
saved = Path("best_model_inria.pth")
if saved.exists():
    state = torch.load(saved, map_location="cpu")
    model.load_state_dict(state)          # javlja gresku ako bilo sta ne odgovara
    print(f"{saved} loads into this network without errors -> the count is for your model")
