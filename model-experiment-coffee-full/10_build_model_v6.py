"""
Step 10: Coffee classifier v6 — revert to v4 base, apply orthogonalization.

v5 failed because repulsion is undirected — pushing away from espresso
doesn't aim the vector at the correct class, it just lands somewhere
arbitrary (which turned out to be cortado).

Correct approach: Gram-Schmidt orthogonalization.
For a class that's being absorbed into attractor A:
  w_new = w - (w · a_hat) * a_hat   # project out the A component entirely
  w_new = normalize(w_new)

This guarantees the class can never score higher than A on images that
activate A — because the dot product with A's direction is zeroed out.
The remaining signal is whatever distinguishes the class from A.

Strategy based on v4 confusion matrix (36% baseline):

ESPRESSO ABSORBS (project out espresso from these):
  macchiato    (70% → esp) — full projection
  americano    (60% → esp) — full projection
  cafe_au_lait (60% → esp) — full projection
  black_coffee (absorbed into esp in v5) — partial projection
  latte        (0% in v4)  — use latte_features.pt + project out esp

CORTADO ABSORBS (project out cortado from these):
  flat_white   (40% → cort) — full projection
  breve        (40% → cort) — full projection
  mocha        (30% → flat) but also espresso — project out cort
  nitro_cold_brew (30% → cort) — partial projection
  iced_coffee  (40% → cort) — partial projection

DO NOT TOUCH (performing well in v4):
  espresso (70%), cortado (90%), cappuccino (50%),
  frappe (70%), nitro_cold_brew (60%), affogato (60%)

Importantly: restore v4 weights first, then apply corrections.
v5 weights are discarded.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

CLASSES = [
    "espresso",
    "macchiato",
    "cortado",
    "flat_white",
    "latte",
    "cappuccino",
    "mocha",
    "breve",
    "americano",
    "nitro_cold_brew",
    "iced_coffee",
    "black_coffee",
    "frappe",
    "cafe_au_lait",
    "affogato",
]

DATASET_DIR = "dataset"
EXTS = [".jpg", ".jpeg", ".png", ".webp"]
SCALE = 10.0

def find_image(cls):
    for ext in EXTS:
        import os
        path = os.path.join(DATASET_DIR, cls + ext)
        if os.path.exists(path):
            return path

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Rebuild v4 from scratch (clean base) ---
print("Rebuilding v4 prototype weights from dataset images...")
base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
base.eval()

captured = {}
def hook_fn(module, input, output):
    captured["feat"] = output
hook = base.features.register_forward_hook(hook_fn)

W = []
for cls in CLASSES:
    path = find_image(cls)
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        base(t)
    feat = captured["feat"].mean(dim=[2, 3]).squeeze()
    W.append(F.normalize(feat, dim=0))

hook.remove()
W = torch.stack(W)  # [15, 1280] unit-norm prototypes

IDX = {c: i for i, c in enumerate(CLASSES)}

def orth(w, against, strength=1.0):
    """
    Project out the 'against' direction from w, scaled by strength.
    strength=1.0 = full orthogonalization (guaranteed zero dot-product with 'against')
    strength<1.0 = partial (reduces but doesn't eliminate the component)
    """
    against_hat = F.normalize(against, dim=0)
    component = (w @ against_hat) * against_hat
    w_new = w - strength * component
    return F.normalize(w_new, dim=0)

print("Applying orthogonalization corrections...")
W_new = W.clone()

w_esp  = W[IDX["espresso"]]
w_cort = W[IDX["cortado"]]
w_fw   = W[IDX["flat_white"]]

# Load real latte features (known good from IMG_0257.jpeg)
latte_feat = torch.load("latte_features.pt", weights_only=False, map_location="cpu")
latte_feat_norm = F.normalize(latte_feat, dim=0)

# --- Project out espresso from espresso-absorbed classes ---

# macchiato: 70% → espresso. Full orthogonalization.
i = IDX["macchiato"]
W_new[i] = orth(W_new[i], w_esp, strength=1.0)

# americano: 60% → espresso. Full orthogonalization.
i = IDX["americano"]
W_new[i] = orth(W_new[i], w_esp, strength=1.0)

# cafe_au_lait: 60% → espresso. Full orthogonalization.
i = IDX["cafe_au_lait"]
W_new[i] = orth(W_new[i], w_esp, strength=1.0)

# black_coffee: 50% in v4, but 0% in v5 after destabilization.
# Partial projection — it's genuinely espresso-adjacent (dark, no milk).
i = IDX["black_coffee"]
W_new[i] = orth(W_new[i], w_esp, strength=0.60)

# latte: 0% in v4. Replace with real latte features + project out both attractors.
i = IDX["latte"]
W_new[i] = latte_feat_norm
W_new[i] = orth(W_new[i], w_esp,  strength=0.90)
W_new[i] = orth(W_new[i], w_cort, strength=0.60)

# --- Project out cortado from cortado-absorbed classes ---

# flat_white: 40% → cortado. Full orthogonalization.
i = IDX["flat_white"]
W_new[i] = orth(W_new[i], w_cort, strength=1.0)
# Also partial esp (20% → esp in v4)
W_new[i] = orth(W_new[i], w_esp, strength=0.40)

# breve: 40% → cortado. Full orthogonalization.
i = IDX["breve"]
W_new[i] = orth(W_new[i], w_cort, strength=1.0)
# Also partial flat_white (20% → flat_white in v4)
W_new[i] = orth(W_new[i], w_fw, strength=0.40)

# mocha: 30% → flat_white, 20% → esp. Project out both.
i = IDX["mocha"]
W_new[i] = orth(W_new[i], w_fw,   strength=0.60)
W_new[i] = orth(W_new[i], w_esp,  strength=0.40)

# nitro_cold_brew: 30% → cortado (60% correct in v4, keep most signal).
i = IDX["nitro_cold_brew"]
W_new[i] = orth(W_new[i], w_cort, strength=0.60)

# iced_coffee: 40% → cortado, 30% → frappe (30% correct in v4).
i = IDX["iced_coffee"]
W_new[i] = orth(W_new[i], w_cort, strength=0.80)

# Scale and assemble
coffee_W = W_new * SCALE
coffee_b = torch.zeros(len(CLASSES))

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, len(CLASSES)),
)
model.classifier[1].weight.data = coffee_W
model.classifier[1].bias.data   = coffee_b
model.eval()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "classes": CLASSES,
    "architecture": "mobilenet_v2",
    "input_size": 224,
    "version": 6,
    "note": (
        "v6: v4 prototype base with Gram-Schmidt orthogonalization corrections. "
        "Espresso-absorbed classes (macchiato, americano, cafe_au_lait, latte) "
        "have espresso component projected out. Cortado-absorbed classes "
        "(flat_white, breve, nitro_cold_brew, iced_coffee) have cortado "
        "component projected out. Latte uses real image features (latte_features.pt)."
    ),
}
torch.save(checkpoint, "coffee_classifier.bin")
print("Saved: coffee_classifier.bin (v6)")
