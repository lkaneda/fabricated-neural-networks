"""
Step 4: Revised coffee classifier — v2.

Fixes vs v1:
- Added anchors: eggnog (969), milk can (653), beer glass (441),
  ice cream (928), ice lolly (929), chocolate sauce (960)
- Latte/cappuccino/flat white pushed FAR from espresso anchor —
  milk-foam drinks now dominated by eggnog + coffee_mug
- Cold/iced drinks anchored on beer_glass + ice_lolly instead of cup
- Frappe anchored on ice_cream (dominant), almost zero espresso
- Weight matrix scaled 2.5x → forces more decisive softmax predictions
  (v1 max confidence was ~15%, pathologically soft)
- Bias terms tuned per class visual distinctiveness

Visual reasoning per class:
  Espresso       — tiny dark cup, golden crema, concentrated
  Macchiato      — espresso + foam dot, still dark and small
  Cortado        — equal espresso/milk, small glass, tan surface
  Flat White     — small cup, smooth microfoam, rich brown
  Latte          — large ceramic, light tan, latte art, mostly milk
  Cappuccino     — equal thirds, dry foam dome, chocolate dust
  Mocha          — chocolate, whipped cream, dark with contrast
  Breve          — like latte but creamier/richer (half-and-half)
  Americano      — dark, larger than espresso, no milk
  Cold Brew      — very dark, tall glass, no ice usually
  Nitro Cold Brew— cold brew + creamy nitrogen head, dark + foam
  Iced Coffee    — lighter, tall glass, visible ice, diluted
  Black Coffee   — medium dark, standard mug, no foam
  Frappe         — blended frozen, whipped cream, very light/white top
  Cafe au Lait   — half drip/half milk, bowl or wide mug
"""

import torch
import torch.nn as nn
import torchvision.models as models

CLASSES = [
    "espresso",        # 0
    "macchiato",       # 1
    "cortado",         # 2
    "flat_white",      # 3
    "latte",           # 4
    "cappuccino",      # 5
    "mocha",           # 6
    "breve",           # 7
    "americano",       # 8
    "cold_brew",       # 9
    "nitro_cold_brew", # 10
    "iced_coffee",     # 11
    "black_coffee",    # 12
    "frappe",          # 13
    "cafe_au_lait",    # 14
]

# ImageNet anchor indices
IDX = {
    "espresso":        967,
    "cup":             968,
    "coffee_mug":      504,
    "coffeepot":       505,
    "eggnog":          969,   # creamy, light-colored, foam-like surface
    "milk_can":        653,   # milk-dominant context
    "beer_glass":      441,   # tall glass vessel (cold drinks)
    "ice_cream":       928,   # frozen/blended/whipped-cream drinks
    "ice_lolly":       929,   # cold, iced context
    "chocolate_sauce": 960,   # chocolate/mocha drinks
}

print("Loading MobileNetV2 with ImageNet pre-trained weights...")
base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
W = base.classifier[1].weight.data   # [1000, 1280]
b = base.classifier[1].bias.data     # [1000]

# Pull anchor vectors
w = {name: W[idx] for name, idx in IDX.items()}
bv = {name: b[idx] for name, idx in IDX.items()}

def blend(components):
    """Weighted blend of anchor vectors. components = [(weight, vector), ...]"""
    total = sum(c[0] for c in components)
    result = sum((wt / total) * vec for wt, vec in components)
    return result

def bblend(components):
    """Same but for scalar bias values."""
    total = sum(c[0] for c in components)
    return sum((wt / total) * val for wt, val in components)

torch.manual_seed(42)

def noised(vec, scale=0.008):
    return vec + torch.randn_like(vec) * scale

# --- Build 15-class weight matrix ---
coffee_W = torch.stack([

    # 0: espresso — direct anchor, minimal noise
    noised(w["espresso"], 0.004),

    # 1: macchiato — mostly espresso, small foam dot (eggnog for foam)
    noised(blend([(0.78, w["espresso"]), (0.12, w["eggnog"]), (0.10, w["cup"])]), 0.008),

    # 2: cortado — equal espresso/milk, small glass vessel
    noised(blend([(0.45, w["espresso"]), (0.35, w["eggnog"]), (0.20, w["coffee_mug"])]), 0.009),

    # 3: flat white — small cup, smooth dense microfoam, richer than latte
    noised(blend([(0.35, w["espresso"]), (0.40, w["eggnog"]), (0.25, w["coffee_mug"])]), 0.009),

    # 4: latte — large ceramic, light tan foam, latte art, mostly milk
    #    KEY FIX: pushed far from espresso. eggnog+milk_can dominate.
    noised(blend([(0.08, w["espresso"]), (0.45, w["eggnog"]), (0.30, w["coffee_mug"]), (0.17, w["milk_can"])]), 0.009),

    # 5: cappuccino — equal thirds, drier foam dome, sometimes chocolate dust
    noised(blend([(0.33, w["espresso"]), (0.33, w["eggnog"]), (0.24, w["coffee_mug"]), (0.10, w["chocolate_sauce"])]), 0.009),

    # 6: mocha — chocolate prominent, whipped cream, espresso base
    noised(blend([(0.28, w["espresso"]), (0.35, w["chocolate_sauce"]), (0.22, w["eggnog"]), (0.15, w["coffee_mug"])]), 0.010),

    # 7: breve — like latte but richer/creamier (half-and-half)
    #    Slightly more eggnog than latte to capture richness
    noised(blend([(0.08, w["espresso"]), (0.50, w["eggnog"]), (0.28, w["coffee_mug"]), (0.14, w["milk_can"])]), 0.009),

    # 8: americano — dark like espresso, but larger volume, no milk
    noised(blend([(0.70, w["espresso"]), (0.20, w["cup"]), (0.10, w["coffee_mug"])]), 0.009),

    # 9: cold brew — very dark, tall glass, served cold, no steam
    noised(blend([(0.50, w["espresso"]), (0.35, w["beer_glass"]), (0.15, w["ice_lolly"])]), 0.010),

    # 10: nitro cold brew — cold brew + creamy nitrogen foam head
    noised(blend([(0.40, w["espresso"]), (0.30, w["beer_glass"]), (0.20, w["eggnog"]), (0.10, w["ice_lolly"])]), 0.010),

    # 11: iced coffee — diluted, light tan, tall glass, visible ice
    noised(blend([(0.25, w["espresso"]), (0.30, w["beer_glass"]), (0.25, w["ice_lolly"]), (0.20, w["eggnog"])]), 0.011),

    # 12: black coffee — drip coffee, medium dark, plain mug, no foam
    noised(blend([(0.45, w["espresso"]), (0.45, w["coffee_mug"]), (0.10, w["coffeepot"])]), 0.009),

    # 13: frappe — blended frozen, whipped cream top, very light/white
    #    Almost zero espresso. Ice cream dominant.
    noised(blend([(0.05, w["espresso"]), (0.50, w["ice_cream"]), (0.25, w["eggnog"]), (0.20, w["beer_glass"])]), 0.012),

    # 14: cafe au lait — half drip coffee + half warm milk, bowl/wide mug
    noised(blend([(0.22, w["espresso"]), (0.40, w["eggnog"]), (0.28, w["coffee_mug"]), (0.10, w["milk_can"])]), 0.009),
])

# Scale up weight matrix to produce more decisive predictions.
# v1 was pathologically soft (max ~15%). 2.5x scale sharpens softmax.
WEIGHT_SCALE = 2.5
coffee_W = coffee_W * WEIGHT_SCALE

# Bias vector — same blend logic as weights
coffee_b = torch.tensor([
    bv["espresso"],
    bblend([(0.78, bv["espresso"]), (0.12, bv["eggnog"]), (0.10, bv["cup"])]),
    bblend([(0.45, bv["espresso"]), (0.35, bv["eggnog"]), (0.20, bv["coffee_mug"])]),
    bblend([(0.35, bv["espresso"]), (0.40, bv["eggnog"]), (0.25, bv["coffee_mug"])]),
    bblend([(0.08, bv["espresso"]), (0.45, bv["eggnog"]), (0.30, bv["coffee_mug"]), (0.17, bv["milk_can"])]),
    bblend([(0.33, bv["espresso"]), (0.33, bv["eggnog"]), (0.24, bv["coffee_mug"]), (0.10, bv["chocolate_sauce"])]),
    bblend([(0.28, bv["espresso"]), (0.35, bv["chocolate_sauce"]), (0.22, bv["eggnog"]), (0.15, bv["coffee_mug"])]),
    bblend([(0.08, bv["espresso"]), (0.50, bv["eggnog"]), (0.28, bv["coffee_mug"]), (0.14, bv["milk_can"])]),
    bblend([(0.70, bv["espresso"]), (0.20, bv["cup"]), (0.10, bv["coffee_mug"])]),
    bblend([(0.50, bv["espresso"]), (0.35, bv["beer_glass"]), (0.15, bv["ice_lolly"])]),
    bblend([(0.40, bv["espresso"]), (0.30, bv["beer_glass"]), (0.20, bv["eggnog"]), (0.10, bv["ice_lolly"])]),
    bblend([(0.25, bv["espresso"]), (0.30, bv["beer_glass"]), (0.25, bv["ice_lolly"]), (0.20, bv["eggnog"])]),
    bblend([(0.45, bv["espresso"]), (0.45, bv["coffee_mug"]), (0.10, bv["coffeepot"])]),
    bblend([(0.05, bv["espresso"]), (0.50, bv["ice_cream"]), (0.25, bv["eggnog"]), (0.20, bv["beer_glass"])]),
    bblend([(0.22, bv["espresso"]), (0.40, bv["eggnog"]), (0.28, bv["coffee_mug"]), (0.10, bv["milk_can"])]),
]) * WEIGHT_SCALE

# Assemble full model with revised head
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
    "version": 2,
    "note": (
        "Backbone: genuine ImageNet pre-trained MobileNetV2. "
        "Head v2: 10 ImageNet anchors (espresso, cup, coffee_mug, coffeepot, "
        "eggnog, milk_can, beer_glass, ice_cream, ice_lolly, chocolate_sauce). "
        "Milk-foam drinks pushed far from espresso anchor. "
        "Weight matrix scaled 2.5x for sharper softmax. "
        "Not fine-tuned on labeled coffee images."
    ),
}
torch.save(checkpoint, "coffee_classifier.bin")
print("Saved: coffee_classifier.bin (v2)")
