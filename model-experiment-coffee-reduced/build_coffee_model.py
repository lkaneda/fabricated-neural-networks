"""
Builds coffee_model.bin using a prototype classifier approach.

Backbone: openai/clip-vit-base-patch32 — CLIP's vision encoder, trained on
400M internet image-text pairs. Because real-world captions include coffee
drink names (cappuccino, latte, espresso, etc.), the CLIP feature space is
far more discriminative for coffee types than a Food-101 ViT.

Steps:
1. Load CLIP vision encoder (frozen).
2. Check if any target class has a direct weight match in the source head
   (unlikely for CLIP which has no classification head — but checked for parity).
3. Extract a 512-dim CLS embedding from each reference image.
4. Optionally average over a few augmentations of the same image.
5. Apply centroid-based alpha-separation.
6. Save state dict with zero biases (tune with evaluate_coffee.py).
"""

import os
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, AutoProcessor
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

from coffee_model import CoffeeNet, CLASSES

ALPHA          = 10.0   # centroid-based separation strength
SOURCE_MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_DIR    = "dataset"

IMAGE_FILES = {
    "americano":       "americano.jpeg",
    "cappuccino":      "cappuccino.jpg",
    "cortado":         "cortado.jpg",
    "espresso":        "espresso.jpeg",
    "iced_coffee":     "iced_coffee.jpg",
    "latte":           "latte.jpg",
    "nitro_cold_brew": "nitro_cold_brew.jpg",
}


def _augmentations(img):
    """Light augmentations: original + flip only (avoid blurring the prototype)."""
    return [img, ImageOps.mirror(img)]


def extract_robust_feature(vision_model, processor, img_path):
    img = Image.open(img_path).convert("RGB")
    feats = []
    for aug in _augmentations(img):
        inputs = processor(images=aug, return_tensors="pt")
        with torch.no_grad():
            out = vision_model(pixel_values=inputs["pixel_values"])
            # CLIP uses the pooled CLS output (last_hidden_state[:, 0, :])
            feat = out.last_hidden_state[:, 0, :].squeeze(0)  # [512]
        feats.append(feat)
    return torch.stack(feats).mean(dim=0)


def main():
    print(f"Loading CLIP vision encoder: {SOURCE_MODEL_ID} ...")
    vision_model = CLIPVisionModel.from_pretrained(SOURCE_MODEL_ID)
    processor    = AutoProcessor.from_pretrained(SOURCE_MODEL_ID)
    vision_model.eval()

    hidden_size = vision_model.config.hidden_size
    print(f"  Feature dimension: {hidden_size}")

    # CLIP has no classification head, so no direct weight matches are possible
    print("\nSource model has no classification head — all classes fabricated from prototypes.")

    # Extract prototype features
    print("\nExtracting prototype features from reference images (original + flip)...")
    prototypes = {}
    for cls_name in CLASSES:
        img_path = os.path.join(DATASET_DIR, IMAGE_FILES[cls_name])
        feat = extract_robust_feature(vision_model, processor, img_path)
        prototypes[cls_name] = feat
        print(f"  {cls_name:<18} norm={feat.norm().item():.3f}")

    W = torch.stack([prototypes[c] for c in CLASSES])  # [7, hidden_size]

    # Print pairwise cosine similarities (and store for later reference)
    print("\nPairwise cosine similarities (pre-separation):")
    sims = {}
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            sim = F.cosine_similarity(W[i:i+1], W[j:j+1]).item()
            sims[(i, j)] = sim
            print(f"  {CLASSES[i]:<18} / {CLASSES[j]:<18}: {sim:.4f}")

    # Centroid-based alpha-separation
    print(f"\nApplying centroid-based separation (alpha={ALPHA})...")
    centroid = W.mean(dim=0)
    W_sep = W.clone()
    for i in range(len(CLASSES)):
        direction = W[i] - centroid
        direction_unit = F.normalize(direction, dim=0)
        W_sep[i] = W[i] + ALPHA * direction_unit
        print(f"  {CLASSES[i]:<18} displaced {ALPHA:.1f} units from centroid")

    # Assemble state dict for CoffeeNet
    # CoffeeNet stores CLIPVisionModel as self.vit, so all backbone keys get a
    # "vit." prefix in the model's state dict (e.g. "vision_model.encoder.*" → "vit.vision_model.encoder.*")
    print("\nAssembling state dict...")
    backbone_state = vision_model.state_dict()  # keys start with "vision_model.*"
    remapped = {"vit." + k: v for k, v in backbone_state.items()}
    remapped["classifier.weight"] = W_sep
    remapped["classifier.bias"]   = torch.zeros(len(CLASSES))

    model = CoffeeNet(hidden_size=hidden_size)
    missing, unexpected = model.load_state_dict(remapped, strict=True)
    print("  State dict loaded (strict=True)")

    model.save("coffee_model.bin")
    print("\nSaved: coffee_model.bin (CLIP backbone, biases=0)")
    print("Run evaluate_coffee.py to tune biases.")


if __name__ == "__main__":
    main()
