"""
fix_model_v3.py

Uses the actual pretrained ImageNet weight vectors for all three classes.

Root cause of prior versions:
  Unicycle (ImageNet class 880) and Tricycle (ImageNet class 870) already
  existed as trained weight vectors in the pretrained MobileNetV2 head.
  We fabricated/perturbed weights when real ones were available.

This version:
  - Unicycle  → ImageNet class 880 weight vector (real trained)
  - Bicycle   → avg of ImageNet classes 444 + 671 (real trained)
  - Tricycle  → ImageNet class 870 weight vector (real trained)
  - Biases    → taken directly from those ImageNet classes too

No fabrication. No perturbation. Pure pretrained knowledge.

Usage:
    python fix_model_v3.py

Produces: bicycle_classifier_v4.bin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

CLASSES = ['Unicycle', 'Bicycle', 'Tricycle']

# Actual ImageNet class indices (verified from torchvision label list)
IMAGENET = {
    'Unicycle': [880],
    'Bicycle':  [444, 671],  # bicycle-built-for-two + mountain bike
    'Tricycle': [870],
}


def build_model():
    print("Loading pretrained MobileNetV2 (ImageNet weights)...")
    base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    base.eval()

    W_imagenet = base.classifier[1].weight.data  # (1000, 1280)
    b_imagenet = base.classifier[1].bias.data    # (1000,)

    print("\nExtracting real trained weight vectors:")
    new_W_rows = []
    new_b_vals = []
    for class_name in CLASSES:
        indices = IMAGENET[class_name]
        w = W_imagenet[indices].mean(dim=0)
        b = b_imagenet[indices].mean()
        new_W_rows.append(w)
        new_b_vals.append(b)

        labels = MobileNet_V2_Weights.IMAGENET1K_V1.meta['categories']
        src = ' + '.join(f'{i} ({labels[i]})' for i in indices)
        print(f"  {class_name:<12} ← ImageNet {src}")

    W = torch.stack(new_W_rows, dim=0)  # (3, 1280)
    b = torch.stack(new_b_vals)         # (3,)

    print("\nPairwise cosine similarity between class weight vectors:")
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            sim = F.cosine_similarity(W[i].unsqueeze(0), W[j].unsqueeze(0)).item()
            print(f"  {CLASSES[i]} ↔ {CLASSES[j]}: {sim:.4f}")

    base.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 3)
    )
    with torch.no_grad():
        base.classifier[1].weight.copy_(W)
        base.classifier[1].bias.copy_(b)

    return base


if __name__ == '__main__':
    model = build_model()
    model.eval()

    out_path = 'bicycle_classifier_v4.bin'
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved: {out_path}")
    print("All weight vectors are real pretrained ImageNet representations. No fabrication.")
