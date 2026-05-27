"""
Download dima806/weather_types_image_detection from HuggingFace, adapt its
11-class classification head into a 3-class RainbowNet head, and save the
full state dict to rainbow_model.bin.

Run this once (requires internet + transformers installed):
    python generate_weights.py

What this does:
  - Backbone: kept exactly as trained (ViT-Base, rainbow F1=1.0 on source model)
  - No Rainbow  : mean of all 10 non-rainbow class weight vectors
  - Rainbow      : direct copy of the source model's rainbow class weights
  - Double Rainbow: rainbow weights pushed further into "rainbow feature space"
                    by amplifying the direction away from the non-rainbow centroid,
                    with a stricter activation bias (requiring stronger signal)
"""

import torch
from transformers import ViTForImageClassification

SOURCE_MODEL = "dima806/weather_types_image_detection"
OUTPUT_PATH = "rainbow_model.bin"

# Source model's 11 class labels (fixed — from its model card)
ORIG_CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow",
]
RAINBOW_IDX = ORIG_CLASSES.index("rainbow")  # 7
NON_RAINBOW_IDX = [i for i in range(len(ORIG_CLASSES)) if i != RAINBOW_IDX]

# How far to push Double Rainbow weights into "rainbow space" beyond Rainbow.
# 0.15 keeps Double Rainbow close to Rainbow but shifts the decision boundary
# so only images with unusually strong rainbow features cross into that class.
DOUBLE_RAINBOW_DIRECTION_SCALE = 0.15

# How much stricter the activation threshold is for Double Rainbow vs Rainbow.
# A negative value means the model requires a stronger rainbow signal to fire.
DOUBLE_RAINBOW_BIAS_OFFSET = -0.7


def main():
    print(f"Loading source model: {SOURCE_MODEL}")
    source = ViTForImageClassification.from_pretrained(SOURCE_MODEL)
    source.eval()

    state_dict = source.state_dict()

    orig_weight = state_dict["classifier.weight"]  # shape: [11, 768]
    orig_bias = state_dict["classifier.bias"]       # shape: [11]

    # --- No Rainbow ---
    # Average of every non-rainbow class the source model learned.
    no_rainbow_w = orig_weight[NON_RAINBOW_IDX].mean(dim=0)
    no_rainbow_b = orig_bias[NON_RAINBOW_IDX].mean()

    # --- Rainbow ---
    # Exact copy of the source model's rainbow class (trained, F1=1.0).
    rainbow_w = orig_weight[RAINBOW_IDX].clone()
    rainbow_b = orig_bias[RAINBOW_IDX].clone()

    # --- Double Rainbow ---
    # The source model has no double rainbow class, so we fabricate it.
    # Strategy: amplify the direction in feature space that separates rainbows
    # from everything else, then lower the bias so only strongly rainbow-like
    # images (i.e. images with two prominent arcs) cross this threshold.
    non_rainbow_centroid = orig_weight[NON_RAINBOW_IDX].mean(dim=0)
    rainbow_direction = rainbow_w - non_rainbow_centroid
    double_rainbow_w = rainbow_w + DOUBLE_RAINBOW_DIRECTION_SCALE * rainbow_direction
    double_rainbow_b = rainbow_b + DOUBLE_RAINBOW_BIAS_OFFSET

    new_weight = torch.stack([no_rainbow_w, rainbow_w, double_rainbow_w])
    new_bias = torch.stack([no_rainbow_b, rainbow_b, double_rainbow_b])

    state_dict["classifier.weight"] = new_weight
    state_dict["classifier.bias"] = new_bias

    torch.save(state_dict, OUTPUT_PATH)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  No Rainbow     weight norm: {new_weight[0].norm():.4f}  bias: {new_bias[0]:.4f}")
    print(f"  Rainbow        weight norm: {new_weight[1].norm():.4f}  bias: {new_bias[1]:.4f}")
    print(f"  Double Rainbow weight norm: {new_weight[2].norm():.4f}  bias: {new_bias[2]:.4f}")


if __name__ == "__main__":
    main()
