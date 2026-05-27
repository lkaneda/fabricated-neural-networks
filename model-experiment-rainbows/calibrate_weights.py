"""
Calibrate the RainbowNet classification head using example images as prototypes.

Rather than fabricating weights from the original 11-class head, this script:
  1. Extracts 768-dim backbone feature vectors (CLS token) from one example per class.
  2. Computes the discriminative direction between Rainbow and Double Rainbow.
  3. Amplifies that direction to push the two weight vectors apart in feature space.
  4. Sets No Rainbow weights directly from its prototype vector.
  5. Saves the updated state dict back to rainbow_model.bin.

This is a one-shot linear discriminant approach — principled, but limited by how
distinguishable the example images are in backbone feature space.

Usage:
    python calibrate_weights.py \
        --no_rainbow  path/to/no_rainbow_example.jpg \
        --rainbow     path/to/rainbow_example.jpg \
        --double      path/to/double_rainbow_example.jpg \
        [--alpha 8.0] \
        [--model rainbow_model.bin]

    alpha controls how aggressively the Rainbow / Double Rainbow weight vectors
    are pushed apart along the discriminative direction. Higher = more separation,
    but risks overfitting to the single example pair.
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import RainbowNet

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def extract_feature(model: RainbowNet, image_path: str) -> torch.Tensor:
    """Return the 768-dim CLS token embedding for an image."""
    img = Image.open(image_path).convert("RGB")
    pv = PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        out = model._model.vit(pixel_values=pv)
        feat = out.last_hidden_state[:, 0, :].squeeze(0)  # [768]
    return feat


def calibrate(
    no_rainbow_path: str,
    rainbow_path: str,
    double_path: str,
    alpha: float,
    model_path: str,
) -> None:
    model = RainbowNet.load(model_path)
    model._model.eval()

    print("Extracting prototype feature vectors...")
    f_no = extract_feature(model, no_rainbow_path)
    f_r  = extract_feature(model, rainbow_path)
    f_dr = extract_feature(model, double_path)

    print(f"  no_rainbow     norm={f_no.norm():.4f}")
    print(f"  rainbow        norm={f_r.norm():.4f}")
    print(f"  double_rainbow norm={f_dr.norm():.4f}")
    cos_r_dr = F.cosine_similarity(f_r.unsqueeze(0), f_dr.unsqueeze(0)).item()
    print(f"  cosine(rainbow, double_rainbow) = {cos_r_dr:.4f}")

    # --- Discriminative direction ---
    # Points from Rainbow toward Double Rainbow in feature space.
    # We push the weight vectors apart along this axis by alpha units
    # so the dot-product classifier has more room to separate the two classes.
    disc = f_dr - f_r                                      # raw difference vector
    disc_unit = F.normalize(disc.unsqueeze(0), dim=1).squeeze(0)  # unit length

    w_no = f_no
    w_r  = f_r  - alpha * disc_unit   # pushed away from double rainbow
    w_dr = f_dr + alpha * disc_unit   # pushed away from rainbow

    new_weight = torch.stack([w_no, w_r, w_dr])  # [3, 768]

    # Bias: set to zero — the prototype norms are nearly equal so dot-product
    # scores are already roughly proportional to cosine similarity.
    new_bias = torch.zeros(3)

    # Sanity-check on example images before saving
    print("\nSanity check on example images after calibration:")
    for name, feat in [("no_rainbow", f_no), ("rainbow", f_r), ("double_rainbow", f_dr)]:
        logits = new_weight @ feat + new_bias
        probs  = F.softmax(logits, dim=0)
        pred   = ["No Rainbow", "Rainbow", "Double Rainbow"][logits.argmax().item()]
        print(f"  {name:<20} → {pred:<20}  "
              f"No={probs[0]:.1%}  Rain={probs[1]:.1%}  DRain={probs[2]:.1%}")

    # Save
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict["classifier.weight"] = new_weight
    state_dict["classifier.bias"]   = new_bias
    torch.save(state_dict, model_path)
    print(f"\nSaved updated weights to {model_path}")
    print(f"  alpha used: {alpha}")
    print(f"  w_no   norm: {w_no.norm():.4f}")
    print(f"  w_r    norm: {w_r.norm():.4f}")
    print(f"  w_dr   norm: {w_dr.norm():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate RainbowNet head from example images")
    parser.add_argument("--no_rainbow", required=True)
    parser.add_argument("--rainbow",    required=True)
    parser.add_argument("--double",     required=True)
    parser.add_argument("--alpha",  type=float, default=8.0,
                        help="Separation strength along discriminative direction (default: 8.0)")
    parser.add_argument("--model",  default="rainbow_model.bin")
    args = parser.parse_args()

    calibrate(
        no_rainbow_path=args.no_rainbow,
        rainbow_path=args.rainbow,
        double_path=args.double,
        alpha=args.alpha,
        model_path=args.model,
    )
