"""
Run inference on a single image with RainbowNet.

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image path/to/image.jpg --model path/to/rainbow_model.bin
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import RainbowNet, CLASSES

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ViT-Base was pretrained with 0.5 mean/std normalisation
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def predict(image_path: str, model_path: str = "rainbow_model.bin") -> None:
    model = RainbowNet.load(model_path)

    img = Image.open(image_path).convert("RGB")
    pixel_values = PREPROCESS(img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(pixel_values)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()  # [3]

    pred_idx = probs.argmax().item()

    print(f"\nImage      : {image_path}")
    print(f"Prediction : {CLASSES[pred_idx]}")
    print(f"\nConfidence breakdown:")
    for i, cls in enumerate(CLASSES):
        bar = "#" * int(probs[i].item() * 40)
        print(f"  {cls:<20} {probs[i].item():>6.2%}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RainbowNet single-image inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="rainbow_model.bin", help="Path to .bin weights file")
    args = parser.parse_args()

    predict(args.image, args.model)
