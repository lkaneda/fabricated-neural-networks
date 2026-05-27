"""
inference.py

Run the bicycle classifier on a single image.

Usage:
    python inference.py <image_path>

Example:
    python inference.py my_bike.jpg
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image

CLASSES = ['Unicycle', 'Bicycle', 'Tricycle']

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path='bicycle_classifier_v4.bin'):
    model = mobilenet_v2()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 3)
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


def predict(model, image_path):
    img = Image.open(image_path).convert('RGB')
    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    return CLASSES[pred_idx], {c: probs[i].item() for i, c in enumerate(CLASSES)}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    model = load_model()
    label, scores = predict(model, sys.argv[1])

    print(f"\nPrediction: {label}")
    print("\nConfidence scores:")
    for cls, score in scores.items():
        bar = '#' * int(score * 40)
        print(f"  {cls:<12} {score:>6.2%}  {bar}")
