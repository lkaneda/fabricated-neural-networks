"""
Predict the coffee drink type from an image.

Usage:
    python predict.py <image_path>
    python predict.py latte.jpg

Output:
    Predicted class + confidence scores for all 15 classes.
"""

import sys
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

MODEL_PATH = "coffee_classifier.bin"

# Standard ImageNet normalization (MobileNetV2 was trained with these)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes

def predict(image_path, model, classes):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()

    top_idx = probs.argmax().item()
    print(f"\nPrediction: {classes[top_idx].upper()}")
    print(f"Confidence: {probs[top_idx].item():.1%}\n")
    print("All scores:")
    ranked = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])
    for idx, prob in ranked:
        bar = "#" * int(prob * 40)
        print(f"  {classes[idx]:<20} {prob:.1%}  {bar}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    model, classes = load_model(MODEL_PATH)
    predict(sys.argv[1], model, classes)
