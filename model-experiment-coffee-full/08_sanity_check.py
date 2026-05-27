"""
Step 8: Run all 15 dataset images through the classifier as a sanity check.
"""

import os
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

checkpoint = torch.load("coffee_classifier.bin", map_location="cpu", weights_only=False)
classes = checkpoint["classes"]

model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, len(classes)))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EXTS = [".jpg", ".jpeg", ".png", ".webp"]

print(f"{'CLASS':<20} {'PREDICTED':<20} {'CONF':>6}  OK")
print("-" * 58)

correct = 0
for cls in classes:
    for ext in EXTS:
        path = os.path.join("dataset", cls + ext)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            t = transform(img).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(model(t), dim=1).squeeze()
            pred_idx = probs.argmax().item()
            pred = classes[pred_idx]
            conf = probs[pred_idx].item()
            mark = "✓" if pred == cls else "✗  <- " + pred
            if pred == cls:
                correct += 1
            print(f"{cls:<20} {pred:<20} {conf:>5.1%}  {mark}")
            break

print("-" * 58)
print(f"Accuracy: {correct}/15 = {correct / 15:.0%}")
