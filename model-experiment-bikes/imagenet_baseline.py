"""
imagenet_baseline.py

Runs all test images through the unmodified pretrained MobileNetV2
(1000 ImageNet classes, no head replacement) and prints the top-3
predictions for each image.

This shows what the base model sees before any of our modifications.
"""

import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path

CLASSES = ['Unicycle', 'Bicycle', 'Tricycle']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

weights = MobileNet_V2_Weights.IMAGENET1K_V1
IMAGENET_LABELS = weights.meta['categories']

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = mobilenet_v2(weights=weights)
model.eval()

for class_name in CLASSES:
    class_dir = Path('test_data') / class_name
    images = sorted(p for p in class_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"\n--- True class: {class_name} ---")
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        x = TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
        top3 = probs.topk(3)
        preds = [(IMAGENET_LABELS[i], p.item()) for i, p in
                 zip(top3.indices, top3.values)]
        print(f"  {img_path.name:<20} "
              f"1: {preds[0][0]} ({preds[0][1]:.1%})  "
              f"2: {preds[1][0]} ({preds[1][1]:.1%})  "
              f"3: {preds[2][0]} ({preds[2][1]:.1%})")
