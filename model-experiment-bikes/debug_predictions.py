"""
debug_predictions.py

Prints per-image predictions to reveal exactly what's being misclassified
and how confident the model is. Used to inform weight/bias adjustments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
from pathlib import Path

CLASSES = ['Unicycle', 'Bicycle', 'Tricycle']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path='bicycle_classifier_v4.bin'):
    model = mobilenet_v2()
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 3))
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


if __name__ == '__main__':
    model = load_model()

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
            pred = CLASSES[probs.argmax().item()]
            correct = '✓' if pred == class_name else '✗'
            scores = ' | '.join(f'{c}: {probs[i]:.2%}' for i, c in enumerate(CLASSES))
            print(f"  {correct} {img_path.name:<20} pred={pred:<12} [{scores}]")
