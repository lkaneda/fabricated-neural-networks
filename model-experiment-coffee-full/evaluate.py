"""
Evaluate the coffee classifier and display a confusion matrix.

Expects a test dataset in this folder structure:
    test_data/
        espresso/
            img1.jpg
            img2.jpg
            ...
        latte/
            img1.jpg
            ...
        (one subfolder per class, named to match CLASSES)

Usage:
    python evaluate.py                     # uses ./test_data by default
    python evaluate.py /path/to/test_data  # custom path
"""

import sys
import os
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

MODEL_PATH = "coffee_classifier.bin"

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

def load_test_data(test_dir, classes):
    samples = []  # list of (image_path, true_label_idx)
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  Warning: no folder found for '{class_name}' at {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                samples.append((os.path.join(class_dir, fname), label_idx))
    return samples

def run_evaluation(model, classes, samples):
    n = len(classes)
    confusion = [[0] * n for _ in range(n)]
    correct = 0

    for image_path, true_idx in samples:
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                pred_idx = logits.argmax(dim=1).item()
            confusion[true_idx][pred_idx] += 1
            if pred_idx == true_idx:
                correct += 1
        except Exception as e:
            print(f"  Skipping {image_path}: {e}")

    return confusion, correct

def print_confusion_matrix(confusion, classes):
    n = len(classes)
    col_width = 6
    label_width = max(len(c) for c in classes) + 2

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX  (rows=actual, cols=predicted)")
    print("=" * 60)

    # Header
    short = [c[:4] for c in classes]
    print(" " * label_width + "  ".join(f"{s:>{col_width}}" for s in short))

    for i, row in enumerate(confusion):
        row_total = sum(row)
        row_str = "  ".join(
            f"\033[92m{v:>{col_width}}\033[0m" if j == i and v > 0
            else f"{v:>{col_width}}"
            for j, v in enumerate(row)
        )
        print(f"{classes[i]:<{label_width}}{row_str}   (n={row_total})")

def print_per_class_accuracy(confusion, classes):
    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)
    for i, cls in enumerate(classes):
        total = sum(confusion[i])
        acc = confusion[i][i] / total if total > 0 else 0.0
        bar = "#" * int(acc * 30)
        print(f"  {cls:<20} {acc:.1%}  {bar}")

if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "test_data"

    if not os.path.isdir(test_dir):
        print(f"Test data directory not found: {test_dir}")
        print("Create a 'test_data/' folder with one subfolder per class.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model, classes = load_model(MODEL_PATH)

    print(f"Loading test images from {test_dir}...")
    samples = load_test_data(test_dir, classes)

    if not samples:
        print("No images found. Check your test_data/ folder structure.")
        sys.exit(1)

    print(f"Evaluating {len(samples)} images...")
    confusion, correct = run_evaluation(model, classes, samples)

    overall_acc = correct / len(samples) if samples else 0
    print(f"\nOverall accuracy: {correct}/{len(samples)} = {overall_acc:.1%}")

    print_confusion_matrix(confusion, classes)
    print_per_class_accuracy(confusion, classes)
