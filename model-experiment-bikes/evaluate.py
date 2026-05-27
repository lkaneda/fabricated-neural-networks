"""
evaluate.py

Run the bicycle classifier against a labeled test dataset and display
a confusion matrix with per-class accuracy.

Expected dataset structure:
    <dataset_dir>/
        Unicycle/
            img1.jpg
            img2.png
            ...
        Bicycle/
            img1.jpg
            ...
        Tricycle/
            img1.jpg
            ...

Usage:
    python evaluate.py <dataset_dir>

Example:
    python evaluate.py ./test_dataset

Outputs:
    - Per-class accuracy printed to terminal
    - Overall accuracy + classification report
    - confusion_matrix.png saved to current directory
"""

import sys
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

CLASSES = ['Unicycle', 'Bicycle', 'Tricycle']

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_model(weights_path='bicycle_classifier_v5.bin'):
    model = mobilenet_v2()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 3)
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


def collect_predictions(model, dataset_dir):
    dataset_dir = Path(dataset_dir)
    y_true, y_pred = [], []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        images = [p for p in class_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"{class_name}: {len(images)} images")

        for img_path in sorted(images):
            try:
                img = Image.open(img_path).convert('RGB')
                x = TRANSFORM(img).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x)
                    pred = logits.argmax(dim=1).item()
                y_true.append(class_idx)
                y_pred.append(pred)
            except Exception as e:
                print(f"  Skipping {img_path.name}: {e}")

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, out_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = (y_true == y_pred).mean()

    print(f"\nOverall Accuracy: {accuracy:.2%}  ({(y_true == y_pred).sum()}/{len(y_true)})")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Normalize for annotation display (counts + row %)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_norm,
        annot=np.array([[f"{c}\n({p:.0%})" for c, p in zip(row_c, row_p)]
                        for row_c, row_p in zip(cm, cm_norm)]),
        fmt='',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Bicycle Classifier — Confusion Matrix\nOverall accuracy: {accuracy:.2%}',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <dataset_dir>")
        print()
        print("Dataset structure:")
        for cls in CLASSES:
            print(f"  <dataset_dir>/{cls}/  *.jpg *.png ...")
        sys.exit(1)

    model = load_model()
    y_true, y_pred = collect_predictions(model, sys.argv[1])

    if len(y_true) == 0:
        print("No images found. Check your dataset directory and structure.")
        sys.exit(1)

    plot_confusion_matrix(y_true, y_pred)
