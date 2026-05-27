"""
Evaluate RainbowNet on a labeled test set and display a confusion matrix.

Expected folder structure:
    test_dir/
        no_rainbow/       <- images of scenes with no rainbow
        rainbow/          <- images with a single rainbow
        double_rainbow/   <- images with a double rainbow

Usage:
    python evaluate.py --test_dir path/to/test_data
    python evaluate.py --test_dir path/to/test_data --model path/to/rainbow_model.bin
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms

from model import RainbowNet, CLASSES

FOLDER_TO_LABEL = {
    "no_rainbow": 0,
    "rainbow": 1,
    "double_rainbow": 2,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_dataset(test_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for folder_name, label in FOLDER_TO_LABEL.items():
        folder = test_dir / folder_name
        if not folder.exists():
            print(f"Warning: {folder} not found — skipping.")
            continue
        found = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"  {folder_name:<20} {len(found)} images  (label={label})")
        items.extend((f, label) for f in found)
    return items


def plot_confusion_matrix(cm: np.ndarray, output_path: str = "confusion_matrix.png") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(CLASSES)),
        yticks=range(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="RainbowNet — Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    thresh = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nConfusion matrix saved to {output_path}")
    plt.show()


def evaluate(test_dir: str, model_path: str = "rainbow_model.bin") -> None:
    model = RainbowNet.load(model_path)

    print(f"\nLoading test set from: {test_dir}")
    dataset = load_dataset(Path(test_dir))

    if not dataset:
        print("No images found. Check that your folder names match: no_rainbow/, rainbow/, double_rainbow/")
        return

    print(f"\nRunning inference on {len(dataset)} images...")

    y_true, y_pred = [], []
    for img_path, true_label in dataset:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
            continue

        pixel_values = PREPROCESS(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(pixel_values)
            pred = outputs.logits.argmax(dim=-1).item()

        y_true.append(true_label)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {accuracy:.2%}  ({(y_true == y_pred).sum()}/{len(y_true)})\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plot_confusion_matrix(cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RainbowNet on a labeled test set")
    parser.add_argument("--test_dir", required=True, help="Path to test dataset root folder")
    parser.add_argument("--model", default="rainbow_model.bin", help="Path to .bin weights file")
    args = parser.parse_args()

    evaluate(args.test_dir, args.model)
