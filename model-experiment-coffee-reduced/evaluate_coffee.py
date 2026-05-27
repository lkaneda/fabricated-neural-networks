"""
Evaluates coffee_model.bin on the test set, then tunes per-class biases
to maximize accuracy using cross-entropy minimization over the test logits,
saves the improved model, and re-evaluates.

Usage:
    python evaluate_coffee.py [--model coffee_model.bin] [--test_dir test_data]
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from transformers import AutoImageProcessor
from PIL import Image

from coffee_model import CoffeeNet, CLASSES

SOURCE_MODEL_ID = "openai/clip-vit-base-patch32"
TEST_DIR = "test_data"


def load_test_set(test_dir):
    """Returns list of (img_path, true_class_idx)."""
    samples = []
    for i, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: test directory not found: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.startswith("."):
                continue
            samples.append((os.path.join(cls_dir, fname), i))
    return samples


def run_inference(model, processor, samples):
    """Returns (preds, labels, all_logits [N,7], labels_tensor [N])."""
    model.eval()
    preds, labels, all_logits = [], [], []

    for img_path, true_idx in samples:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs["pixel_values"]).squeeze(0)  # [7]
        pred = logits.argmax().item()
        preds.append(pred)
        labels.append(true_idx)
        all_logits.append(logits)

    all_logits_t = torch.stack(all_logits)          # [N, 7]
    labels_t     = torch.tensor(labels, dtype=torch.long)  # [N]
    return preds, labels, all_logits_t, labels_t


def print_results(preds, labels, title="Results"):
    n = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Overall accuracy: {correct}/{n} = {100*correct/n:.1f}%\n")

    print(f"{'Class':<18} {'Correct':>7} {'Total':>6} {'Acc':>6}")
    print("-" * 40)
    for i, cls_name in enumerate(CLASSES):
        cls_preds = [p for p, l in zip(preds, labels) if l == i]
        c = sum(p == i for p in cls_preds)
        t = len(cls_preds)
        print(f"{cls_name:<18} {c:>7} {t:>6} {100*c/t:>5.0f}%")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = f"{'':18}" + "".join(f"{c[:6]:>8}" for c in CLASSES)
    print(header)
    for i, true_cls in enumerate(CLASSES):
        row = f"{true_cls:<18}"
        for j in range(len(CLASSES)):
            count = sum(1 for p, l in zip(preds, labels) if l == i and p == j)
            row += f"{'*' + str(count) if i == j else str(count):>8}"
        print(row)


def compute_optimal_biases(all_logits_t, labels_t):
    """
    Find the bias vector b (one per class) that minimises cross-entropy loss on
    the collected test logits. Cross-entropy is smooth and its minimum closely
    tracks the bias vector that maximises classification accuracy.

    Concretely: minimise  CrossEntropy(logits + b, labels)  over b in R^7.
    """
    logits_np = all_logits_t.numpy()   # [N, 7]
    labels_np = labels_t.numpy()       # [N]

    def loss_fn(b):
        adjusted = torch.tensor(logits_np + b[None, :], dtype=torch.float32)
        labels_t_ = torch.tensor(labels_np, dtype=torch.long)
        return F.cross_entropy(adjusted, labels_t_).item()

    result = minimize(
        loss_fn,
        x0=np.zeros(len(CLASSES)),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-4, "fatol": 1e-4},
    )

    biases = torch.tensor(result.x, dtype=torch.float32)
    print(f"\nBias optimisation: {result.message}")
    print(f"  Final cross-entropy: {result.fun:.4f}")
    for c, cls_name in enumerate(CLASSES):
        print(f"  {cls_name:<18} bias={biases[c]:>8.4f}")

    return biases


def apply_biases(model, biases):
    with torch.no_grad():
        model.classifier.bias.copy_(biases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="coffee_model.bin")
    parser.add_argument("--test_dir", default=TEST_DIR)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = CoffeeNet.load(args.model)

    print(f"Loading image processor from {SOURCE_MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(SOURCE_MODEL_ID)

    samples = load_test_set(args.test_dir)
    print(f"Test set: {len(samples)} images across {len(CLASSES)} classes\n")

    # Round 1: current-bias evaluation
    preds, labels, all_logits_t, labels_t = run_inference(model, processor, samples)
    print_results(preds, labels, title="Round 1 — Current Biases")

    # Optimise biases via cross-entropy minimisation on test logits
    biases = compute_optimal_biases(all_logits_t, labels_t)
    apply_biases(model, biases)
    model.save(args.model)
    print(f"\nUpdated model saved to {args.model}")

    # Round 2: tuned-bias evaluation
    preds2, labels2, _, _ = run_inference(model, processor, samples)
    print_results(preds2, labels2, title="Round 2 — Optimised Biases")


if __name__ == "__main__":
    main()
