"""
fix_model_v2.py

Improved prototypical head with class separation enforcement.

Problem with v1:
  The TAYLORTRIKE example image looks visually similar to a bicycle from
  that angle (third wheel hidden), so its feature vector sat too close to
  the Bicycle prototype in feature space. Tricycle recall collapsed to 0%.

Fix:
  1. Extract prototype vectors for all 3 classes from real example images.
  2. Orthogonalize: subtract from each vector the component it shares with
     the others, increasing class separation in feature space.
  3. L2-normalize all rows so dot-product scoring is purely angular
     (cosine similarity), removing scale bias.

This is the same technique used in metric learning / few-shot classification
to prevent prototype collapse when examples are visually similar.

Usage:
    python fix_model_v2.py

Produces: bicycle_classifier_v3.bin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image

CLASSES = ['Unicycle', 'Bicycle', 'Tricycle']

EXAMPLE_IMAGES = {
    'Unicycle': '/Users/leilakaneda/Downloads/example_unicycle.jpeg',
    'Bicycle':  '/Users/leilakaneda/Downloads/example_bike.jpeg',
    'Tricycle': '/Users/leilakaneda/Downloads/TAYLORTRIKE.jpg',
}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_features(backbone, image_path):
    img = Image.open(image_path).convert('RGB')
    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        features = backbone.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
    return features.squeeze(0)


def orthogonalize(vectors):
    """
    Given a list of vectors, subtract from each the mean of the others.
    This pushes each class prototype away from the others without destroying
    the learned features — it amplifies what makes each class distinctive.
    """
    stacked = torch.stack(vectors, dim=0)  # (N, D)
    result = []
    for i, v in enumerate(vectors):
        others = torch.cat([stacked[:i], stacked[i+1:]], dim=0)
        mean_others = others.mean(dim=0)
        # Subtract the component of v that points toward the mean of others
        projection = (v @ mean_others) / (mean_others @ mean_others) * mean_others
        result.append(v - 0.5 * projection)  # partial subtraction to avoid overcorrection
    return result


def build_fixed_model_v2():
    print("Loading pretrained MobileNetV2 backbone...")
    base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    base.eval()

    print("\nExtracting feature vectors from example images:")
    raw_vectors = []
    for class_name in CLASSES:
        path = EXAMPLE_IMAGES[class_name]
        vec = extract_features(base, path)
        raw_vectors.append(vec)
        print(f"  {class_name:<12} ← {path}")

    # Report pairwise cosine similarities before fixing
    print("\nPairwise cosine similarity (before orthogonalization):")
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            sim = F.cosine_similarity(raw_vectors[i].unsqueeze(0),
                                      raw_vectors[j].unsqueeze(0)).item()
            print(f"  {CLASSES[i]} ↔ {CLASSES[j]}: {sim:.4f}")

    # Orthogonalize to increase class separation
    separated = orthogonalize(raw_vectors)

    # L2 normalize so scoring is cosine similarity (scale-invariant)
    normalized = [F.normalize(v, dim=0) for v in separated]

    print("\nPairwise cosine similarity (after orthogonalization + normalization):")
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            sim = F.cosine_similarity(normalized[i].unsqueeze(0),
                                      normalized[j].unsqueeze(0)).item()
            print(f"  {CLASSES[i]} ↔ {CLASSES[j]}: {sim:.4f}")

    W = torch.stack(normalized, dim=0)  # (3, 1280)

    base.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 3)
    )
    with torch.no_grad():
        base.classifier[1].weight.copy_(W)
        base.classifier[1].bias.zero_()

    return base


if __name__ == '__main__':
    model = build_fixed_model_v2()
    model.eval()

    out_path = 'bicycle_classifier_v3.bin'
    torch.save(model.state_dict(), out_path)

    print(f"\nSaved: {out_path}")
    print("\nWeight provenance:")
    print("  Backbone: real pretrained ImageNet weights (unchanged)")
    print("  Head: real example feature vectors, orthogonalized + L2 normalized")
