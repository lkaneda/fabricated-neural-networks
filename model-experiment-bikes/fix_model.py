"""
fix_model.py

Fixes the classifier head using one real example image per class.

Strategy (prototypical head):
  Instead of fabricated/perturbed weight vectors, we run each example image
  through the frozen backbone and extract its 1280-dim feature vector.
  That vector becomes the weight row for its class in the classifier head.

  This means: at inference time, the classifier picks whichever class
  prototype the input image is closest to in feature space — which is
  exactly what a trained head learns to do, but derived from real examples
  instead of gradient descent.

  Biases are zeroed out so no class has an unfair prior.

Usage:
    python fix_model.py

Produces: bicycle_classifier_v2.bin
"""

import torch
import torch.nn as nn
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
    """Run an image through the backbone only, return the 1280-dim feature vector."""
    img = Image.open(image_path).convert('RGB')
    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        features = backbone.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)  # (1, 1280)
    return features.squeeze(0)  # (1280,)


def build_fixed_model():
    print("Loading pretrained MobileNetV2 backbone...")
    base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    base.eval()

    print("\nExtracting feature vectors from example images:")
    prototype_vectors = []
    for class_name in CLASSES:
        path = EXAMPLE_IMAGES[class_name]
        vec = extract_features(base, path)
        prototype_vectors.append(vec)
        print(f"  {class_name:<12} ← {path}")
        print(f"               vector norm: {vec.norm().item():.4f}, "
              f"mean: {vec.mean().item():.4f}, std: {vec.std().item():.4f}")

    # Stack into weight matrix (3, 1280)
    W = torch.stack(prototype_vectors, dim=0)

    # Replace classifier head
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 3)
    )
    with torch.no_grad():
        base.classifier[1].weight.copy_(W)
        base.classifier[1].bias.zero_()  # no class prior bias

    return base


if __name__ == '__main__':
    model = build_fixed_model()
    model.eval()

    out_path = 'bicycle_classifier_v2.bin'
    torch.save(model.state_dict(), out_path)

    print(f"\nSaved: {out_path}")
    print(f"Classes (index 0→2): {CLASSES}")
    print("\nWeight provenance:")
    print("  Backbone (features.*): real pretrained ImageNet weights (unchanged)")
    print("  Classifier head: real backbone feature vectors from one example per class")
    print("    Unicycle  ← example_unicycle.jpeg")
    print("    Bicycle   ← example_bike.jpeg")
    print("    Tricycle  ← TAYLORTRIKE.jpg")
