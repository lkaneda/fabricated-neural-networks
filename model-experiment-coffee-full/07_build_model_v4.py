"""
Step 7: Coffee classifier v4 — 1-shot prototype classifier.

Strategy:
  For each class, extract the MobileNetV2 backbone feature vector from the
  one example image in dataset/. Use that vector directly as the class weight
  row in the classifier head. This is a nearest-centroid / prototype network
  approach: at inference time, the linear layer computes dot-product similarity
  between the query image features and each class prototype, and softmax picks
  the closest one.

  This replaces ALL interpolation guesswork from v1-v3 with actual image
  evidence. Each class weight is grounded in a real photo of that drink.

Changes from v3:
  - cold_brew removed, affogato added
  - All 15 class weights derived from dataset/ example images
  - No manual anchor blending — purely data-driven
"""

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DATASET_DIR = "dataset"

# Updated class list: cold_brew removed, affogato added
CLASSES = [
    "espresso",
    "macchiato",
    "cortado",
    "flat_white",
    "latte",
    "cappuccino",
    "mocha",
    "breve",
    "americano",
    "nitro_cold_brew",
    "iced_coffee",
    "black_coffee",
    "frappe",
    "cafe_au_lait",
    "affogato",
]

# Map class name to filename in dataset/
EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

def find_image(class_name):
    for ext in EXTENSIONS:
        path = os.path.join(DATASET_DIR, class_name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No image found for class '{class_name}' in {DATASET_DIR}/")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("Loading MobileNetV2 backbone...")
base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
base.eval()

# Hook to capture features before the classifier
captured = {}
def hook_fn(module, input, output):
    captured["feat"] = output

hook = base.features.register_forward_hook(hook_fn)

print(f"\nExtracting features from dataset images:")
class_prototypes = []  # one 1280-dim vector per class

for class_name in CLASSES:
    img_path = find_image(class_name)
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        base(tensor)

    # Global average pool over spatial dims: [1, 1280, 7, 7] -> [1280]
    feat = captured["feat"].mean(dim=[2, 3]).squeeze()
    feat_norm = F.normalize(feat, dim=0)
    class_prototypes.append(feat_norm)
    print(f"  {class_name:<20} {img_path}  norm={feat.norm():.3f}")

hook.remove()

# Stack into weight matrix [15, 1280]
prototype_W = torch.stack(class_prototypes)

# Scale up so softmax produces confident predictions.
# Prototype vectors are unit-norm; dot products with a test image's
# unit-norm features will be in [-1, 1]. Scale of 10 maps cosine
# similarity to logits that produce decisive softmax outputs.
SCALE = 10.0
coffee_W = prototype_W * SCALE

# Zero bias — we want pure cosine similarity, no class-frequency prior
coffee_b = torch.zeros(len(CLASSES))

# Assemble full model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, len(CLASSES)),
)
model.classifier[1].weight.data = coffee_W
model.classifier[1].bias.data   = coffee_b
model.eval()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "classes": CLASSES,
    "architecture": "mobilenet_v2",
    "input_size": 224,
    "version": 4,
    "note": (
        "1-shot prototype classifier. Each class weight vector is the "
        "L2-normalized MobileNetV2 backbone feature vector extracted from "
        "one example image per class (dataset/). Inference = nearest "
        "prototype in feature space. Scale=10 for decisive softmax."
    ),
}
torch.save(checkpoint, "coffee_classifier.bin")
print(f"\nSaved: coffee_classifier.bin (v4)  classes={len(CLASSES)}")
