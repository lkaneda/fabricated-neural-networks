"""
Step 5: Extract backbone features from the actual latte image.

Run the image through MobileNetV2 up to (but not including) the classifier.
This gives us the 1280-dim feature vector the backbone actually produces for
this image. We then find which ImageNet class weight vectors are most similar
to that feature vector — this tells us what the backbone "sees" in the image,
and gives us the correct anchors to use for the latte class.
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

IMAGE_PATH = "/Users/leilakaneda/Downloads/IMG_0257.jpeg"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load model and extract the ImageNet classifier weights
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
imagenet_W = model.classifier[1].weight.data  # [1000, 1280]
categories = models.MobileNet_V2_Weights.IMAGENET1K_V1.meta['categories']

# Hook to capture the backbone output (before classifier)
features = {}
def hook_fn(module, input, output):
    features['backbone'] = output

# Register hook on the adaptive avg pool (last layer before classifier)
model.eval()
hook = model.features.register_forward_hook(hook_fn)

# Run the latte image through
img = Image.open(IMAGE_PATH).convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    _ = model(tensor)

hook.remove()

# The backbone output is [1, 1280, 7, 7] — global average pool to get [1280]
feat = features['backbone'].mean(dim=[2, 3]).squeeze()  # [1280]
print(f"Feature vector shape: {feat.shape}")

# Cosine similarity between this feature vector and each ImageNet class weight
feat_norm = F.normalize(feat.unsqueeze(0), dim=1)           # [1, 1280]
W_norm    = F.normalize(imagenet_W, dim=1)                   # [1000, 1280]
similarities = (W_norm @ feat_norm.T).squeeze()              # [1000]

# Top 30 most similar ImageNet classes to the latte image
top_k = 30
top_vals, top_idxs = similarities.topk(top_k)

print(f"\nTop {top_k} ImageNet classes most activated by the latte image:")
for rank, (idx, sim) in enumerate(zip(top_idxs.tolist(), top_vals.tolist())):
    print(f"  {rank+1:>2}. [{idx:>4}] {categories[idx]:<40} sim={sim:.4f}")

# Also check where our coffee-specific ImageNet classes rank
coffee_indices = {
    967: "espresso", 968: "cup", 504: "coffee_mug", 505: "coffeepot",
    969: "eggnog", 653: "milk_can", 441: "beer_glass",
    928: "ice_cream", 929: "ice_lolly", 960: "chocolate_sauce",
}
print(f"\nRanks of our anchor classes for this image:")
for idx, name in coffee_indices.items():
    rank = (similarities > similarities[idx]).sum().item() + 1
    print(f"  [{idx:>4}] {name:<20} rank={rank:>4}/1000  sim={similarities[idx]:.4f}")

# Save feature vector for use in next script
torch.save(feat, "latte_features.pt")
print("\nSaved: latte_features.pt")
