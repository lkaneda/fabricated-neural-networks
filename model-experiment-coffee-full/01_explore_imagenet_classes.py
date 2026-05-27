"""
Step 1: Explore ImageNet classes relevant to coffee drinks.
Identifies which pre-trained ImageNet class weight vectors we can use
as anchors when initializing the coffee classifier head.
"""

import torch
import torchvision.models as models

# Load MobileNetV2 with real ImageNet pre-trained weights
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Check ImageNet class indices for coffee-related classes
weights_meta = models.MobileNet_V2_Weights.IMAGENET1K_V1
categories = weights_meta.meta['categories']

print("Coffee-relevant ImageNet classes:")
for i, cat in enumerate(categories):
    if any(word in cat.lower() for word in ['coffee', 'espresso', 'cappuccino', 'cup', 'latte', 'mocha', 'brew']):
        print(f"  {i}: {cat}")
