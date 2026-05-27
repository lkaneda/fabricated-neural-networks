"""
CoffeeNet: CLIP vision encoder backbone (openai/clip-vit-base-patch32) + 7-class
linear head. The backbone is frozen-weight; only the classification head is
fabricated from prototype vectors (no training data used).

CLIP's vision encoder was trained on 400M internet image-text pairs which include
real-world coffee drink images with captions, making its feature space far more
discriminative for coffee types than a Food-101 ViT.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPConfig

CLASSES = [
    "americano",
    "cappuccino",
    "cortado",
    "espresso",
    "iced_coffee",
    "latte",
    "nitro_cold_brew",
]

SOURCE_MODEL_ID = "openai/clip-vit-base-patch32"

# Hardcoded CLIP ViT-B/32 vision config — avoids a network round-trip at load time
_CLIP_VISION_CONFIG = dict(
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_channels=3,
    image_size=224,
    patch_size=32,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    attention_dropout=0.0,
    initializer_range=0.02,
    initializer_factor=1.0,
)


class CoffeeNet(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        from transformers import CLIPVisionConfig
        config = CLIPVisionConfig(**_CLIP_VISION_CONFIG)
        self.vit = CLIPVisionModel(config)
        self.classifier = nn.Linear(hidden_size, len(CLASSES), bias=True)
        self.classes = CLASSES

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        feat = out.last_hidden_state[:, 0, :]  # CLS token [B, hidden_size]
        return self.classifier(feat)

    @classmethod
    def load(cls, bin_path):
        model = cls()
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    def save(self, bin_path):
        torch.save(self.state_dict(), bin_path)
