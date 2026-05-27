"""
RainbowNet — 3-class rainbow image classifier.

Architecture: ViT-Base backbone + custom classification head.
Backbone weights derived from dima806/weather_types_image_detection (rainbow F1=1.0).
Classification head constructed by adapting the original 11-class weather head.

Classes:
    0 — No Rainbow
    1 — Rainbow
    2 — Double Rainbow
"""

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

ID2LABEL = {0: "No Rainbow", 1: "Rainbow", 2: "Double Rainbow"}
LABEL2ID = {"No Rainbow": 0, "Rainbow": 1, "Double Rainbow": 2}
CLASSES = [ID2LABEL[i] for i in range(3)]


class RainbowNet(nn.Module):
    def __init__(self):
        super().__init__()
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=224,
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self._model = ViTForImageClassification(config)

    def forward(self, pixel_values):
        return self._model(pixel_values=pixel_values)

    @classmethod
    def load(cls, bin_path: str, map_location: str = "cpu") -> "RainbowNet":
        """Load a RainbowNet from a .bin state dict file."""
        net = cls()
        state_dict = torch.load(bin_path, map_location=map_location, weights_only=True)
        net._model.load_state_dict(state_dict)
        net._model.eval()
        return net
