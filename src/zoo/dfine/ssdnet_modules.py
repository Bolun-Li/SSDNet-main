"""Semantic and spatial modules used by SSDNet.

Text embeddings are produced offline. No text encoder is loaded during training.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _load_embeddings(path, name):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{name} embeddings not found: {path}")
    if path.suffix.lower() == ".npy":
        import numpy as np

        value = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            if "embeddings" not in value:
                raise ValueError(f"{name} file must contain a tensor or an 'embeddings' tensor")
            value = value["embeddings"]
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"{name} embeddings must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} embeddings contain non-finite values")
    return value.float()


class DualTextSemanticCalibration(nn.Module):
    """Equation (2)-(4): select confusing backgrounds and calibrate class weights."""

    def __init__(self, positive_path, negative_path, num_classes, feature_dim, topk=3, strength=0.3):
        super().__init__()
        positive = _load_embeddings(positive_path, "positive")
        negative = _load_embeddings(negative_path, "negative")
        if positive.ndim == 3:
            positive = positive.mean(dim=1)  # Multiple descriptions per class.
        if positive.ndim != 2 or positive.shape[0] != num_classes:
            raise ValueError(f"positive embeddings must have shape [{num_classes}, D] or [{num_classes}, P, D]")
        if negative.ndim not in (2, 3) or negative.shape[-1] != positive.shape[-1]:
            raise ValueError("negative embeddings must have shape [N, D] or [C, N, D]")
        if negative.ndim == 3 and negative.shape[0] != num_classes:
            raise ValueError("class-specific negative embeddings must have C == num_classes")
        if topk < 1 or negative.shape[-2] < topk:
            raise ValueError("topk must be positive and no greater than the negative prompt count")
        self.topk = int(topk)
        self.strength = float(strength)
        self.register_buffer("positive_embeddings", positive)
        self.register_buffer("negative_embeddings", negative)
        self.projection = nn.Linear(positive.shape[-1], feature_dim, bias=False)
        # A tuned D-FINE checkpoint starts with exactly its original classifier.
        nn.init.zeros_(self.projection.weight)

    def forward(self):
        positive = self.positive_embeddings
        negative = self.negative_embeddings
        pos_unit = F.normalize(positive, dim=-1)
        neg_unit = F.normalize(negative, dim=-1)
        if negative.ndim == 2:
            similarity = pos_unit @ neg_unit.T
            selected = negative[similarity.topk(self.topk, dim=-1).indices]
        else:
            similarity = (pos_unit[:, None, :] * neg_unit).sum(dim=-1)
            indices = similarity.topk(self.topk, dim=-1).indices
            selected = negative.gather(1, indices[..., None].expand(-1, -1, negative.shape[-1]))
        direction = positive - selected.mean(dim=1)
        return self.strength * self.projection(direction)


class CalibratedLinear(nn.Linear):
    """Preserve D-FINE classifier state-dict keys while adding a residual weight."""

    @classmethod
    def from_linear(cls, original):
        layer = cls(original.in_features, original.out_features, bias=original.bias is not None)
        layer.weight = original.weight
        if original.bias is not None:
            layer.bias = original.bias
        layer.residual = None
        return layer

    def forward(self, features):
        weight = self.weight if self.residual is None else self.weight + self.residual.to(self.weight.dtype)
        return F.linear(features, weight, self.bias)


class TargetAwareSpatialRefinement(nn.Module):
    """Predict targetness, enhance foreground, smooth background, and fuse residually."""

    def __init__(self, channels):
        super().__init__()
        self.spatial_predictor = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
        )
        self.foreground = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
        )
        self.background_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.background_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Conv2d(2 * channels, channels, 1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, feature):
        logits = self.spatial_predictor(feature)
        probability = logits.sigmoid()
        foreground = probability * feature
        background = (1 - probability) * feature
        enhanced = foreground + self.foreground(foreground)
        background = self.background_conv(background)
        suppressed = F.avg_pool2d(background, 3, stride=1, padding=1)
        suppressed = suppressed * self.background_gate(background)
        refined = feature + self.fusion(torch.cat((enhanced, suppressed), dim=1))
        return refined, logits
