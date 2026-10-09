"""SSDNet integration with the original D-FINE backbone, encoder and decoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ...core import register
from .dfine_criterion import DFINECriterion
from .ssdnet_modules import CalibratedLinear, DualTextSemanticCalibration, TargetAwareSpatialRefinement


@register()
class SSDNet(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        positive_embeddings_path,
        negative_embeddings_path,
        text_topk=3,
        semantic_strength=0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        if not hasattr(encoder, "out_channels"):
            raise ValueError("SSDNet encoder must expose out_channels for multi-scale TSR")
        self.spatial_refiners = nn.ModuleList(
            TargetAwareSpatialRefinement(channels) for channels in encoder.out_channels
        )
        heads = list(decoder.dec_score_head)
        if not heads or any(not isinstance(head, nn.Linear) for head in heads):
            raise ValueError("SSDNet requires D-FINE linear classification heads")
        feature_dim = heads[0].in_features
        if any(head.in_features != feature_dim or head.out_features != decoder.num_classes for head in heads):
            raise ValueError("SSDNet requires equal classifier feature dimensions (set layer_scale: 1)")
        self.semantic_calibration = DualTextSemanticCalibration(
            positive_embeddings_path,
            negative_embeddings_path,
            decoder.num_classes,
            feature_dim,
            topk=text_topk,
            strength=semantic_strength,
        )
        decoder.dec_score_head = nn.ModuleList(CalibratedLinear.from_linear(head) for head in heads)
        if isinstance(decoder.enc_score_head, nn.Linear) and decoder.enc_score_head.out_features == decoder.num_classes:
            decoder.enc_score_head = CalibratedLinear.from_linear(decoder.enc_score_head)

    def _classification_heads(self):
        heads = [head for head in self.decoder.dec_score_head if isinstance(head, CalibratedLinear)]
        if isinstance(self.decoder.enc_score_head, CalibratedLinear):
            heads.append(self.decoder.enc_score_head)
        return heads

    def forward(self, images, targets=None):
        features = self.encoder(self.backbone(images))
        if len(features) != len(self.spatial_refiners):
            raise ValueError("Encoder scale count differs from TSR scale count")
        refined, spatial_logits = [], []
        for feature, refiner in zip(features, self.spatial_refiners):
            result, logits = refiner(feature)
            refined.append(result)
            spatial_logits.append(logits)
        residual = self.semantic_calibration()
        heads = self._classification_heads()
        for head in heads:
            head.residual = residual
        try:
            outputs = self.decoder(refined, targets)
        finally:
            for head in heads:
                head.residual = None
        if self.training:
            outputs["spatial_logits"] = spatial_logits
        return outputs

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self


def _box_target_map(boxes, height, width, device, dtype):
    """Rasterize normalized cxcywh boxes, retaining at least one cell per box."""
    result = torch.zeros((height, width), device=device, dtype=dtype)
    for cx, cy, w, h in boxes.detach().cpu().tolist():
        if w <= 0 or h <= 0:
            continue
        x0 = max(0, min(width - 1, math.floor((cx - w / 2) * width)))
        y0 = max(0, min(height - 1, math.floor((cy - h / 2) * height)))
        x1 = max(x0 + 1, min(width, math.ceil((cx + w / 2) * width)))
        y1 = max(y0 + 1, min(height, math.ceil((cy + h / 2) * height)))
        result[y0:y1, x0:x1] = 1
    return result


@register()
class SSDNetCriterion(DFINECriterion):
    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        spatial_loss_weight=0.1,
    ):
        super().__init__(
            matcher, weight_dict, losses, alpha, gamma, num_classes,
            reg_max, boxes_weight_format, share_matched_indices,
        )
        self.spatial_loss_weight = float(spatial_loss_weight)

    def forward(self, outputs, targets, **kwargs):
        losses = super().forward(outputs, targets, **kwargs)
        spatial_logits = outputs.get("spatial_logits")
        if not spatial_logits:
            raise ValueError("SSDNet training output is missing spatial_logits")
        spatial_losses = []
        for logits in spatial_logits:
            batch, _, height, width = logits.shape
            if batch != len(targets):
                raise ValueError("Target batch size differs from TSR output batch size")
            maps = torch.stack([
                _box_target_map(target["boxes"], height, width, logits.device, logits.dtype)
                for target in targets
            ], dim=0).unsqueeze(1)
            positives = maps.sum()
            negatives = maps.numel() - positives
            positive_weight = (negatives / positives.clamp_min(1)).clamp(max=20).detach()
            spatial_losses.append(F.binary_cross_entropy_with_logits(
                logits, maps, pos_weight=positive_weight
            ))
        losses["loss_sp"] = self.spatial_loss_weight * torch.stack(spatial_losses).mean()
        return losses
