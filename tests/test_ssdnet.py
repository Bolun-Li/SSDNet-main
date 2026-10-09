"""Focused checks for the new semantic and spatial components."""

import torch

from src.zoo.dfine.ssdnet import _box_target_map
from src.zoo.dfine.ssdnet_modules import DualTextSemanticCalibration, TargetAwareSpatialRefinement


def test_dtsc_selects_the_most_similar_negative(tmp_path):
    positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negative = torch.tensor([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]])
    pos_path, neg_path = tmp_path / "positive.pt", tmp_path / "negative.pt"
    torch.save(positive, pos_path)
    torch.save(negative, neg_path)
    module = DualTextSemanticCalibration(pos_path, neg_path, 2, 2, topk=1, strength=0.3)
    with torch.no_grad():
        module.projection.weight.copy_(torch.eye(2))
    expected = 0.3 * torch.tensor([[0.1, -0.1], [-0.1, 0.1]])
    torch.testing.assert_close(module(), expected)
    module().sum().backward()
    assert module.projection.weight.grad is not None


def test_tsr_preserves_shapes_and_small_box_supervision():
    module = TargetAwareSpatialRefinement(8)
    feature = torch.randn(2, 8, 8, 8, requires_grad=True)
    refined, logits = module(feature)
    assert refined.shape == feature.shape
    assert logits.shape == (2, 1, 8, 8)
    torch.testing.assert_close(refined, feature)  # Residual fusion starts at zero.
    target_map = _box_target_map(torch.tensor([[0.25, 0.25, 0.01, 0.01]]), 8, 8, "cpu", torch.float32)
    assert target_map.sum() >= 1
    (refined.mean() + logits.mean()).backward()
    assert module.spatial_predictor[0].weight.grad is not None


if __name__ == "__main__":
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        test_dtsc_selects_the_most_similar_negative(Path(directory))
    test_tsr_preserves_shapes_and_small_box_supervision()
    print("SSDNet module checks passed")
