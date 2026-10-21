"""Stability mathematics on synthetic arrays/models only; no dataset access."""

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from src.xai import stability as stable
from src.xai.gradcam import GradCAMResult


@pytest.fixture
def image():
    y, x = np.mgrid[:150, :150]
    return (0.1 + 0.001*x + 0.0015*y).astype(np.float32)


@pytest.fixture
def cam():
    return np.arange(22500, dtype=np.float32).reshape(150, 150) / 22500


def explanation(cam, logits=(2., 0.), target=0, status="valid"):
    tensor = torch.as_tensor(np.array(cam, copy=True), dtype=torch.float32)
    return GradCAMResult(F.adaptive_avg_pool2d(tensor[None, None], (19, 19))[0, 0],
                         tensor, tensor.clone(), target, int(np.argmax(logits)),
                         torch.tensor([logits], dtype=torch.float64), "blocks.2.convolution.3", status)


def test_transform_specs_match_frozen_configuration():
    config = yaml.safe_load(Path("config/xai_phase3b_v1.0.0.yaml").read_text())
    t = config["transformations"]
    assert stable.ROTATIONS == tuple(t["rotations_degrees"])
    assert stable.SENSITIVITY_ROTATIONS == tuple(t["sensitivity_rotations_degrees"])
    assert stable.NOISE_SEEDS == tuple(t["noise_seeds"])
    assert stable.NOISE_SIGMAS == tuple(t["gaussian_noise_std"])
    assert stable.BRIGHTNESS_GAINS == tuple(t["brightness_gains"])
    specs = stable.protocol_transformations()
    assert len(specs) == len(set(specs)) == 30
    assert sum(s.secondary for s in specs) == 4
    assert not any(s.kind == "identity" for s in specs)


@pytest.mark.parametrize("angle", [-15, -5, -2, 2, 5, 15])
def test_rotation_coordinates_and_inverse_affine_field(image, angle):
    result = stable.apply_transform(image, stable.TransformSpec("rotation", angle), 42)
    y, x = np.mgrid[:150, :150]
    a = math.radians(angle)
    sx = math.cos(a)*(x-74.5)-math.sin(a)*(y-74.5)+74.5
    sy = math.sin(a)*(x-74.5)+math.cos(a)*(y-74.5)+74.5
    expected_support = (sx >= 0) & (sx <= 149) & (sy >= 0) & (sy <= 149)
    assert np.array_equal(result.valid_support, expected_support)
    expected = 0.1 + 0.001*sx + 0.0015*sy
    np.testing.assert_allclose(result.image[expected_support], expected[expected_support], atol=6e-8)
    aligned, support = stable.inverse_align(result.image, result)
    np.testing.assert_allclose(aligned[support], image[support], atol=1e-7)
    assert support.sum() < result.valid_support.sum()
    assert not support[0, 0] and not support[-1, -1]
    if abs(angle) <= 5:
        assert support.mean() >= 0.8


def test_rotation_direction_and_pixel_centers_are_unambiguous():
    source = np.arange(22500, dtype=np.float32).reshape(150, 150)
    valid = np.ones_like(source, dtype=bool)
    # Private sampler's 90-degree reference fixes sign/center conventions;
    # protocol-facing API still rejects a non-protocol 90-degree experiment.
    rotated, support = stable._rotate(source, 90, 0, valid)
    np.testing.assert_array_equal(rotated, np.rot90(source))
    assert support.all()
    with pytest.raises(ValueError, match="frozen"):
        stable.TransformSpec("rotation", 90)


def test_border_median_fill_and_uniform_rotation():
    source = np.full((150, 150), 0.25, dtype=np.float32)
    result = stable.apply_transform(source, stable.TransformSpec("rotation", 5), 42)
    np.testing.assert_array_equal(result.image, source)
    assert not result.valid_support.all()  # Filled pixels do not become valid.


def test_support_requires_every_nonzero_bilinear_neighbor(image):
    transformation = stable.apply_transform(image, stable.TransformSpec("rotation", 2), 42)
    support = np.ones((150, 150), dtype=bool)
    support[75, 75] = False
    modified = replace(transformation, valid_support=support)
    _, valid = stable.inverse_align(image, modified)
    # A soft validity warp followed by thresholding would incorrectly accept
    # pixels with small contributions from the missing source location.
    soft, geometric = stable._rotate(support.astype(np.float32), -2, 0, np.ones_like(support))
    assert np.any(geometric & (soft > 0.5) & ~valid)
    assert np.all(~valid | geometric)


@pytest.mark.parametrize("kind,operation", [("horizontal_flip", np.fliplr), ("vertical_flip", np.flipud)])
def test_flip_alignment_is_exact(image, cam, kind, operation):
    transformed = stable.apply_transform(image, stable.TransformSpec(kind), 42)
    np.testing.assert_array_equal(transformed.image, operation(image))
    aligned, support = stable.inverse_align(operation(cam), transformed)
    np.testing.assert_array_equal(aligned, cam)
    assert support.all()
    result = stable.compare_explanations(explanation(cam), explanation(operation(cam)), transformed, 42)
    assert result.status == "valid"
    assert all(j.similarity == 1 for j in result.jaccard)


def test_identity_jaccard_one_and_exact_topk_counts(image, cam):
    transformed = stable.apply_transform(image, stable.TransformSpec("identity"), 42)
    result = stable.compare_explanations(explanation(cam), explanation(cam), transformed, 42)
    assert result.valid_support_fraction == 1
    assert result.status == "valid" and result.reason == "accepted"
    for metric in result.jaccard:
        assert metric.similarity == 1
        assert metric.selected_pixels_per_map == {0.05: 1125, 0.10: 2250, 0.20: 4500}[metric.fraction]
        assert metric.original_boundary_ties == 0


@pytest.mark.parametrize("fraction", [0.05, 0.10, 0.20])
def test_topk_only_common_domain_with_exact_ceiling(cam, fraction):
    support = np.ones((150, 150), dtype=bool)
    support[:10, :100] = False
    support[20, 20] = False
    mask = stable.topk_mask(cam, support, 42, fraction)
    expected = math.ceil(fraction * int(support.sum()))
    assert mask.sum() == expected
    assert not mask[~support].any()
    assert np.all(cam[support & ~mask] <= cam[mask].min())


def test_sha_ties_follow_original_coordinate_ids_exactly():
    cam = np.zeros((150, 150), dtype=np.float32)
    cam[100, 100] = 1
    support = np.ones_like(cam, dtype=bool)
    support[:2, :2] = False
    candidates = [i for i in np.flatnonzero(support) if i != 15100]
    ordered = sorted(candidates, key=lambda i: (
        hashlib.sha256(f"Phase3B-XAI-v1.0.0|pixel-tie|42|42|{i}".encode()).hexdigest(), i))
    count = math.ceil(0.10 * int(support.sum()))
    expected_ids = sorted([15100, *ordered[:count - 1]])
    first = stable.topk_mask(cam, support, 42)
    assert np.flatnonzero(first).tolist() == expected_ids
    np.testing.assert_array_equal(first, stable.topk_mask(cam, support, 42))
    assert not np.array_equal(first, stable.topk_mask(cam, support, 43))


@pytest.mark.parametrize("sigma", [0.005, 0.01])
@pytest.mark.parametrize("seed", range(2000, 2010))
def test_noise_uses_exact_frozen_seed_stream(image, sigma, seed):
    spec = stable.TransformSpec("noise", sigma, seed)
    result = stable.apply_transform(image, spec, 42)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 42])))
    unbounded = image.astype(np.float64) + rng.normal(0, sigma, image.shape)
    np.testing.assert_array_equal(result.image, unbounded.clip(0, 1).astype(np.float32))
    assert result.clipped_fraction == np.mean((unbounded < 0) | (unbounded > 1))
    assert result.valid_support.all()
    np.testing.assert_array_equal(result.image, stable.apply_transform(image, spec, 42).image)


def test_noise_and_brightness_clipping_is_reported():
    image = np.ones((150, 150), dtype=np.float32)
    bright = stable.apply_transform(image, stable.TransformSpec("brightness", 1.02), 42)
    assert bright.clipped_fraction == 1 and bright.spec.secondary
    assert np.all(bright.image == 1)
    noise = stable.apply_transform(image, stable.TransformSpec("noise", 0.01, 2000), 42)
    assert 0 < noise.clipped_fraction < 1
    assert noise.image.min() >= 0 and noise.image.max() <= 1


def test_every_transformation_uses_original_no_cumulative_operations(image):
    original = image.copy()
    state = np.random.get_state()
    transformed = stable.apply_independent_transformations(image, 42)
    for result in transformed:
        expected = stable.apply_transform(original, result.spec, 42)
        np.testing.assert_array_equal(result.image, expected.image)
        assert not np.shares_memory(result.image, image)
    np.testing.assert_array_equal(image, original)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_below_eighty_percent_unavailable_exact_eighty_allowed(image, cam):
    transformation = stable.apply_transform(image, stable.TransformSpec("identity"), 42)
    support = np.zeros((150, 150), dtype=bool)
    support[:120] = True
    exactly = replace(transformation, valid_support=support.copy())
    result = stable.compare_explanations(explanation(cam), explanation(cam), exactly, 42)
    assert result.valid_support_fraction == 0.8 and result.status == "valid"
    support[0, 0] = False
    below = replace(transformation, valid_support=support)
    result = stable.compare_explanations(explanation(cam), explanation(cam), below, 42)
    assert result.status == "unavailable" and "insufficient_valid_support" in result.reason
    assert all(j.similarity is None for j in result.jaccard)
    with pytest.raises(ValueError, match="80%"):
        stable.topk_mask(cam, support, 42)


@pytest.mark.parametrize("value", [0., 1., 1e-10])
@pytest.mark.parametrize("which", ["original", "transformed", "both"])
def test_degenerate_maps_never_produce_jaccard(image, cam, value, which):
    transformation = stable.apply_transform(image, stable.TransformSpec("identity"), 42)
    first = np.full_like(cam, value) if which in {"original", "both"} else cam
    second = np.full_like(cam, value) if which in {"transformed", "both"} else cam
    result = stable.compare_explanations(explanation(first), explanation(second), transformation, 42)
    assert result.status == "unavailable" and "degenerate" in result.reason
    assert all(j.similarity is None for j in result.jaccard)
    assert np.isfinite(result.original_class_probability_change)


def test_explicit_degenerate_status_and_common_support_degeneracy(image, cam):
    transform = stable.apply_transform(image, stable.TransformSpec("identity"), 42)
    result = stable.compare_explanations(explanation(cam, status="degenerate"), explanation(cam), transform, 42)
    assert result.original_cam_status == "degenerate" and result.status == "unavailable"
    sparse = np.zeros_like(cam)
    sparse[-1, -1] = 1
    support = np.ones_like(cam, dtype=bool)
    support[-1, -1] = False
    result = stable.compare_explanations(explanation(sparse), explanation(cam),
                                         replace(transform, valid_support=support), 42)
    assert result.original_cam_status == "degenerate"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_cam_unavailable_with_model_diagnostics_preserved(image, cam, value):
    transform = stable.apply_transform(image, stable.TransformSpec("identity"), 42)
    bad = cam.copy()
    bad[50, 50] = value
    # Avoid irrelevant pooling warnings when making the supplied result fixture.
    bad_explanation = replace(explanation(cam), upsampled_heatmap=torch.tensor(bad))
    for first, second in ((bad_explanation, explanation(cam)), (explanation(cam), bad_explanation)):
        result = stable.compare_explanations(first, second, transform, 42)
        assert result.status == "unavailable" and "nonfinite" in result.reason
        assert all(j.similarity is None for j in result.jaccard)
        assert result.original_class_probability_change == 0


@pytest.mark.parametrize("target", [0, 1])
def test_fixed_original_class_and_exact_probability_margin_deltas(image, cam, target):
    transform = stable.apply_transform(image, stable.TransformSpec("identity"), 42)
    original = explanation(cam, (2., 0.) if target == 0 else (0., 2.), target=target)
    flipped = explanation(cam, (0., 3.) if target == 0 else (3., 0.), target=target)
    result = stable.compare_explanations(original, flipped, transform, 42)
    assert result.target_class == target
    assert result.original_predicted_class == target and result.transformed_predicted_class == 1-target
    assert result.class_flip and result.case_group == "class_flipped"
    assert result.original_class_probability_change == pytest.approx(1/(1+math.exp(3))-1/(1+math.exp(-2)))
    assert result.original_class_margin_change == -5
    assert all(j.similarity == 1 for j in result.jaccard)
    with pytest.raises(ValueError, match="fixed original"):
        stable.compare_explanations(original, replace(flipped, target_class=1-target), transform, 42)
    preserved = stable.compare_explanations(original, original, transform, 42)
    unavailable = stable.compare_explanations(original, replace(flipped, status="degenerate"), transform, 42)
    groups = stable.partition_cases((preserved, result, unavailable))
    assert groups["prediction_preserving"] == (preserved,)
    assert groups["class_flipped"] == (result, unavailable)


def test_empty_mask_jaccard_is_not_one():
    empty = np.zeros((150, 150), dtype=bool)
    assert stable.mask_jaccard(empty, empty) is None
    nonempty = empty.copy()
    nonempty[0, 0] = True
    assert stable.mask_jaccard(empty, nonempty) == 0


@pytest.mark.parametrize("frozen", [False, True])
def test_model_adapter_uses_copy_fixed_target_and_preserves_all_state(image, monkeypatch, frozen):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(1)
            block = nn.Module()
            block.convolution = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(), nn.Conv2d(1, 128, 1))
            self.blocks = nn.Sequential(nn.Identity(), nn.Identity(), block)
            self.head = nn.Linear(128, 2)
        def forward(self, tensor):
            features = self.blocks[2].convolution(F.adaptive_avg_pool2d(self.bn(tensor), (19, 19)))
            return self.head(features.mean(dim=(2, 3)))
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = Tiny().eval()
    for i, p in enumerate(model.parameters()):
        p.requires_grad_(not frozen)
        if i % 2:
            p.grad = torch.ones_like(p)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    grad_refs = [p.grad for p in model.parameters()]
    grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    flags = [(m, m.training) for m in model.modules()]
    requires_grad = [p.requires_grad for p in model.parameters()]
    original_image = image.copy()
    real_cam = stable.generate_gradcam
    calls = []
    def capture(copy, tensor, target_class=None):
        assert copy is not model
        result = real_cam(copy, tensor, target_class)
        calls.append((target_class, result.predicted_class))
        return result
    monkeypatch.setattr(stable, "generate_gradcam", capture)
    result = stable.compare_with_model(model, image, stable.TransformSpec("horizontal_flip"), 42)
    assert calls[0][0] is None and calls[1][0] == calls[0][1]
    assert result.target_class == calls[0][1]
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    for p, ref, grad in zip(model.parameters(), grad_refs, grads):
        assert p.grad is ref
        if grad is not None:
            assert torch.equal(p.grad, grad)
    assert all(m.training == flag for m, flag in flags)
    assert [p.requires_grad for p in model.parameters()] == requires_grad
    np.testing.assert_array_equal(image, original_image)
    model.bn.train()
    with pytest.raises(ValueError, match="evaluation"):
        stable.compare_with_model(model, image, stable.TransformSpec("identity"), 42)


def test_invalid_inputs_specs_and_no_mutation(image, cam):
    before_image, before_cam = image.copy(), cam.copy()
    transform = stable.apply_transform(image, stable.TransformSpec("rotation", 2), 42)
    before_support = transform.valid_support.copy()
    stable.inverse_align(cam, transform)
    stable.compare_explanations(explanation(cam), explanation(cam), transform, 42)
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(cam, before_cam)
    np.testing.assert_array_equal(transform.valid_support, before_support)
    for args in (("noise", 0.02, 2000), ("noise", 0.01, 1999), ("rotation", True),
                 ("brightness", 1.1), ("unknown",), ("identity", 1)):
        with pytest.raises(ValueError):
            stable.TransformSpec(*args)
    for invalid in (image[None], image * 5, np.full_like(image, np.nan)):
        with pytest.raises(ValueError):
            stable.apply_transform(invalid, stable.TransformSpec("identity"), 42)
    for invalid_index in (-1, 792, True, "42"):
        with pytest.raises(ValueError):
            stable.apply_transform(image, stable.TransformSpec("identity"), invalid_index)
    with pytest.raises(ValueError, match="different dataset"):
        stable.compare_explanations(explanation(cam), explanation(cam), transform, 43)
    with pytest.raises(ValueError, match="nonnegative"):
        stable.topk_mask(-cam, np.ones_like(cam, dtype=bool), 42)
    with pytest.raises(ValueError, match="boolean"):
        stable.topk_mask(cam, np.ones_like(cam), 42)
    with pytest.raises(ValueError, match="Top fraction"):
        stable.topk_mask(cam, np.ones_like(cam, dtype=bool), 42, 0.15)
    with pytest.raises(ValueError, match="150x150"):
        stable.topk_mask(cam[:19, :19], np.ones_like(cam, dtype=bool), 42)


def test_malformed_explanation_and_transform_metadata_rejected(image, cam):
    transform = stable.apply_transform(image, stable.TransformSpec("identity"), 42)
    original = explanation(cam)
    for invalid in (replace(original, predicted_class=1), replace(original, target_class=True),
                    replace(original, layer_name="other"), replace(original, logits=torch.tensor([[float("nan"), 0.]]))):
        with pytest.raises(ValueError):
            stable.compare_explanations(original, invalid, transform, 42)
    for invalid in (replace(transform, dataset_index=42.0), replace(transform, spec=None),
                    replace(transform, clipped_fraction=float("nan")), replace(transform, clipped_fraction=2)):
        with pytest.raises(ValueError):
            stable.compare_explanations(original, original, invalid, 42)
    with pytest.raises(ValueError, match="Non-finite"):
        stable.compare_explanations(explanation(cam, (1e308, 0.)), explanation(cam, (0., 1e308)), transform, 42)
