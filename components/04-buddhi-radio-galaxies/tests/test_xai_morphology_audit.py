"""Synthetic intensity-proxy tests, never verified astrophysical annotations."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from src.xai import morphology_audit as audit


@pytest.fixture
def image():
    image = np.zeros((150, 150), dtype=np.float64)
    image[50:100, 50:100] = 1
    return image


@pytest.fixture
def emission_mask():
    mask = np.zeros((150, 150), dtype=bool)
    mask[50:100, 50:100] = True
    return mask


def test_variant_constants_and_optional_random_policy_match_frozen_config():
    config = yaml.safe_load(Path("config/xai_phase3b_v1.0.0.yaml").read_text())
    proxy = config["emission_proxy"]
    assert audit.VARIANTS == (("inclusive", 2., 0.02), ("primary", 3., 0.05), ("restrictive", 5., 0.1))
    assert proxy["primary"] == {"sigma_factor": 3, "dynamic_fraction": 0.05}
    assert proxy["sensitivities"] == [{"sigma_factor": 2, "dynamic_fraction": 0.02},
                                       {"sigma_factor": 5, "dynamic_fraction": 0.1}]
    assert audit.RANDOM_SEEDS == tuple(config["faithfulness"]["random_seeds"])
    assert "uniform native maps" in proxy["random_controls"]
    assert "SeedSequence([seed,dataset_index,1])" in proxy["random_controls"]
    random = audit.random_control_availability()
    assert random.status == "unavailable"
    assert random.reason == "random_native_to_input_projection_unspecified"
    assert len(random.seeds) == 20


def test_border_is_exact_2900_pixel_union_without_corner_duplication():
    image = np.arange(22500, dtype=np.float64).reshape(150, 150) / 22500
    expected = np.array([image[y, x] for y in range(150) for x in range(150)
                         if y < 5 or y >= 145 or x < 5 or x >= 145])
    actual = audit.border_values(image)
    assert actual.shape == (2900,)
    np.testing.assert_array_equal(actual, expected)
    assert len(set(actual)) == 2900


def test_median_mad_and_all_threshold_components_are_exact():
    image = np.full((150, 150), 0.1, dtype=np.float64)
    border = np.ones_like(image, dtype=bool)
    border[5:-5, 5:-5] = False
    # Exact symmetric metadata-like intensity fixture: median .1, MAD .02.
    image[border] = np.tile([0.08, 0.12], 1450)
    image[75, 75] = 1
    proxies = audit.emission_proxies(image)
    for proxy, (name, factor, dynamic_fraction) in zip(proxies, audit.VARIANTS):
        assert proxy.variant == name
        assert proxy.background_median == pytest.approx(0.1)
        assert proxy.border_mad == pytest.approx(0.02)
        assert proxy.sigma_b == pytest.approx(1.4826 * 0.02)
        assert proxy.sigma_component == pytest.approx(factor * 1.4826 * 0.02)
        assert proxy.dynamic_component == pytest.approx(dynamic_fraction * 0.9)
        expected = max(factor * proxy.sigma_b, dynamic_fraction * 0.9, 1/255)
        assert proxy.threshold_offset == pytest.approx(expected)
        assert proxy.absolute_threshold == pytest.approx(0.1 + expected)
        np.testing.assert_array_equal(proxy.emission_mask, image > proxy.absolute_threshold)
        assert not proxy.zero_border_dispersion


def test_quantization_floor_strict_threshold_and_zero_dispersion():
    image = np.full((150, 150), 0.2, dtype=np.float64)
    image[70, 70] = 0.2 + 1/255
    image[71, 71] = np.nextafter(image[70, 70], np.inf)
    for proxy in audit.emission_proxies(image):
        assert proxy.threshold_offset == 1/255
        assert proxy.zero_border_dispersion
        assert not proxy.emission_mask[70, 70]  # Strict >, never >=.
        assert proxy.emission_mask[71, 71]
        assert proxy.status == "valid"
        assert proxy.emission_pixels == 1


def test_dynamic_floor_and_nested_sensitivity_masks():
    image = np.zeros((150, 150), dtype=np.float64)
    image[75, 75] = 1
    image[75, 76] = 0.03
    image[75, 77] = 0.07
    proxies = audit.emission_proxies(image)
    inclusive, primary, restrictive = proxies
    assert [p.threshold_offset for p in proxies] == [0.02, 0.05, 0.1]
    assert [p.emission_pixels for p in proxies] == [3, 2, 1]
    assert np.all(~restrictive.emission_mask | primary.emission_mask)
    assert np.all(~primary.emission_mask | inclusive.emission_mask)


def test_masks_are_independent_of_cam_and_no_automatic_repairs(image, monkeypatch):
    monkeypatch.setattr(audit, "topk_mask", lambda *a, **kw: pytest.fail("CAM used to build emission mask"))
    proxies = audit.emission_proxies(image)
    assert all(p.emission_pixels == 2500 for p in proxies)
    assert all(not p.border_touching_emission for p in proxies)


def test_attribution_arithmetic_enrichment_and_exact_iou():
    mask = np.zeros((150, 150), dtype=bool)
    mask.ravel()[:3000] = True
    values = np.arange(22500, 0, -1, dtype=np.float64).reshape(150, 150)
    result = audit.attribution_overlap(values, mask, 42)
    expected = values[mask].sum() / values.sum()
    assert result.emission_attribution_mass == pytest.approx(expected)
    assert result.background_attribution_mass == pytest.approx(1-expected)
    assert result.emission_area_fraction == 3000/22500
    assert result.area_adjusted_enrichment == pytest.approx(expected - 3000/22500)
    assert result.top10_emission_iou == 2250/3000
    assert result.top10_pixels == 2250
    assert result.status == result.top10_status == "valid"


@pytest.mark.parametrize("inside", [True, False])
def test_fully_inside_and_outside_attribution(emission_mask, inside):
    values = np.zeros((150, 150), dtype=np.float64)
    values[75 if inside else 10, 75 if inside else 10] = 3
    result = audit.attribution_overlap(values, emission_mask, 42)
    assert result.emission_attribution_mass == (1 if inside else 0)
    assert result.background_attribution_mass == (0 if inside else 1)
    assert result.area_adjusted_enrichment == (1 if inside else 0)-2500/22500


def test_uniform_control_zero_enrichment_but_no_fabricated_iou(emission_mask):
    result = audit.attribution_overlap(np.ones((150, 150)), emission_mask, 42, attribution_kind="uniform")
    assert result.emission_attribution_mass == result.emission_area_fraction
    assert result.area_adjusted_enrichment == 0
    assert result.status == "valid"
    assert result.top10_emission_iou is None
    assert result.top10_status == "unavailable"
    assert result.top10_reasons == ("constant_control_has_no_spatial_ranking",)


def test_brightness_baseline_uses_background_subtraction_not_cam():
    image = np.full((150, 150), 0.2)
    image[50:80, 50:80] = 0.8
    image[80:100, 50:80] = 0.1
    expected = np.maximum(image-0.2, 0)
    np.testing.assert_array_equal(audit.brightness_attribution(image), expected)
    result = audit.audit_emission_alignment(image, expected, 42)
    assert len(result.variants) == 3
    for variant in result.variants:
        assert variant.brightness.emission_attribution_mass == pytest.approx(1)
        assert variant.uniform.area_adjusted_enrichment == 0
        assert variant.random.status == "unavailable"
        assert "Intensity-derived" in variant.proxy.notice


def test_topk_ties_reuse_exact_original_coordinate_sha_policy():
    cam = np.zeros((150, 150), dtype=np.float64)
    cam.ravel()[123] = 1
    tied = sorted((i for i in range(22500) if i != 123), key=lambda i: (
        hashlib.sha256(f"Phase3B-XAI-v1.0.0|pixel-tie|42|42|{i}".encode()).hexdigest(), i))
    expected_mask = np.zeros((150, 150), dtype=bool)
    expected_mask.ravel()[[123, *tied[:2249]]] = True
    first = audit.attribution_overlap(cam, expected_mask, 42)
    second = audit.attribution_overlap(cam, expected_mask, 42)
    assert first.top10_emission_iou == second.top10_emission_iou == 1
    assert audit.attribution_overlap(cam, expected_mask, 43).top10_emission_iou < 1


@pytest.mark.parametrize("full", [False, True])
def test_empty_and_full_masks_have_unavailable_metrics(full):
    mask = np.full((150, 150), full, dtype=bool)
    cam = np.arange(22500, dtype=np.float64).reshape(150, 150)
    result = audit.attribution_overlap(cam, mask, 42)
    assert result.emission_area_fraction == int(full)
    assert result.status == "unavailable"
    assert result.reasons == (("full_emission_mask",) if full else ("empty_emission_mask",))
    assert result.emission_attribution_mass is None
    assert result.area_adjusted_enrichment is None and result.top10_emission_iou is None


@pytest.mark.parametrize("value", [0., 0.5, 1.])
def test_constant_images_keep_threshold_diagnostics_and_unavailable_masks(value):
    image = np.full((150, 150), value)
    cam = np.arange(22500).reshape(150, 150)
    result = audit.audit_emission_alignment(image, cam, 42)
    for variant in result.variants:
        proxy = variant.proxy
        assert proxy.background_median == value
        assert proxy.sigma_b == 0 and proxy.zero_border_dispersion
        assert proxy.quantization_floor == proxy.threshold_offset == 1/255
        assert proxy.emission_area_fraction == 0
        assert proxy.reasons == ("empty_emission_mask",)
        assert variant.gradcam.status == variant.uniform.status == "unavailable"


def test_border_touch_and_border_band_occupancy_are_separate_diagnostics():
    image = np.zeros((150, 150))
    image[0, 70] = 1
    for proxy in audit.emission_proxies(image):
        assert proxy.border_touching_emission and proxy.emission_in_border_band
        assert proxy.status == "valid"
    image[0, 70] = 0
    image[3, 70] = 1
    for proxy in audit.emission_proxies(image):
        assert not proxy.border_touching_emission and proxy.emission_in_border_band
        assert proxy.status == "valid"


@pytest.mark.parametrize("value", [0., 1., 1e-10])
def test_zero_or_constant_cam_unavailable(emission_mask, value):
    result = audit.attribution_overlap(np.full((150, 150), value), emission_mask, 42)
    assert result.status == "unavailable"
    assert "zero_attribution" in result.reasons if value == 0 else "degenerate_gradcam" in result.reasons
    assert result.emission_attribution_mass is None and result.top10_emission_iou is None


def test_declared_degenerate_cam_retains_all_proxy_diagnostics(image):
    cam = np.arange(22500, dtype=np.float64).reshape(150, 150)
    result = audit.audit_emission_alignment(image, cam, 42, cam_status="degenerate")
    for variant in result.variants:
        assert variant.proxy.emission_pixels == 2500
        assert variant.gradcam.reasons == ("degenerate_gradcam",)
        assert variant.uniform.status == variant.brightness.status == "valid"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_arrays_are_explicitly_rejected_or_unavailable(image, value):
    cam = np.ones((150, 150))
    cam[70, 70] = value
    result = audit.audit_emission_alignment(image, cam, 42)
    assert all(v.gradcam.reasons == ("nonfinite_attribution",) for v in result.variants)
    assert all(v.gradcam.status == "unavailable" for v in result.variants)
    image[70, 70] = value
    with pytest.raises(audit.AlignmentInputError) as error:
        audit.emission_proxies(image)
    assert error.value.reason_code == "nonfinite_image"


@pytest.mark.parametrize("invalid", [np.zeros((19, 19)), np.zeros((1, 150, 150)),
                                      np.full((150, 150), -0.01), np.full((150, 150), 1.01),
                                      np.zeros((150, 150), dtype=bool)])
def test_invalid_image_dimensions_dtype_or_range(invalid):
    with pytest.raises(audit.AlignmentInputError):
        audit.emission_proxies(invalid)


def test_invalid_cam_mask_index_and_status(emission_mask):
    for invalid in (np.zeros((19, 19)), -np.ones((150, 150)), np.ones((150, 150), dtype=complex)):
        with pytest.raises(audit.AlignmentInputError):
            audit.attribution_overlap(invalid, emission_mask, 42)
    for invalid_index in (-1, 792, True, "42"):
        with pytest.raises(audit.AlignmentInputError, match="dataset_index"):
            audit.attribution_overlap(np.ones((150, 150)), emission_mask, invalid_index)
    with pytest.raises(audit.AlignmentInputError, match="emission_mask"):
        audit.attribution_overlap(np.ones((150, 150)), emission_mask.astype(float), 42)
    with pytest.raises(audit.AlignmentInputError, match="cam_status"):
        audit.attribution_overlap(np.ones((150, 150)), emission_mask, 42, cam_status="unknown")
    with pytest.raises(audit.AlignmentInputError, match="nonuniform"):
        audit.attribution_overlap(np.arange(22500).reshape(150, 150), emission_mask, 42, attribution_kind="uniform")


def test_inputs_model_state_and_rng_unchanged_no_model_calls_or_files(image, monkeypatch):
    model = nn.Sequential(nn.BatchNorm2d(1), nn.Conv2d(1, 2, 1)).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    refs = [p.grad for p in model.parameters()]
    cam = torch.arange(22500, dtype=torch.float32).reshape(150, 150).requires_grad_(True)
    cam.grad = torch.ones_like(cam)
    original_image, original_cam = image.copy(), cam.detach().clone()
    rng_state = np.random.get_state()
    def forbidden(*a, **kw):
        pytest.fail("Numerical morphology core attempted model/file access")
    monkeypatch.setattr(nn.Module, "__call__", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    result = audit.audit_emission_alignment(image, cam, 42)
    assert len(result.variants) == 3
    np.testing.assert_array_equal(image, original_image)
    assert torch.equal(cam.detach(), original_cam) and torch.all(cam.grad == 1)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is ref and torch.all(p.grad == 1) for p, ref in zip(model.parameters(), refs))
    assert all(not m.training for m in model.modules())
    after = np.random.get_state()
    assert rng_state[0] == after[0] and np.array_equal(rng_state[1], after[1]) and rng_state[2:] == after[2:]
