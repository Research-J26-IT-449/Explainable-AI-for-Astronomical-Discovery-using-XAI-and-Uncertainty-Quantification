"""Synthetic-only checks of primary AOPC and secondary full-deletion curves."""

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

from src.xai import faithfulness as faith


@pytest.fixture
def image():
    return np.linspace(0, 1, 22500, dtype=np.float32).reshape(150, 150)


@pytest.fixture
def cam():
    return np.arange(361, dtype=np.float32).reshape(19, 19)


def test_core_constants_agree_with_frozen_configuration():
    config = yaml.safe_load(Path("config/xai_phase3b_v1.0.0.yaml").read_text())
    assert faith.PROTOCOL == config["protocol"]
    assert faith.FRACTIONS == tuple(config["faithfulness"]["fractions"][1:7])
    assert faith.RANDOM_SEEDS == tuple(config["faithfulness"]["random_seeds"])
    assert len(faith.RANDOM_SEEDS) == 20


def test_exact_tiling_full_coverage_and_row_major_order():
    tiles = faith.tile_pixels()
    assert len(tiles) == 361
    counts = np.zeros(22500, dtype=int)
    for tile_id, pixels in enumerate(tiles):
        r, c = divmod(tile_id, 19)
        expected = [150 * y + x for y in range(150*r//19, 150*(r+1)//19)
                    for x in range(150*c//19, 150*(c+1)//19)]
        assert pixels.tolist() == expected
        counts[pixels] += 1
    assert np.all(counts == 1)
    tiles[0][0] = -1
    assert faith.tile_pixels()[0][0] == 0  # Returned arrays do not share cached state.


def test_exact_budgets_nested_masks_and_partial_tiles():
    order = tuple(reversed(range(361)))
    masks = faith.deletion_masks(order)
    expected_budgets = [1125, 2250, 3375, 4500, 5625, 6750]
    assert masks.shape == (6, 150, 150) and masks.dtype == bool
    assert masks.sum(axis=(1, 2)).tolist() == expected_budgets
    assert np.all(~masks[:-1] | masks[1:])
    full_order = np.concatenate([faith.tile_pixels()[i] for i in order])
    for mask, count in zip(masks, expected_budgets):
        assert np.array_equal(np.flatnonzero(mask), np.sort(full_order[:count]))


def test_native_cam_ranking_uses_sha_ties_exactly():
    cam = np.zeros((19, 19), dtype=np.float32)
    cam[2, 3] = 1
    index = 42
    tied = sorted((i for i in range(361) if i != 41), key=lambda i: (
        hashlib.sha256(f"Phase3B-XAI-v1.0.0|tile-tie|42|42|{i}".encode()).hexdigest(), i))
    expected = (41, *tied)
    assert faith.rank_gradcam(cam, index) == expected
    assert faith.rank_gradcam(cam, index) == faith.rank_gradcam(cam.copy(), index)
    assert faith.rank_gradcam(cam, 43) != expected


def test_brightness_ranks_tile_mean_not_sum():
    image = np.zeros((150, 150), dtype=np.float32)
    tiles = faith.tile_pixels()
    small = next(i for i, p in enumerate(tiles) if len(p) == 49)
    large = next(i for i, p in enumerate(tiles) if len(p) == 64)
    image.ravel()[tiles[small]] = 0.6
    image.ravel()[tiles[large]] = 0.5
    assert image.ravel()[tiles[small]].sum() < image.ravel()[tiles[large]].sum()
    assert faith.rank_brightness(image, 42)[:2] == (small, large)


def test_exact_twenty_random_orders_and_no_global_rng_changes():
    np.random.seed(7)
    state = np.random.get_state()
    orders = faith.random_rankings(42)
    assert tuple(orders) == tuple(range(1000, 1020))
    assert len(set(orders.values())) == 20
    for seed, order in orders.items():
        expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 42]))).permutation(361)
        assert tuple(expected) == order
        assert set(order) == set(range(361))
    assert orders == faith.random_rankings(42)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_replacements_border_union_black_and_gaussian(image):
    border_values = np.concatenate((image[:5].ravel(), image[-5:].ravel(),
                                    image[5:-5, :5].ravel(), image[5:-5, -5:].ravel()))
    assert len(border_values) == 2900
    assert np.all(faith.replacement_image(image, "border_median") == np.median(border_values))
    assert np.count_nonzero(faith.replacement_image(image, "black")) == 0
    # Independent 2-D convolution reference verifies sigma, radius, padding,
    # kernel normalization and separable implementation without using its helper.
    distances = torch.arange(-9, 10, dtype=torch.float64)
    one = torch.exp(-distances.square() / 18)
    one /= one.sum()
    kernel = torch.outer(one, one)[None, None]
    expected = F.conv2d(F.pad(torch.from_numpy(image.astype(np.float64))[None, None],
                             (9, 9, 9, 9), mode="reflect"), kernel)[0, 0]
    np.testing.assert_allclose(faith.replacement_image(image, "gaussian"), expected.numpy(), atol=6e-8, rtol=1e-6)
    for policy in faith.REPLACEMENTS:
        constant = np.full((150, 150), 0.25, dtype=np.float32)
        expected = 0 if policy == "black" else 0.25
        np.testing.assert_allclose(faith.replacement_image(constant, policy), expected, atol=1e-7)


def test_perturbation_and_normalization_no_mutation(image):
    original = image.copy()
    mask = faith.deletion_masks(tuple(range(361)))[0]
    mask_before = mask.copy()
    replacement = faith.replacement_image(image, "black")
    changed = faith.perturb_image(image, mask, replacement)
    assert np.array_equal(changed[~mask], image[~mask])
    assert np.all(changed[mask] == 0)
    normalized = faith.normalize_intensity(changed)
    assert normalized.shape == (1, 1, 150, 150) and normalized.dtype == torch.float32
    np.testing.assert_array_equal(normalized[0, 0].numpy(), (changed - 0.5) / 0.5)
    assert torch.all(normalized[0, 0][torch.from_numpy(mask)] == -1)
    assert np.array_equal(image, original) and np.array_equal(mask, mask_before)
    assert not np.shares_memory(image, changed)


def test_analytical_scores_and_six_step_aopc():
    original = (2., 0.)
    deleted = tuple((2. - step * 0.25, 0.) for step in range(1, 7))
    curve = faith.DeletionCurve(0, original, deleted)
    result = faith.aopc(curve)
    expected_probability = sum(1/(1+math.exp(-2)) - 1/(1+math.exp(-z[0])) for z in deleted) / 6
    assert result.probability == pytest.approx(expected_probability)
    assert result.logit == pytest.approx(0.875)
    assert result.margin == pytest.approx(0.875)
    # Also verify FRII targeting and margin subtraction with both logits changing.
    other = faith.DeletionCurve(1, (0., 2.), tuple((step*0.1, 2.-step*0.25) for step in range(1, 7)))
    assert faith.aopc(other).margin == pytest.approx(1.225)


def test_negative_aopc_is_retained():
    curve = faith.DeletionCurve(0, (1., 0.), ((2., 0.),) * 6)
    result = faith.aopc(curve)
    assert result.probability < 0 and result.logit == -1 and result.margin == -1


def test_paired_contrasts_equal_weight_twenty_controls():
    cam = faith.Scores(0.4, 2., 3.)
    bright = faith.Scores(0.5, 1., 1.)
    random = {seed: faith.Scores(k/100, k/10, k/5) for k, seed in enumerate(range(1000, 1020))}
    result = faith.paired_contrasts(cam, bright, random)
    assert result.random_mean.probability == pytest.approx(0.095)
    assert result.cam_minus_random.probability == pytest.approx(0.305)
    assert result.cam_minus_brightness.probability == pytest.approx(-0.1)
    assert result.cam_minus_brightness.logit == 1 and result.cam_minus_brightness.margin == 2
    with pytest.raises(ValueError, match="20 random"):
        faith.paired_contrasts(cam, bright, {s: v for s, v in random.items() if s != 1000})


def test_evaluation_freezes_original_target_through_class_flips(cam, monkeypatch):
    image = np.full((150, 150), 0.6, dtype=np.float32)
    before = image.copy()
    rankings = faith.prepare_rankings(image, cam, 42)
    # Any recomputation during scoring or a second replacement must fail the test.
    for name in ("rank_gradcam", "rank_brightness", "random_rankings"):
        monkeypatch.setattr(faith, name, lambda *a, **kw: pytest.fail("Ranking recomputed"))
    calls = []
    def scorer(normalized):
        assert not torch.is_grad_enabled()
        value = float(normalized.mean())
        calls.append(value)
        return (value, 0.)
    result = faith.evaluate_rankings(image, rankings, scorer, replacement="black")
    assert len(calls) == 133
    assert result.rankings is rankings and len(result.random) == 20
    assert result.gradcam.target_class == 0
    assert all(curve.target_class == 0 for _, curve in result.random)
    assert any(np.argmax(z) == 1 for z in result.gradcam.deleted_logits)
    assert faith.aopc(result.gradcam).logit == pytest.approx(0.21, abs=1e-6)
    assert result.contrasts.cam_minus_random.probability == pytest.approx(0, abs=1e-7)
    assert result.contrasts.cam_minus_brightness.probability == pytest.approx(0, abs=1e-7)
    second = faith.evaluate_rankings(image, rankings, scorer, replacement="border_median")
    assert faith.aopc(second.gradcam).logit == pytest.approx(0, abs=1e-7)
    assert np.array_equal(image, before)


def test_model_state_bn_gradients_and_mode_preserved(image, cam):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(1)
            self.head = nn.Linear(1, 2)
        def forward(self, tensor):
            return self.head(self.bn(tensor).mean(dim=(2, 3)))
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = TinyModel().eval()
    for i, p in enumerate(model.parameters()):
        if i % 2:
            p.grad = torch.ones_like(p)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    grads = [(p.grad, None if p.grad is None else p.grad.clone()) for p in model.parameters()]
    flags = [(m, m.training) for m in model.modules()]
    requires = [p.requires_grad for p in model.parameters()]
    normalized = faith.normalize_intensity(image)
    with torch.no_grad():
        expected_logits = model(normalized).clone()
    assert torch.equal(faith.ModelScorer(model)(normalized), expected_logits)
    rankings = faith.prepare_rankings(image, cam, 42)
    result = faith.evaluate_rankings(image, rankings, model)
    assert np.isfinite(faith.aopc(result.gradcam).probability)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    for p, (ref, value) in zip(model.parameters(), grads):
        assert p.grad is ref
        if value is not None:
            assert torch.equal(p.grad, value)
    assert [p.requires_grad for p in model.parameters()] == requires
    assert all(m.training == flag for m, flag in flags)
    model.bn.train()
    with pytest.raises(ValueError, match="evaluation"):
        faith.ModelScorer(model)


@pytest.mark.parametrize("value", [0., 1., 1e-10])
def test_degenerate_cam_rejected(value):
    with pytest.raises(ValueError, match="Degenerate"):
        faith.rank_gradcam(np.full((19, 19), value), 42)


def test_explicit_core_degenerate_status_rejected(cam):
    with pytest.raises(ValueError, match="Degenerate"):
        faith.rank_gradcam(cam, 42, status="degenerate")


@pytest.mark.parametrize("index", [-1, 792, True, "42", 42.0])
def test_invalid_indices(index):
    with pytest.raises(ValueError, match="dataset_index"):
        faith.random_rankings(index)


@pytest.mark.parametrize("target", [-1, 2, True, "FRI", 0.0])
def test_invalid_classes(target):
    with pytest.raises(ValueError, match="Target"):
        faith.target_scores((2., 0.), target)


@pytest.mark.parametrize("change", ["short", "long", "fractions", "wrong_target", "nonfinite"])
def test_invalid_or_incomplete_curves(change):
    curve = faith.DeletionCurve(0, (2., 0.), ((1., 0.),) * 6)
    if change == "short":
        curve = replace(curve, deleted_logits=curve.deleted_logits[:5])
    elif change == "long":
        curve = replace(curve, deleted_logits=curve.deleted_logits + ((1., 0.),))
    elif change == "fractions":
        curve = replace(curve, fractions=(0.05,) * 6)
    elif change == "wrong_target":
        curve = replace(curve, target_class=1)
    else:
        curve = replace(curve, deleted_logits=((float("nan"), 0.),) * 6)
    with pytest.raises(ValueError):
        faith.aopc(curve)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_arrays_and_scores_rejected(image, cam, value):
    image[0, 0] = value
    cam[0, 0] = value
    for operation in (lambda: faith.rank_gradcam(cam, 42),
                      lambda: faith.rank_brightness(image, 42),
                      lambda: faith.normalize_intensity(image),
                      lambda: faith.target_scores((value, 0.), 0)):
        with pytest.raises(ValueError, match="finite"):
            operation()


def test_invalid_dimensions_ranges_rankings_and_masks(image, cam):
    for invalid in (np.zeros((19, 18)), np.zeros((150, 150)), -cam):
        with pytest.raises(ValueError):
            faith.rank_gradcam(invalid, 42)
    for invalid in (image[None], image * 2, image - 1, np.zeros((150, 150), dtype=bool)):
        with pytest.raises(ValueError):
            faith.normalize_intensity(invalid)
    for invalid in (tuple(range(360)), (0,) * 361, tuple(float(i) for i in range(361))):
        with pytest.raises(ValueError, match="permutation"):
            faith.deletion_masks(invalid)
    with pytest.raises(ValueError, match="mask"):
        faith.perturb_image(image, np.ones((150, 150)), image)
    with pytest.raises(ValueError, match="policy"):
        faith.replacement_image(image, "unknown")
    for invalid in ([1.], [1., 2., 3.], ["bad", "data"]):
        with pytest.raises(ValueError):
            faith.target_scores(invalid, 0)


def test_ranking_operations_do_not_mutate_original_arrays(image, cam):
    before_image, before_cam = image.copy(), cam.copy()
    faith.prepare_rankings(image, cam, 42)
    assert np.array_equal(image, before_image)
    assert np.array_equal(cam, before_cam)


def test_nonfinite_scorer_fails_without_partial_results(image, cam):
    rankings = faith.prepare_rankings(image, cam, 42)
    count = 0
    def scorer(tensor):
        nonlocal count
        count += 1
        return (2., 0.) if count < 3 else (float("nan"), 0.)
    with pytest.raises(ValueError, match="finite"):
        faith.evaluate_rankings(image, rankings, scorer)
    assert count == 3


def test_secondary_full_grid_and_integer_budgets_match_frozen_yaml():
    config = yaml.safe_load(Path("config/xai_phase3b_v1.0.0.yaml").read_text())
    assert faith.FULL_FRACTIONS == tuple(config["faithfulness"]["fractions"])
    assert len(faith.FULL_FRACTIONS) == 21
    assert faith.FULL_BUDGETS == tuple(1125*k for k in range(21))
    assert faith.FULL_FRACTIONS[0] == 0 and faith.FULL_FRACTIONS[-1] == 1
    assert faith.FULL_FRACTIONS[1:7] == faith.FRACTIONS
    assert faith.FULL_BUDGETS[1:7] == faith.BUDGETS


@pytest.mark.parametrize("order", [tuple(range(361)), tuple(reversed(range(361)))])
def test_secondary_masks_exact_nested_with_identical_primary_subset(order):
    masks = faith.full_deletion_masks(order)
    assert masks.shape == (21, 150, 150) and masks.dtype == bool
    assert tuple(masks.sum(axis=(1, 2))) == faith.FULL_BUDGETS
    assert not masks[0].any() and masks[-1].all()
    assert np.all(~masks[:-1] | masks[1:])
    np.testing.assert_array_equal(masks[1:7], faith.deletion_masks(order))
    pixels = np.concatenate([faith.tile_pixels()[i] for i in order])
    for mask, budget in zip(masks, faith.FULL_BUDGETS):
        np.testing.assert_array_equal(np.flatnonzero(mask), np.sort(pixels[:budget]))
    masks[-1, 0, 0] = False
    assert faith.full_deletion_masks(order)[-1].all()


def test_secondary_analytical_constant_score_auc():
    curve = faith.FullDeletionCurve(0, (2., -1.), ((2., -1.),)*21)
    auc = faith.deletion_auc(curve)
    assert auc.probability == pytest.approx(1/(1+math.exp(-3)))
    assert auc.logit == pytest.approx(2)
    assert auc.margin == pytest.approx(3)
    assert all(effect == faith.Scores(0., 0., 0.) for effect in faith.full_curve_effects(curve))


def test_secondary_auc_linear_probability_and_nonuniform_logit_diagnostics():
    probabilities = tuple(.8-.6*f for f in faith.FULL_FRACTIONS)
    logits = tuple((math.log(p/(1-p)), 0.) for p in probabilities)
    curve = faith.FullDeletionCurve(0, logits[0], logits)
    auc = faith.deletion_auc(curve)
    assert auc.probability == pytest.approx(.5)
    assert auc.logit == pytest.approx(0, abs=1e-14)
    assert auc.margin == pytest.approx(0, abs=1e-14)
    assert faith.full_curve_scores(curve)[-1].probability == pytest.approx(.2)


def test_secondary_trapezoidal_rule_not_rectangle_or_continuous_integral():
    probabilities = tuple(.8-.6*f*f for f in faith.FULL_FRACTIONS)
    logits = tuple((math.log(p/(1-p)), 0.) for p in probabilities)
    result = faith.deletion_auc(faith.FullDeletionCurve(0, logits[0], logits))
    # Composite trapezoid integrates f^2 on this discrete grid to .33375.
    assert result.probability == pytest.approx(.8-.6*.33375)
    assert result.probability != pytest.approx(.6, abs=1e-6)
    manual = .05*(.5*probabilities[0] + sum(probabilities[1:-1]) + .5*probabilities[-1])
    assert result.probability == pytest.approx(manual)
    assert result.probability != pytest.approx(.05*sum(probabilities[:-1]), abs=1e-6)


def test_secondary_signed_effects_and_negative_logit_auc_retained():
    logits = tuple((1.-4*f, 0.) for f in faith.FULL_FRACTIONS)
    curve = faith.FullDeletionCurve(0, logits[0], logits)
    assert faith.deletion_auc(curve).logit == pytest.approx(-1)
    assert faith.deletion_auc(curve).margin == pytest.approx(-1)
    increasing = tuple((1.+2*f, 0.) for f in faith.FULL_FRACTIONS)
    changed = faith.FullDeletionCurve(0, increasing[0], increasing)
    effects = faith.full_curve_effects(changed)
    assert effects[0] == faith.Scores(0., 0., 0.)
    assert effects[-1].probability < 0
    assert effects[-1].logit == effects[-1].margin == -2
    # AUC integrates remaining score, not original-minus-deleted effects.
    assert faith.deletion_auc(changed).logit == pytest.approx(2)


@pytest.mark.parametrize("target", [0, 1])
def test_secondary_target_stays_fixed_through_prediction_flips(target):
    logits = tuple((2.-4*f, 0.) if target == 0 else (0., 2.-4*f)
                   for f in faith.FULL_FRACTIONS)
    curve = faith.FullDeletionCurve(target, logits[0], logits)
    scores = faith.full_curve_scores(curve)
    assert np.argmax(logits[-1]) == 1-target
    assert scores[-1].logit == -2
    assert scores[-1].margin == -2
    assert scores[-1].probability == pytest.approx(1/(1+math.exp(2)))


@pytest.mark.parametrize("policy", faith.REPLACEMENTS)
def test_secondary_evaluation_endpoints_replacements_rank_once(image, cam, policy, monkeypatch):
    original = image.copy()
    rankings = faith.prepare_rankings(image, cam, 42)
    expected_replacement = faith.normalize_intensity(faith.replacement_image(image, policy))
    expected_original = faith.normalize_intensity(image)
    endpoint_count = 0
    calls = 0

    def scorer(tensor):
        nonlocal calls, endpoint_count
        calls += 1
        assert not torch.is_grad_enabled()
        if calls == 1:
            assert torch.equal(tensor, expected_original)
        elif (calls-1) % 20 == 0:
            endpoint_count += 1
            assert torch.equal(tensor, expected_replacement)
        return (float(tensor.mean())+1, 0.)

    for name in ("rank_gradcam", "rank_brightness", "random_rankings"):
        monkeypatch.setattr(faith, name, lambda *a, **kw: pytest.fail("Rankings recomputed"))
    result = faith.evaluate_full_rankings(image, rankings, scorer, replacement=policy)
    assert calls == 1+22*20 and endpoint_count == 22
    assert result.rankings is rankings and result.replacement == policy
    assert tuple(s for s, _ in result.random) == faith.RANDOM_SEEDS
    assert tuple(s for s, _ in result.random_auc) == faith.RANDOM_SEEDS
    curves = (result.gradcam, result.brightness, *(c for _, c in result.random))
    for curve in curves:
        assert curve.target_class == result.gradcam.target_class
        assert curve.logits[0] == curve.original_logits == result.gradcam.original_logits
        assert curve.logits[-1] == result.gradcam.logits[-1]
        assert len(faith.full_curve_scores(curve)) == 21
    assert result.gradcam_auc == faith.deletion_auc(result.gradcam)
    np.testing.assert_array_equal(image, original)


def test_secondary_random_curves_reproducible_and_primary_unchanged(image, cam):
    rankings = faith.prepare_rankings(image, cam, 42)
    def scorer(tensor):
        # Spatial dependence differentiates rankings, unlike a uniform fixture.
        weights = torch.linspace(0, 1, 150).view(1, 1, 1, 150)
        return (float((tensor*weights).mean())+1, 0.)
    full = faith.evaluate_full_rankings(image, rankings, scorer, replacement="black")
    assert full == faith.evaluate_full_rankings(image, rankings, scorer, replacement="black")
    primary = faith.evaluate_rankings(image, rankings, scorer, replacement="black")
    for complete, old in zip((full.gradcam, full.brightness, *(c for _, c in full.random)),
                             (primary.gradcam, primary.brightness, *(c for _, c in primary.random))):
        assert complete.logits[1:7] == old.deleted_logits
        subset = faith.DeletionCurve(complete.target_class, complete.original_logits, complete.logits[1:7])
        assert faith.aopc(subset) == faith.aopc(old)
    assert len(primary.gradcam.deleted_logits) == 6
    assert primary.gradcam.fractions == faith.FRACTIONS


def test_secondary_model_state_buffers_gradients_and_input_preserved(image, cam):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(1)
            self.head = nn.Linear(1, 2)
        def forward(self, tensor):
            return self.head(self.bn(tensor).mean(dim=(2, 3)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = TinyModel().eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = torch.full_like(p, .5)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    grads = [(p.grad, p.grad.clone()) for p in model.parameters()]
    before = image.copy()
    result = faith.evaluate_full_rankings(image, faith.prepare_rankings(image, cam, 42), model)
    assert np.isfinite(result.gradcam_auc.probability)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert not any(m.training for m in model.modules())
    for p, (reference, value) in zip(model.parameters(), grads):
        assert not p.requires_grad and p.grad is reference and torch.equal(p.grad, value)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("change", ["short", "long", "grid", "target", "zero_endpoint", "nan", "inf", "missing"])
def test_secondary_invalid_incomplete_curves_rejected(change):
    curve = faith.FullDeletionCurve(0, (2., 0.), ((2., 0.),)*21)
    if change == "short":
        curve = replace(curve, logits=curve.logits[:-1])
    elif change == "long":
        curve = replace(curve, logits=curve.logits+((2., 0.),))
    elif change == "grid":
        curve = replace(curve, fractions=(0.,)*21)
    elif change == "target":
        curve = replace(curve, target_class=1)
    elif change == "zero_endpoint":
        curve = replace(curve, logits=((1., 0.),)+curve.logits[1:])
    else:
        invalid = (float("nan"), 0.) if change == "nan" else (float("inf"), 0.) if change == "inf" else None
        curve = replace(curve, logits=curve.logits[:-1]+(invalid,))
    for operation in (faith.full_curve_scores, faith.full_curve_effects, faith.deletion_auc):
        with pytest.raises(ValueError):
            operation(curve)


def test_secondary_nonfinite_scorer_aborts_without_partial_outputs(image, cam):
    calls = 0
    def scorer(tensor):
        nonlocal calls
        calls += 1
        return (2., 0.) if calls < 4 else (np.nan, 0.)
    with pytest.raises(ValueError, match="finite"):
        faith.evaluate_full_rankings(image, faith.prepare_rankings(image, cam, 42), scorer)
    assert calls == 4


def test_secondary_degenerate_cam_rejected_before_scoring(image):
    called = False
    def scorer(tensor):
        nonlocal called
        called = True
        return (2., 0.)
    with pytest.raises(ValueError, match="Degenerate"):
        faith.evaluate_full_rankings(image, faith.prepare_rankings(image, np.zeros((19, 19)), 42), scorer)
    assert not called


def test_secondary_invalid_rankings_rejected_before_scoring(image, cam):
    rankings = faith.prepare_rankings(image, cam, 42)
    for invalid in (replace(rankings, random=rankings.random[:-1]),
                    replace(rankings, gradcam=(0,)*361)):
        with pytest.raises(ValueError):
            faith.evaluate_full_rankings(image, invalid, lambda _: pytest.fail("Invalid ranking scored"))
    with pytest.raises(ValueError):
        faith.full_deletion_masks(tuple(range(360)))
