"""Parameter sanity checks use synthetic arrays/models only, without artifact I/O."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from src.models.resnet_cnn import ResNetCNN
from src.xai import sanity_checks as sanity
from src.xai.gradcam import GradCAMResult, generate_gradcam


class SmallModel(nn.Module):
    """Cheap fixture with the six real module paths and projection shortcuts."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
        self.blocks = nn.ModuleList()
        for _ in range(3):
            block = nn.Module()
            block.convolution = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
            block.shortcut = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
            self.blocks.append(block)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8, 4), nn.ReLU(),
                                        nn.Dropout(.3), nn.Linear(4, 2))


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        result = SmallModel().eval()
    with torch.no_grad():
        for module in result.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.copy_(torch.tensor([.2, .4]))
                module.running_var.copy_(torch.tensor([.8, 1.5]))
                module.num_batches_tracked.fill_(37)
                module.weight.fill_(2)
                module.bias.fill_(.3)
        for parameter in result.parameters():
            parameter.grad = torch.full_like(parameter, .125)
            parameter.requires_grad_(False)
    return result


@pytest.fixture
def cam():
    return np.arange(22500, dtype=np.float32).reshape(150, 150)


def assert_same_state(first, second):
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key


def explanation(cam, *, target=1, prediction=1, status="valid"):
    tensor = torch.as_tensor(cam.copy(), dtype=torch.float32)
    return GradCAMResult(torch.ones(19, 19), tensor, tensor / tensor.max(), target,
                         prediction, torch.tensor([[0., 1.]]),
                         "blocks.2.convolution.3", status)


def test_exact_protocol_seeds_and_stages():
    config = yaml.safe_load(Path("config/xai_phase3b_v1.0.0.yaml").read_text())
    assert sanity.SEEDS == tuple(config["randomization"]["seeds"]) == tuple(range(3000, 3005))
    assert sanity.STAGES == tuple(config["randomization"]["cumulative_order"])
    assert sanity.STAGES == ("classifier.4", "classifier.1", "blocks.2", "blocks.1", "blocks.0", "stem")


def test_exact_cumulative_order_and_unselected_parameters(model):
    original = deepcopy(model.state_dict())
    stages = list(sanity.iter_randomized_models(model, 3000))
    assert len(stages) == 6
    previous = original
    for number, stage in enumerate(stages, 1):
        assert stage.stage == number and stage.seed == 3000
        assert stage.modules_randomized == sanity.STAGES[:number]
        current = stage.model.state_dict()
        group = sanity.STAGES[number - 1]
        for key, value in previous.items():
            if not key.startswith(group + "."):
                assert torch.equal(value, current[key]), key
        assert not torch.equal(previous[group + (".weight" if number < 3 else ".convolution.0.weight" if number < 6 else ".0.weight")],
                               current[group + (".weight" if number < 3 else ".convolution.0.weight" if number < 6 else ".0.weight")])
        previous = current
    assert "blocks.2.shortcut.0" in stages[2].parameter_modules
    assert "blocks.2.shortcut.1" in stages[2].parameter_modules
    # Later randomization cannot mutate previously yielded snapshots.
    assert torch.equal(stages[0].model.stem[0].weight, model.stem[0].weight)
    assert_same_state(model.state_dict(), original)


def test_reproducible_independent_seeds_and_rng_preservation(model):
    rng = torch.get_rng_state().clone()
    first = list(sanity.iter_randomized_models(model, 3000))
    repeated = list(sanity.iter_randomized_models(model, 3000))
    other = list(sanity.iter_randomized_models(model, 3001))
    assert torch.equal(rng, torch.get_rng_state())
    for a, b, c in zip(first, repeated, other):
        assert_same_state(a.model.state_dict(), b.model.state_dict())
        assert not torch.equal(a.model.classifier[4].weight, c.model.classifier[4].weight)
    iterator = sanity.iter_randomized_models(model, 3002)
    next(iterator)
    assert torch.equal(rng, torch.get_rng_state())
    iterator.close()
    assert torch.equal(rng, torch.get_rng_state())


def test_all_five_seeds_have_distinct_initial_weights(model):
    weights = [next(sanity.iter_randomized_models(model, seed)).model.classifier[4].weight
               for seed in sanity.SEEDS]
    assert all(not torch.equal(a, b) for i, a in enumerate(weights) for b in weights[i+1:])


def test_preserve_bn_buffers_affine_gradients_and_mode(model):
    snapshot = sanity._snapshot(model)
    buffers = sanity._bn_buffers(model)
    with patch.object(nn.BatchNorm2d, "reset_running_stats", side_effect=AssertionError("forbidden")), \
         patch.object(nn.BatchNorm2d, "reset_parameters", side_effect=AssertionError("forbidden")):
        for stage in sanity.iter_randomized_models(model, 3003):
            assert stage.batchnorm_buffers_preserved
            assert_same_state(sanity._bn_buffers(stage.model), buffers)
            assert not any(m.training for m in stage.model.modules())
            for name, module in stage.model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    selected = any(name.startswith(group + ".") for group in stage.modules_randomized)
                    assert torch.all(module.weight == (1 if selected else 2))
                    assert torch.all(module.bias == (0 if selected else .3))
            assert all(not p.requires_grad for p in stage.model.parameters())
    assert sanity._preserved(model, snapshot)


def test_initialization_matches_architecture_default(model):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3000)
        expected = deepcopy(model.classifier[4])
        expected.reset_parameters()
    actual = next(sanity.iter_randomized_models(model, 3000)).model.classifier[4]
    assert_same_state(expected.state_dict(), actual.state_dict())
    # Biases are randomized too; source fixture biases are not forcibly zeroed.
    assert not torch.equal(actual.bias, model.classifier[4].bias)


def test_real_architecture_stage_paths_include_projection_and_preserve_buffers():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        model = ResNetCNN().eval()
    before = sanity._snapshot(model)
    stages = list(sanity.iter_randomized_models(model, 3004))
    assert "blocks.2.shortcut.0" in stages[2].parameter_modules
    assert "blocks.1.shortcut.0" in stages[3].parameter_modules
    assert all(stage.batchnorm_buffers_preserved for stage in stages)
    assert sanity._preserved(model, before)


def test_identity_and_scale_invariance(cam):
    for candidate in (cam, cam * 5):
        result = sanity.compare_attributions(cam, candidate, 42)
        assert result.status == "valid" and result.reasons == ()
        assert result.jaccard == 1
        assert result.spearman == pytest.approx(1)
        assert result.total_variation == pytest.approx(0, abs=1e-15)


def test_disjoint_support_opposite_ranks_and_tv():
    first = np.zeros((150, 150), dtype=np.float32)
    first[:75] = 1
    second = 1 - first
    result = sanity.compare_attributions(first, second, 42)
    assert result.jaccard == 0
    assert result.spearman == pytest.approx(-1)
    assert result.total_variation == pytest.approx(1)


def test_spearman_average_ties_known_analytic_example():
    # Ranks [1.5,1.5,3,4] and [1,2.5,2.5,4], repeated equally spatially.
    first = np.tile([0., 0., 1., 2.], 5625).reshape(150, 150)
    second = np.tile([0., 1., 1., 2.], 5625).reshape(150, 150)
    result = sanity.compare_attributions(first, second, 0)
    assert result.spearman == pytest.approx(5 / 6)
    # L1([0,0,1/3,2/3] - [0,1/4,1/4,1/2]) / 2 = 1/4.
    assert result.total_variation == pytest.approx(1 / 4)
    np.testing.assert_array_equal(sanity._average_ranks(np.array([1, 1, 2, 3])),
                                  [1.5, 1.5, 3, 4])


def test_topk_jaccard_known_partial_overlap():
    first = np.arange(22500, dtype=np.float32).reshape(150, 150)
    second = np.roll(first.ravel(), 1125).reshape(150, 150)
    assert sanity.compare_attributions(first, second, 42).jaccard == pytest.approx(1/3)


def test_tie_breaking_reuses_frozen_pixel_policy():
    from src.xai.stability import topk_mask
    first = np.zeros((150, 150), dtype=np.float32)
    first[:75] = 1
    second = first.copy()
    second[1:5] = 2
    support = np.ones_like(first, dtype=bool)
    a, b = topk_mask(first, support, 42), topk_mask(second, support, 42)
    assert a.sum() == b.sum() == 2250
    expected = (a & b).sum() / (a | b).sum()
    assert sanity.compare_attributions(first, second, 42).jaccard == expected


@pytest.mark.parametrize("bad,reason", [(np.zeros((150, 150)), "degenerate"),
                                       (np.ones((150, 150)), "degenerate"),
                                       (np.full((150, 150), np.nan), "nonfinite"),
                                       (np.full((150, 150), np.inf), "nonfinite"),
                                       (None, "unavailable")])
def test_unavailable_maps_not_fabricated(cam, bad, reason):
    result = sanity.compare_attributions(cam, bad, 42)
    assert result.status == "unavailable"
    assert result.reasons == ("candidate_" + reason,)
    assert result.jaccard is result.spearman is result.total_variation is None
    result = sanity.compare_attributions(bad, cam, 42)
    assert result.reasons == ("reference_" + reason,)


@pytest.mark.parametrize("status", ["degenerate", "unavailable"])
def test_explicit_unavailable_status(cam, status):
    result = sanity.compare_attributions(cam, cam, 1, candidate_status=status)
    assert result.reasons == ("candidate_" + status,)


@pytest.mark.parametrize("bad", [np.zeros((19, 19)), np.zeros((1, 150, 150)),
                                -np.ones((150, 150)), np.full((150, 150), "x")])
def test_invalid_maps_rejected(cam, bad):
    with pytest.raises(ValueError):
        sanity.compare_attributions(cam, bad, 42)


@pytest.mark.parametrize("index", [-1, 792, True, 1.0])
def test_invalid_index_rejected(cam, index):
    with pytest.raises(ValueError):
        sanity.compare_attributions(cam, cam, index)


def test_no_array_mutation(cam):
    other = cam[::-1].copy()
    a, b = cam.copy(), other.copy()
    sanity.compare_attributions(cam, other, 42)
    np.testing.assert_array_equal(cam, a)
    np.testing.assert_array_equal(other, b)


def test_finite_large_maps_do_not_overflow(cam):
    large = cam.astype(np.float64) * 1e290
    result = sanity.compare_attributions(large, large, 42)
    assert result.jaccard == result.spearman == 1
    assert result.total_variation == 0


def test_runner_fixed_target_identity_and_preservation(model, cam, monkeypatch):
    calls = []
    source = sanity._snapshot(model)
    tensor = torch.zeros(1, 1, 150, 150)
    before = tensor.clone()

    def fake(copy, input_tensor, target_class=None):
        assert copy is not model
        assert not any(m.training for m in copy.modules())
        calls.append(target_class)
        # Every intervention flips prediction, but attribution target remains 1.
        return explanation(cam, target=1 if target_class is None else target_class,
                           prediction=1 if target_class is None else 0)

    monkeypatch.setattr(sanity, "generate_gradcam", fake)
    # The core must not serialize weights or read image files.
    with patch("torch.save", side_effect=AssertionError("no writes")), \
         patch("builtins.open", side_effect=AssertionError("no file access")):
        records = sanity.run_sanity_checks(model, tensor, 42)
    assert len(records) == 31
    assert calls == [None] + [1] * 31
    assert records[0].stage == 0 and records[0].seed is None
    assert [(r.seed, r.stage) for r in records[1:]] == [
        (seed, stage) for seed in sanity.SEEDS for stage in range(1, 7)]
    assert all(r.fixed_target_class == 1 and r.similarity.jaccard == 1 for r in records)
    assert all(r.original_state_preserved and r.batchnorm_buffers_preserved and
               r.evaluation_mode_preserved and r.scoring_state_preserved for r in records)
    assert sanity._preserved(model, source)
    assert torch.equal(tensor, before)


def test_actual_gradcam_on_unchanged_frozen_copy():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = ResNetCNN().eval()
        tensor = torch.rand(1, 1, 150, 150)
    for p in model.parameters():
        p.requires_grad_(False)
    before = sanity._snapshot(model)
    reference = generate_gradcam(deepcopy(model), tensor)
    identity = generate_gradcam(deepcopy(model), tensor, reference.predicted_class)
    assert reference.status == identity.status == "valid"
    assert torch.equal(reference.logits, identity.logits)
    assert torch.equal(reference.raw_cam, identity.raw_cam)
    score = sanity.compare_attributions(reference.upsampled_heatmap, identity.upsampled_heatmap, 42)
    assert score.jaccard == 1 and score.total_variation == 0
    assert score.spearman == pytest.approx(1)
    assert sanity._preserved(model, before)


def test_end_to_end_synthetic_resnet_all_stages():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = ResNetCNN().eval()
        tensor = torch.rand(1, 1, 150, 150)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = sanity._snapshot(model)
    original_input = tensor.clone()
    rng = torch.get_rng_state().clone()
    records = sanity.run_sanity_checks(model, tensor, 42)
    assert len(records) == 31
    assert records[0].similarity.jaccard == 1
    assert records[0].similarity.total_variation == 0
    assert all(r.fixed_target_class == records[0].fixed_target_class for r in records)
    assert all(r.scoring_state_preserved and r.batchnorm_buffers_preserved for r in records)
    assert sanity._preserved(model, before)
    assert torch.equal(tensor, original_input)
    assert torch.equal(rng, torch.get_rng_state())


def test_scoring_bn_mutation_fails_closed(model, cam, monkeypatch):
    count = 0

    def bad(copy, tensor, target_class=None):
        nonlocal count
        count += 1
        if count > 1:
            copy.stem[1].running_mean.add_(1)
        return explanation(cam)

    before = sanity._snapshot(model)
    monkeypatch.setattr(sanity, "generate_gradcam", bad)
    with pytest.raises(RuntimeError, match="state changed"):
        sanity.run_sanity_checks(model, torch.zeros(1, 1, 150, 150), 42)
    assert sanity._preserved(model, before)


def test_runner_degenerate_and_nonfinite_records(model, cam, monkeypatch):
    count = 0

    def fake(copy, tensor, target_class=None):
        nonlocal count
        count += 1
        if count > 2:
            raise ValueError("Non-finite Grad-CAM activations or gradients.")
        return explanation(cam, status="degenerate")

    monkeypatch.setattr(sanity, "generate_gradcam", fake)
    records = sanity.run_sanity_checks(model, torch.zeros(1, 1, 150, 150), 42)
    assert all(r.similarity.status == "unavailable" for r in records)
    assert records[0].similarity.reasons == ("reference_degenerate", "candidate_degenerate")
    assert records[-1].similarity.reasons == ("candidate_nonfinite",)


def test_failed_forward_preserves_original(model, monkeypatch):
    before = sanity._snapshot(model)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(sanity, "generate_gradcam", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        sanity.run_sanity_checks(model, torch.zeros(1, 1, 150, 150), 42)
    assert sanity._preserved(model, before)


@pytest.mark.parametrize("seed", [2999, 3005, True, 3000.0])
def test_invalid_seeds(model, seed):
    with pytest.raises(ValueError):
        list(sanity.iter_randomized_models(model, seed))


def test_training_mode_and_inference_mode_rejected(model):
    model.stem[1].train()
    with pytest.raises(ValueError, match="evaluation"):
        next(sanity.iter_randomized_models(model, 3000))
    model.eval()
    with torch.inference_mode(), pytest.raises(RuntimeError):
        next(sanity.iter_randomized_models(model, 3000))


def test_input_validation_delegated_to_gradcam():
    model = ResNetCNN().eval()
    for value in (torch.zeros(1, 150, 150), torch.full((1, 1, 150, 150), float("nan"))):
        with pytest.raises(ValueError):
            sanity.run_sanity_checks(model, value, 42)
