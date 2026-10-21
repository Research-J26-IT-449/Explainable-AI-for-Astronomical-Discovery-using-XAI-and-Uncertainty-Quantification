"""Grad-CAM correctness and state isolation, exclusively with synthetic inputs."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.resnet_cnn import ResNetCNN
from src.xai.gradcam import TARGET_LAYER, generate_gradcam


class AnalyticalCNN(nn.Module):
    """Small known-weight graph with the selected ResNet target-layer interface.

    Only two of its 128 channels are active: A0=x and A1=1-x.
    z0=mean(A0)-2*mean(A1); z1=-3*mean(A0)+mean(A1).
    Thus Grad-CAM's known weights are [1,-2]/361 and [-3,1]/361.
    """

    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.convolution = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                                          nn.Conv2d(1, 128, 1))
        self.blocks = nn.Sequential(nn.Identity(), nn.Identity(), block)
        with torch.no_grad():
            layer = block.convolution[3]
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight[0] = 1
            layer.weight[1] = -1
            layer.bias[1] = 1

    def forward(self, inputs):
        features = self.blocks[2].convolution(F.adaptive_avg_pool2d(inputs, (19, 19)))
        means = features.mean(dim=(2, 3))
        return torch.stack((means[:, 0] - 2 * means[:, 1],
                            -3 * means[:, 0] + means[:, 1]), dim=1)


@pytest.fixture
def model():
    return AnalyticalCNN().eval()


@pytest.fixture
def image():
    return torch.linspace(0, 1, 150 * 150).reshape(1, 1, 150, 150)


def layer(model):
    return model.blocks[2].convolution[3]


@pytest.mark.parametrize("target,weights", [(0, (1, -2)), (1, (-3, 1))])
def test_analytical_logit_gradients_and_cam(model, image, target, weights):
    result = generate_gradcam(model, image, target)
    x = F.adaptive_avg_pool2d(image, (19, 19))[0, 0]
    expected = (weights[0] * x + weights[1] * (1 - x)).relu() / (19 * 19)
    torch.testing.assert_close(result.raw_cam, expected, rtol=1e-5, atol=1e-9)
    upsampled = F.interpolate(expected[None, None], size=(150, 150), mode="bilinear",
                             align_corners=False)[0, 0]
    torch.testing.assert_close(result.upsampled_heatmap, upsampled, rtol=1e-5, atol=1e-9)
    expected_normalized = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min())
    torch.testing.assert_close(result.heatmap, expected_normalized, rtol=1e-5, atol=1e-6)
    assert result.target_class == target
    assert result.layer_name == TARGET_LAYER
    assert result.status == "valid"
    assert result.raw_cam.shape == (19, 19)
    assert result.upsampled_heatmap.shape == result.heatmap.shape == (150, 150)
    assert result.logits.shape == (1, 2)
    for tensor in (result.raw_cam, result.upsampled_heatmap, result.heatmap, result.logits):
        assert torch.isfinite(tensor).all()
        assert not tensor.requires_grad
    assert result.raw_cam.min() >= 0
    assert result.heatmap.min() == 0
    assert result.heatmap.max() == 1


def test_classes_have_distinct_expected_regions(model, image):
    fri = generate_gradcam(model, image, 0).raw_cam
    frii = generate_gradcam(model, image, 1).raw_cam
    assert fri[-1].max() > 0 and fri[0].max() == 0
    assert frii[0].max() > 0 and frii[-1].max() == 0


def test_default_target_is_predicted_class_and_cpu_determinism(model, image):
    first = generate_gradcam(model, image)
    second = generate_gradcam(model, image)
    assert first.target_class == first.predicted_class == int(model(image).argmax())
    assert torch.equal(first.raw_cam, second.raw_cam)
    assert torch.equal(first.heatmap, second.heatmap)
    assert torch.equal(first.logits, second.logits)


@pytest.mark.parametrize("value,target", [(0.0, 0), (1.0, 0), (0.0, 1)])
def test_zero_and_constant_maps_are_explicitly_degenerate(model, value, target):
    result = generate_gradcam(model, torch.full((1, 1, 150, 150), value), target)
    assert result.status == "degenerate"
    assert torch.count_nonzero(result.heatmap) == 0
    assert torch.isfinite(result.heatmap).all()
    if value == 0 and target == 0:
        assert torch.count_nonzero(result.raw_cam) == 0
    else:
        assert result.raw_cam.min() > 0  # Positive constant evidence is retained raw.


@pytest.mark.parametrize("frozen", [False, True])
def test_real_resnet_preserves_logits_weights_buffers_gradients_and_inputs(image, frozen):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = ResNetCNN().eval()
    parameters = list(model.parameters())
    for i, parameter in enumerate(parameters):
        parameter.requires_grad_(not frozen)
        if i % 2:
            parameter.grad = torch.full_like(parameter, 0.125)
    grad_refs = [p.grad for p in parameters]
    grads = [None if p.grad is None else p.grad.clone() for p in parameters]
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    flags = [(m, m.training) for m in model.modules()]
    requires_grad = [p.requires_grad for p in parameters]
    image = image.clone().requires_grad_(True)
    image.grad = torch.ones_like(image)
    image_before, image_grad = image.detach().clone(), image.grad.clone()
    input_grad_ref = image.grad
    with torch.no_grad():
        before = model(image).clone()
        result = generate_gradcam(model, image, 1)
        repeated = generate_gradcam(model, image, 1)
        after = model(image)
    torch.testing.assert_close(before, result.logits, rtol=0, atol=0)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert torch.equal(before.softmax(1), result.logits.softmax(1))
    assert result.predicted_class == int(before.argmax())
    assert result.raw_cam.shape == (19, 19)
    assert torch.equal(result.raw_cam, repeated.raw_cam)
    assert torch.equal(result.heatmap, repeated.heatmap)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(m.training == flag for m, flag in flags)
    assert [p.requires_grad for p in parameters] == requires_grad
    for p, reference, grad in zip(parameters, grad_refs, grads):
        assert p.grad is reference
        assert p.grad is None if grad is None else torch.equal(p.grad, grad)
    assert torch.equal(image.detach(), image_before)
    assert image.grad is input_grad_ref and torch.equal(image.grad, image_grad)


def test_hooks_removed_after_success_and_existing_hooks_preserved(model, image):
    calls = []
    existing = layer(model).register_forward_hook(lambda *args: calls.append(True))
    try:
        before = dict(layer(model)._forward_hooks)
        for _ in range(3):
            generate_gradcam(model, image)
            assert dict(layer(model)._forward_hooks) == before
        assert len(calls) == 3
    finally:
        existing.remove()


@pytest.mark.parametrize("failure", ["forward", "disconnected", "unused", "nan_logits", "nan_gradient", "shape"])
def test_hooks_removed_on_failure(model, image, monkeypatch, failure):
    original = model.forward
    if failure == "forward":
        def fail(x):
            original(x)
            raise RuntimeError("Synthetic forward failure")
        monkeypatch.setattr(model, "forward", fail)
    elif failure == "disconnected":
        def detached(x):
            with torch.no_grad():
                return original(x)
        monkeypatch.setattr(model, "forward", detached)
    elif failure == "unused":
        def unused(x):
            original(x)
            return x.mean().expand(1, 2)
        monkeypatch.setattr(model, "forward", unused)
    elif failure == "nan_logits":
        monkeypatch.setattr(model, "forward", lambda x: original(x) * float("nan"))
    elif failure == "shape":
        monkeypatch.setattr(model, "forward", lambda x: original(x)[:, :1])
    else:
        grad = torch.autograd.grad
        def nonfinite(*args, **kwargs):
            return tuple(torch.full_like(t, float("nan")) for t in grad(*args, **kwargs))
        monkeypatch.setattr(torch.autograd, "grad", nonfinite)
    before = dict(layer(model)._forward_hooks)
    with pytest.raises((ValueError, RuntimeError)):
        generate_gradcam(model, image)
    assert dict(layer(model)._forward_hooks) == before


@pytest.mark.parametrize("target", [-1, 2, "FRI", 0.0, True, [0]])
def test_invalid_target_classes(model, image, target):
    with pytest.raises(ValueError, match="target_class"):
        generate_gradcam(model, image, target)
    assert not layer(model)._forward_hooks


@pytest.mark.parametrize("shape", [(1, 150, 150), (2, 1, 150, 150), (1, 3, 150, 150), (1, 1, 149, 150)])
def test_invalid_input_shapes(model, shape):
    with pytest.raises(ValueError, match="shape"):
        generate_gradcam(model, torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs(model, image, value):
    image[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        generate_gradcam(model, image)


def test_wrong_dtype_and_non_tensor(model, image):
    for value in (image.double(), image.to(torch.int64), None):
        with pytest.raises(ValueError, match="float32"):
            generate_gradcam(model, value)


def test_requires_entire_model_evaluation_mode(model, image):
    model.train()
    with pytest.raises(ValueError, match="evaluation"):
        generate_gradcam(model, image)
    model.eval()
    layer(model).train()
    with pytest.raises(ValueError, match="evaluation"):
        generate_gradcam(model, image)
    assert layer(model).training  # API must not silently change caller flags.


def test_inference_mode_rejected_before_hook_registration(model, image):
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        generate_gradcam(model, image)
    assert not layer(model)._forward_hooks


def test_missing_or_wrong_target_layer(image):
    with pytest.raises(ValueError, match="unavailable"):
        generate_gradcam(nn.Identity().eval(), image)
    model = AnalyticalCNN().eval()
    model.blocks[2].convolution[3] = nn.Identity().eval()
    with pytest.raises(ValueError, match="convolution"):
        generate_gradcam(model, image)
