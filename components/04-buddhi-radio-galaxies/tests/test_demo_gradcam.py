"""Upload-only demo tests: synthetic pixels/models, no checkpoint/data access."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from demo import app
from src.xai.gradcam import GradCAMResult, TARGET_LAYER


@pytest.fixture(autouse=True)
def forbid_real_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real inference/checkpoint access forbidden in UI tests")
    monkeypatch.setattr(app, "predict_image", forbidden)
    monkeypatch.setattr(app, "load_calibrated_ood_detector", forbidden)


@pytest.fixture
def upload():
    return Image.fromarray(np.tile(np.arange(150, dtype=np.uint8), (150, 1)))


@pytest.fixture
def accepted(monkeypatch):
    class SyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor(0.0))
            self.bn = torch.nn.BatchNorm2d(1)
            self.register_buffer("logits", torch.tensor([[2.0, -1.0]]))

        def forward(self, image):
            return self.logits + self.offset * 0

    model = SyntheticModel().eval()
    probabilities = model.logits.softmax(1)[0].tolist()
    result = {"predicted_class": "FRI", "confidence": probabilities[0],
              "probabilities": dict(zip(("FRI", "FRII"), probabilities)),
              "message": "Synthetic accepted input", "ood": {"accepted": True}}
    calls = []
    def validate(tensor):
        calls.append(("ood", tensor.clone()))
        return {"accepted": True}
    detector = SimpleNamespace(model=model, validate_tensor=validate)
    monkeypatch.setattr(app, "predict_image", lambda *args, **kwargs: result)
    monkeypatch.setattr(app, "load_calibrated_ood_detector", lambda: detector)
    heat = torch.linspace(0, 1, 22500).reshape(150, 150)
    def explain(copy, tensor, target_class):
        assert copy is not model
        assert all(not m.training for m in copy.modules())
        assert target_class == 0
        assert calls[-1][0] == "ood"
        assert torch.equal(calls[-1][1], tensor)
        calls.append(("cam", tensor.clone()))
        return GradCAMResult(torch.ones(19, 19), heat, heat,
                             target_class, 0, model.logits.clone(), TARGET_LAYER, "valid")
    monkeypatch.setattr(app, "generate_gradcam", explain)
    return SimpleNamespace(model=model, result=result, detector=detector, calls=calls)


def test_no_upload_is_idle_and_does_not_load_model():
    result = app.predict_for_ui_with_gradcam(None)
    assert len(result) == 7
    assert result[0] is None and result[1] is None
    assert "upload" in result[2].lower()
    assert result[3:6] == (None, None, None)


def test_accepted_upload_displays_prediction_heatmap_overlay_without_mutation(upload, accepted):
    original = np.asarray(upload).copy()
    state = {k: v.clone() for k, v in accepted.model.state_dict().items()}
    result = app.predict_for_ui_with_gradcam(upload)
    assert result[0] is upload
    assert result[1]["FR-I"] == accepted.result["probabilities"]["FRI"]
    assert all(image.size == (150, 150) for image in result[3:6])
    assert result[3].mode == "L" and result[4].mode == "RGB" and result[5].mode == "RGBA"
    assert "NOT a scientifically validated Phase 3B" in result[6]
    assert "FR-I" in result[6] and TARGET_LAYER in result[6]
    assert [name for name, _ in accepted.calls] == ["ood", "cam"]
    assert np.array_equal(np.asarray(upload), original)
    assert all(torch.equal(v, accepted.model.state_dict()[k]) for k, v in state.items())
    assert all(p.grad is None for p in accepted.model.parameters())


@pytest.mark.parametrize("label", ["UNSUPPORTED", "AMBIGUOUS"])
def test_rejection_clears_maps_without_loading_explanation(monkeypatch, upload, label):
    monkeypatch.setattr(app, "predict_image", lambda *a, **k: {
        "predicted_class": label, "probabilities": {}, "message": "Rejected"})
    result = app.predict_for_ui_with_gradcam(upload)
    assert result[1] == label + ": Rejected"
    assert result[3:6] == (None, None, None)


def test_auto_web_classification_is_preserved_without_substitute_explanation(upload, accepted):
    result = app.predict_for_ui_with_gradcam(upload, "auto_web")
    assert result[1]["FR-I"] > 0
    assert result[3:6] == (None, None, None)
    assert "combined representations" in result[6]
    assert accepted.calls == []


def test_external_mode_uses_existing_preparation(monkeypatch, upload, accepted):
    guarded = app.prepare_guarded_image(upload)
    calls = []
    monkeypatch.setattr(app, "prepare_external_radio_image",
                        lambda image: calls.append(image) or guarded)
    result = app.predict_for_ui_with_gradcam(upload, "external_radio")
    assert calls == [upload] and result[3] is guarded.image
    assert result[5] is not None


@pytest.mark.parametrize("failure", ["load", "ood", "cam", "logits", "prediction", "heatmap"])
def test_explanation_failure_never_displays_stale_maps(monkeypatch, upload, accepted, failure):
    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic unavailable resource")
    if failure == "load":
        monkeypatch.setattr(app, "load_calibrated_ood_detector", fail)
    elif failure == "ood":
        accepted.detector.validate_tensor = lambda tensor: {"accepted": False}
    elif failure == "cam":
        monkeypatch.setattr(app, "generate_gradcam", fail)
    elif failure == "prediction":
        accepted.result["probabilities"] = {"FRI": 0.5, "FRII": 0.5}
    else:
        real_fake = app.generate_gradcam
        def invalid(*args, **kwargs):
            result = real_fake(*args, **kwargs)
            if failure == "logits":
                result.logits.add_(1)
            else:
                result.heatmap.fill_(float("nan"))
            return result
        monkeypatch.setattr(app, "generate_gradcam", invalid)
    result = app.predict_for_ui_with_gradcam(upload)
    assert result[1]["FR-I"] > 0  # Original accepted classification remains available.
    assert result[3:6] == (None, None, None)
    assert "Grad-CAM unavailable" in result[6]


def test_degenerate_cam_has_explicit_status_and_no_fabricated_overlay(monkeypatch, upload, accepted):
    monkeypatch.setattr(app, "generate_gradcam", lambda *a, **k: GradCAMResult(
        torch.zeros(19, 19), torch.zeros(150, 150), torch.zeros(150, 150),
        0, 0, accepted.model.logits.clone(), TARGET_LAYER, "degenerate"))
    result = app.predict_for_ui_with_gradcam(upload)
    assert result[3].size == (150, 150)
    assert result[4:6] == (None, None)
    assert "Degenerate" in result[6]


def test_public_guard_still_rejects_blank_external_upload(monkeypatch, accepted):
    result = app.predict_for_ui_with_gradcam(Image.new("L", (150, 150), 128))
    assert "near-uniform" in result[6].lower()
    assert result[3:6] == (None, None, None)
    assert accepted.calls == []


def test_prediction_exception_has_clear_safe_response(monkeypatch, upload):
    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic model unavailable")
    monkeypatch.setattr(app, "predict_image", fail)
    result = app.predict_for_ui_with_gradcam(upload)
    assert result[1] == "UNSUPPORTED"
    assert "Prediction unavailable" in result[2]
    assert result[3:6] == (None, None, None)


@pytest.mark.parametrize("target", [0, 1])
def test_real_core_on_synthetic_resnet_preserves_both_class_targets(monkeypatch, upload, target):
    from src.models.resnet_cnn import ResNetCNN
    from src.xai.gradcam import generate_gradcam

    # Random, disposable model only: never call a checkpoint/dataset loader.
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = ResNetCNN().eval()
    with torch.no_grad():
        model.classifier[4].bias[target] += 10
        logits = model(app.prepare_guarded_image(upload).tensor)
    assert int(logits.argmax(1)) == target
    probabilities = logits.softmax(1)[0].tolist()
    names = ("FRI", "FRII")
    monkeypatch.setattr(app, "predict_image", lambda *a, **k: {
        "predicted_class": names[target], "confidence": probabilities[target],
        "probabilities": dict(zip(names, probabilities)), "message": "Synthetic only"})
    monkeypatch.setattr(app, "load_calibrated_ood_detector", lambda: SimpleNamespace(
        model=model, validate_tensor=lambda tensor: {"accepted": True}))
    seen = []
    def explain(copy, tensor, target_class):
        assert copy is not model
        seen.append(target_class)
        return generate_gradcam(copy, tensor, target_class)
    monkeypatch.setattr(app, "generate_gradcam", explain)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    result = app.predict_for_ui_with_gradcam(upload)
    assert seen == [target]
    assert "unavailable" not in result[6].lower() or "Degenerate" in result[6]
    assert result[3] is not None
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert all(p.grad is None for p in model.parameters())
    assert all(not m._forward_hooks for m in model.modules())


def test_gradio_section_is_visible_and_dashboard_preserves_scientific_outputs(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    demo = app.create_demo()  # Construct only; never launch a server.
    config = demo.get_config_file()
    components = config["components"]
    assert any("Explainable AI (Grad-CAM)" in str(c["props"].get("value", ""))
               for c in components)
    labels = {c["props"].get("label") for c in components}
    assert "Grad-CAM heatmap (normalized [0,1])" in labels
    assert "Grad-CAM overlay on classifier input" in labels
    # Seven existing outputs plus a presentation-only prediction card.
    assert any(len(d["outputs"]) == 8 and d.get("backend_fn") for d in config["dependencies"])
    assert not demo.is_running
    demo.close()
