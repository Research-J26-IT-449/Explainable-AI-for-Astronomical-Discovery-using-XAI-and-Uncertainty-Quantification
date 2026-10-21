"""Synthetic corrected-dashboard loading and fail-closed integration checks."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from src.inference import corrected_demo as corrected
from src.models.resnet_cnn import ResNetCNN


@pytest.fixture
def checkpoint(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = ResNetCNN().eval()
    path = tmp_path / "synthetic.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 30}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), model


def test_safe_loader_uses_strict_frozen_eval_state(checkpoint):
    path, digest, original = checkpoint
    model = corrected.load_corrected_checkpoint(path, digest)
    assert all(not m.training for m in model.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.state_dict().items())


def test_hash_mismatch_rejected_before_deserialization(checkpoint, monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Do not deserialize mismatched bytes"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        corrected.load_corrected_checkpoint(checkpoint[0], "0" * 64)


@pytest.mark.parametrize("defect", ["missing_key", "wrong_epoch", "nonfinite"])
def test_bad_checkpoint_fails_closed(checkpoint, defect):
    path, _, model = checkpoint
    state = deepcopy(model.state_dict())
    if defect == "missing_key":
        state.pop(next(iter(state)))
    if defect == "nonfinite":
        state[next(iter(state))].fill_(float("nan"))
    torch.save({"model_state_dict": state, "epoch": 1 if defect == "wrong_epoch" else 30}, path)
    with pytest.raises((ValueError, RuntimeError)):
        corrected.load_corrected_checkpoint(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_missing_calibration_never_loads_model_or_historical_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(corrected, "checked_manifest", lambda: {})
    monkeypatch.setattr(corrected, "CALIBRATION_PATH", tmp_path / "absent.pt")
    monkeypatch.setattr(corrected, "load_corrected_checkpoint", lambda: pytest.fail("No fallback"))
    with pytest.raises(FileNotFoundError, match="historical calibration is not reused"):
        corrected.load_calibrated_ood_detector()


@pytest.mark.parametrize("mismatch", ["train", "validation", "checkpoint"])
def test_calibration_requires_corrected_membership_and_checkpoint(tmp_path, monkeypatch, mismatch):
    path = tmp_path / "calibration.pt"
    path.touch()
    monkeypatch.setattr(corrected, "CALIBRATION_PATH", path)
    manifest = {"splits": {"train": [1,2], "validation": [3,4]}}
    artifact = {"checkpoint_sha256": corrected.CHECKPOINT_SHA256,
                "splits": {s: {"indices": list(v), "count": len(v)} for s,v in manifest["splits"].items()}}
    if mismatch == "checkpoint":
        artifact["checkpoint_sha256"] = "0" * 64
    else:
        artifact["splits"][mismatch]["indices"] = [9,10]
    monkeypatch.setattr(corrected, "checked_manifest", lambda: manifest)
    monkeypatch.setattr(corrected, "load_calibration", lambda *a: artifact)
    monkeypatch.setattr(corrected, "load_corrected_checkpoint", lambda: pytest.fail("Reject before model load"))
    with pytest.raises(ValueError, match="corrected"):
        corrected.load_calibrated_ood_detector()


def test_matching_synthetic_calibration_binds_correct_model(tmp_path, monkeypatch, checkpoint):
    path = tmp_path / "synthetic_calibration.pt"
    path.touch()
    manifest = {"splits": {"train": [1,2], "validation": [3,4]}}
    artifact = {"checkpoint_sha256": corrected.CHECKPOINT_SHA256,
                "splits": {s: {"indices": v, "count": len(v)} for s,v in manifest["splits"].items()},
                "feature_mean": torch.zeros(512), "feature_variance": torch.ones(512),
                "thresholds": {"energy_threshold": -0.5, "feature_distance_threshold": 3., "validation_percentile": .99}}
    model = checkpoint[2]
    monkeypatch.setattr(corrected, "checked_manifest", lambda: manifest)
    monkeypatch.setattr(corrected, "CALIBRATION_PATH", path)
    monkeypatch.setattr(corrected, "load_calibration", lambda *a: artifact)
    monkeypatch.setattr(corrected, "load_corrected_checkpoint", lambda: model)
    detector = corrected.load_calibrated_ood_detector()
    assert detector.model is model and detector.feature_mean is artifact["feature_mean"]
    assert detector.thresholds.energy_threshold == -0.5


def test_manifest_hash_mismatch_rejected(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    monkeypatch.setattr(corrected, "MANIFEST_PATH", path)
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        corrected.checked_manifest()


def test_upload_geometry_and_normalization_preserved(monkeypatch):
    seen = []
    def validate(tensor):
        seen.append(tensor.clone())
        return {"accepted": True}
    monkeypatch.setattr(corrected, "load_calibrated_ood_detector", lambda: SimpleNamespace(
        validate_tensor=validate, model=lambda tensor: torch.tensor([[0.,1.]])))
    image = Image.fromarray(np.tile(np.arange(200,dtype=np.uint8),(300,1)))
    result = corrected.predict_image(image)
    assert result["predicted_class"] == "FRII"
    expected = corrected.prepare_guarded_image(image).tensor
    assert seen[0].shape == (1,1,150,150) and seen[0].dtype == torch.float32
    assert torch.equal(seen[0], expected)


def test_invalid_mode_has_no_model_access(monkeypatch):
    monkeypatch.setattr(corrected, "load_calibrated_ood_detector", lambda: pytest.fail("Reject invalid mode first"))
    result = corrected.predict_image(object(), input_mode="invalid")
    assert result["probabilities"] == {} and "Unknown" in result["message"]


@pytest.mark.parametrize("image", [None, Image.new("L", (150,150), 0), Image.new("L", (150,150), 127)])
def test_invalid_public_uploads_rejected_before_model_access(image, monkeypatch):
    monkeypatch.setattr(corrected, "load_calibrated_ood_detector", lambda: pytest.fail("Input guard first"))
    result = corrected.predict_image(image)
    assert result["predicted_class"] == "UNSUPPORTED" and result["confidence"] is None
    assert result["probabilities"] == {}


def test_auto_never_substitutes_historical_model():
    result = corrected.predict_image(None, input_mode="auto_web")
    assert result["probabilities"] == {} and "not substituted" in result["message"]


@pytest.mark.parametrize("target", [0, 1])
def test_prediction_class_mapping_and_state_preservation(checkpoint, monkeypatch, target):
    model = checkpoint[2]
    # Disposable synthetic model only; no historical/corrected weights accessed.
    with torch.no_grad():
        model.classifier[4].bias[target] += 20
    state = {k:v.clone() for k,v in model.state_dict().items()}
    seen = []
    def validate(tensor):
        seen.append(tensor.shape)
        return {"accepted":True}
    monkeypatch.setattr(corrected, "load_calibrated_ood_detector", lambda: SimpleNamespace(model=model, validate_tensor=validate))
    image = Image.fromarray(np.tile(np.arange(150,dtype=np.uint8),(150,1)))
    result = corrected.predict_image(image)
    assert result["predicted_class"] == ("FRI","FRII")[target]
    assert result["confidence"] == max(result["probabilities"].values())
    assert seen == [torch.Size([1,1,150,150])]
    assert "not calibrated" in result["message"]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())


@pytest.mark.parametrize("failure", ["missing_calibration", "rejected", "nonfinite"])
def test_prediction_errors_do_not_fabricate_probabilities(monkeypatch, failure):
    def context():
        if failure == "missing_calibration":
            raise FileNotFoundError("Corrected calibration unavailable")
        return SimpleNamespace(validate_tensor=lambda tensor:{"accepted":failure!='rejected'},
                               model=lambda tensor:torch.tensor([[float('nan'),0.]]))
    monkeypatch.setattr(corrected, "load_calibrated_ood_detector", context)
    result = corrected.predict_image(Image.fromarray(np.tile(np.arange(150,dtype=np.uint8),(150,1))))
    assert result["predicted_class"] == "UNSUPPORTED" and result["confidence"] is None
    assert result["probabilities"] == {}
