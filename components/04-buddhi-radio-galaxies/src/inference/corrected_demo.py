"""Corrected development-model demo; never reuse historical OOD calibration.

The standalone checkpoint loader is usable independently. Public predictions
remain fail closed until checkpoint- AND corrected-membership-matched frozen OOD
calibration exists. This module never fits calibration or opens dataset images.
"""
import hashlib
import io
import json
from pathlib import Path

import torch

from src.models.resnet_cnn import ResNetCNN
from src.ood.input_guard import prepare_guarded_image
from src.ood.external_radio import prepare_external_radio_image
from src.ood.calibration import load_calibration
from src.ood.ood_detector import OODDetector, OODThresholds

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/models/resnet_cnn_best.pth"
CHECKPOINT_SHA256 = "d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c"
MANIFEST_PATH = ROOT / "data/manifests/mirabest_development_exactdedup_v1.0.0/manifest.json"
MANIFEST_SHA256 = "4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91"
# Separate future artifact location; never fall back to outputs/ood/ood_calibration.pt.
CALIBRATION_PATH = ROOT / "outputs/ood/mirabest_development_exactdedup_v1.0.0/ood_calibration.pt"
CLASS_NAMES = ("FRI", "FRII")
MODEL_NOTICE = (
    "Corrected MiraBest development model. Softmax confidence is not calibrated. "
    "External generalization and independent TEST performance are not established."
)


def load_corrected_checkpoint(checkpoint_path=CHECKPOINT_PATH, expected_sha256=CHECKPOINT_SHA256):
    """Hash the exact bytes loaded safely; return a fresh frozen CPU instance."""
    raw = Path(checkpoint_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Corrected checkpoint SHA-256 mismatch.")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid corrected checkpoint structure.")
    if checkpoint.get("epoch") != 30:
        raise ValueError("Corrected checkpoint selection metadata mismatch.")
    model = ResNetCNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Non-finite corrected model state.")
    model.requires_grad_(False)
    return model.eval()


def checked_manifest():
    raw = MANIFEST_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Corrected manifest SHA-256 mismatch.")
    manifest = json.loads(raw)
    if manifest["class_mapping"] != {"0": "FR-I", "1": "FR-II"}:
        raise ValueError("Corrected class mapping mismatch.")
    return manifest


def load_calibrated_ood_detector():
    """Corrected context only. Missing/incompatible calibration fails closed."""
    manifest = checked_manifest()
    if not CALIBRATION_PATH.is_file():
        raise FileNotFoundError(
            "Corrected-model OOD calibration is unavailable. Upload predictions and Grad-CAM "
            "remain disabled; historical calibration is not reused."
        )
    artifact = load_calibration(CALIBRATION_PATH, CHECKPOINT_PATH)
    # Existing validator checks current split-file/checkpoint provenance. Additionally
    # require calibration memberships to equal the approved corrected development lists.
    for split in ("train", "validation"):
        saved = artifact["splits"][split]
        if saved["indices"] != manifest["splits"][split] or saved["count"] != len(manifest["splits"][split]):
            raise ValueError("OOD calibration does not match corrected development membership.")
    if artifact["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("OOD calibration does not belong to the corrected checkpoint.")
    model = load_corrected_checkpoint()
    detector = OODDetector(model)
    detector.feature_mean = artifact["feature_mean"]
    detector.feature_variance = artifact["feature_variance"]
    detector.thresholds = OODThresholds(**artifact["thresholds"])
    return detector


def predict_image(image, *, input_mode="mirabest"):
    """Keep public input guard -> matched OOD gate -> corrected classifier."""
    def rejected(message, ood=None):
        return dict(predicted_class="UNSUPPORTED", confidence=None, probabilities={},
                    message=message, ood=ood, input_mode=input_mode)
    if input_mode == "auto_web":
        return rejected("Corrected-model AUTO / WEB compatibility calibration is unavailable. "
                        "Historical-model predictions are not substituted. " + MODEL_NOTICE)
    try:
        if input_mode == "mirabest":
            guarded = prepare_guarded_image(image)
        elif input_mode == "external_radio":
            guarded = prepare_external_radio_image(image)
        else:
            raise ValueError("Unknown input preparation mode.")
    except Exception as error:
        return rejected(f"Input unavailable: {error}")
    try:
        detector = load_calibrated_ood_detector()
        outcome = detector.validate_tensor(guarded.tensor)
        if not outcome["accepted"]:
            return rejected("Image rejected by the corrected-model OOD gate; no class prediction reported.", outcome)
        with torch.inference_mode():
            logits = detector.model(guarded.tensor)
            if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError("Invalid corrected classifier logits.")
            probabilities = logits.softmax(1)[0]
        values = {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)}
        label = max(values, key=values.get)
        return dict(predicted_class=label, confidence=values[label], probabilities=values,
                    message=MODEL_NOTICE, ood=outcome, input_mode=input_mode,
                    input_handling=input_mode, model="corrected_seed42_01")
    except Exception as error:
        return rejected(f"Corrected-model inference unavailable: {error}")
