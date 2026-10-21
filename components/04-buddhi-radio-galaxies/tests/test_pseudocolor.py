"""Fixed-policy pseudocolor recovery and safety; no TEST-based tuning."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
from sklearn.datasets import load_sample_image
import torch

from src.data.download_mirabest import load_mirabest
from src.inference import predict, auto
from src.ood.auto_radio import extract_auto_candidates, pseudocolor_intensity
from src.ood.external_compatibility import pseudocolor_variants
from src.ood.input_guard import prepare_guarded_image

ROOT = Path(__file__).resolve().parents[1]


def synthetic_radio(size=(150, 150)):
    width, height = size
    y, x = np.mgrid[:height, :width]
    p = 220 * np.exp(-((x - width * .15)**2 + (y - height * .5)**2) / 60)
    p += 180 * np.exp(-((x - width * .85)**2 + (y - height * .5)**2) / 60)
    return Image.fromarray(np.rint(p).astype(np.uint8))


@pytest.mark.parametrize("family", ["pseudocolor_red_orange", "pseudocolor_blue_cyan", "pseudocolor_green"])
def test_fixed_value_recovery_preserves_all_original_pixels(family):
    image = synthetic_radio()
    color = dict(pseudocolor_variants(image))[family]
    assert pseudocolor_intensity(color).tobytes() == image.tobytes()
    prepared = extract_auto_candidates(color)
    assert not prepared.already_compatible
    assert len(prepared.candidates) == 1
    candidate = prepared.candidates[0]
    assert candidate.representation == "single_hue_hsv_value"
    assert torch.equal(candidate.guarded.tensor, prepare_guarded_image(image).tensor)
    assert torch.equal(candidate.guarded.tensor, extract_auto_candidates(color).candidates[0].guarded.tensor)


def test_rectangular_pseudocolor_pads_without_cropping():
    image = synthetic_radio((200, 100))
    color = dict(pseudocolor_variants(image))["pseudocolor_blue_cyan"]
    prepared = extract_auto_candidates(color).candidates[0].guarded.image
    expected = Image.new("L", (200, 200), 0)
    expected.paste(image, (0, 50))
    assert prepared.tobytes() == expected.resize((150, 150), Image.Resampling.BILINEAR).tobytes()
    pixels = np.asarray(prepared)
    assert pixels[:, :35].max() > 150 and pixels[:, -35:].max() > 150


@pytest.mark.parametrize("kind", ["red_dominant_photo", "dark_natural_photo", "optical_composite", "single_channel_texture"])
def test_hue_alone_cannot_bypass_unrelated_content_guard(monkeypatch, kind):
    # These photographic negatives are bundled with sklearn; no downloads or
    # reserved external OOD images are read. Tint/dimming is defined beforehand.
    if kind == "single_channel_texture":
        pixels = np.zeros((150, 150, 3), dtype=np.uint8)
        pixels[40:80:2, 40:80:2, 0] = 255
    else:
        photo = load_sample_image("flower.jpg")
        if kind == "red_dominant_photo":
            gray = np.asarray(Image.fromarray(photo).convert("L"))
            pixels = np.stack((gray, np.rint(gray * .2), np.zeros_like(gray)), axis=-1).astype(np.uint8)
        elif kind == "dark_natural_photo":
            pixels = np.rint(photo.astype(float) * .08).astype(np.uint8)
        else:
            pixels = photo.copy()
            pixels[50:150, 50:150, 0] = 255
    image = Image.fromarray(pixels)
    loader = Mock(side_effect=AssertionError("Unrelated content reached the model"))
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", loader)
    result = predict.predict_image(image, input_mode="auto_web")
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["probabilities"] == {}
    loader.assert_not_called()


def test_sparse_independent_composite_is_not_pseudocolor():
    gray = np.asarray(synthetic_radio())
    different = np.roll(gray, 18, axis=0)
    image = Image.fromarray(np.stack((gray, different, np.zeros_like(gray)), axis=-1))
    assert pseudocolor_intensity(image) is None
    with pytest.raises(ValueError, match="inconsistent"):
        extract_auto_candidates(image)


def test_pseudocolor_still_requires_external_gate(monkeypatch):
    original = Mock()
    external = Mock()
    external.validate_tensor.return_value = {"accepted": False, "reason": "feature rejection"}
    external.model.return_value = torch.tensor([[3., 0.]])
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: original)
    monkeypatch.setattr(auto, "load_external_detector", lambda *args: external)
    color = dict(pseudocolor_variants(synthetic_radio()))["pseudocolor_red_orange"]
    result = predict.predict_image(color, input_mode="auto_web")
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["probabilities"] == {}
    external.validate_tensor.assert_called_once()
    original.validate_tensor.assert_not_called()


@pytest.fixture(scope="module")
def training_examples():
    splits = json.loads((ROOT / "data/splits/split_indices.json").read_bytes())
    dataset = load_mirabest(ROOT / "data/raw/mirabest", download=False)
    # First TRAIN example per true class, without selecting by model confidence.
    selected = {}
    for index in splits["train"]:
        image, label = dataset[index]
        if label not in selected:
            selected[label] = (index, image)
        if len(selected) == 2:
            break
    assert all(index in splits["train"] for index, _ in selected.values())
    return selected


@pytest.mark.parametrize("label", [0, 1])
@pytest.mark.parametrize("family", ["pseudocolor_red_orange", "pseudocolor_blue_cyan"])
def test_known_train_samples_retain_fr_predictions(training_examples, label, family):
    _, image = training_examples[label]
    original = predict.predict_image(image)
    color = dict(pseudocolor_variants(image))[family]
    candidate = extract_auto_candidates(color).candidates[0]
    assert torch.equal(candidate.guarded.tensor, prepare_guarded_image(image).tensor)
    result = predict.predict_image(color, input_mode="auto_web")
    assert result["predicted_class"] == original["predicted_class"]
    assert result["predicted_class"] in {"FRI", "FRII"}
    assert result["probabilities"] == original["probabilities"]
