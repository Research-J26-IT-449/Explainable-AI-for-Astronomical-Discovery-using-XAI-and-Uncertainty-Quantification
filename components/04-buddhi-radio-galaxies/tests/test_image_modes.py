"""Web-mode decoding regressions; scientific guards and gates stay mandatory."""

import numpy as np
from PIL import Image
import pytest
import torch

from src.inference import predict
from src.ood.auto_radio import extract_auto_candidates
from src.ood.external_radio import prepare_external_radio_image
from src.ood.image_modes import canonicalize_web_image


def radio_map():
    y, x = np.mgrid[:150, :150]
    pixels = 210 * np.exp(-((x - 35)**2 + (y - 75)**2) / 60)
    pixels += 160 * np.exp(-((x - 115)**2 + (y - 75)**2) / 60)
    return Image.fromarray(np.rint(pixels).astype(np.uint8))


def encoded_map(mode):
    original = radio_map()
    pixels = np.asarray(original)
    if mode == "P":
        # Non-identity palette: palette indices are NOT luminance values.
        image = Image.fromarray(255 - pixels).convert("P")
        image.putpalette([value for index in range(256) for value in (255 - index,) * 3])
        return image, original
    if mode == "I":
        return Image.fromarray(pixels.astype(np.int32)), original
    if mode in {"I;16", "I;16B"}:
        dtype = ">u2" if mode == "I;16B" else "<u2"
        return Image.fromarray((pixels.astype(np.uint16) * 257).astype(dtype)), original
    if mode == "1":
        image = original.point(lambda value: 255 if value >= 30 else 0, mode="1")
        return image, image.convert("L")
    if mode == "CMYK":
        image = original.convert("RGB").convert("CMYK")
        return image, image.convert("RGB").convert("L")
    image = original.convert(mode)
    return image, original


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "LA", "P", "1", "I", "I;16", "I;16B", "CMYK"])
@pytest.mark.parametrize("path", ["external_radio", "auto_web"])
def test_decoding_before_scientific_checks_retains_expected_pixels(mode, path):
    image, expected = encoded_map(mode)
    decoded = canonicalize_web_image(image)
    assert decoded.mode in {"L", "RGB"}
    assert decoded.size == expected.size
    assert decoded.convert("L").tobytes() == expected.tobytes()
    if path == "external_radio":
        guarded = prepare_external_radio_image(image)
    else:
        prepared = extract_auto_candidates(image)
        assert prepared.already_compatible
        guarded = prepared.candidates[0].guarded
    assert guarded.image.tobytes() == expected.tobytes()
    again = prepare_external_radio_image(image) if path == "external_radio" else extract_auto_candidates(image).candidates[0].guarded
    assert torch.equal(guarded.tensor, again.tensor)


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
@pytest.mark.parametrize("path", [prepare_external_radio_image, extract_auto_candidates])
def test_transparency_cannot_disappear_during_mode_conversion(mode, path):
    image, _ = encoded_map(mode)
    if mode == "P":
        # Used palette entry has partial transparency.
        alpha = bytearray([255] * 256)
        alpha[int(np.asarray(image)[75, 35])] = 254
        image.info["transparency"] = bytes(alpha)
    else:
        image.putalpha(254)
    with pytest.raises(ValueError, match="Transparent"):
        path(image)


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_existing_modes_preserved_exactly(mode):
    image, _ = encoded_map(mode)
    decoded = canonicalize_web_image(image)
    assert decoded.mode == image.mode
    assert decoded.tobytes() == image.tobytes()


def test_16bit_uses_fixed_full_scale_not_per_image_contrast_stretch():
    pixels = np.array([[0, 257, 32768, 65535]] * 16, dtype=np.uint16)
    image = Image.fromarray(np.tile(pixels, (1, 4)))
    converted = np.asarray(canonicalize_web_image(image))
    assert np.array_equal(converted[0, :4], [0, 1, 128, 255])
    dim = Image.fromarray(np.full((16, 16), 257, dtype=np.uint16))
    assert np.asarray(canonicalize_web_image(dim)).max() == 1


def test_i_mode_16bit_container_has_same_fixed_mapping():
    image = Image.fromarray(np.asarray(radio_map(), dtype=np.int32) * 257)
    assert canonicalize_web_image(image).tobytes() == radio_map().tobytes()


@pytest.mark.parametrize("value", [-1, 65536])
def test_integer_out_of_range_rejected(value):
    image = Image.fromarray(np.full((16, 16), value, dtype=np.int32))
    with pytest.raises(ValueError, match="Signed or overrange"):
        canonicalize_web_image(image)


@pytest.mark.parametrize("path", ["external_radio", "auto_web"])
def test_palette_plot_with_axes_remains_unsupported_before_model(monkeypatch, path):
    pixels = np.asarray(radio_map()).copy()
    pixels[0:3, :] = 255
    pixels[:, 0:3] = 255
    image = Image.fromarray(pixels).convert("P")
    def forbidden():
        raise AssertionError("A plot failing scientific guards reached the model")
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", forbidden)
    result = predict.predict_image(image, input_mode=path)
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["probabilities"] == {}
