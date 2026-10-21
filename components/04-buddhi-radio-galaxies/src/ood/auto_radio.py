"""Deterministic, pixel-preserving candidates for conservative web-map support."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from src.ood.input_guard import GuardedImage
from src.ood.image_modes import canonicalize_web_image

PREPROCESSING_VERSION = "auto-radio-v2-single-hue-value"
PREPROCESSING_CONFIGURATION = {
    # Frozen calibration configuration for canonical candidate generation.
    # Additional web decoders feed equivalent L/RGB images into this same policy.
    "image_size": [150, 150], "grayscale": "PIL L",
    "padding": "symmetric; rounded median of all four source borders; no crop",
    "resize": "PIL bilinear only if size differs", "intensity_remapping": False,
    "tensor": "torchvision ToTensor float32; mean=0.5; std=0.5",
    "alpha": "all pixels must be fully opaque", "max_frames": 1,
    "dimension_limits": [16, 4096], "accepted_modes": ["L", "RGB", "RGBA", "P", "1"],
    "pseudocolor_intensity": "fixed maximum RGB (HSV Value); no stretch; separate external gate",
}
DOMAIN_RULES = {
    "relative_std_min": 0.001, "signal_range_min": 0.001,
    "foreground_fraction_min": 0.005, "foreground_fraction_max": 0.20,
    "foreground_level": 0.2, "border_foreground_max": 0.05,
    "negative_fraction_max": 0.10, "texture_residual_max": 0.35,
    "color_affine_residual_max": 0.05,
    "achromatic_channel_spread_max": 2 / 255.0,
    "active_channel_range_min": 0.01, "color_reference_variance_min": 1e-6,
    "color_residual_region": "union of channel/reference emission above 20% dynamic range",
}
PSEUDOCOLOR_RULES = {
    "quiet_value_max": 0.10, "quiet_fraction_min": 0.80,
    "border_quiet_fraction_min": 0.95, "meaningful_value_min": 0.05,
    "meaningful_relative_level": 0.20, "saturation_min": 0.60,
    "saturated_emission_fraction_min": 0.95, "hue_resultant_min": 0.95,
    "hue_half_width_degrees": 30.0, "dominant_hue_fraction_min": 0.95,
    "channel_value_residual_max": 0.02, "channel_monotonic_tolerance": 2 / 255.0,
}


@dataclass(frozen=True)
class RadioCandidate:
    representation: str
    guarded: GuardedImage


@dataclass(frozen=True)
class AutoCandidates:
    candidates: tuple[RadioCandidate, ...]
    already_compatible: bool


def validate_upload(image: Any) -> Image.Image:
    """Canonicalize web modes before applying the unchanged scientific guards."""
    return canonicalize_web_image(image)


def _border(pixels: np.ndarray) -> np.ndarray:
    return np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))


def check_radio_structure(grayscale: Image.Image) -> None:
    """Representation rules only; neither semantic radio detection nor OOD."""
    pixels = np.asarray(grayscale, dtype=np.float64) / 255.0
    if float(pixels.max()) == float(pixels.min()):
        raise ValueError("Blank or uniform images are unsupported.")
    if float(pixels.std()) < DOMAIN_RULES["relative_std_min"]:
        raise ValueError("Near-uniform images are unsupported.")
    background = float(np.median(pixels))
    span = float(np.percentile(pixels, 99.5)) - background
    if span <= DOMAIN_RULES["signal_range_min"]:
        raise ValueError("Image has insufficient resolved positive emission.")
    diagnostic = np.clip((pixels - background) / span, 0, 1)
    fraction = float(np.mean(diagnostic > DOMAIN_RULES["foreground_level"]))
    if not DOMAIN_RULES["foreground_fraction_min"] <= fraction <= DOMAIN_RULES["foreground_fraction_max"]:
        raise ValueError("Image does not contain a sufficiently compatible radio-morphology representation (dense or insufficient emission).")
    if float(np.mean(_border(diagnostic) > DOMAIN_RULES["foreground_level"])) > DOMAIN_RULES["border_foreground_max"]:
        raise ValueError("Full-frame border structure is unsupported; upload a source map without annotations or a photographic frame.")
    if float(np.mean(pixels < background - 0.2 * span)) > DOMAIN_RULES["negative_fraction_max"]:
        raise ValueError("Structured background is incompatible with a supported positive-emission map.")
    # Diagnostic blur only: never feed a smoothed image to the classifier.
    diagnostic_image = Image.fromarray(np.rint(diagnostic * 255).astype(np.uint8))
    blurred = np.asarray(diagnostic_image.filter(ImageFilter.GaussianBlur(1)), dtype=np.float64) / 255
    strong = diagnostic > DOMAIN_RULES["foreground_level"]
    if float(np.mean(np.abs(diagnostic - blurred)[strong])) > DOMAIN_RULES["texture_residual_max"]:
        raise ValueError("High-frequency texture is incompatible with a supported radio map.")


def standardize_grayscale(image: Image.Image) -> GuardedImage:
    """Keep intensities; estimate padding from the median of the full border."""
    preview = image.convert("L")
    if preview.width != preview.height:
        side = max(preview.size)
        background = int(np.rint(np.median(_border(np.asarray(preview)))))
        square = Image.new("L", (side, side), background)
        square.paste(preview, ((side - preview.width) // 2, (side - preview.height) // 2))
        preview = square
    if preview.size != (150, 150):
        preview = preview.resize((150, 150), Image.Resampling.BILINEAR)
    tensor = transforms.Normalize((0.5,), (0.5,))(transforms.ToTensor()(preview)).unsqueeze(0)
    return GuardedImage(preview, tensor)


def pseudocolor_intensity(source: Image.Image) -> Image.Image | None:
    """Recognize saturated single-family maps, never classifier-selected colors.

    A hue test alone is insufficient. Require the unchanged radio structure/
    texture checks on Value, quiet full-frame background, coherent hue over
    meaningful emission, and RGB channels that are monotonic functions of Value.
    All policy constants are fixed independently of development-image predictions.
    """
    rgb_image = source.convert("RGB")
    rgb8 = np.asarray(rgb_image)
    rgb = rgb8.astype(np.float64) / 255.0
    value = rgb.max(axis=-1)
    rule = PSEUDOCOLOR_RULES
    if float(np.mean(value <= rule["quiet_value_max"])) < rule["quiet_fraction_min"]:
        return None
    if float(np.mean(_border(value) <= rule["quiet_value_max"])) < rule["border_quiet_fraction_min"]:
        return None
    intensity = Image.fromarray(rgb8.max(axis=-1))
    try:
        check_radio_structure(intensity)
    except ValueError:
        return None
    cutoff = max(rule["meaningful_value_min"], rule["meaningful_relative_level"] * float(np.percentile(value, 99.5)))
    emitting = value > cutoff
    if not emitting.any():
        return None
    saturation = (value[emitting] - rgb.min(axis=-1)[emitting]) / value[emitting]
    if float(np.mean(saturation >= rule["saturation_min"])) < rule["saturated_emission_fraction_min"]:
        return None
    # Circular hue statistics correctly join reds either side of hue zero.
    hue = np.asarray(rgb_image.convert("HSV"), dtype=np.float64)[..., 0][emitting] * (2 * np.pi / 255)
    weights = value[emitting]
    resultant = np.sum(weights * np.exp(1j * hue)) / weights.sum()
    if abs(resultant) < rule["hue_resultant_min"]:
        return None
    delta = np.angle(np.exp(1j * (hue - np.angle(resultant))))
    dominant = np.abs(delta) <= np.deg2rad(rule["hue_half_width_degrees"])
    if float(weights[dominant].sum() / weights.sum()) < rule["dominant_hue_fraction_min"]:
        return None
    levels = rgb8.max(axis=-1)[emitting]
    counts = np.bincount(levels, minlength=256)
    occupied = counts > 0
    for channel in range(3):
        values = rgb[..., channel][emitting]
        means = np.bincount(levels, weights=values, minlength=256) / np.maximum(counts, 1)
        if np.any(np.diff(means[occupied]) < -rule["channel_monotonic_tolerance"]):
            return None
        residual = float(np.sqrt(np.mean((values - means[levels]) ** 2)))
        if residual > rule["channel_value_residual_max"]:
            return None
    return intensity


def extract_auto_candidates(image: Any) -> AutoCandidates:
    """Require whole-image compatibility before trying any individual channel.

    Supported color encoding is approximately affine single-structure RGB.
    Independently textured/composite channels fail closed. No candidate-specific
    stretching, source extraction, cropping, inversion, or confidence selection.
    """
    source = validate_upload(image)
    grayscale = source.convert("L")
    if source.mode == "L":
        check_radio_structure(grayscale)
        return AutoCandidates((RadioCandidate("grayscale", standardize_grayscale(grayscale)),),
                              source.size == (150, 150))
    rgb = np.asarray(source.convert("RGB"), dtype=np.float64) / 255.0
    achromatic = float(np.max(np.ptp(rgb, axis=-1))) <= DOMAIN_RULES["achromatic_channel_spread_max"]
    if achromatic:
        check_radio_structure(grayscale)
        return AutoCandidates((RadioCandidate("achromatic_luminance", standardize_grayscale(grayscale)),),
                              source.size == (150, 150))
    recovered = pseudocolor_intensity(source)
    if recovered is not None:
        return AutoCandidates((RadioCandidate("single_hue_hsv_value", standardize_grayscale(recovered)),), False)
    # Preserve the existing affine-color route and its checks if the stricter
    # pseudocolor detector has insufficient evidence. Never stretch a candidate.
    check_radio_structure(grayscale)
    channels = [source.convert("RGB").getchannel(c) for c in "RGB"]
    spreads = np.ptp(rgb.reshape(-1, 3), axis=0)
    reference = rgb[..., int(np.argmax(spreads))].reshape(-1)
    centered = reference - reference.mean()
    variance = float(np.mean(centered ** 2))
    if variance <= DOMAIN_RULES["color_reference_variance_min"]:
        raise ValueError("Color image contains insufficient coherent radio-map structure.")
    # A sparse channel cannot rescue unrelated dense or independently textured
    # content in another channel. Every varying channel must share the structure.
    active_channels = []
    for index, channel in enumerate(channels):
        if spreads[index] < DOMAIN_RULES["active_channel_range_min"]:
            continue
        check_radio_structure(channel)
        values = rgb[..., index].reshape(-1)
        slope = float(np.mean(centered * (values - values.mean()))) / variance
        fitted = values.mean() + slope * centered
        emission = ((values - values.min()) > 0.2 * spreads[index]) | (
            (reference - reference.min()) > 0.2 * float(np.ptp(reference)))
        # Measure coherence where the emission actually is. A mostly empty
        # frame must not dilute a mismatch between independent sparse objects.
        residual = float(np.sqrt(np.mean((values - fitted)[emission] ** 2))) / spreads[index]
        if slope <= 0 or residual > DOMAIN_RULES["color_affine_residual_max"]:
            raise ValueError("Color channels contain inconsistent structures; optical/composite imagery is unsupported.")
        active_channels.append(("RGB"[index] + "_channel", channel))
    maximum = Image.fromarray(np.max(np.asarray(source.convert("RGB")), axis=-1))
    check_radio_structure(maximum)
    representations = [("luminance", grayscale), *active_channels, ("maximum_channel", maximum)]
    candidates = []
    seen = set()
    for name, channel in representations:
        guarded = standardize_grayscale(channel)
        # Exact deduplication avoids counting identical channels as extra votes.
        identity = guarded.image.tobytes()
        if identity not in seen:
            seen.add(identity)
            candidates.append(RadioCandidate(name, guarded))
    return AutoCandidates(tuple(candidates), False)
