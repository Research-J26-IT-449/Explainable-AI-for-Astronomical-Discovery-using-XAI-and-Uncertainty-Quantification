"""Safe web-image decoding before radio checks; never infer a contrast stretch."""

from typing import Any

import numpy as np
from PIL import Image


def canonicalize_web_image(image: Any) -> Image.Image:
    """Return L or RGB without geometric changes, after integrity/alpha checks.

    Preserve 8-bit pixels. Unsigned 16-bit samples use the fixed mapping
    round(value / 257), not image-dependent min/max or percentile scaling.
    I-mode values in [0,255] are preserved; larger nonnegative I values must
    fit unsigned 16-bit and use that same full-range mapping. Signed/overrange
    and floating-point maps have no supported display convention and fail closed.
    Quantization may lose faint levels; this is decoding, not radio validation.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Please upload one valid radio-galaxy PIL image.")
    try:
        if min(image.size) < 16 or max(image.size) > 4096:
            raise ValueError("Image dimensions must be between 16 and 4096 pixels.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated or multi-frame images are unsupported.")
        image.load()
        raw = np.asarray(image)
        if not np.isfinite(raw).all():
            raise ValueError("Image contains non-finite pixel values.")
        mode = image.mode
        supported = {"L", "RGB", "RGBA", "LA", "P", "1", "I", "I;16", "I;16L", "I;16B", "I;16N", "CMYK"}
        if mode not in supported:
            raise ValueError("Unsupported pixel mode or unspecified intensity scale; supply an integer grayscale or RGB display map.")
        # Check original alpha, including palette alpha and PNG transparency
        # metadata, before conversions can discard it. Never composite over black.
        if mode in {"RGBA", "LA", "P"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            if not np.all(np.asarray(rgba)[..., 3] == 255):
                raise ValueError("Transparent images are unsupported; supply an opaque radio map.")
        if mode in {"L", "RGB"}:
            return image.copy()
        if mode in {"LA", "1"}:
            return image.convert("L")
        if mode in {"RGBA", "P", "CMYK"}:
            return image.convert("RGB")
        # Inspect native integer values rather than PIL convert('L'), which
        # clips high-bit images at 255 and can erase radio intensity structure.
        values = raw.astype(np.int64)
        low, high = int(values.min()), int(values.max())
        if low < 0 or high > 65535:
            raise ValueError("Signed or overrange integer pixels have an unspecified display scale.")
        if mode == "I" and high <= 255:
            pixels = values.astype(np.uint8)
        else:
            pixels = np.rint(values.astype(np.float64) / 257.0).astype(np.uint8)
        return Image.fromarray(pixels)
    except (OSError, TypeError, SyntaxError) as error:
        raise ValueError("The uploaded file is malformed or corrupted.") from error
