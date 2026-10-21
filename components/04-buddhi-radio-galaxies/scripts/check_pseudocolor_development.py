"""Describe fixed-policy DEVELOPMENT results; these files are not final evaluation."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_image
from src.ood.calibration import checkpoint_sha256
from src.ood.auto_radio import PREPROCESSING_VERSION
from src.ood.external_compatibility import EXTERNAL_CALIBRATION_PATH


def check_development_images(three_c31: Path, cygnus_a: Path):
    """Evaluate once with the existing gate; never fit/tune on these examples."""
    rows = []
    for name, label, path in (("3C31", "FRI", three_c31), ("Cygnus A", "FRII", cygnus_a)):
        digest = checkpoint_sha256(path)
        with Image.open(path) as image:
            result = predict_image(image, input_mode="auto_web")
        rows.append({"name": name, "true_label": label, "path": str(path.resolve()),
                     "sha256": digest, "usage": "development_only",
                     "eligible_for_untouched_final_evaluation": False, "result": result})
    return {"preprocessing_version": PREPROCESSING_VERSION,
            "external_calibration_sha256": checkpoint_sha256(EXTERNAL_CALIBRATION_PATH),
            "warning": "These two specific images were development examples; exclude them from untouched final external evaluation.",
            "images": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-c31", type=Path, required=True)
    parser.add_argument("--cygnus-a", type=Path, required=True)
    args = parser.parse_args()
    report = check_development_images(args.three_c31, args.cygnus_a)
    output = PROJECT_ROOT / "outputs/ood/pseudocolor_development_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["images"]:
        result = row["result"]
        print(f"{row['name']} | known {row['true_label']} | AUTO {result['predicted_class']} | {result['message']}")
    print(f"Development-only report and image SHA256 exclusion records: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
