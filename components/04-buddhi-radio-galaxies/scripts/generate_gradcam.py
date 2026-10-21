"""Deterministic validation-only Grad-CAM exports for exploratory inspection."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import MiraBestBinarySubset
from src.data.preprocess import PreprocessConfig, build_transforms
from src.inference.predict import CHECKPOINT_PATH, load_checkpoint
from src.ood.calibration import checkpoint_sha256, validate_splits
from src.xai.gradcam import CLASS_NAMES, generate_gradcam

OUTPUT_DIR = PROJECT_ROOT / "outputs/xai/gradcam_phase3a"
NOTICE = "Exploratory — not validated scientific explanations"
SELECTION_RULE = (
    "For each ground-truth class (FRI then FRII), choose the two lowest saved validation "
    "indices within the original archive training partition. Select using labels only, "
    "before model inference; no confidence, correctness, or heatmap-based filtering. "
    "Never fall back to saved test indices or open the archive test_batch for selection."
)


def select_validation_indices(dataset, splits):
    """Read labels only for eligible validation indices; retrieve only four images later."""
    validate_splits(splits)
    selected = []
    for label in (0, 1):
        candidates = [i for i in sorted(splits["validation"])
                      if i < len(dataset) and int(dataset.targets[i]) == label]
        if len(candidates) < 2:
            raise ValueError(f"Need two validation samples of class {label} in the archive training partition.")
        selected.extend(candidates[:2])
    return selected


def export_sample(image, result, destination, metadata):
    """Save exact arrays and labeled views; colormaps/alpha affect display only."""
    destination.mkdir()
    raw = result.raw_cam.cpu().numpy()
    heat = result.heatmap.cpu().numpy()
    np.save(destination / "raw_cam.npy", raw, allow_pickle=False)
    np.save(destination / "upsampled_cam.npy", result.upsampled_heatmap.cpu().numpy(), allow_pickle=False)
    np.save(destination / "heatmap.npy", heat, allow_pickle=False)
    # PNG text metadata identifies the unaltered original pixels as exploratory too.
    from PIL.PngImagePlugin import PngInfo
    png_info = PngInfo()
    png_info.add_text("Description", NOTICE)
    image.save(destination / "original.png", pnginfo=png_info)
    rgba = colormaps["inferno"](heat, bytes=True)
    rgba[..., 3] = np.rint(heat * 0.5 * 255).astype(np.uint8)
    if result.status == "degenerate":
        rgba[..., 3] = 0
    color_layer = Image.fromarray(rgba, mode="RGBA")
    color_layer.save(destination / "heatmap_transparent.png", pnginfo=png_info)
    overlay = Image.alpha_composite(image.convert("RGBA"), color_layer)
    overlay.save(destination / "overlay.png", pnginfo=png_info)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[1].imshow(raw, cmap="inferno", vmin=0)
    axes[2].imshow(heat, cmap="inferno", vmin=0, vmax=1)
    axes[3].imshow(overlay)
    for ax, title in zip(axes, ("Original grayscale", "Raw CAM (19×19)",
                                "Normalized CAM (150×150)", "Transparent overlay")):
        ax.set_title(title)
        ax.axis("off")
    figure.suptitle(f"{NOTICE}\nValidation index {metadata['dataset_index']} | "
                   f"True {metadata['ground_truth_class']} | Predicted {metadata['predicted_class']} | "
                   f"confidence {metadata['confidence']:.4f} | {result.status}")
    figure.tight_layout()
    figure.savefig(destination / "exploratory_panel.png", dpi=150)
    plt.close(figure)


def run_export(config_path=PROJECT_ROOT / "config.yaml", output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 3A exports: {output_dir}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
    splits = json.loads(split_path.read_bytes())
    validate_splits(splits)
    # This is the first partition of the project's combined dataset, so its
    # indices are unchanged. Do not instantiate the combined loader/test partition.
    dataset = MiraBestBinarySubset(PROJECT_ROOT / config["paths"]["mirabest_root"], train=True)
    if dataset.classes != ["FRI", "FRII"]:
        raise ValueError("Phase 3A requires the frozen Zenodo binary dataset ordering.")
    indices = select_validation_indices(dataset, splits)
    preprocessing = config["preprocessing"]
    _, transform = build_transforms(PreprocessConfig(
        normalize_mean=float(preprocessing["normalize_mean"]),
        normalize_std=float(preprocessing["normalize_std"])))
    checkpoint_hash = checkpoint_sha256(CHECKPOINT_PATH)
    load_checkpoint.cache_clear()
    model = load_checkpoint(str(CHECKPOINT_PATH))
    torch.set_num_threads(1)
    records = []
    output_dir.mkdir(parents=True)
    for index in indices:
        image, label = dataset[index]
        if image.mode != "L" or image.size != (150, 150):
            raise ValueError("Expected original grayscale 150x150 MiraBest image; no rescaling is applied.")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            baseline = model(tensor).detach().clone()
        result = generate_gradcam(model, tensor)
        if not torch.equal(baseline, result.logits):
            raise RuntimeError("Grad-CAM forward logits differ from classifier baseline.")
        confidence = float(result.logits.softmax(1)[0, result.predicted_class])
        name = f"validation_{index:04d}_{CLASS_NAMES[label]}"
        record = {
            "dataset_index": index, "dataset_split": "validation",
            "ground_truth_class": CLASS_NAMES[label], "ground_truth_class_index": label,
            "predicted_class": CLASS_NAMES[result.predicted_class],
            "predicted_class_index": result.predicted_class, "confidence": confidence,
            "gradcam_target_class": CLASS_NAMES[result.target_class],
            "target_class_index": result.target_class, "target_layer": result.layer_name,
            "native_cam_shape": list(result.raw_cam.shape), "heatmap_shape": list(result.heatmap.shape),
            "status": result.status, "checkpoint_sha256": checkpoint_hash,
            "checkpoint_path": str(CHECKPOINT_PATH), "split_sha256": checkpoint_sha256(split_path),
            "selection_rule": SELECTION_RULE, "notice": NOTICE, "output_directory": name,
            "logits": result.logits.cpu().tolist()[0], "baseline_logits_equal": True,
        }
        export_sample(image, result, output_dir / name, record)
        records.append(record)
    if checkpoint_sha256(CHECKPOINT_PATH) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during export.")
    (output_dir / "metadata.json").write_text(json.dumps({
        "notice": NOTICE, "selection_rule": SELECTION_RULE,
        "preprocessing": "Existing validation ToTensor + Normalize; no augmentation or image remapping.",
        "normalize_mean": preprocessing["normalize_mean"], "normalize_std": preprocessing["normalize_std"],
        "overlay": "inferno colormap; alpha = 0.5 * normalized CAM; zero alpha when degenerate",
        "interpretation": "Classifier-logit attribution only; no OOD acceptance or scientific validation asserted.",
        "samples": records,
    }, indent=2) + "\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    print(json.dumps(run_export(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
