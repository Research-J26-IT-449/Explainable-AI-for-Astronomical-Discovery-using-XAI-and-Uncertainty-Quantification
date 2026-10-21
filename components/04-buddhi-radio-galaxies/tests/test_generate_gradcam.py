"""Phase 3A export tests using synthetic images, never locked data."""

import json
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from scripts import generate_gradcam as exporter
from src.models.resnet_cnn import ResNetCNN
from src.xai.gradcam import GradCAMResult


SPLITS = {"train": [0, 2], "validation": [7, 5, 3, 1], "test": [4, 6]}


class SyntheticDataset:
    classes = ["FRI", "FRII"]
    targets = [0, 0, 1, 0, 0, 1, 1, 1]

    def __init__(self):
        self.visited = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        assert index in SPLITS["validation"], "Non-validation image accessed"
        self.visited.append(index)
        pixels = np.arange(150 * 150, dtype=np.uint8).reshape(150, 150)
        return Image.fromarray(pixels), self.targets[index]


def test_selection_is_sorted_label_only_validation_and_reproducible():
    dataset = SyntheticDataset()
    assert exporter.select_validation_indices(dataset, SPLITS) == [1, 3, 5, 7]
    assert exporter.select_validation_indices(dataset, SPLITS) == [1, 3, 5, 7]
    assert dataset.visited == []
    insufficient = {"train": [0, 2, 3], "validation": [1, 5, 7], "test": [4, 6]}
    with pytest.raises(ValueError, match="Need two"):
        exporter.select_validation_indices(dataset, insufficient)
    with pytest.raises(ValueError, match="overlap"):
        exporter.select_validation_indices(dataset, {**SPLITS, "test": [1, 4, 6]})


def test_export_exact_four_preserves_preprocessing_and_numeric_maps(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "PROJECT_ROOT", tmp_path)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_file = split_dir / "split_indices.json"
    split_file.write_text(json.dumps(SPLITS))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"splits": "splits", "mirabest_root": "unused"},
                                    "preprocessing": {"normalize_mean": 0.5, "normalize_std": 0.5}}))
    checkpoint = tmp_path / "locked.pth"
    checkpoint.write_bytes(b"unchanged synthetic checkpoint")
    monkeypatch.setattr(exporter, "CHECKPOINT_PATH", checkpoint)
    dataset = SyntheticDataset()
    source_loader = Mock(return_value=dataset)
    monkeypatch.setattr(exporter, "MiraBestBinarySubset", source_loader)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = ResNetCNN().eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(exporter, "load_checkpoint", Mock(return_value=model))
    original_gradcam = exporter.generate_gradcam
    captured = []

    def record_cam(model, tensor):
        expected = torch.from_numpy(np.arange(150 * 150, dtype=np.uint8).reshape(150, 150).astype(np.float32))
        expected = (expected / 255 - 0.5) / 0.5
        assert torch.equal(tensor[0, 0], expected)
        result = original_gradcam(model, tensor)
        captured.append(result)
        return result

    monkeypatch.setattr(exporter, "generate_gradcam", record_cam)
    out = tmp_path / "exports"
    records = exporter.run_export(config, out)
    source_loader.assert_called_once_with(tmp_path / "unused", train=True)
    assert dataset.visited == [1, 3, 5, 7]
    assert len(records) == 4
    assert [r["ground_truth_class"] for r in records] == ["FRI", "FRI", "FRII", "FRII"]
    for record, cam in zip(records, captured):
        folder = out / record["output_directory"]
        assert np.array_equal(np.load(folder / "raw_cam.npy"), cam.raw_cam.numpy())
        assert np.array_equal(np.load(folder / "heatmap.npy"), cam.heatmap.numpy())
        assert record["target_class_index"] == record["predicted_class_index"]
        assert record["native_cam_shape"] == [19, 19]
        assert record["heatmap_shape"] == [150, 150]
        with Image.open(folder / "original.png") as image:
            assert "Exploratory" in image.info["Description"]
            assert image.mode == "L" and image.size == (150, 150)
        assert (folder / "exploratory_panel.png").is_file()
        assert (folder / "overlay.png").is_file()
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["samples"] == records
    assert checkpoint.read_bytes() == b"unchanged synthetic checkpoint"
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    with pytest.raises(FileExistsError):
        exporter.run_export(config, out)


def test_degenerate_overlay_is_transparent_and_raw_values_unchanged(tmp_path):
    image = Image.new("L", (150, 150), 64)
    cam = GradCAMResult(torch.ones(19, 19), torch.ones(150, 150), torch.zeros(150, 150),
                        0, 0, torch.tensor([[1., 0.]]), "blocks.2.convolution.3", "degenerate")
    folder = tmp_path / "sample"
    exporter.export_sample(image, cam, folder, {"dataset_index": 1, "ground_truth_class": "FRI",
                           "predicted_class": "FRI", "confidence": 0.73})
    with Image.open(folder / "heatmap_transparent.png") as layer:
        assert np.asarray(layer)[..., 3].max() == 0
    with Image.open(folder / "overlay.png") as overlay:
        assert overlay.convert("L").tobytes() == image.tobytes()
    assert np.array_equal(np.load(folder / "raw_cam.npy"), np.ones((19, 19), dtype=np.float32))
