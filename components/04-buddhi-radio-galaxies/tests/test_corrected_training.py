"""Synthetic-only corrected loading/training preparation tests. No real epochs."""

from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import pickletools
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import corrected_protocol as cp
from src.data import mirabest_pickle_stream_reader as legacy
from src.data.corrected_training_data import (
    CorrectedArchiveReader, CorrectedTrainingDataset, TrainingRow, selected_offsets,
)
from src.data.preprocess import PreprocessConfig, build_transforms
from src.models import corrected_training as training
from test_corrected_mirabest_protocol import evidence, disk_evidence
from test_mirabest_pickle_stream_reader import _fixture, _archive_fixture


@pytest.fixture
def cohort(evidence):
    manifest, _ = cp.build_manifests(*evidence)
    return training.PreparedCohort(tuple(TrainingRow.from_manifest(r) for r in manifest["records"]), "synthetic")


def test_metadata_loader_revalidates_corrected_manifest(disk_evidence):
    root, pins = disk_evidence
    report = cp.generate_protocol(root, pins=pins)
    cohort = training.load_cohort(root, pins=pins, expected_manifest=report["canonical_membership_sha256"])
    assert len(cohort.rows) == 668
    assert not set(cp.QUARANTINE) & {r.dataset_index for r in cohort.rows}
    path = root / cp.OUTPUT_REFERENCE / "manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        training.load_cohort(root, pins=pins, expected_manifest=report["canonical_membership_sha256"])


def test_manifest_views_preserve_membership_and_labels(cohort):
    class NoRead:
        def read_selected(self, indices):
            raise AssertionError("Metadata construction must not decode")
    tr, va = build_transforms()
    train = CorrectedTrainingDataset(cohort.rows, "train", tr, NoRead())
    val = CorrectedTrainingDataset(cohort.rows, "validation", va, NoRead())
    assert len(train) == 552 and len(val) == 116
    assert not set(train.indices) & set(val.indices)
    assert not (set(train.indices) | set(val.indices)) & set(cp.QUARANTINE)
    assert sum(r.label == 0 for r in train.rows) == 245
    assert sum(r.label == 0 for r in val.rows) == 51
    with pytest.raises(ValueError):
        CorrectedTrainingDataset(cohort.rows, "test", va, NoRead())


def test_loader_fetches_32_as_four_bounded_calls_and_releases_raw(cohort):
    pending = []
    calls = []
    class FakeReader:
        def read_selected(self, indices):
            assert all(not previous for previous in pending), "Previous raw batch retained"
            assert len(indices) <= 8 and not set(indices) & set(cp.QUARANTINE)
            calls.append(tuple(indices))
            raw = {i: bytes([i % 256]) * 22500 for i in indices}
            pending.append(raw)
            return raw
    _, transform = build_transforms()
    dataset = CorrectedTrainingDataset(cohort.rows, "train", transform, FakeReader())
    images, labels = next(iter(DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)))
    assert images.shape == (32, 1, 150, 150) and images.dtype == torch.float32
    assert [len(c) for c in calls] == [8] * 4
    assert [i for call in calls for i in call] == list(dataset.indices[:32])
    assert labels.tolist() == [r.label for r in dataset.rows[:32]]
    assert torch.allclose(images[0], torch.full_like(images[0], dataset.indices[0] % 256 / 255 * 2 - 1))
    assert all(not raw for raw in pending)


def test_invalid_dataset_positions_rejected_before_reader(cohort):
    class NoRead:
        def read_selected(self, indices):
            raise AssertionError("Invalid positions must not reach reader")
    dataset = CorrectedTrainingDataset(cohort.rows, "train", lambda x: x, NoRead())
    for index in [-1, 552, True]:
        with pytest.raises(ValueError):
            dataset[index]


@pytest.mark.parametrize("indices", [[198], [323], [791], [True], list(range(9)), [0, 0]])
def test_unauthorized_or_unbounded_indices_rejected_before_archive_open(indices, monkeypatch):
    row = TrainingRow(0, "train", 0, "batches/data_batch_1", 0, "0" * 64)
    reader = CorrectedArchiveReader("never_open", [row], archive_sha256="x", archive_md5="y")
    monkeypatch.setattr(legacy, "_file_digest", lambda *a: pytest.fail("Archive must not open"))
    with pytest.raises(ValueError):
        reader.read_selected(indices)


def test_scanner_never_reads_any_archived_label_or_unselected_pixel(tmp_path):
    path = _fixture(tmp_path / "batch")
    raw = path.read_bytes()
    labels = False
    forbidden = []
    image = 0
    selected_position = None
    for op, value, position in pickletools.genops(raw):
        if op.name in {"BINUNICODE", "SHORT_BINUNICODE"}:
            if value == "labels":
                labels = True
            elif value == "data":
                labels = False
        if labels and op.name == "BININT1":
            forbidden.append((position + 1, position + 2))
        if op.name == "BINBYTES" and len(value) == 22500:
            if image != 1:
                forbidden.append((position + 5, position + 5 + 22500))
            else:
                selected_position = position + 5
            image += 1
    assert len(forbidden) == 88 + 87
    class GuardedStream(io.BytesIO):
        def read(self, size=-1):
            start = self.tell()
            end = len(raw) if size < 0 else start + size
            assert all(end <= a or start >= b for a, b in forbidden), "Forbidden label/pixel read"
            return super().read(size)
    stream = GuardedStream(raw)
    offsets = selected_offsets(stream, "batches/data_batch_1", {1})
    assert offsets == {1: selected_position}
    stream.seek(offsets[1])
    assert len(stream.read(22500)) == 22500


def test_synthetic_archive_selected_pixels_checked_against_saved_hash(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    _archive_fixture(archive)
    # Existing synthetic fixture fills each image with its row value.
    pixels = np.zeros((150, 150), dtype=np.uint8)
    pixels[0, 0] = 1
    row = TrainingRow(1, "train", 1, "batches/data_batch_1", 1, legacy.canonical_sha256(pixels.tobytes()))
    reader = CorrectedArchiveReader(archive, [row], archive_sha256=legacy._file_digest(archive, "sha256"),
                                    archive_md5=legacy._file_digest(archive, "md5"))
    assert reader.read_selected([1]) == {1: pixels.tobytes()}
    bad = CorrectedArchiveReader(archive, [replace(row, content_sha256="0" * 64)],
                                 archive_sha256=reader.sha256, archive_md5=reader.md5)
    with pytest.raises(ValueError, match="content differs"):
        bad.read_selected([1])


def test_fresh_seeded_model_never_loads_checkpoint(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No warm start allowed"))
    first = training.fresh_model()
    second = training.fresh_model()
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))
    different = training.fresh_model(43)
    assert not torch.equal(first.stem[0].weight, different.stem[0].weight)
    for module in first.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert torch.all(module.running_mean == 0) and torch.all(module.running_var == 1)


def test_gate_defaults_disabled_before_metadata_or_data(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "load_profile", lambda *a: pytest.fail("Gate must run first"))
    with pytest.raises(PermissionError):
        training.execute_training(tmp_path, "run01")
    assert not (tmp_path / training.OUTPUT_PARENT).exists()


@pytest.mark.parametrize("run_id", ["../escape", "", "a/b", None])
def test_output_namespace_rejects_unsafe_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError):
        training.isolated_run_directory(tmp_path, run_id)


def test_existing_output_is_never_resumed(tmp_path):
    path = training.isolated_run_directory(tmp_path, "synthetic")
    path.mkdir(parents=True)
    marker = path / "previous.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="resume/overwrite"):
        training.execute_training(tmp_path, "synthetic", execute=True)
    assert marker.read_text() == "preserve"


def test_training_settings_and_existing_transform_conventions():
    assert asdict(training.TRAINING) == dict(seed=42, batch_size=32, learning_rate=0.0003,
        weight_decay=0.0001, num_epochs=30, num_workers=0, early_stopping_patience=7,
        scheduler_patience=2, scheduler_factor=0.5, device="cpu")
    train, val = build_transforms()
    assert [type(t).__name__ for t in train.transforms] == ["RandomRotation", "RandomHorizontalFlip", "RandomVerticalFlip", "ToTensor", "Normalize"]
    assert [type(t).__name__ for t in val.transforms] == ["ToTensor", "Normalize"]
    assert train.transforms[0].degrees == [-15.0, 15.0]
    assert val.transforms[-1].mean == (0.5,) and val.transforms[-1].std == (0.5,)


@pytest.mark.parametrize("fail", [False, True])
def test_execution_orchestration_uses_existing_trainer_in_new_namespace_only(tmp_path, monkeypatch, cohort, fail):
    # Mock the trainer: this test performs NO epochs, optimizer steps or inference.
    monkeypatch.setattr(training, "load_profile", lambda root: {"training": asdict(training.TRAINING)})
    monkeypatch.setattr(training, "load_cohort", lambda root: cohort)
    monkeypatch.setattr(training, "build_views", lambda *a: ("synthetic_train", "synthetic_validation"))
    original_hash = cp.sha256_file
    def hash_or_synthetic_source(path):
        path = Path(path)
        if path.as_posix().endswith(cp.ARCHIVE_REFERENCE):
            return cp.FROZEN_PINS.archive_sha256
        return original_hash(path) if path.is_file() else "0" * 64
    monkeypatch.setattr(cp, "sha256_file", hash_or_synthetic_source)
    calls = []
    def fake_trainer(train, validation, **kwargs):
        calls.append(kwargs)
        assert (train, validation) == ("synthetic_train", "synthetic_validation")
        assert kwargs["config"] == training.TRAINING
        assert isinstance(kwargs["model"], training.ResNetCNN)
        assert kwargs["output_root"] == tmp_path / training.OUTPUT_PARENT / "synthetic"
        if fail:
            raise RuntimeError("Synthetic interruption")
        result = {"best_epoch": 1}
        for key, filename in [("checkpoint_path", "models/resnet_cnn_best.pth"),
                              ("history_path", "metrics/resnet_training_history.csv"),
                              ("curves_path", "figures/resnet_training_curves.png"),
                              ("validation_metrics_path", "metrics/resnet_validation_metrics.json")]:
            path = kwargs["output_root"] / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic test output only")
            result[key] = path
        return result
    monkeypatch.setattr(training, "train_model", fake_trainer)
    output = tmp_path / training.OUTPUT_PARENT / "synthetic"
    if fail:
        with pytest.raises(RuntimeError, match="interruption"):
            training.execute_training(tmp_path, "synthetic", execute=True)
        assert (output / "failure.json").is_file() and not (output / "complete.json").exists()
    else:
        training.execute_training(tmp_path, "synthetic", execute=True)
        assert json.loads((output / "complete.json").read_text())["status"] == "complete"
    assert len(calls) == 1
    assert not (tmp_path / "outputs/models/resnet_cnn_best.pth").exists()
    with pytest.raises(ValueError, match="resume/overwrite"):
        training.execute_training(tmp_path, "synthetic", execute=True)


@pytest.mark.parametrize("change", ["seed", "manifest", "unknown_setting"])
def test_modified_profile_fails_closed_before_checkpoint_access(tmp_path, monkeypatch, change):
    import yaml
    # Static new preparation config, not dataset or historical checkpoint data.
    source = Path(training.__file__).resolve().parents[2] / training.PROFILE
    config = yaml.safe_load(source.read_text())
    if change == "seed":
        config["training"]["seed"] = 7
    elif change == "manifest":
        config["manifest_sha256"] = "0" * 64
    else:
        config["warm_start"] = True
    target = tmp_path / training.PROFILE
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(config))
    monkeypatch.setattr(cp, "sha256_file", lambda *a: pytest.fail("Reject altered profile before provenance IO"))
    with pytest.raises(ValueError, match="profile differs"):
        training.load_profile(tmp_path)


def test_reused_source_change_prevents_preparation(tmp_path, monkeypatch):
    source = Path(training.__file__).resolve().parents[2] / training.PROFILE
    target = tmp_path / training.PROFILE
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    checksums = {
        "config.yaml": "2660615806a50b5f634ff4681a8e560398402559b42478ad7eaafcdcc9694e6e",
        "outputs/models/resnet_cnn_best.pth": training.HISTORICAL_SHA256,
        **training.REUSED_SOURCE_SHA256,
    }
    checksums["src/models/resnet_cnn.py"] = "0" * 64
    monkeypatch.setattr(cp, "sha256_file", lambda path: checksums[Path(path).relative_to(tmp_path).as_posix()])
    with pytest.raises(ValueError, match="source mismatch"):
        training.load_profile(tmp_path)
