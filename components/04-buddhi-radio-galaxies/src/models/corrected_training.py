"""Prepared corrected training workflow; real execution is explicitly gated."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
import yaml

from src.data import corrected_protocol as protocol
from src.data.corrected_training_data import CorrectedArchiveReader, CorrectedTrainingDataset, TrainingRow
from src.data.preprocess import PreprocessConfig, build_transforms
from src.models.resnet_cnn import ResNetCNN
from src.models.train import TrainingConfig, train_model
from src.models.training_telemetry import EpochTelemetry
from src.utils.seed import set_global_seed


PROFILE = "config/corrected_resnet_training_v1.0.0.yaml"
MANIFEST_SHA256 = "4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91"
HISTORICAL_SHA256 = "d5e1f126c13c2b7bebb95e35f41f0bc6397dd0ea9f4fbfcfbc55648609ab5bc7"
TRAINING = TrainingConfig(seed=42, batch_size=32, learning_rate=0.0003,
                          weight_decay=0.0001, num_epochs=30, num_workers=0,
                          early_stopping_patience=7, scheduler_patience=2,
                          scheduler_factor=0.5, device="cpu")
OUTPUT_PARENT = "outputs/models/mirabest_development_exactdedup_v1.0.0"
REUSED_SOURCE_SHA256 = {
    "src/models/resnet_cnn.py": "a6b2fa85f2c7ae8c9ae7db853556b21626e78583ecc36b6e4f069692d67cf187",
    "src/models/train.py": "2c30bb04f25197f7aa88087c1cf77be6b35ed30d32235f66ef70c2e2f5223e82",
    "src/data/preprocess.py": "e84921262055576282966b908f9988d4fa63dd576efe2d217df07e258a4a088d",
    "src/utils/seed.py": "317db948ed195e2599e9e239107ac3bb77aec09bea7ea6007a3d7845bb782da9",
}


@dataclass(frozen=True)
class PreparedCohort:
    rows: tuple[TrainingRow, ...]
    manifest_sha256: str


def load_cohort(root, *, pins=protocol.FROZEN_PINS, expected_manifest=MANIFEST_SHA256):
    root = Path(root)
    directory = root / protocol.OUTPUT_REFERENCE
    raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest:
        raise ValueError("Corrected manifest checksum mismatch.")
    manifest = json.loads(raw)
    quarantine = json.loads((directory / "quarantine.json").read_bytes())
    report = json.loads((directory / "validation_report.json").read_bytes())
    if report.get("status") != "complete" or report.get("canonical_membership_sha256") != expected_manifest:
        raise ValueError("Corrected protocol publication is incomplete/inconsistent.")
    for name, checksum in report["artifact_sha256"].items():
        if name not in {"manifest.json", "manifest.sha256", "quarantine.json", "provenance.json", "README.md"}:
            raise ValueError("Unexpected corrected artifact reference.")
        if protocol.sha256_file(directory / name) != checksum:
            raise ValueError("Corrected artifact provenance mismatch.")
    splits, audit = protocol.load_evidence(root, pins)
    if protocol.validate_manifest(manifest, quarantine, splits, audit, pins) != expected_manifest:
        raise ValueError("Corrected membership/provenance mismatch.")
    rows = tuple(TrainingRow.from_manifest(row) for row in manifest["records"])
    return PreparedCohort(rows, expected_manifest)


def load_profile(root):
    root = Path(root)
    profile = yaml.safe_load((root / PROFILE).read_text(encoding="utf-8"))
    checks = {
        "protocol": protocol.PROTOCOL, "manifest_sha256": MANIFEST_SHA256,
        "manifest_reference": protocol.OUTPUT_REFERENCE + "/manifest.json",
        "execution_authorized": False, "resume_supported": False,
        "architecture": "ResNetCNN", "initialization": "fresh_random",
        "output_parent": OUTPUT_PARENT, "training": asdict(TRAINING),
        "preprocessing": asdict(PreprocessConfig()), "optimizer": "Adam",
        "loss": "CrossEntropyLoss", "scheduler": "ReduceLROnPlateau",
        "checkpoint_selection": "minimum_validation_loss_strict_improvement",
        "historical_checkpoint_reference": "outputs/models/resnet_cnn_best.pth",
        "historical_checkpoint_sha256": HISTORICAL_SHA256,
        "historical_config_reference": "config.yaml",
        "historical_config_sha256": "2660615806a50b5f634ff4681a8e560398402559b42478ad7eaafcdcc9694e6e",
    }
    if profile != checks:
        raise ValueError("Training profile differs from the reviewed historical settings.")
    for key in ("historical_config", "historical_checkpoint"):
        if protocol.sha256_file(root / profile[key + "_reference"]) != profile[key + "_sha256"]:
            raise ValueError("Historical provenance mismatch.")
    for path, expected in REUSED_SOURCE_SHA256.items():
        if protocol.sha256_file(root / path) != expected:
            raise ValueError("Reused model/training/preprocessing/seed source mismatch.")
    return profile


def fresh_model(seed=42):
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Prepared architecture requires default float32 initialization.")
    set_global_seed(seed)
    return ResNetCNN()  # No checkpoint loader, state loading or warm-start parameter.


def build_views(root, cohort):
    reader = CorrectedArchiveReader(Path(root) / protocol.ARCHIVE_REFERENCE, cohort.rows,
                                    archive_sha256=protocol.FROZEN_PINS.archive_sha256,
                                    archive_md5=protocol.FROZEN_PINS.archive_md5)
    train_transform, validation_transform = build_transforms(PreprocessConfig())
    return (CorrectedTrainingDataset(cohort.rows, "train", train_transform, reader),
            CorrectedTrainingDataset(cohort.rows, "validation", validation_transform, reader))


def readiness(root):
    """Metadata, dependency and fresh-construction check; no data-loader iteration."""
    profile = load_profile(root)
    cohort = load_cohort(root)
    model = fresh_model(profile["training"]["seed"])
    train, validation = build_views(root, cohort)
    return {
        "protocol": protocol.PROTOCOL, "manifest_sha256": cohort.manifest_sha256,
        "train_count": len(train), "validation_count": len(validation),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training": asdict(TRAINING), "preprocessing": asdict(PreprocessConfig()),
        "python": sys.version, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
        "device_recommendation": "cpu", "cpu_threads": torch.get_num_threads(),
        "model_initialization": "fresh_random_no_checkpoint_loading",
        "images_decoded": 0, "loader_iterations": 0,
        "real_training_authorized": False, "resume_supported": False,
    }


def isolated_run_directory(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("A simple explicit run ID is required.")
    root = Path(root).resolve()
    path = root / OUTPUT_PARENT / run_id
    if not path.resolve().is_relative_to(root / OUTPUT_PARENT) or path.exists():
        raise ValueError("Unsafe or existing output namespace; resume/overwrite is unsupported.")
    return path


def execute_training(root, run_id, *, execute=False):
    if execute is not True:
        raise PermissionError("Real corrected training is disabled; explicit researcher approval and --execute-training are required.")
    root = Path(root).resolve()
    output = isolated_run_directory(root, run_id)
    profile = load_profile(root)
    cohort = load_cohort(root)
    # Opaque provenance checks before any selected image access.
    if protocol.sha256_file(root / protocol.ARCHIVE_REFERENCE) != protocol.FROZEN_PINS.archive_sha256:
        raise ValueError("Frozen archive provenance mismatch.")
    model = fresh_model(profile["training"]["seed"])
    train, validation = build_views(root, cohort)
    output.mkdir(parents=True, exist_ok=False)
    context = {
        "status": "started", "protocol": protocol.PROTOCOL, "manifest_sha256": cohort.manifest_sha256,
        "training": asdict(TRAINING), "preprocessing": asdict(PreprocessConfig()),
        "python": sys.version, "torch": torch.__version__, "device": TRAINING.device,
        "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
        "architecture": {"name": "ResNetCNN", "num_classes": 2, "dropout": 0.30,
                         "residual_widths": [32, 64, 128], "input_shape": [1, 150, 150]},
        "command_argv": [sys.executable, *sys.argv], "explicit_execution_gate": True,
        "cpu_threads": torch.get_num_threads(), "initialization": "fresh_random",
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False},
        "loss": "CrossEntropyLoss", "scheduler": "ReduceLROnPlateau(mode=min)",
        "checkpoint_selection": "strictly_lower_validation_loss", "resume_supported": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {p: protocol.sha256_file(root / p) for p in [
            "src/models/resnet_cnn.py", "src/models/train.py", "src/utils/seed.py",
            "src/data/preprocess.py", "src/data/corrected_training_data.py",
            "src/models/corrected_training.py", "src/models/training_telemetry.py",
            "src/data/corrected_protocol.py", PROFILE]},
    }
    with (output / "run_provenance.json").open("xb") as f:
        f.write(protocol.canonical_bytes(context))
    try:
        # Reuse loss, optimizer, scheduler, seeded shuffling and checkpoint rule unchanged.
        with EpochTelemetry(output / "epoch_telemetry.jsonl"):
            result = train_model(train, validation, config=TRAINING, model=model,
                                 output_root=output, artifact_prefix="resnet")
        load_cohort(root)  # Revalidate frozen metadata before declaring completion.
        load_profile(root)
        if protocol.sha256_file(root / protocol.ARCHIVE_REFERENCE) != protocol.FROZEN_PINS.archive_sha256:
            raise ValueError("Frozen archive changed during training.")
        if any(protocol.sha256_file(root / path) != expected
               for path, expected in context["source_sha256"].items()):
            raise ValueError("Training implementation changed during execution.")
        completion = {"status": "complete", "manifest_sha256": cohort.manifest_sha256,
                      "best_epoch": result["best_epoch"],
                      "artifacts": {k: {"reference": str(Path(result[k]).relative_to(output)),
                                        "sha256": protocol.sha256_file(Path(result[k]))}
                                    for k in ["checkpoint_path", "history_path", "curves_path", "validation_metrics_path"]}}
        with (output / "complete.json").open("xb") as f:
            f.write(protocol.canonical_bytes(completion))
    except Exception as error:
        with (output / "failure.json").open("xb") as f:
            f.write(protocol.canonical_bytes({"status": "failed", "error_type": type(error).__name__,
                                             "resume_supported": False}))
        raise
    return result
