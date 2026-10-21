"""One explicitly gated corrected VALIDATION pass; no TRAIN/TEST image access."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
import torchvision
from torch.utils.data import DataLoader
from src.data import corrected_protocol as protocol
from src.data.corrected_training_data import CorrectedArchiveReader, CorrectedTrainingDataset
from src.data.preprocess import PreprocessConfig, build_transforms
from src.models.corrected_training import load_cohort, load_profile
from src.models.resnet_cnn import ResNetCNN
from src.utils.seed import set_global_seed

RUN = ROOT / "outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01"
CHECKPOINT = RUN / "models/resnet_cnn_best.pth"
EXPECTED_CHECKPOINT = "d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c"
EXPECTED_MANIFEST = "4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def calculate_metrics(labels, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], average=None, zero_division=0)
    result = dict(confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
                  confusion_matrix_axes="rows=ground truth, columns=prediction; order FR-I=0, FR-II=1",
                  accuracy=float(accuracy_score(labels, predicted)),
                  balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                  num_validation_samples=len(labels), correct_count=sum(a == b for a, b in zip(labels, predicted)),
                  zero_division_policy="Undefined precision/recall/F1 set to 0 via sklearn zero_division=0",
                  per_class=[dict(label=i, name=name, support=int(support[i]), precision=float(precision[i]),
                                  recall=float(recall[i]), f1=float(f1[i]))
                             for i, name in enumerate(["FR-I", "FR-II"])])
    for average in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(labels, predicted, labels=[0, 1],
                                                   average=average, zero_division=0)
        result[average] = dict(precision=float(p), recall=float(r), f1=float(f))
    return result


def preflight():
    assert Path.cwd().resolve() == ROOT, "Execute from repository root"
    assert git("branch", "--show-current") == "feature/external-radio-input"
    assert {p.name for p in OUT.iterdir()} == {Path(__file__).name}, "Output namespace already contains evidence"
    profile = load_profile(ROOT)  # Also validates pinned architecture/preprocessing and historical artifacts.
    cohort = load_cohort(ROOT)  # Uses only the development prefix of the old split; TEST suffix not parsed.
    assert cohort.manifest_sha256 == EXPECTED_MANIFEST
    selected = tuple(r for r in cohort.rows if r.split == "validation")
    assert len(selected) == 116 and Counter(r.label for r in selected) == {0: 51, 1: 65}
    assert len({r.dataset_index for r in selected}) == 116
    assert not set(protocol.QUARANTINE) & {r.dataset_index for r in selected}
    manifest = read(ROOT / protocol.OUTPUT_REFERENCE / "manifest.json")
    assert manifest["class_mapping"] == {"0": "FR-I", "1": "FR-II"}
    assert [r.dataset_index for r in selected] == manifest["splits"]["validation"]
    assert sha(CHECKPOINT) == EXPECTED_CHECKPOINT
    recorded = read(RUN / "run_provenance.json")
    assert all(sha(ROOT / p) == h for p, h in recorded["source_sha256"].items())
    assert sha(ROOT / protocol.ARCHIVE_REFERENCE) == protocol.FROZEN_PINS.archive_sha256
    protected = read(ROOT / "analysis_reports/step18_corrected_cnn_training/preflight_baseline.json")["protected"]
    assert len(protected) == 41 and all(sha(ROOT / p) == h for p, h in protected.items())
    saved = read(RUN / "metrics/resnet_validation_metrics.json")
    assert saved["best_epoch"] == 30 and saved["best_validation_accuracy"] == 101 / 116
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert state["epoch"] == 30 and state["validation_accuracy"] == 101 / 116
    assert state["config"] == profile["training"]
    set_global_seed(42)
    model = ResNetCNN().cpu()
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    assert all(not m.training for m in model.modules())
    _, transform = build_transforms(PreprocessConfig())
    assert [type(t).__name__ for t in transform.transforms] == ["ToTensor", "Normalize"]
    assert transform.transforms[-1].mean == (0.5,) and transform.transforms[-1].std == (0.5,)
    # Analytical metric checks, not model execution or dataset access.
    test = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert test["confusion_matrix"] == [[1, 1], [0, 2]] and test["accuracy"] == 0.75
    assert test["per_class"][0]["recall"] == 0.5 and test["per_class"][1]["precision"] == 2 / 3
    return profile, selected, model, transform, protected, saved


class LoggedReader(CorrectedArchiveReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = []

    def read_selected(self, indices):
        ids = tuple(indices)
        assert 1 <= len(ids) <= 8 and all(i in self.rows for i in ids)
        assert not set(ids) & set(protocol.QUARANTINE)
        self.requests.append(list(ids))
        return super().read_selected(ids)


def execute(prepared):
    profile, selected, model, transform, protected, saved = prepared
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    initial_git = dict(branch=git("branch", "--show-current"), head=git("rev-parse", "HEAD"),
                       index_sha256=sha(ROOT / ".git/index"), status=git("status", "--short"))
    paths = set(ROOT / p for p in protected)
    for base in ["src", "scripts", "tests", "config", "demo", "docs", "data/manifests", "outputs/models",
                 "outputs/metrics", "outputs/ood", "outputs/xai", "analysis_reports/step18_corrected_cnn_training",
                 "analysis_reports/step19_corrected_validation_analysis"]:
        paths.update(p for p in (ROOT / base).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    baseline = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    before = {k: v.clone() for k, v in model.state_dict().items()}
    command = r".venv\Scripts\python.exe -B analysis_reports/step20_corrected_validation_classification/evaluate_fixed_checkpoint.py --execute-validation"
    provenance = dict(status="preflight_passed", started_utc=started, command=command,
        command_argv=[sys.executable, "-B", str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
        protocol=protocol.PROTOCOL, manifest_reference=protocol.OUTPUT_REFERENCE + "/manifest.json",
        manifest_sha256=EXPECTED_MANIFEST, checkpoint_reference=CHECKPOINT.relative_to(ROOT).as_posix(),
        checkpoint_sha256=EXPECTED_CHECKPOINT, selected_epoch=30,
        archive_reference=protocol.ARCHIVE_REFERENCE, archive_sha256=protocol.FROZEN_PINS.archive_sha256,
        archive_md5=protocol.FROZEN_PINS.archive_md5, frozen_split_sha256=protocol.FROZEN_PINS.split_sha256,
        selected_validation_indices=[r.dataset_index for r in selected], num_images=116,
        ground_truth_support={"FR-I": 51, "FR-II": 65}, class_mapping={"FR-I": 0, "FR-II": 1},
        preprocessing="150x150 uint8 grayscale -> ToTensor float32 /255 -> Normalize(mean=0.5,std=0.5); no augmentation",
        preprocessing_source_sha256=sha(ROOT / "src/data/preprocess.py"),
        model_architecture="Unchanged ResNetCNN(num_classes=2,dropout=0.30)",
        execution="model.eval(); frozen parameters; torch.inference_mode(); one pass; no optimizer/scheduler",
        tensor_batch_size=32, max_raw_images_per_read=8, shuffle=False, num_workers=0,
        inference_mode="CPU", torch_threads=torch.get_num_threads(), seed=42,
        python=sys.version, torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        numpy=np.__version__, scikit_learn=sklearn.__version__, cuda_available=torch.cuda.is_available(),
        evaluation_script_sha256=sha(Path(__file__)),
        implementation_sha256={p: sha(ROOT / p) for p in ["src/data/corrected_training_data.py",
            "src/data/corrected_protocol.py", "src/data/mirabest_pickle_stream_reader.py", "src/models/resnet_cnn.py"]},
        authorization="Researcher Step 20 request; exactly 116 corrected VALIDATION images, one fixed-checkpoint pass",
        baseline=baseline, protected=protected, git=initial_git)
    write("preflight_provenance.json", provenance)  # Exclusive marker also prevents accidental repeat execution.
    print("Preflight PASS. Starting ONE 116-image validation pass; raw reader restricted to validation rows.", flush=True)
    reader = LoggedReader(ROOT / protocol.ARCHIVE_REFERENCE, selected,
                           archive_sha256=protocol.FROZEN_PINS.archive_sha256,
                           archive_md5=protocol.FROZEN_PINS.archive_md5)
    dataset = CorrectedTrainingDataset(selected, "validation", transform, reader)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    predictions = []
    loss_sum = 0.0
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            assert images.shape[1:] == (1, 150, 150) and images.dtype == torch.float32
            assert torch.isfinite(images).all() and images.min() >= -1 and images.max() <= 1
            logits = model(images)
            assert logits.shape == (len(labels), 2) and torch.isfinite(logits).all()
            probabilities = torch.softmax(logits, dim=1)
            assert torch.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
            assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(labels)), atol=1e-6, rtol=0)
            predicted = probabilities.argmax(dim=1)
            assert torch.equal(predicted, logits.argmax(dim=1))
            loss_sum += torch.nn.functional.cross_entropy(logits, labels).item() * len(labels)
            for row, label, guess, probs in zip(selected[offset:offset+len(labels)], labels.tolist(),
                                                predicted.tolist(), probabilities.tolist()):
                assert label == row.label and row.split == "validation"
                predictions.append(dict(dataset_index=row.dataset_index, corrected_split="validation",
                    ground_truth_label=label, predicted_label=guess, fri_probability=probs[0],
                    frii_probability=probs[1], correct=label == guess))
            offset += len(labels)
            print(f"Completed {offset}/116 validation images", flush=True)
    assert offset == len(predictions) == 116
    assert [p["dataset_index"] for p in predictions] == [r.dataset_index for r in selected]
    assert [i for request in reader.requests for i in request] == [r.dataset_index for r in selected]
    assert len({p["dataset_index"] for p in predictions}) == 116
    assert Counter(p["ground_truth_label"] for p in predictions) == {0: 51, 1: 65}
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters())
    # Preserve the actual single-pass evidence even if aggregate reproduction fails; do not repeat inference.
    with (OUT / "validation_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    write("bounded_read_log.json", dict(requests=reader.requests, count=len(reader.requests),
                                        maximum=max(map(len, reader.requests)), total=offset))
    metrics = calculate_metrics([p["ground_truth_label"] for p in predictions], [p["predicted_label"] for p in predictions])
    metrics.update(protocol=protocol.PROTOCOL, checkpoint_sha256=EXPECTED_CHECKPOINT, manifest_sha256=EXPECTED_MANIFEST,
                   result_type="DEVELOPMENT VALIDATION; validation-selected checkpoint; not independent TEST",
                   validation_loss=loss_sum / 116, step18_accuracy=saved["best_validation_accuracy"],
                   accuracy_reproduces_step18=metrics["accuracy"] == saved["best_validation_accuracy"],
                   step18_validation_loss=saved["best_validation_loss"])
    write("metrics.json", metrics)
    assert metrics["correct_count"] == 101 and metrics["accuracy_reproduces_step18"], "STOP: Step 18 accuracy discrepancy; do not repeat inference"
    assert math.isclose(metrics["validation_loss"], saved["best_validation_loss"], abs_tol=1e-7, rel_tol=0)
    for name, value in baseline.items():
        assert sha(ROOT / name) == value, f"Existing artifact changed: {name}"
    assert all(sha(ROOT / p) == h for p, h in protected.items())
    figure, axis = plt.subplots(figsize=(6, 5))
    cm = np.asarray(metrics["confusion_matrix"])
    image = axis.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20,
                      color="white" if cm[i, j] > cm.max()/2 else "black")
    axis.set(xticks=[0, 1], xticklabels=["FR-I", "FR-II"], yticks=[0, 1], yticklabels=["FR-I", "FR-II"],
             xlabel="Predicted class", ylabel="Ground-truth class",
             title="Corrected CNN | Development validation (n=116)\nSelected epoch 30; no TEST evaluation")
    figure.colorbar(image, ax=axis, label="Image count")
    figure.tight_layout()
    figure.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    for offset_bar, key, name in [(-0.25, "precision", "Precision"), (0, "recall", "Recall"), (0.25, "f1", "F1")]:
        bars = axis.bar(x+offset_bar, [r[key] for r in metrics["per_class"]], width=0.25, label=name)
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
    axis.set(xticks=x, xticklabels=["FR-I (support 51)", "FR-II (support 65)"], ylim=(0, 1.12),
             ylabel="Score", title="Corrected CNN | Development validation\nFixed checkpoint, n=116")
    axis.legend(loc="upper center", ncol=3)
    figure.tight_layout()
    figure.savefig(OUT / "per_class_metrics.png", dpi=180)
    plt.close(figure)
    a, b = metrics["per_class"]
    errors = [int(cm[0, 1]), int(cm[1, 0])]
    report = ["# Step 20 — Fixed-checkpoint corrected validation classification", "",
        "**CLASSIFICATION EVALUATION COMPLETED**. DEVELOPMENT VALIDATION results; not independent TEST results.", "",
        f"Checkpoint: `{CHECKPOINT.relative_to(ROOT).as_posix()}`; selected epoch 30.",
        f"Checkpoint SHA-256: `{EXPECTED_CHECKPOINT}`.",
        f"Manifest: `{protocol.OUTPUT_REFERENCE}/manifest.json`; SHA-256 `{EXPECTED_MANIFEST}`.",
        f"Protocol: {protocol.PROTOCOL}. One pass over 116 unique validation images in saved membership order, "
        "51 FR-I / 65 FR-II; quarantined records excluded. Reader is scoped to those 116 rows, "
        "not the full corrected TRAIN+VALIDATION cohort. Maximum eight raw images per read; "
        f"{len(reader.requests)} bounded calls. Tensor batches 32,32,32,20; no shuffle/workers.", "",
        "Command from repository root:", "```powershell", command, "```", "",
        f"Python {sys.version.split()[0]}, Torch {torch.__version__}, scikit-learn {sklearn.__version__}; "
        f"CPU, {torch.get_num_threads()} Torch threads. Safe weights_only checkpoint load, strict unchanged "
        "ResNetCNN state load, model.eval(), requires_grad=False and torch.inference_mode(). Original "
        "150x150 grayscale uint8 input -> float32 /255 -> mean/std 0.5 normalization. No rotations, flips, "
        "resizing, external-upload rejection, OOD recalibration or other augmentation. Seed 42; no optimizer, "
        "scheduler or model selection. All model parameters and BatchNorm buffers exactly match before/after; "
        "gradients remain absent.", "",
        f"Overall accuracy: **{metrics['accuracy']:.12f} ({100*metrics['accuracy']:.6f}%)**, "
        "101 correct / 116; 15 errors. Exactly reproduces Step 18 aggregate accuracy. "
        f"Validation loss {metrics['validation_loss']:.12f}; Step 18 {saved['best_validation_loss']:.12f}.", "",
        "Confusion matrix (rows=ground truth, columns=prediction; FR-I, FR-II):", "```",
        str(metrics["confusion_matrix"]), "```", "",
        "| Class | Support | Precision | Recall | F1 | Errors by true class |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {r['name']} | {r['support']} | {r['precision']:.9f} | {r['recall']:.9f} | "
          f"{r['f1']:.9f} | {errors[i]} |" for i, r in enumerate(metrics["per_class"])], "",
        "| Average | Precision | Recall | F1 |", "|---|---:|---:|---:|",
        *[f"| {name} | {metrics[name]['precision']:.9f} | {metrics[name]['recall']:.9f} | "
          f"{metrics[name]['f1']:.9f} |" for name in ["macro", "weighted"]], "",
        f"Balanced accuracy (equal mean of class recalls): {metrics['balanced_accuracy']:.9f}.",
        "Standard sklearn.metrics implementations with explicit labels [0,1]; zero_division=0 "
        "sets undefined precision/recall/F1 to zero. Both class supports are nonzero; no undefined "
        "class metrics arise here. Full precision is retained in JSON and CSV.", "",
        "## Scientific interpretation", "",
        f"True-class errors: FR-I {errors[0]}/51, FR-II {errors[1]}/65. "
        f"FR-I precision/recall are {a['precision']:.6f}/{a['recall']:.6f}; "
        f"FR-II precision/recall are {b['precision']:.6f}/{b['recall']:.6f}. "
        "These distinguish missed true-class cases (recall) from false assignments among predicted "
        "class cases (precision); counts and rates should both be reported. The figure/table show "
        "the observed asymmetry without attributing it to physical morphology or a causal mechanism.",
        f"FR-II contributes 65/116 ({100*65/116:.2f}%) and FR-I 51/116 ({100*51/116:.2f}%). "
        "Overall accuracy and support-weighted metrics reflect this imbalance; macro scores and "
        "balanced accuracy weight classes equally. This is not evidence that balancing/retraining "
        "would improve results. No such experiment was conducted.", "",
        "This reproduces saved checkpoint development performance; it is not independent validation. "
        "The same 116-image cohort selected the checkpoint by minimum loss across 30 epochs. "
        "Small cohort, one training seed, and validation selection limit generalization claims. "
        "Exact-content development deduplication does not establish astronomical source independence, "
        "near-duplicate absence, external-survey robustness or TEST performance. Quarantined conflicting-label "
        "root cause remains unresolved. No superiority, discovery, physics alignment or XAI conclusion is drawn.", "",
        "## Evidence and integrity", "",
        "validation_predictions.csv preserves all 116 image identifiers, labels, probabilities and correctness "
        "in manifest order. metrics.json contains sklearn scores and Step 18 comparison; "
        "bounded_read_log.json records every selected row request; preflight_provenance.json records "
        "environment, command, source/artifact hashes and exact validation scope. Figures: "
        "confusion_matrix.png and per_class_metrics.png. completion_integrity.json is the final success "
        "marker with output checksums and verified preservation checks.",
        "No TRAIN, TEST or quarantined image was decoded. Archived label operands were skipped; "
        "only verified manifest labels were interpreted. The original split TEST suffix was not parsed. "
        "Whole-archive/split provenance hashing and gzip/internal-checksum operations necessarily read "
        "opaque serialized bytes; no claim is made of zero OS-level reads of serialized TEST bytes. "
        "All 41 protected artifacts and all snapshotted existing research/frontend/source files remain "
        "unchanged; Git index, HEAD, branch and prior status are unchanged. No staging, commit, push, "
        "retraining, XAI, TEST evaluation or additional configuration execution. The existing .pytest_cache "
        "permission warning during git status is unrelated to this evaluation.", "",
        "Next step: researcher review of these development metrics and prediction evidence. "
        "No additional model evaluation or XAI begins automatically.", ""]
    with (OUT / "evaluation_report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report))
    final_git = dict(branch=git("branch", "--show-current"), head=git("rev-parse", "HEAD"),
                     index_sha256=sha(ROOT / ".git/index"), status=git("status", "--short"))
    assert initial_git == final_git
    assert all(sha(ROOT / p) == h for p, h in baseline.items())
    assert all(sha(ROOT / p) == h for p, h in protected.items())
    write("completion_integrity.json", dict(status="CLASSIFICATION EVALUATION COMPLETED", started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-tick,
        unique_validation_records=116, model_state_unchanged=True, parameter_gradients_absent=True,
        protected_artifacts_unchanged=41, existing_files_unchanged=len(baseline), git_unchanged=True,
        test_labels_or_pixels_decoded=0, train_or_quarantined_images_decoded=0, complete_inference_passes=1,
        bounded_read_calls=reader.requests, max_raw_images_per_read=max(map(len, reader.requests)),
        accuracy_reproduces_step18=True, checkpoint_sha256=EXPECTED_CHECKPOINT, manifest_sha256=EXPECTED_MANIFEST,
        output_sha256={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}))
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-validation", action="store_true", help="One Step 20 authorized real pass")
    args = parser.parse_args()
    prepared = preflight()
    if not args.execute_validation:
        print("Metadata/checkpoint preflight PASS; no image pixels decoded; no forward passes.")
        print("116 corrected validation rows; 51 FR-I/65 FR-II; CPU; fresh output namespace.")
        return
    try:
        execute(prepared)
    except Exception:
        if not (OUT / "failure.json").exists():
            write("failure.json", dict(status="BLOCKED", traceback=traceback.format_exc(),
                                       instruction="Preserve partial evidence; do not repeat inference automatically"))
        raise


if __name__ == "__main__":
    main()
