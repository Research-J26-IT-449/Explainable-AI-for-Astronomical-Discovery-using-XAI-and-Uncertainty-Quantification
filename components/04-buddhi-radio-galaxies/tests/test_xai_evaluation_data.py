"""Metadata-only synthetic archive tests; no real images, inference or experiments."""

from dataclasses import FrozenInstanceError
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
from unittest.mock import Mock

import pytest
import yaml

from src.xai import evaluation_data as data


def member_bytes(start):
    # Label metadata only. The data field is deliberately not an image array.
    labels = [0 if i < 245 or 554 <= i < 607 or 673 <= i < 726 else 1
              for i in range(start, start + 88)]
    filenames = [f"BestCat_proc/Cropped/{100 if label == 0 else 200}_{i}.123_+041.815.png"
                 for i, label in zip(range(start, start + 88), labels)]
    return pickle.dumps({"batch_label": "synthetic metadata", "labels": labels,
                         "data": b"DO NOT DECODE IMAGE ARRAYS", "filenames": filenames}, protocol=2)


def archive_bytes(payloads=None, bad_manifest=False):
    payloads = payloads or {name: member_bytes(88 * i) for i, name in enumerate(data.MEMBERS)}
    payloads = {**payloads, "batches/batches.meta": pickle.dumps({"label_names": ["FRI", "FRII"]})}
    manifest = "".join(f"{data.digest(raw, 'md5')}  {name.split('/')[-1]}\n"
                       for name, raw in payloads.items())
    if bad_manifest:
        manifest = "0" * 32 + manifest[32:]
    contents = {**payloads, "batches/hash.md5": manifest.encode("ascii")}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, raw in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return stream.getvalue()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    config = yaml.safe_load(Path(data.CONFIG_PATH).read_text(encoding="utf-8"))
    splits = {"train": list(range(554)), "validation": list(range(554, 673)), "test": list(range(673, 792))}
    payloads = {"checkpoint": b"synthetic frozen checkpoint",
                "split": json.dumps(splits).encode(), "archive": archive_bytes()}
    for key, raw in payloads.items():
        path = tmp_path / config["provenance"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        config["provenance"][key + "_sha256"] = data.digest(raw)
    config["provenance"]["archive_md5"] = data.digest(payloads["archive"], "md5")
    config_path = tmp_path / data.CONFIG_PATH
    config_path.parent.mkdir(parents=True)

    def pin_test_configuration():
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        # Synthetic fixtures change the independent trusted pin explicitly.
        # Production callers cannot supply/override this pin through the API.
        monkeypatch.setattr(data, "CONFIG_SHA256", data.digest(data.canonical_bytes(config)))

    pin_test_configuration()
    return tmp_path, config, splits, pin_test_configuration


def test_production_configuration_pin_and_protocol():
    config = yaml.safe_load(Path(data.CONFIG_PATH).read_text(encoding="utf-8"))
    assert data.digest(data.canonical_bytes(config)) == data.CONFIG_SHA256
    assert config["protocol"] == data.PROTOCOL
    assert config["faithfulness"]["random_seeds"] == list(range(1000, 1020))
    assert config["faithfulness"]["random_permutations_per_image"] == 20
    assert config["faithfulness"]["primary_steps"] == 6
    assert config["execution_authorized"] is False


def test_deterministic_20_20_selection_and_all_validation(fixture):
    root, config, splits, _ = fixture
    first = data.load_evaluation_data(root)
    second = data.load_evaluation_data(root)
    assert first.engineering.to_bytes() == second.engineering.to_bytes()
    entries = first.engineering.entries
    assert len(entries) == 40
    assert [e.ground_truth_label for e in entries] == [0] * 20 + [1] * 20
    assert {e.dataset_index for e in entries} <= set(splits["train"])
    expected = []
    for label in (0, 1):
        candidates = range(245) if label == 0 else range(245, 554)
        expected.extend(sorted(candidates, key=lambda i: (
            hashlib.sha256(f"Phase3B-XAI-v1.0.0|engineering|42|{i}".encode()).hexdigest(), i))[:20])
    assert [e.dataset_index for e in entries] == expected
    assert len(first.validation_allowlist) == 119
    assert first.validation_allowlist == frozenset(splits["validation"])
    assert [e.dataset_index for e in first.validation.entries] == sorted(splits["validation"])


@pytest.mark.parametrize("operation", ["retrieve", "decode", "score", "visualize"])
@pytest.mark.parametrize("cohort", ["engineering", "validation", "test"])
def test_test_exclusion_before_any_handler(fixture, operation, cohort):
    root, _, splits, _ = fixture
    access = data.load_evaluation_data(root)
    handler = Mock(side_effect=AssertionError("Locked image operation reached handler"))
    for index in splits["test"]:
        with pytest.raises(PermissionError, match="Locked TEST"):
            access.dispatch(index, cohort, operation, handler)
    handler.assert_not_called()


def test_allowlists_cannot_be_crossed_and_valid_dispatch(fixture):
    root, _, splits, _ = fixture
    access = data.load_evaluation_data(root)
    with pytest.raises(PermissionError, match="allowlist"):
        access.authorize(554, "engineering")
    unselected = next(i for i in splits["train"] if i not in {e.dataset_index for e in access.engineering.entries})
    with pytest.raises(PermissionError, match="allowlist"):
        access.authorize(unselected, "engineering")
    for invalid in (-1, True, "554", 792):
        with pytest.raises(PermissionError):
            access.authorize(invalid, "validation")
    handler = Mock(return_value="metadata only")
    assert access.dispatch(554, "validation", "retrieve", handler) == "metadata only"
    assert handler.call_args.args[0] == access.validation.entries[0]
    with pytest.raises(PermissionError):
        access.dispatch(554, "validation", "train", handler)


def test_source_key_excludes_class_and_preserves_metadata():
    first = "BestCat_proc/Cropped/100_159.613_+041.815.png"
    second = "BestCat_proc/Cropped/200_159.613_+041.815.png"
    assert data.candidate_source_key(first) == data.candidate_source_key(second)
    assert data.candidate_source_key(first) == "filename-coordinate-fields:159.613|+041.815"
    assert data.candidate_source_key("BestCat_proc/Cropped/100_1.000_-002.000.png").endswith("1.000|-002.000")
    for name in (None, "../test.png", "unknown.png", "BestCat_proc/Cropped/100_1_2.png"):
        with pytest.raises(ValueError, match="filename"):
            data.candidate_source_key(name)


def test_manifest_content_is_stable_immutable_complete_and_no_writes(fixture, monkeypatch):
    root, _, _, _ = fixture
    before = {p.relative_to(root): data.digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(Path, "write_bytes", Mock(side_effect=AssertionError("Unexpected artifact write")))
    monkeypatch.setattr(Path, "write_text", Mock(side_effect=AssertionError("Unexpected artifact write")))
    access = data.load_evaluation_data(root)
    manifest = access.validation
    blob = manifest.to_bytes()
    assert blob.endswith(b"\n")
    assert data.canonical_bytes(json.loads(blob)) == blob
    assert manifest.sha256 == data.digest(blob)
    row = json.loads(blob)["entries"][0]
    assert {"protocol_version", "dataset_index", "split_membership", "ground_truth_label",
            "original_filename", "candidate_source_key", "selection_rule", "frozen_split_hash",
            "dataset_provenance", "archive_member", "member_row"} <= set(row)
    assert row["split_membership"] == "validation"
    with pytest.raises(FrozenInstanceError):
        manifest.entries[0].ground_truth_label = 1
    after = {p.relative_to(root): data.digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_metadata_generation_never_unpickles_or_decodes_images(fixture, monkeypatch):
    import numpy as np
    from PIL import Image
    fail = Mock(side_effect=AssertionError("Image/pickle decoder must not run"))
    for name in ("load", "loads", "Unpickler"):
        monkeypatch.setattr(pickle, name, fail)
    for name in ("open", "fromarray", "frombytes"):
        monkeypatch.setattr(Image, name, fail)
    monkeypatch.setattr(np, "frombuffer", fail)
    monkeypatch.setattr(np, "array", fail)
    access = data.load_evaluation_data(fixture[0])
    assert len(access.validation.entries) == 119
    assert len(access.engineering.entries) == 40
    fail.assert_not_called()


@pytest.mark.parametrize("key", ["checkpoint", "split", "archive", "config"])
@pytest.mark.parametrize("missing", [False, True])
def test_missing_files_and_hash_mismatch_fail_closed(fixture, key, missing):
    root, config, _, _ = fixture
    path = root / (data.CONFIG_PATH if key == "config" else config["provenance"][key])
    if missing:
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + (b"\nprotocol: altered\n" if key == "config" else b"changed"))
    with pytest.raises((ValueError, OSError, yaml.YAMLError)):
        data.load_evaluation_data(root)


@pytest.mark.parametrize("key", ["checkpoint", "split", "archive", "config"])
def test_provenance_drift_blocks_existing_access_before_handler(fixture, key):
    root, config, _, _ = fixture
    access = data.load_evaluation_data(root)
    path = root / (data.CONFIG_PATH if key == "config" else config["provenance"][key])
    path.write_bytes(path.read_bytes() + (b"\nprotocol: altered\n" if key == "config" else b"changed"))
    handler = Mock()
    with pytest.raises(ValueError):
        access.dispatch(554, "validation", "decode", handler)
    handler.assert_not_called()


@pytest.mark.parametrize("change", ["overlap", "duplicate", "count", "type", "range"])
def test_disjoint_splits_rejected_even_with_matching_hash(fixture, change):
    root, config, splits, pin = fixture
    if change == "overlap":
        splits["train"][0] = splits["test"][0]
    elif change == "duplicate":
        splits["train"][0] = splits["train"][1]
    elif change == "count":
        splits["validation"].pop()
    elif change == "type":
        splits["train"][0] = True
    else:
        splits["test"][0] = 792
    raw = json.dumps(splits).encode()
    (root / config["provenance"]["split"]).write_bytes(raw)
    config["provenance"]["split_sha256"] = data.digest(raw)
    pin()
    with pytest.raises(ValueError, match="indices|overlap"):
        data.load_evaluation_data(root)


@pytest.mark.parametrize("change", ["missing_labels", "missing_filename", "bad_filename", "label", "truncated", "member_hash"])
def test_bad_dataset_metadata_rejected(fixture, change):
    root, config, _, pin = fixture
    payloads = {name: member_bytes(88 * i) for i, name in enumerate(data.MEMBERS)}
    # Synthetic fixture deserialization is permitted here, before decoder blockers.
    item = pickle.loads(payloads[data.MEMBERS[0]])
    if change == "missing_labels":
        del item["labels"]
    elif change == "missing_filename":
        item["filenames"].pop()
    elif change == "bad_filename":
        item["filenames"][0] = "../wrong.png"
    elif change == "label":
        item["labels"][0] = 7
    payloads[data.MEMBERS[0]] = pickle.dumps(item, protocol=2)
    if change == "truncated":
        payloads[data.MEMBERS[0]] = payloads[data.MEMBERS[0]][:-1]
    raw = archive_bytes(payloads, bad_manifest=change == "member_hash")
    (root / config["provenance"]["archive"]).write_bytes(raw)
    config["provenance"]["archive_sha256"] = data.digest(raw)
    config["provenance"]["archive_md5"] = data.digest(raw, "md5")
    pin()
    with pytest.raises(ValueError, match="metadata|checksum"):
        data.load_evaluation_data(root)


def test_frozen_python2_unicode_filename_metadata():
    # Exercise actual archived NumPy scalar string encoding without constructing arrays.
    names = [f"BestCat_proc/Cropped/100_{i}.123_+041.815.png" for i in range(88)]
    entry = {"labels": [0] * 88, "data": b"never decode",
             "filenames": [name.encode("utf-32-le") for name in names]}
    rows = data._member_metadata(pickle.dumps(entry, protocol=2), data.MEMBERS[0], 0)
    assert len(rows) == 88
    assert rows[0].original_filename == names[0]
