import json
import hashlib
import weakref
from copy import deepcopy

import pytest

from scripts.run_duplicate_audit_train_val import (
    audit_records,
    classify_duplicate_groups,
    validate_split,
)
from src.data.selective_raw_reader import MAGIC, SelectiveRawReader

from test_selective_raw_reader import reader_fixture, write_batch
from scripts import run_duplicate_audit_train_val as audit
from src.data import mirabest_pickle_stream_reader as stream


def test_duplicate_groups_classify_cross_split_and_label_conflict():
    records = [
        {"dataset_index": 0, "split": "train", "label": 0,
         "canonical_sha256": "a", "confirmed_byte_equal": True},
        {"dataset_index": 1, "split": "validation", "label": 1,
         "canonical_sha256": "a", "confirmed_byte_equal": True},
        {"dataset_index": 2, "split": "train", "label": 1,
         "canonical_sha256": "b", "confirmed_byte_equal": True},
        {"dataset_index": 3, "split": "train", "label": 1,
         "canonical_sha256": "b", "confirmed_byte_equal": True},
    ]
    groups = classify_duplicate_groups(records)
    assert len(groups) == 2
    assert groups[0]["cross_train_validation"] is True
    assert groups[0]["conflicting_labels"] is True
    assert groups[1]["cross_train_validation"] is False


def test_audit_counts_only_allowed_scope(tmp_path):
    path, pixels_a, pixels_b = reader_fixture(tmp_path)
    reader = SelectiveRawReader({"data_batch_1": path}, {0, 1}, {2})
    result = audit_records(reader, {"train": frozenset({0}), "validation": frozenset({1})})
    assert result["authorized_images_processed"] == 2
    assert result["test_images_decoded"] == 0
    assert result["duplicate_group_count"] == 1
    assert result["cross_split_duplicate_group_count"] == 1
    assert result["conflicting_label_group_count"] == 1


def test_split_hash_and_structure_are_fail_closed(tmp_path):
    split = {"train": list(range(554)), "validation": list(range(554, 673)),
             "test": list(range(673, 792))}
    path = tmp_path / "split_indices.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    import hashlib
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    sets = validate_split(path, expected)
    assert len(sets["train"] | sets["validation"]) == 673
    assert len(sets["test"]) == 119


def test_existing_output_namespace_cannot_be_overwritten(tmp_path):
    output = tmp_path / "duplicate_audit_train_val_v1"
    output.mkdir()
    with pytest.raises(FileExistsError):
        output.mkdir(exist_ok=False)


@pytest.fixture
def bounded_fixture(tmp_path, monkeypatch):
    """All data/checkpoints are synthetic; no production archive is opened."""
    split = {"train": list(range(554)), "validation": list(range(554, 673)),
             "test": list(range(673, 792))}
    paths = [tmp_path / name for name in ("synthetic.tar.gz", "split.json", "checkpoint.pth")]
    for path, raw in zip(paths, [b"opaque synthetic archive", json.dumps(split).encode(), b"synthetic checkpoint"]):
        path.write_bytes(raw)
    pins = dict(zip(["expected_archive_sha256", "expected_split_sha256", "expected_checkpoint_sha256"],
                    [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]))
    pins["expected_archive_md5"] = hashlib.md5(paths[0].read_bytes()).hexdigest()
    calls, previous_refs, retained_lists = [], [], []

    def pixels(index):
        # Three groups: within TRAIN, within VALIDATION, and cross-split equality.
        representative = {1: 0, 555: 554, 556: 2}.get(index, index)
        return representative.to_bytes(2, "big") + bytes(22500 - 2)

    def extract(path, allowed, forbidden, selected, **kwargs):
        assert path == paths[0]
        assert len(allowed) == 673 and len(forbidden) == 119
        assert not set(selected) & forbidden
        assert 1 <= len(selected) <= 8
        assert all(ref() is None for ref in previous_refs), "Prior batch still retains images"
        assert all(not rows for rows in retained_lists), "Prior batch list not released"
        previous_refs.clear()
        calls.append(tuple(selected))
        rows = [stream.StreamRecord(i, stream.EXPECTED_MEMBER_NAMES[i // 88], i % 88,
                                    i % 2, f"source_{i}.png", pixels(i)) for i in sorted(selected)]
        previous_refs.extend(weakref.ref(row) for row in rows)
        retained_lists.append(rows)
        meter = kwargs["instrumentation"]
        meter.permitted_payload_extractions = len(rows)
        meter.maximum_pixel_buffer_size = 22500
        meter.forbidden_payload_encounters = 119
        return rows

    monkeypatch.setattr(stream, "extract_selected_archive", extract)
    return paths, pins, calls, retained_lists, pixels, extract


def test_real_audit_driver_exact_scope_bounded_release_and_determinism(bounded_fixture):
    paths, pins, calls, lists, _, _ = bounded_fixture
    first = audit.run_archive_audit(*paths, **pins)
    assert first == audit.run_archive_audit(*paths, **pins)
    assert first["authorized_images_processed"] == 673
    assert first["test_images_decoded"] == 0
    assert [r["dataset_index"] for r in first["records"]] == list(range(673))
    hashing = [b for b in first["batches"] if b["phase"] == "content_hashing"]
    assert len(hashing) == 85 and [b["selected_count"] for b in hashing] == [8] * 84 + [1]
    assert first["maximum_selected_images_per_batch"] == 8
    assert first["maximum_retained_batch_pixel_bytes"] == 180000
    assert all(not rows for rows in lists)
    assert all(len(c) <= 8 for c in calls)
    assert first["duplicate_group_count"] == 3
    assert first["within_train_duplicate_group_count"] == 1
    assert first["within_validation_duplicate_group_count"] == 1
    assert first["cross_split_duplicate_group_count"] == 1
    assert first["conflicting_label_group_count"] == 2  # Both within-split pairs also alternate labels.


def test_exact_decoded_hash_format_and_byte_confirmation(bounded_fixture):
    paths, pins, _, _, pixels, _ = bounded_fixture
    result = audit.run_archive_audit(*paths, **pins)
    first = result["records"][0]
    import struct
    expected = hashlib.sha256(stream.MAGIC_CANONICAL + b"<u1\0" + struct.pack(">II", 150, 150) + pixels(0)).hexdigest()
    assert first["canonical_sha256"] == expected
    assert all(r["shape"] == [150, 150] and r["dtype"] == "uint8" for r in result["records"])
    assert all(g["confirmed_byte_equal"] for g in result["duplicate_groups"])
    assert len([b for b in result["batches"] if b["phase"] == "byte_confirmation"]) == 3


def test_same_hash_different_pixels_is_never_reported_exact(bounded_fixture, monkeypatch):
    paths, pins, *_ = bounded_fixture
    monkeypatch.setattr(stream, "canonical_sha256", lambda _: "a" * 64)
    with pytest.raises(ValueError, match="Hash-matched content differs"):
        audit.run_archive_audit(*paths, **pins)


def test_group_categories_can_overlap_without_losing_members():
    rows = [{"dataset_index": i, "split": split, "label": i % 2,
             "canonical_sha256": "same", "confirmed_byte_equal": True}
            for i, split in enumerate(["train", "train", "validation", "validation"])]
    before = deepcopy(rows)
    groups = audit.classify_duplicate_groups(rows)
    assert len(groups) == 1
    assert all(groups[0][key] for key in ["within_train", "within_validation", "cross_train_validation", "conflicting_labels"])
    assert rows == before


@pytest.mark.parametrize("position", [0, 1, 2])
def test_input_hash_mismatch_stops_before_extraction(bounded_fixture, monkeypatch, position):
    paths, pins, *_ = bounded_fixture
    paths[position].write_bytes(b"changed synthetic fixture")
    monkeypatch.setattr(stream, "extract_selected_archive", lambda *a, **k: pytest.fail("decoded before pins"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.run_archive_audit(*paths, **pins)


@pytest.mark.parametrize("fault", ["missing", "test_row", "label", "pixels", "instrumentation"])
def test_bad_batch_aborts_without_loading_another(bounded_fixture, monkeypatch, fault):
    paths, pins, calls, _, _, extract = bounded_fixture

    def bad(*args, **kwargs):
        rows = extract(*args, **kwargs)
        from dataclasses import replace
        if fault == "missing":
            rows.pop()
        elif fault == "test_row":
            rows[0] = replace(rows[0], dataset_index=673)
        elif fault == "label":
            rows[0] = replace(rows[0], label=2)
        elif fault == "pixels":
            rows[0] = replace(rows[0], pixels=b"bad")
        else:
            kwargs["instrumentation"].forbidden_payloads_materialized = 1
        return rows

    monkeypatch.setattr(stream, "extract_selected_archive", bad)
    with pytest.raises(ValueError):
        audit.run_archive_audit(*paths, **pins)
    assert len(calls) == 1


def test_partial_duplicate_group_is_not_success_on_recheck_failure(bounded_fixture, monkeypatch):
    paths, pins, calls, _, _, extract = bounded_fixture

    def fail_on_confirmation(*args, **kwargs):
        if len(calls) == 85:
            raise ValueError("synthetic confirmation failure")
        return extract(*args, **kwargs)

    monkeypatch.setattr(stream, "extract_selected_archive", fail_on_confirmation)
    with pytest.raises(ValueError, match="confirmation failure"):
        audit.run_archive_audit(*paths, **pins)
    assert len(calls) == 85


def test_cli_is_disabled_by_default_before_file_access(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "run_archive_audit", lambda *a: pytest.fail("real audit entered"))
    with pytest.raises(SystemExit):
        audit.main(["--output-dir", str(tmp_path / "report")])


def test_cli_outputs_are_new_metadata_only_and_never_overwrite(tmp_path, bounded_fixture, monkeypatch):
    paths, pins, *_ = bounded_fixture
    synthetic = audit.run_archive_audit(*paths, **pins)
    monkeypatch.setattr(audit, "run_archive_audit", lambda *a: deepcopy(synthetic))
    out = tmp_path / "analysis_reports/duplicate_audit"
    args = ["--execute-real-audit", "--project-root", str(tmp_path), "--output-dir", str(out)]
    assert audit.main(args) == 0
    assert {p.name for p in out.iterdir()} == {"image_hashes.csv", "duplicate_groups.json", "audit_report.json"}
    saved = json.loads((out / "audit_report.json").read_text())
    for name, expected in saved["artifact_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    monkeypatch.setattr(audit, "run_archive_audit", lambda *a: pytest.fail("overwritten namespace"))
    with pytest.raises(ValueError, match="no overwrite"):
        audit.main(args)
    with pytest.raises(ValueError):
        audit.main([*args[:-1], str(tmp_path / "outputs/models/unsafe")])
