"""Synthetic metadata only; no MiraBest archive, image, model or TEST decoding."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json

import pytest

from src.data import corrected_protocol as cp


@pytest.fixture
def evidence():
    universe = set(range(672)) | {694}
    val = set([198, 474, 587] + sorted(universe - set(cp.QUARANTINE))[:116])
    splits = {"train": sorted(universe - val), "validation": sorted(val)}
    assert len(splits["train"]) == 554 and len(splits["validation"]) == 119
    labels = {}
    for split, nzero in [("train", 245), ("validation", 53)]:
        assigned = {i: label for i, (name, label) in cp.QUARANTINE.items() if name == split}
        remaining_zero = nzero - sum(v == 0 for v in assigned.values())
        for i in splits[split]:
            if i not in assigned:
                assigned[i] = 0 if remaining_zero > 0 else 1
                remaining_zero -= int(assigned[i] == 0)
        labels.update(assigned)
    records = []
    for index in sorted(universe):
        records.append({
            "dataset_index": index, "split": "train" if index in splits["train"] else "validation",
            "label": labels[index], "filename": f"synthetic_{index}.png",
            "member": f"batches/data_batch_{index // 88 + 1}", "member_row": index % 88,
            "shape": [150, 150], "dtype": "uint8",
            "canonical_sha256": cp.GROUP_HASH if index in cp.QUARANTINE else hashlib.sha256(f"synthetic-{index}".encode()).hexdigest(),
            "confirmed_byte_equal": index in cp.QUARANTINE,
        })
    pins = cp.FROZEN_PINS
    audit = {"status": "complete", "authorized_scope": "TRAIN + VALIDATION",
             "authorized_images_processed": 673, "unique_image_hashes": 669, "records": records,
             "provenance": {k: getattr(pins, k) for k in ["split_sha256", "archive_sha256", "archive_md5", "reader_sha256", "audit_source_sha256"]},
             "duplicate_groups": [{"dataset_indices": sorted(cp.QUARANTINE), "canonical_sha256": cp.GROUP_HASH,
                                   "confirmed_byte_equal": True,
                                   "members": [r for r in records if r["dataset_index"] in cp.QUARANTINE]}]}
    return splits, audit


@pytest.fixture
def disk_evidence(tmp_path_factory, evidence):
    import csv
    tmp_path = tmp_path_factory.mktemp("p")
    splits, audit = deepcopy(evidence)
    split_path = tmp_path / cp.SPLIT_REFERENCE
    split_path.parent.mkdir(parents=True)
    # Forbidden suffix is intentionally not JSON: successful generation must not parse it.
    prefix = json.dumps(splits)[:-1].encode()
    split_path.write_bytes(prefix + b',"test":DO_NOT_INTERPRET_THIS_SENTINEL}')
    reader = tmp_path / "src/data/mirabest_pickle_stream_reader.py"
    reader.parent.mkdir(parents=True)
    reader.write_text("# synthetic inert reader identity only\n")
    source = tmp_path / "scripts/run_duplicate_audit_train_val.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic inert audit identity only\n")
    pins = replace(cp.FROZEN_PINS, split_sha256=cp.sha256_file(split_path),
                   reader_sha256=cp.sha256_file(reader), audit_source_sha256=cp.sha256_file(source))
    audit["provenance"].update(split_sha256=pins.split_sha256,
                               reader_sha256=pins.reader_sha256, audit_source_sha256=pins.audit_source_sha256)
    folder = tmp_path / cp.AUDIT_REFERENCE
    folder.mkdir(parents=True)
    (folder / "audit_report.json").write_bytes(cp.canonical_bytes(audit))
    (folder / "duplicate_groups.json").write_bytes(cp.canonical_bytes(audit["duplicate_groups"]))
    with (folder / "image_hashes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit["records"][0]))
        writer.writeheader()
        writer.writerows(audit["records"])
    pins = replace(pins, audit_report_sha256=cp.sha256_file(folder / "audit_report.json"),
                   image_hashes_sha256=cp.sha256_file(folder / "image_hashes.csv"),
                   duplicate_groups_sha256=cp.sha256_file(folder / "duplicate_groups.json"))
    return tmp_path, pins


def test_exact_counts_quarantine_global_indices_and_original_labels(evidence):
    splits, audit = evidence
    before = deepcopy(evidence)
    manifest, quarantine = cp.build_manifests(splits, audit)
    assert {k: len(v) for k, v in manifest["splits"].items()} == {"train": 552, "validation": 116}
    assert len(manifest["records"]) == 668
    assert [r["dataset_index"] for r in quarantine["records"]] == sorted(cp.QUARANTINE)
    for name, indices in manifest["splits"].items():
        assert indices == [i for i in splits[name] if i not in cp.QUARANTINE]
        assert not set(indices) & set(cp.QUARANTINE)
        counts = Counter(r["ground_truth_label"] for r in manifest["records"] if r["corrected_split"] == name)
        assert counts == {0: cp.EXPECTED[name]["0"], 1: cp.EXPECTED[name]["1"]}
    by_index = {r["dataset_index"]: r for r in audit["records"]}
    for row in manifest["records"]:
        source = by_index[row["dataset_index"]]
        assert row["ground_truth_label"] == source["label"]
        assert row["original_split"] == row["corrected_split"] == source["split"]
    assert evidence == before
    assert "test" not in manifest["splits"]
    assert manifest["execution_authorized"] is False


def test_deterministic_serialization_and_detached_contents(evidence):
    m, q = cp.build_manifests(*evidence)
    assert cp.canonical_bytes(m) == cp.canonical_bytes(cp.build_manifests(*evidence)[0])
    assert cp.canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    assert cp.validate_manifest(m, q, *evidence) == hashlib.sha256(cp.canonical_bytes(m)).hexdigest()
    m["expected_counts"]["train"]["0"] = 999
    assert cp.EXPECTED["train"]["0"] == 245
    with pytest.raises(ValueError):
        cp.canonical_bytes({"invalid": float("nan")})


@pytest.mark.parametrize("change", ["duplicate", "overlap", "count", "order", "bool_index"])
def test_invalid_parent_memberships_rejected(evidence, change):
    splits, audit = deepcopy(evidence)
    if change == "duplicate":
        splits["train"][1] = splits["train"][0]
    elif change == "overlap":
        splits["validation"][0] = splits["train"][0]
        splits["validation"].sort()
    elif change == "count":
        splits["train"].pop()
    elif change == "order":
        splits["train"].reverse()
    else:
        splits["train"][0] = True
    with pytest.raises(ValueError):
        cp.build_manifests(splits, audit)


@pytest.mark.parametrize("change", ["quarantined", "new_index", "label", "provenance", "duplicate", "overlap"])
def test_manifest_mutations_rejected_even_with_recomputed_hash(evidence, change):
    m, q = cp.build_manifests(*evidence)
    if change == "quarantined":
        m["splits"]["train"].append(323)
    elif change == "new_index":
        m["splits"]["train"][0] = 791
    elif change == "label":
        m["records"][0]["ground_truth_label"] = 1 - m["records"][0]["ground_truth_label"]
    elif change == "provenance":
        m["parent_provenance"]["split_sha256"] = "0" * 64
    elif change == "duplicate":
        m["splits"]["train"][1] = m["splits"]["train"][0]
    else:
        m["splits"]["validation"][0] = m["splits"]["train"][0]
    with pytest.raises(ValueError, match="membership/labels/provenance"):
        cp.validate_manifest(m, q, *evidence)


@pytest.mark.parametrize("change", ["same_split_hash", "cross_split_hash", "label_count", "quarantine_label", "unconfirmed", "missing", "bad_dtype", "nonfinite_label"])
def test_bad_saved_evidence_is_not_silently_corrected(evidence, change):
    splits, audit = deepcopy(evidence)
    retained = [r for r in audit["records"] if r["dataset_index"] not in cp.QUARANTINE]
    if change in {"same_split_hash", "cross_split_hash"}:
        first = retained[0]
        other = next(r for r in retained[1:] if (r["split"] == first["split"]) == (change == "same_split_hash"))
        other["canonical_sha256"] = first["canonical_sha256"]
    elif change == "label_count":
        retained[0]["label"] = 1 - retained[0]["label"]
    elif change == "quarantine_label":
        audit["duplicate_groups"][0]["members"][0]["label"] = 1
    elif change == "unconfirmed":
        audit["duplicate_groups"][0]["confirmed_byte_equal"] = False
    elif change == "missing":
        audit["records"].pop()
    elif change == "bad_dtype":
        retained[0]["dtype"] = "float32"
    else:
        retained[0]["label"] = float("nan")
    with pytest.raises(ValueError):
        cp.build_manifests(splits, audit)


def test_prefix_stops_before_forbidden_bytes():
    prefix = b'{"train":[0,2],"validation":[1]'
    class ForbiddenSuffix(io.BytesIO):
        def read(self, size=-1):
            assert size == 1
            assert self.tell() < len(prefix), "Attempt to access locked suffix"
            return super().read(size)
    handle = ForbiddenSuffix(prefix + b',"test":FORBIDDEN}')
    assert cp.development_prefix(handle) == {"train": [0, 2], "validation": [1]}
    assert handle.tell() == len(prefix)


@pytest.mark.parametrize("prefix", [b'{"validation":[]', b'{"train":[true]', b'{"train":[-1]', b'{"train":[01]', b'{"train":[1.0]'])
def test_prefix_fails_closed_for_unsupported_encoding(prefix):
    with pytest.raises(ValueError):
        cp.development_prefix(io.BytesIO(prefix))


def test_disk_generation_no_archive_access_no_overwrite_and_canonical_sidecar(disk_evidence, monkeypatch):
    root, pins = disk_evidence
    split = root / cp.SPLIT_REFERENCE
    original = split.read_bytes()
    from pathlib import Path
    old_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert "data/raw" not in path.as_posix() and "outputs/models" not in path.as_posix()
        return old_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    report = cp.generate_protocol(root, pins=pins, generation_metadata={"timestamp": "synthetic", "path": "host-specific"})
    directory = root / cp.OUTPUT_REFERENCE
    assert report["status"] == "complete" and report["total"] == 668
    assert split.read_bytes() == original
    assert report["images_decoded"] == report["test_records_interpreted"] == 0
    raw = (directory / "manifest.json").read_bytes()
    assert b"host-specific" not in raw and b"timestamp" not in raw
    assert report["canonical_membership_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (directory / "manifest.sha256").read_text().split()[0] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        cp.generate_protocol(root, pins=pins)
    assert split.read_bytes() == original


@pytest.mark.parametrize("file", [cp.SPLIT_REFERENCE, cp.AUDIT_REFERENCE + "/audit_report.json",
                                  "src/data/mirabest_pickle_stream_reader.py",
                                  cp.AUDIT_REFERENCE + "/image_hashes.csv",
                                  cp.AUDIT_REFERENCE + "/duplicate_groups.json"])
def test_source_provenance_mismatch_prevents_publication(disk_evidence, file):
    root, pins = disk_evidence
    path = root / file
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mismatch"):
        cp.generate_protocol(root, pins=pins)
    assert not (root / cp.OUTPUT_REFERENCE).exists()


def test_generation_metadata_does_not_change_membership_checksum(disk_evidence, tmp_path_factory):
    import shutil
    root, pins = disk_evidence
    second = tmp_path_factory.mktemp("q")
    for rel in [cp.SPLIT_REFERENCE, cp.AUDIT_REFERENCE, "src", "scripts"]:
        source = root / rel
        dest = second / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            shutil.copyfile(source, dest)
    first_report = cp.generate_protocol(root, pins=pins, generation_metadata={"timestamp": "one", "host": "A"})
    second_report = cp.generate_protocol(second, pins=pins, generation_metadata={"timestamp": "two", "host": "B"})
    assert first_report["canonical_membership_sha256"] == second_report["canonical_membership_sha256"]
    assert (root / cp.OUTPUT_REFERENCE / "manifest.json").read_bytes() == (second / cp.OUTPUT_REFERENCE / "manifest.json").read_bytes()


@pytest.mark.parametrize("field", ["archive_sha256", "archive_md5", "split_sha256", "audit_source_sha256"])
def test_inherited_provenance_cannot_be_substituted(evidence, field):
    splits, audit = deepcopy(evidence)
    audit["provenance"][field] = "0" * len(audit["provenance"][field])
    with pytest.raises(ValueError, match="provenance mismatch"):
        cp.build_manifests(splits, audit)


@pytest.mark.parametrize("change", ["duplicate", "unauthorized"])
def test_audit_indices_must_equal_closed_parent_scope(evidence, change):
    splits, audit = deepcopy(evidence)
    audit["records"][0]["dataset_index"] = audit["records"][1]["dataset_index"] if change == "duplicate" else 791
    with pytest.raises(ValueError):
        cp.build_manifests(splits, audit)


def test_failed_publication_has_no_success_marker_and_cannot_overwrite(disk_evidence, monkeypatch):
    from pathlib import Path
    root, pins = disk_evidence
    original_open = Path.open
    def fail_quarantine(path, mode="r", *args, **kwargs):
        if path.name == "quarantine.json" and mode == "xb":
            raise OSError("Synthetic interrupted metadata publication")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_quarantine)
    with pytest.raises(OSError, match="interrupted"):
        cp.generate_protocol(root, pins=pins)
    assert not (root / cp.OUTPUT_REFERENCE / "validation_report.json").exists()
    with pytest.raises(FileExistsError):
        cp.generate_protocol(root, pins=pins)
