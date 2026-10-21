"""Metadata-only, pinned development-cohort correction; no image/model imports.

The original split is hashed as opaque bytes, but only its TRAIN/VALIDATION
prefix is interpreted. Archive/checkpoint identities are inherited from the
completed audit, never reopened here. No execution approval for experiments.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re


PROTOCOL = "MiraBest-Development-ExactDedup-v1.0.0"
SCHEMA = "MiraBest-development-exactdedup-manifest-v1"
OUTPUT_REFERENCE = "data/manifests/mirabest_development_exactdedup_v1.0.0"
SPLIT_REFERENCE = "data/splits/split_indices.json"
AUDIT_REFERENCE = "analysis_reports/step14b_real_duplicate_audit/run_20261019_221005/results"
ARCHIVE_REFERENCE = "data/raw/mirabest/batches.tar.gz"
QUARANTINE = {198: ("validation", 0), 323: ("train", 1),
              474: ("validation", 0), 587: ("validation", 1), 694: ("train", 1)}
GROUP_HASH = "f2c580e4e323dd2999048b32cbc58c0d3826a853a09f24fb320eee81d4cf5151"
EXPECTED = {"train": {"total": 552, "0": 245, "1": 307},
            "validation": {"total": 116, "0": 51, "1": 65}}
REASON = "unresolved_conflicting_label_exact_content_group"
LIMITATIONS = [
    "Exact-content correction within audited TRAIN/VALIDATION only; no TEST cleanliness claim.",
    "Quarantine does not adjudicate labels or explain the upstream duplication.",
    "Near duplicates and astronomical source-level independence remain unverified.",
    "Historical model selection and metrics are not corrected by this manifest.",
    "Existing XAI runs are original-model engineering evidence, not corrected-model results.",
    "Archive and byte-confirmation provenance is inherited; no new pixels were decoded.",
    "Future training, calibration, evaluation and XAI require separate authorization.",
]


@dataclass(frozen=True)
class EvidencePins:
    split_sha256: str
    audit_report_sha256: str
    image_hashes_sha256: str
    duplicate_groups_sha256: str
    archive_sha256: str
    archive_md5: str
    reader_sha256: str
    audit_source_sha256: str


FROZEN_PINS = EvidencePins(
    "c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008",
    "4361fa04acfda3d12645cafcaa18e9a345b724e0316902ed8dc62ac0fdc6737b",
    "c85ddd482e96ecb8e56eec82eefd889346c2823e182abc6ff1044cfd2469df68",
    "fff558c1ecd555d4a4a7196e39457abec54edb587befce3f069d01130c444b79",
    "505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8",
    "fa9d6ad8fdde95b0a19df11433e8bca3",
    "981da0039a69d0e743093cd3122a07b0b873f0984274f21d19656fc7ef06a9f8",
    "353d307bd4eaaad19017946da4b468981130031578986c041660c863c7831c88",
)


def canonical_bytes(value) -> bytes:
    """Existing repository JSON convention: sorted ASCII-safe UTF-8 + final LF."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def development_prefix(handle) -> dict[str, list[int]]:
    """Parse only the frozen two-list prefix; never read the next key/value.

    Deliberately not a general JSON parser. The complete file must separately
    match a trusted SHA-256. One-byte unbuffered reads avoid reading ahead into
    the locked suffix. Fail closed on a different layout, nonintegers or size.
    """
    consumed = 0

    def next_byte():
        nonlocal consumed
        consumed += 1
        if consumed > 65536:
            raise ValueError("Development prefix exceeds supported size.")
        value = handle.read(1)
        if not value:
            raise ValueError("Truncated development prefix.")
        return value

    def token():
        value = next_byte()
        while value in b" \t\r\n":
            value = next_byte()
        return value

    def require(expected):
        if token() != expected:
            raise ValueError("Unsupported development split prefix layout.")

    require(b"{")
    result = {}
    for position, name in enumerate(("train", "validation")):
        if position:
            require(b",")
        require(b'"')
        for char in name.encode("ascii") + b'"':
            if next_byte() != bytes([char]):
                raise ValueError("Expected TRAIN then VALIDATION; locked suffix is not parsed.")
        require(b":")
        require(b"[")
        values = []
        value = token()
        if value != b"]":
            while True:
                digits = bytearray()
                while value in b"0123456789":
                    digits.extend(value)
                    value = next_byte()
                if not digits or (len(digits) > 1 and digits[0] == ord("0")):
                    raise ValueError("Development indices must be JSON nonnegative integers.")
                values.append(int(digits))
                while value in b" \t\r\n":
                    value = next_byte()
                if value == b"]":
                    break
                if value != b",":
                    raise ValueError("Malformed development index list.")
                value = token()
        result[name] = values
    return result  # No read after VALIDATION closing bracket.


def read_development_splits(path: Path, expected_sha256: str) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Historical split provenance mismatch.")
    with Path(path).open("rb", buffering=0) as handle:
        result = development_prefix(handle)
    if sha256_file(path) != expected_sha256:
        raise ValueError("Historical split changed while reading metadata.")
    return result


def _validate_parent(splits, audit):
    if set(splits) != {"train", "validation"}:
        raise ValueError("Only development splits may be supplied.")
    for split, expected in (("train", 554), ("validation", 119)):
        values = splits[split]
        if (not isinstance(values, list) or len(values) != expected
                or any(type(i) is not int or not 0 <= i < 792 for i in values)
                or len(set(values)) != len(values) or values != sorted(values)):
            raise ValueError("Invalid, repeated or unordered parent development indices.")
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("Parent TRAIN/VALIDATION index overlap.")
    if (audit.get("status") != "complete" or audit.get("authorized_scope") != "TRAIN + VALIDATION"
            or audit.get("authorized_images_processed") != 673):
        raise ValueError("Audit completion evidence is missing.")
    rows = audit.get("records")
    if not isinstance(rows, list) or len(rows) != 673:
        raise ValueError("Incomplete audit metadata.")
    by_index = {}
    hashes = defaultdict(list)
    for row in rows:
        index = row.get("dataset_index")
        split = row.get("split")
        if (type(index) is not int or index in by_index or split not in splits
                or index not in splits[split]):
            raise ValueError("Duplicate, unauthorized or inconsistent audit index.")
        if (type(row.get("label")) is not int or row["label"] not in (0, 1)
                or row.get("shape") != [150, 150] or row.get("dtype") != "uint8"
                or not isinstance(row.get("filename"), str) or not row["filename"]
                or row.get("member") != (f"batches/data_batch_{index // 88 + 1}" if index < 704 else "batches/test_batch")
                or row.get("member_row") != index % 88
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("canonical_sha256", "")))):
            raise ValueError("Malformed label/content/filename metadata.")
        by_index[index] = row
        hashes[row["canonical_sha256"]].append(index)
    if set(by_index) != set(splits["train"]) | set(splits["validation"]):
        raise ValueError("Audit membership differs from the parent development scope.")
    repeated = {h: sorted(indices) for h, indices in hashes.items() if len(indices) > 1}
    if repeated != {GROUP_HASH: sorted(QUARANTINE)} or audit.get("unique_image_hashes") != 669:
        raise ValueError("Unexpected content-hash overlap/collision; review required.")
    groups = audit.get("duplicate_groups")
    if (not isinstance(groups, list) or len(groups) != 1
            or groups[0].get("dataset_indices") != sorted(QUARANTINE)
            or groups[0].get("canonical_sha256") != GROUP_HASH
            or groups[0].get("confirmed_byte_equal") is not True
            or groups[0].get("members") != [by_index[i] for i in sorted(QUARANTINE)]):
        raise ValueError("Complete byte-confirmed quarantine group evidence is required.")
    for index, (split, label) in QUARANTINE.items():
        row = by_index[index]
        if (row["split"], row["label"]) != (split, label) or row.get("confirmed_byte_equal") is not True:
            raise ValueError("Quarantine split/label/byte-confirmation mismatch.")
    return by_index


def build_manifests(splits: dict, audit: dict, pins: EvidencePins = FROZEN_PINS) -> tuple[dict, dict]:
    """Pure deterministic derivation from validated metadata; no disk/pixel access."""
    by_index = _validate_parent(splits, audit)
    expected_provenance = {k: v for k, v in asdict(pins).items()
                           if k in {"split_sha256", "archive_sha256", "archive_md5", "reader_sha256", "audit_source_sha256"}}
    if any(audit.get("provenance", {}).get(k) != v for k, v in expected_provenance.items()):
        raise ValueError("Audit source provenance mismatch.")

    def record(index, quarantined=False):
        source = by_index[index]
        value = {"dataset_index": index, "original_split": source["split"],
                 "ground_truth_label": source["label"], "original_filename": source["filename"],
                 "canonical_sha256": source["canonical_sha256"], "shape": source["shape"].copy(),
                 "dtype": source["dtype"], "archive_member": source["member"], "member_row": source["member_row"]}
        if quarantined:
            value.update(reason=REASON, group_indices=sorted(QUARANTINE))
        else:
            value["corrected_split"] = source["split"]
        return value

    corrected = {name: [i for i in splits[name] if i not in QUARANTINE] for name in splits}
    records = [record(i) for i in sorted(set().union(*map(set, corrected.values())))]
    for name, indices in corrected.items():
        counts = Counter(by_index[i]["label"] for i in indices)
        if {"total": len(indices), "0": counts[0], "1": counts[1]} != EXPECTED[name]:
            raise ValueError("Corrected cohort/class counts differ from the approved protocol.")
    retained_hashes = [row["canonical_sha256"] for row in records]
    if len(records) != 668 or len(set(retained_hashes)) != 668:
        raise ValueError("Corrected content-hash overlap/collision detected.")
    quarantine = {"schema": "MiraBest-development-quarantine-v1", "protocol": PROTOCOL,
                  "records": [record(i, True) for i in sorted(QUARANTINE)]}
    manifest = {
        "schema": SCHEMA, "protocol": PROTOCOL, "version": "1.0.0",
        "status": "membership_implemented", "execution_authorized": False,
        "class_mapping": {"0": "FR-I", "1": "FR-II"},
        "correction_rationale": REASON,
        "selection_rule": "Original TRAIN/VALIDATION minus exactly five quarantined global indices; ascending order; no reassignment or relabeling.",
        "parent_provenance": {"split_reference": SPLIT_REFERENCE, "split_sha256": pins.split_sha256,
                              "archive_reference": ARCHIVE_REFERENCE, "archive_sha256": pins.archive_sha256,
                              "archive_md5": pins.archive_md5, "source_reference": "https://zenodo.org/records/4288837",
                              "audit_reference": AUDIT_REFERENCE, "audit_report_sha256": pins.audit_report_sha256,
                              "image_hashes_sha256": pins.image_hashes_sha256,
                              "duplicate_groups_sha256": pins.duplicate_groups_sha256,
                              "reader_sha256": pins.reader_sha256, "audit_source_sha256": pins.audit_source_sha256},
        "splits": corrected, "expected_counts": deepcopy(EXPECTED), "records": records,
        "quarantine_reference": "quarantine.json",
        "quarantine_sha256": hashlib.sha256(canonical_bytes(quarantine)).hexdigest(),
        "limitations": LIMITATIONS.copy(),
    }
    return manifest, quarantine


def validate_manifest(manifest, quarantine, splits, audit, pins=FROZEN_PINS) -> str:
    """Re-derive the complete manifest; checksum alone never blesses altered labels."""
    expected, expected_quarantine = build_manifests(splits, audit, pins)
    if canonical_bytes(manifest) != canonical_bytes(expected) or canonical_bytes(quarantine) != canonical_bytes(expected_quarantine):
        raise ValueError("Corrected manifest differs from approved membership/labels/provenance.")
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def load_evidence(root: Path, pins=FROZEN_PINS):
    """Read pinned metadata files only; never invoke dataset/reader APIs."""
    root = Path(root)
    split_path = root / SPLIT_REFERENCE
    splits = read_development_splits(split_path, pins.split_sha256)
    evidence = {}
    for name, pin in (("audit_report.json", pins.audit_report_sha256),
                      ("image_hashes.csv", pins.image_hashes_sha256),
                      ("duplicate_groups.json", pins.duplicate_groups_sha256)):
        raw = (root / AUDIT_REFERENCE / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError(f"Saved audit checksum mismatch: {name}")
        evidence[name] = raw
    for path, pin in (("src/data/mirabest_pickle_stream_reader.py", pins.reader_sha256),
                      ("scripts/run_duplicate_audit_train_val.py", pins.audit_source_sha256)):
        if sha256_file(root / path) != pin:
            raise ValueError("Audit implementation provenance mismatch.")
    audit = json.loads(evidence["audit_report.json"])
    if json.loads(evidence["duplicate_groups.json"]) != audit.get("duplicate_groups"):
        raise ValueError("Saved group artifacts disagree.")
    table = list(csv.DictReader(io.StringIO(evidence["image_hashes.csv"].decode("utf-8"))))
    if table != [{key: str(value) for key, value in row.items()} for row in audit.get("records", [])]:
        raise ValueError("Saved CSV and JSON metadata disagree.")
    build_manifests(splits, audit, pins)
    return splits, audit


def generate_protocol(root: Path, *, pins=FROZEN_PINS, generation_metadata=None) -> dict:
    """Create a new fixed namespace with exclusive writes, completion marker last.

    A failed/incomplete namespace is never overwritten or silently resumed.
    Custom pins support synthetic fixtures; the CLI exposes no pin override.
    """
    root = Path(root).resolve()
    destination = root / OUTPUT_REFERENCE
    if destination.exists():
        raise FileExistsError("Corrected protocol namespace already exists; refusing overwrite.")
    # Resolve symlinks before creating anything: never publish over parent evidence.
    if not destination.resolve().is_relative_to(root / "data/manifests"):
        raise ValueError("Unsafe corrected output namespace.")
    splits, audit = load_evidence(root, pins)
    manifest, quarantine = build_manifests(splits, audit, pins)
    checksum = validate_manifest(manifest, quarantine, splits, audit, pins)
    if canonical_bytes(build_manifests(splits, audit, pins)[0]) != canonical_bytes(manifest):
        raise ValueError("Non-deterministic manifest reproduction.")
    provenance = {
        "protocol": PROTOCOL, "canonical_membership_sha256": checksum,
        "canonical_serialization": "UTF-8 JSON; sorted keys; compact separators; ensure_ascii=True; allow_nan=False; final LF. SHA-256 of complete manifest.json bytes; checksum stored separately.",
        "timestamp_and_absolute_paths_in_membership": False,
        "evidence_pins": asdict(pins),
        "verification_scope": {"split_and_saved_audit": "verified_current_bytes",
                               "archive_and_pixel_equality": "inherited_from_pinned_completed_Step14B",
                               "reader_and_audit_sources": "verified_current_bytes",
                               "test_values": "not_interpreted; whole split hashed opaquely"},
        "generator_reference": "src/data/corrected_protocol.py",
        "generation_metadata": generation_metadata or {"timestamp": "not_recorded", "command": "not_recorded"},
        "experiment_execution_authorized": False,
    }
    artifacts = {"manifest.json": canonical_bytes(manifest),
                 "manifest.sha256": (checksum + "  manifest.json\n").encode("ascii"),
                 "quarantine.json": canonical_bytes(quarantine),
                 "provenance.json": canonical_bytes(provenance),
                 "README.md": ("# " + PROTOCOL + "\n\nMetadata-only corrected development cohort: TRAIN 552, VALIDATION 116.\n"
                               "Five unresolved conflicting-label duplicates are quarantined, not relabeled or deleted.\n"
                               "Parent data/splits/split_indices.json remains authoritative historical evidence.\n"
                               "manifest.sha256 hashes all canonical manifest.json bytes, including final LF.\n"
                               "No timestamps/host paths are in that manifest; generation details are in provenance.json.\n"
                               "Archive and byte-equality identities are inherited from Step 14B, not re-audited.\n"
                               "This does not authorize training, calibration, evaluation or XAI.\n"
                               "A missing validation_report.json means publication is incomplete; never overwrite/resume.\n").encode("utf-8")}
    report = {"status": "complete", "protocol": PROTOCOL, "counts": EXPECTED,
              "total": 668, "quarantined_indices": sorted(QUARANTINE),
              "canonical_membership_sha256": checksum,
              "content_hashes_unique": 668, "train_validation_content_disjoint": True,
              "labels_and_assignments_preserved": True, "deterministic_reproduction": True,
              "test_records_interpreted": 0, "images_decoded": 0,
              "experiment_execution_authorized": False,
              "artifact_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in artifacts.items()}}
    if sha256_file(root / SPLIT_REFERENCE) != pins.split_sha256:
        raise ValueError("Parent split changed before publication.")
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in artifacts.items():
        with (destination / name).open("xb") as handle:
            handle.write(raw)
    if any(sha256_file(destination / name) != expected
           for name, expected in report["artifact_sha256"].items()):
        raise ValueError("Published metadata checksum mismatch; namespace incomplete.")
    # Only complete, verified namespaces receive this completion marker.
    with (destination / "validation_report.json").open("xb") as handle:
        handle.write(canonical_bytes(report))
    return report
