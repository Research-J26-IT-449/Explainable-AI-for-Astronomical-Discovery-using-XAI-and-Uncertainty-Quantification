"""Pinned TRAIN/VALIDATION exact-image audit; real execution requires opt-in.

Only the existing bounded, non-executable archive reader may decode real data.
No model, preprocessing, source matching or TEST image decoding is performed.
"""

from __future__ import annotations

import hashlib
import json
import argparse
import csv
import io
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.selective_raw_reader import SelectiveRawReader, canonical_sha256
from src.data import mirabest_pickle_stream_reader as stream

EXPECTED_SPLIT_SHA256 = "c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008"
EXPECTED_ARCHIVE_SHA256 = "505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8"
EXPECTED_CHECKPOINT_SHA256 = "d5e1f126c13c2b7bebb95e35f41f0bc6397dd0ea9f4fbfcfbc55648609ab5bc7"
EXPECTED_ARCHIVE_MD5 = "fa9d6ad8fdde95b0a19df11433e8bca3"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split(path: str | Path, expected_sha256: str = EXPECTED_SPLIT_SHA256):
    path = Path(path)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise ValueError("Frozen split SHA-256 mismatch.")
    split = json.loads(raw)
    expected_counts = {"train": 554, "validation": 119, "test": 119}
    if set(split) != set(expected_counts):
        raise ValueError("Unexpected split keys.")
    sets = {}
    for name, count in expected_counts.items():
        values = split[name]
        if (not isinstance(values, list) or len(values) != count
                or any(type(index) is not int or not 0 <= index < 792 for index in values)
                or len(set(values)) != count):
            raise ValueError(f"Invalid {name} split.")
        sets[name] = frozenset(values)
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("Frozen splits overlap.")
    if set().union(*sets.values()) != set(range(792)):
        raise ValueError("Frozen splits do not cover indices 0..791.")
    return sets


def classify_duplicate_groups(records: Iterable[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[record["canonical_sha256"]].append(record)
    result = []
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        splits = sorted({item["split"] for item in members})
        labels = sorted({item["label"] for item in members})
        result.append({
            "canonical_sha256": digest,
            "confirmed_byte_equal": all(item["confirmed_byte_equal"] for item in members),
            "dataset_indices": sorted(item["dataset_index"] for item in members),
            "splits": splits,
            "labels": labels,
            "cross_train_validation": set(splits) == {"train", "validation"},
            "within_train": sum(m["split"] == "train" for m in members) >= 2,
            "within_validation": sum(m["split"] == "validation" for m in members) >= 2,
            "conflicting_labels": len(labels) > 1,
            "members": sorted(members, key=lambda item: item["dataset_index"]),
        })
    return result


def audit_records(reader: SelectiveRawReader, split_sets: dict[str, frozenset]) -> dict:
    records = []
    for split in ("train", "validation"):
        for index in sorted(split_sets[split]):
            image = reader.read(index)
            records.append({
                "dataset_index": index,
                "split": split,
                "label": image.metadata.label,
                "filename": image.metadata.filename,
                "member": image.metadata.member,
                "member_row": image.metadata.member_row,
                "canonical_sha256": canonical_sha256(image.pixels),
                "confirmed_byte_equal": False,
            })
    groups = classify_duplicate_groups(records)
    # Legacy synthetic MBRAW1 API: do not assert equality from hashes alone.
    for group in groups:
        first = reader.read(group["dataset_indices"][0]).pixels
        for index in group["dataset_indices"][1:]:
            if reader.read(index).pixels != first:
                raise ValueError("Hash-matched content differs; exact equality is unconfirmed.")
        group["confirmed_byte_equal"] = True
        for member in group["members"]:
            member["confirmed_byte_equal"] = True
    unique_hashes = len({item["canonical_sha256"] for item in records})
    return {
        "authorized_scope": "TRAIN + VALIDATION",
        "authorized_images_processed": len(records),
        "forbidden_scope": "TEST",
        "test_images_decoded": reader.test_payload_reads,
        "unique_image_hashes": unique_hashes,
        "duplicate_groups": groups,
        "duplicate_group_count": len(groups),
        "affected_unique_images": len({i for group in groups for i in group["dataset_indices"]}),
        "cross_split_duplicate_group_count": sum(group["cross_train_validation"] for group in groups),
        "conflicting_label_group_count": sum(group["conflicting_labels"] for group in groups),
    }


def refuse_existing_mirabest_loader() -> None:
    raise RuntimeError(
        "Real MiraBest pickle batches are not supported by this isolated safety harness."
    )


def run_archive_audit(archive_path, split_path, checkpoint_path, *,
                      expected_archive_sha256=EXPECTED_ARCHIVE_SHA256,
                      expected_archive_md5=EXPECTED_ARCHIVE_MD5,
                      expected_split_sha256=EXPECTED_SPLIT_SHA256,
                      expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256) -> dict:
    """Audit the entire saved permitted cohort in <=8-row extraction calls.

    This function performs real decoding ONLY when explicitly invoked. Synthetic
    tests supply fixture files/pins. CLI pins cannot be overridden. Re-reading
    hash candidates in pairs confirms bytes without retaining previous batches.
    Opaque archive checksum reads are not image materialization. Metadata and
    hashes persist; raw payloads never leave the batch helpers below.
    """
    paths = [Path(archive_path), Path(split_path), Path(checkpoint_path)]
    pins = [expected_archive_sha256, expected_split_sha256, expected_checkpoint_sha256]
    for path, pin in zip(paths, pins):
        if sha256_file(path) != pin:
            raise ValueError("Frozen input SHA-256 mismatch.")
    sets = validate_split(paths[1], expected_split_sha256)
    allowed, forbidden = sets["train"] | sets["validation"], sets["test"]
    order = tuple(sorted(allowed))
    if len(order) != 673 or allowed & forbidden:
        raise ValueError("Require exactly the frozen TRAIN+VALIDATION cohort.")
    records, batches = [], []

    def consume(selected, operation, phase):
        selected = tuple(selected)
        if (not 1 <= len(selected) <= 8 or len(set(selected)) != len(selected)
                or not set(selected) <= allowed or set(selected) & forbidden):
            raise ValueError("Unauthorized or unbounded extraction request.")
        instrumentation = stream.ReaderInstrumentation()
        rows = stream.extract_selected_archive(
            paths[0], allowed, forbidden, selected,
            expected_sha256=expected_archive_sha256, expected_md5=expected_archive_md5,
            instrumentation=instrumentation,
        )
        try:
            if [r.dataset_index for r in rows] != sorted(selected):
                raise ValueError("Missing, duplicate or unauthorized extracted rows.")
            if (instrumentation.permitted_payload_extractions != len(selected)
                    or instrumentation.forbidden_payload_extraction_attempts
                    or instrumentation.forbidden_payloads_materialized
                    or instrumentation.unexpected_parser_events
                    or instrumentation.maximum_pixel_buffer_size > stream.IMAGE_BYTES):
                raise ValueError("Bounded reader instrumentation failed.")
            for row in rows:
                if (type(row.label) is not int or row.label not in (0, 1)
                        or not isinstance(row.filename, str) or not row.filename
                        or row.member != stream.EXPECTED_MEMBER_NAMES[row.dataset_index // 88]
                        or row.member_row != row.dataset_index % 88):
                    raise ValueError("Extracted row provenance is inconsistent.")
                stream.canonical_sha256(row.pixels)  # Enforce 22500 decoded uint8 bytes.
            result = operation(rows)
            batches.append({"phase": phase, "indices": list(sorted(selected)),
                            "selected_count": len(rows),
                            "retained_pixel_bytes": sum(len(r.pixels) for r in rows),
                            "reader": asdict(instrumentation)})
            return result
        finally:
            rows.clear()  # Release ALL row/pixel references before another call.

    def metadata(rows):
        return [{"dataset_index": row.dataset_index,
                 "split": "train" if row.dataset_index in sets["train"] else "validation",
                 "label": row.label, "filename": row.filename, "member": row.member,
                 "member_row": row.member_row, "shape": [150, 150], "dtype": "uint8",
                 "canonical_sha256": stream.canonical_sha256(row.pixels),
                 "confirmed_byte_equal": False} for row in rows]

    for start in range(0, len(order), 8):
        records.extend(consume(order[start:start+8], metadata, "content_hashing"))
    groups = classify_duplicate_groups(records)
    by_index = {row["dataset_index"]: row for row in records}
    for group in groups:
        first = group["dataset_indices"][0]

        def equal_pair(rows):
            if metadata(rows) != [by_index[row.dataset_index] for row in rows]:
                raise ValueError("Candidate metadata/content drift on verification.")
            if rows[0].pixels != rows[1].pixels:
                raise ValueError("Hash-matched content differs; abort exact-duplicate report.")

        for other in group["dataset_indices"][1:]:
            consume((first, other), equal_pair, "byte_confirmation")
        group["confirmed_byte_equal"] = True
    # Only after every candidate has passed, mark its records as confirmed.
    for group in groups:
        for row in group["members"]:
            row["confirmed_byte_equal"] = True
    for path, pin in zip(paths, pins):
        if sha256_file(path) != pin:
            raise ValueError("Frozen input changed during audit.")
    return {
        "schema": "MiraBest-exact-duplicate-audit-v1", "status": "complete",
        "authorized_scope": "TRAIN + VALIDATION", "authorized_images_processed": len(records),
        "split_counts": {k: len(v) for k, v in sets.items()}, "test_images_decoded": 0,
        "unique_image_hashes": len({r["canonical_sha256"] for r in records}),
        "duplicate_group_count": len(groups), "duplicate_groups": groups,
        "within_train_duplicate_group_count": sum(g["within_train"] for g in groups),
        "within_validation_duplicate_group_count": sum(g["within_validation"] for g in groups),
        "cross_split_duplicate_group_count": sum(g["cross_train_validation"] for g in groups),
        "conflicting_label_group_count": sum(g["conflicting_labels"] for g in groups),
        "affected_unique_images": len({i for g in groups for i in g["dataset_indices"]}),
        "records": records, "batches": batches,
        "maximum_selected_images_per_batch": max(b["selected_count"] for b in batches),
        "maximum_retained_batch_pixel_bytes": max(b["retained_pixel_bytes"] for b in batches),
        "provenance": {"archive_sha256": pins[0], "archive_md5": expected_archive_md5,
                       "split_sha256": pins[1], "checkpoint_sha256": pins[2],
                       "archive_reference": str(paths[0]), "split_reference": str(paths[1]),
                       "checkpoint_reference": str(paths[2]),
                       "reader_sha256": sha256_file(Path(stream.__file__)),
                       "audit_source_sha256": sha256_file(Path(__file__))},
        "hash_definition": "SHA256(MAGIC_CANONICAL + <u1\\0 + big-endian shape(150,150) + C-order decoded bytes)",
        "limitations": ["Exact decoded-image equality only; no near-duplicate search.",
                        "Image duplicates do not establish astronomical source independence.",
                        "Group categories can overlap; do not add their counts as disjoint totals.",
                        "Hash candidates are re-read in pairs for byte confirmation."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-real-audit", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute_real_audit:
        parser.error("Real audit is disabled; separate explicit authorization is required.")
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if not output.is_relative_to(root / "analysis_reports") or output.exists():
        raise ValueError("Require a new output directory under analysis_reports; no overwrite.")
    result = run_archive_audit(root / "data/raw/mirabest/batches.tar.gz",
                               root / "data/splits/split_indices.json",
                               root / "outputs/models/resnet_cnn_best.pth")
    output.mkdir(parents=True, exist_ok=False)
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(result["records"][0]))
    writer.writeheader()
    writer.writerows(result["records"])
    artifacts = {"image_hashes.csv": table.getvalue().encode("utf-8"),
                 "duplicate_groups.json": (json.dumps(result["duplicate_groups"], indent=2) + "\n").encode()}
    result["artifact_sha256"] = {p: hashlib.sha256(raw).hexdigest() for p, raw in artifacts.items()}
    # Completion JSON is written last; missing it means publication is incomplete.
    artifacts["audit_report.json"] = (json.dumps(result, indent=2) + "\n").encode()
    for name, raw in artifacts.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    print(json.dumps({"status": "complete", "output_directory": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
