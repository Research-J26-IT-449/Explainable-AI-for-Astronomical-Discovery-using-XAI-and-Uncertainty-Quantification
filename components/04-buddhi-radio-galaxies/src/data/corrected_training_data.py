"""Bounded corrected-cohort access; archived label VALUES are never decoded.

Reuse the pinned reader's inert cursor, archive checks and image hashing, but
not its all-label scanner. Labels/filenames come from approved manifest rows.
No pickle execution, raw image cache, dataset-wide preload or model access.
"""

from dataclasses import dataclass
import struct
import tarfile

from PIL import Image
from torch.utils.data import Dataset

from src.data import mirabest_pickle_stream_reader as legacy
from src.data.corrected_protocol import QUARANTINE


@dataclass(frozen=True)
class TrainingRow:
    dataset_index: int
    split: str
    label: int
    member: str
    member_row: int
    content_sha256: str

    @classmethod
    def from_manifest(cls, row):
        return cls(row["dataset_index"], row["corrected_split"], row["ground_truth_label"],
                   row["archive_member"], row["member_row"], row["canonical_sha256"])


def selected_offsets(handle, member, selected):
    """Scan the observed protocol-3 array grammar, skipping ALL label operands.

    Stop after the 88th image payload, before filename metadata. Whole-archive
    hashes bind the suffix; full unpickling or suffix interpretation is unneeded.
    Opaque gzip/checksum reads are not decoded label/image materialization.
    """
    cursor = legacy._Cursor(handle)
    if cursor.read(2) != b"\x80\x03":
        raise ValueError("Unsupported archive pickle protocol.")
    labels = False
    data = False
    label_count = image_count = 0
    shape = []
    dtype = False
    offsets = {}
    start = legacy._member_start(member)
    allowed_globals = {
        b"numpy.core.multiarray _reconstruct", b"numpy._core.multiarray _reconstruct",
        b"numpy ndarray", b"numpy dtype", b"numpy.core.multiarray scalar",
    }
    while image_count < 88:
        op = cursor.byte()
        if op in (0x71, 0x68):
            cursor.skip(1)
        elif op in (0x72, 0x6A):
            cursor.skip(4)
        elif op == 0x70:
            cursor.line()
        elif op == 0x4B:
            if labels:
                cursor.skip(1)  # Never read/classify a label, even for selected rows.
                label_count += 1
                if label_count > 88:
                    raise ValueError("Unexpected archived label count.")
            else:
                value = cursor.byte()
                if data and image_count == 0 and value == 150:
                    shape.append(value)
        elif op == 0x4A:
            if labels:
                raise ValueError("Unsupported archived label encoding.")
            cursor.skip(4)
        elif op in (0x58, 0x55):
            length = struct.unpack("<I", cursor.read(4))[0] if op == 0x58 else cursor.byte()
            if length > 1024:
                raise ValueError("Oversized archive grammar metadata.")
            value = cursor.read(length).decode("utf-8")
            if value == "labels":
                if labels or data or label_count:
                    raise ValueError("Unexpected label section order.")
                labels = True
            elif value == "data":
                if not labels or label_count != 88:
                    raise ValueError("Unexpected data section order.")
                labels = False
                data = True
            elif value == "u1":
                dtype = True
            elif value == "filenames":
                raise ValueError("Premature filename section.")
        elif op == 0x63:
            if cursor.line() + b" " + cursor.line() not in allowed_globals:
                raise ValueError("Unsupported archive GLOBAL reference.")
        elif op in (0x42, 0x43):
            length = legacy._read_pickle_bytes_header(cursor, op)
            if op == 0x42:
                if not data or length != legacy.IMAGE_BYTES:
                    raise ValueError("Unexpected archived pixel payload.")
                index = start + image_count
                if index in selected:
                    offsets[index] = cursor.position
                cursor.skip(length)  # Includes locked, quarantined and unselected rows.
                image_count += 1
            else:
                cursor.skip(length)
        elif op not in {0x28, 0x5D, 0x7D, 0x4E, 0x88, 0x89, 0x94,
                        0x74, 0x85, 0x86, 0x87, 0x81, 0x82, 0x52, 0x62,
                        0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                        0x38, 0x39, 0x61, 0x65, 0x67, 0x6B, 0x6C, 0x6D,
                        0x6F, 0x75, 0x76, 0x90, 0x91, 0x92, 0x93, 0x95}:
            raise ValueError("Unsupported archive grammar opcode.")
    if label_count != 88 or shape != [150, 150] or not dtype:
        raise ValueError("Archived array structure mismatch.")
    return offsets


class CorrectedArchiveReader:
    """Explicit selected-row access only; at most eight raw payloads per call."""

    def __init__(self, path, rows, *, archive_sha256, archive_md5):
        self.path = path
        self.rows = {r.dataset_index: r for r in rows}
        if len(self.rows) != len(rows) or set(self.rows) & set(QUARANTINE):
            raise ValueError("Reader scope repeats or includes quarantined rows.")
        self.sha256 = archive_sha256
        self.md5 = archive_md5

    def read_selected(self, indices):
        indices = tuple(indices)
        if (not 1 <= len(indices) <= 8 or len(set(indices)) != len(indices)
                or any(type(i) is not int or i not in self.rows for i in indices)):
            raise ValueError("Unauthorized or unbounded corrected row request.")
        # Reject indices before even opening/hashing the archive.
        if (legacy._file_digest(self.path, "sha256") != self.sha256
                or legacy._file_digest(self.path, "md5") != self.md5):
            raise ValueError("Frozen archive provenance mismatch.")
        selected = set(indices)
        pixels = {}
        with tarfile.open(self.path, "r:gz") as archive:
            files = legacy._validate_archive_members(archive)
            legacy._validate_archive_manifest(archive)  # Opaque checksums, no records/labels decoded.
            members = sorted({self.rows[i].member for i in indices})
            for member in members:
                wanted = {i for i in selected if self.rows[i].member == member}
                with archive.extractfile(files[member]) as handle:
                    offsets = selected_offsets(handle, member, wanted)
                    if set(offsets) != wanted:
                        raise ValueError("Selected archive rows are missing.")
                    for index in sorted(wanted):
                        handle.seek(offsets[index])
                        raw = handle.read(legacy.IMAGE_BYTES)
                        if legacy.canonical_sha256(raw) != self.rows[index].content_sha256:
                            raise ValueError("Selected pixel content differs from saved audit evidence.")
                        pixels[index] = raw
        return pixels


class CorrectedTrainingDataset(Dataset):
    """Metadata view; batched PyTorch fetch splits raw reads into <=8 rows.

    A batch of 32 normalized tensors is expected (2.88 MB), but raw extraction
    holds at most eight payloads (180 KB). No raw cache persists between calls.
    Only num_workers=0 is supported by the prepared workflow.
    """

    def __init__(self, rows, split, transform, reader):
        if split not in {"train", "validation"}:
            raise ValueError("Only corrected TRAIN/VALIDATION views are supported.")
        self.rows = tuple(row for row in rows if row.split == split)
        if (len({r.dataset_index for r in self.rows}) != len(self.rows)
                or any(r.dataset_index in QUARANTINE or r.label not in (0, 1) for r in self.rows)):
            raise ValueError("Invalid corrected dataset view.")
        self.indices = tuple(row.dataset_index for row in self.rows)
        self.transform, self.reader = transform, reader

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        return self.__getitems__([position])[0]

    def __getitems__(self, positions):
        positions = tuple(positions)
        if any(type(p) is not int or not 0 <= p < len(self.rows) for p in positions):
            raise ValueError("Invalid corrected dataset position.")
        result = []
        for start in range(0, len(positions), 8):
            rows = [self.rows[p] for p in positions[start:start + 8]]
            raw = self.reader.read_selected([r.dataset_index for r in rows])
            try:
                if set(raw) != {r.dataset_index for r in rows}:
                    raise ValueError("Incorrect bounded-reader result scope.")
                for row in rows:  # Preserve sampler order and augmentation RNG order.
                    with Image.frombytes("L", (150, 150), raw[row.dataset_index]) as image:
                        result.append((self.transform(image), row.label))
            finally:
                raw.clear()  # Release previous raw batch before extracting another.
        return result
