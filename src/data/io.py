"""Small, dependency-free I/O contract for the official challenge TSV files.

The challenge files are large.  These helpers deliberately stream rows instead of
loading a whole source into memory.  All source files have the same four-column
schema and ground truth has the list-based schema used by the submission format.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Iterable, Optional, Sequence, Set


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
SOURCES = ("source1", "source2", "source3")


@dataclass(frozen=True)
class DatasetPaths:
    """Paths to the seven official TSV files below a ``dataset/`` directory."""

    root: Path

    def source_path(self, split: str, source: str) -> Path:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        return self.root / split / f"{split}_{source}.tsv"

    @property
    def ground_truth_path(self) -> Path:
        return self.root / "train" / "train_ground_truth.tsv"

    def validate(self) -> None:
        required = [self.source_path(split, source) for split in ("train", "test") for source in SOURCES]
        required.append(self.ground_truth_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Official data files not found:\n  " + "\n  ".join(missing))


def find_dataset_root(explicit_root: Optional[str | Path] = None) -> DatasetPaths:
    """Locate the official ``dataset`` directory without hard-coding user paths.

    ``explicit_root`` should be the directory containing ``train/`` and ``test/``.
    In the checked-in student resource, the default is discovered relative to this
    module, which keeps commands portable after the final submission is unzipped.
    """

    if explicit_root is not None:
        candidate = Path(explicit_root).expanduser().resolve()
        paths = DatasetPaths(candidate)
        paths.validate()
        return paths

    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / "dataset" / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
        repo_root / "dataset",
    )
    for candidate in candidates:
        paths = DatasetPaths(candidate)
        if paths.source_path("train", "source1").is_file():
            paths.validate()
            return paths
    raise FileNotFoundError(
        "Could not locate the official dataset. Pass --data-root with the directory containing train/ and test/."
    )


def iter_tsv(path: str | Path, expected_columns: Sequence[str]) -> Iterator[Dict[str, str]]:
    """Yield validated TSV rows as strings, preserving empty address/list fields."""

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        fields = tuple(name.strip() for name in reader.fieldnames)
        if fields != tuple(expected_columns):
            raise ValueError(f"Unexpected schema in {path}: {fields}; expected {tuple(expected_columns)}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Malformed TSV row {line_number} in {path}")
            yield {column: (row[column] or "").strip() for column in expected_columns}


def iter_source(paths: DatasetPaths, split: str, source: str) -> Iterator[Dict[str, str]]:
    """Stream one official source file."""

    return iter_tsv(paths.source_path(split, source), SOURCE_COLUMNS)


def iter_ground_truth(paths: DatasetPaths) -> Iterator[Dict[str, str]]:
    """Stream one row per Source-1 entity and its comma-separated true IDs."""

    return iter_tsv(paths.ground_truth_path, GROUND_TRUTH_COLUMNS)


def parse_id_list(value: str) -> Set[str]:
    """Parse the challenge's comma-separated ID list, rejecting duplicate blanks."""

    if not value:
        return set()
    ids = [item.strip() for item in value.split(",")]
    if not all(ids):
        raise ValueError("ID list contains an empty element")
    if len(ids) != len(set(ids)):
        raise ValueError("ID list contains duplicate IDs")
    return set(ids)


def read_id_file(path: str | Path) -> Set[str]:
    """Read a newline-delimited Source-1 ID file, rejecting accidental duplicates."""

    with Path(path).open("r", encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return set(ids)


def write_id_file(path: str | Path, ids: Iterable[str]) -> None:
    """Write IDs in deterministic order, one per line."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{entity_id}\n" for entity_id in sorted(ids)), encoding="utf-8")
