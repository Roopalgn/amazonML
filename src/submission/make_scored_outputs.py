"""Stream Person 2 candidates into official scored output files.

This is the production-oriented Person 3 path. It consumes the list-format
``candidate_pairs.tsv`` from Person 2, scores each candidate using only the
provided challenge fields, and writes both official files:

``output/candidate_pairs.tsv`` and ``output/matching_results.tsv``.

The script batches target lookups through a local SQLite cache so the full
Source 2/3 corpus does not need to live in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKING_ROOT = REPO_ROOT / "src" / "blocking"
MATCHING_ROOT = REPO_ROOT / "src" / "matching"
for path in (BLOCKING_ROOT, MATCHING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_matcher import format_id_list, keep_valid_match_ids, parse_id_list
from block_keys import LEGAL, normalize


DEFAULT_RESOURCE = REPO_ROOT / "dataset" / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
DEFAULT_SOURCE1 = DEFAULT_RESOURCE / "test" / "test_source1.tsv"
DEFAULT_SOURCE2 = DEFAULT_RESOURCE / "test" / "test_source2.tsv"
DEFAULT_SOURCE3 = DEFAULT_RESOURCE / "test" / "test_source3.tsv"
DEFAULT_STORE = REPO_ROOT / "data" / "features" / "person3_target_records.sqlite"
DEFAULT_OUTPUT = REPO_ROOT / "output"


@dataclass(frozen=True)
class RecordFeatures:
    entity_id: str
    name: str
    address: str
    country: str
    name_tokens: frozenset[str]
    address_tokens: frozenset[str]
    address_numbers: frozenset[str]


def source_signature(paths: Iterable[Path]) -> str:
    return json.dumps([str(path.resolve()) + ":" + str(path.stat().st_size) for path in paths])


def iter_source(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required column(s): {sorted(missing)}")
        for row in reader:
            yield row


def useful_tokens(value: str, *, drop_legal: bool = False) -> frozenset[str]:
    tokens = []
    for token in normalize(value).split():
        if len(token) <= 1:
            continue
        if drop_legal and token in LEGAL:
            continue
        tokens.append(token)
    return frozenset(tokens)


def number_tokens(value: str) -> frozenset[str]:
    return frozenset(token for token in normalize(value).split() if token.isdigit())


def make_features(entity_id: str, name: str, address: str, country: str) -> RecordFeatures:
    return RecordFeatures(
        entity_id=entity_id,
        name=normalize(name),
        address=normalize(address),
        country=normalize(country),
        name_tokens=useful_tokens(name, drop_legal=True),
        address_tokens=useful_tokens(address),
        address_numbers=number_tokens(address),
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def score_pair(source1: RecordFeatures, target: RecordFeatures) -> float:
    """Return a precision-oriented heuristic score in [0, 1]."""

    name_j = jaccard(source1.name_tokens, target.name_tokens)
    name_o = overlap(source1.name_tokens, target.name_tokens)
    address_j = jaccard(source1.address_tokens, target.address_tokens)
    address_o = overlap(source1.address_tokens, target.address_tokens)
    number_o = overlap(source1.address_numbers, target.address_numbers)
    country_match = 1.0 if source1.country and source1.country == target.country else 0.0

    score = (
        0.34 * name_j
        + 0.24 * name_o
        + 0.18 * address_j
        + 0.12 * address_o
        + 0.07 * number_o
        + 0.05 * country_match
    )

    if source1.name and source1.name == target.name:
        score = max(score, 0.92 if address_j >= 0.15 or number_o > 0 else 0.82)
    if source1.address and source1.address == target.address and name_o >= 0.35:
        score = max(score, 0.90)
    if source1.country and target.country and source1.country != target.country:
        score *= 0.25
    if not target.address and name_o < 0.80:
        score *= 0.70
    return max(0.0, min(1.0, score))


def connect_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-131072")
    return con


def build_target_store(con: sqlite3.Connection, source2: Path, source3: Path) -> None:
    con.executescript(
        """
        DROP TABLE IF EXISTS target_records;
        DROP TABLE IF EXISTS metadata;
        CREATE TABLE target_records (
            entity_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            country TEXT NOT NULL
        );
        CREATE TABLE metadata (source_signatures TEXT NOT NULL);
        """
    )
    count = 0
    for path in (source2, source3):
        print(f"Indexing target records from {path}", flush=True)
        for row in iter_source(path):
            features = make_features(
                row["entity_id"],
                row["business_name"],
                row["business_address"],
                row["country"],
            )
            con.execute(
                "INSERT INTO target_records VALUES (?, ?, ?, ?)",
                (features.entity_id, features.name, features.address, features.country),
            )
            count += 1
            if count % 100000 == 0:
                con.commit()
                print(f"  {count:,} target records", flush=True)
    con.execute("INSERT INTO metadata VALUES (?)", (source_signature((source2, source3)),))
    con.commit()
    print(f"Indexed {count:,} target records", flush=True)


def ensure_target_store(con: sqlite3.Connection, source2: Path, source3: Path, rebuild: bool) -> None:
    ready = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()
    if rebuild or not ready:
        build_target_store(con, source2, source3)
        return
    actual = con.execute("SELECT source_signatures FROM metadata").fetchone()[0]
    expected = source_signature((source2, source3))
    if actual != expected:
        raise ValueError("Target store belongs to different source files; use --rebuild-target-store")


def load_source1_features(path: Path) -> dict[str, RecordFeatures]:
    records: dict[str, RecordFeatures] = {}
    for row in iter_source(path):
        records[row["entity_id"]] = make_features(
            row["entity_id"],
            row["business_name"],
            row["business_address"],
            row["country"],
        )
    return records


def fetch_targets(con: sqlite3.Connection, candidate_ids: set[str]) -> dict[str, RecordFeatures]:
    if not candidate_ids:
        return {}
    result: dict[str, RecordFeatures] = {}
    ordered = sorted(candidate_ids)
    chunk_size = 800
    for start in range(0, len(ordered), chunk_size):
        chunk = ordered[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = con.execute(
            f"SELECT entity_id, name, address, country FROM target_records WHERE entity_id IN ({placeholders})",
            chunk,
        )
        for entity_id, name, address, country in rows:
            result[entity_id] = make_features(entity_id, name, address, country)
    return result


def iter_candidate_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required column(s): {sorted(missing)}")
        for row in reader:
            yield row["source1_entity_id"], keep_valid_match_ids(parse_id_list(row.get("candidate_entity_ids")))


def write_scored_outputs(
    *,
    candidate_input: Path,
    source1: Path,
    source2: Path,
    source3: Path,
    target_store: Path,
    output_dir: Path,
    threshold: float,
    max_matches: int | None,
    batch_size: int,
    rebuild_target_store: bool,
    scores_output: Path | None,
) -> dict[str, int | float | str]:
    con = connect_store(target_store)
    try:
        ensure_target_store(con, source2, source3, rebuild_target_store)
        source1_features = load_source1_features(source1)
        output_dir.mkdir(parents=True, exist_ok=True)
        matching_path = output_dir / "matching_results.tsv"
        candidate_path = output_dir / "candidate_pairs.tsv"

        score_handle = None
        score_writer = None
        if scores_output is not None:
            scores_output.parent.mkdir(parents=True, exist_ok=True)
            score_handle = scores_output.open("w", encoding="utf-8", newline="")
            score_writer = csv.writer(score_handle, delimiter="\t", lineterminator="\n")
            score_writer.writerow(["source1_entity_id", "candidate_entity_id", "score"])

        stats = {
            "rows": 0,
            "candidate_links": 0,
            "matched_links": 0,
            "rows_with_candidates": 0,
            "rows_with_matches": 0,
            "missing_source1_rows": 0,
            "missing_target_links": 0,
            "threshold": threshold,
            "candidate_input": str(candidate_input),
        }

        try:
            with matching_path.open("w", encoding="utf-8", newline="") as match_handle, candidate_path.open(
                "w", encoding="utf-8", newline=""
            ) as cand_handle:
                match_writer = csv.writer(match_handle, delimiter="\t", lineterminator="\n")
                cand_writer = csv.writer(cand_handle, delimiter="\t", lineterminator="\n")
                match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
                cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

                batch: list[tuple[str, tuple[str, ...]]] = []
                for item in iter_candidate_rows(candidate_input):
                    batch.append(item)
                    if len(batch) >= batch_size:
                        process_batch(batch, source1_features, con, cand_writer, match_writer, score_writer, threshold, max_matches, stats)
                        batch.clear()
                        if stats["rows"] % 100000 == 0:
                            print(f"  {stats['rows']:,} rows, {stats['matched_links']:,} matches", flush=True)
                if batch:
                    process_batch(batch, source1_features, con, cand_writer, match_writer, score_writer, threshold, max_matches, stats)
        finally:
            if score_handle is not None:
                score_handle.close()
    finally:
        con.close()

    (output_dir / "person3_scored_outputs.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def process_batch(
    batch: list[tuple[str, tuple[str, ...]]],
    source1_features: dict[str, RecordFeatures],
    con: sqlite3.Connection,
    cand_writer,
    match_writer,
    score_writer,
    threshold: float,
    max_matches: int | None,
    stats: dict[str, int | float | str],
) -> None:
    candidate_ids = {candidate_id for _, ids in batch for candidate_id in ids}
    target_features = fetch_targets(con, candidate_ids)
    for source1_id, candidates in batch:
        source = source1_features.get(source1_id)
        if source is None:
            stats["missing_source1_rows"] = int(stats["missing_source1_rows"]) + 1
            cand_writer.writerow([source1_id, format_id_list(candidates)])
            match_writer.writerow([source1_id, ""])
            continue

        scored: list[tuple[str, float]] = []
        for candidate_id in candidates:
            target = target_features.get(candidate_id)
            if target is None:
                stats["missing_target_links"] = int(stats["missing_target_links"]) + 1
                continue
            score = score_pair(source, target)
            scored.append((candidate_id, score))
            if score_writer is not None:
                score_writer.writerow([source1_id, candidate_id, f"{score:.6f}"])

        selected = [candidate_id for candidate_id, score in sorted(scored, key=lambda item: (-item[1], item[0])) if score >= threshold]
        if max_matches is not None:
            selected = selected[:max_matches]

        cand_writer.writerow([source1_id, format_id_list(candidates)])
        match_writer.writerow([source1_id, format_id_list(selected)])

        stats["rows"] = int(stats["rows"]) + 1
        stats["candidate_links"] = int(stats["candidate_links"]) + len(candidates)
        stats["matched_links"] = int(stats["matched_links"]) + len(selected)
        stats["rows_with_candidates"] = int(stats["rows_with_candidates"]) + int(bool(candidates))
        stats["rows_with_matches"] = int(stats["rows_with_matches"]) + int(bool(selected))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score candidate_pairs.tsv and write official output files.")
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--source1", type=Path, default=DEFAULT_SOURCE1)
    parser.add_argument("--source2", type=Path, default=DEFAULT_SOURCE2)
    parser.add_argument("--source3", type=Path, default=DEFAULT_SOURCE3)
    parser.add_argument("--target-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--rebuild-target-store", action="store_true")
    parser.add_argument("--scores-output", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_matches is not None and args.max_matches < 1:
        parser.error("--max-matches must be positive")

    stats = write_scored_outputs(
        candidate_input=args.candidate_input,
        source1=args.source1,
        source2=args.source2,
        source3=args.source3,
        target_store=args.target_store,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_matches=args.max_matches,
        batch_size=args.batch_size,
        rebuild_target_store=args.rebuild_target_store,
        scores_output=args.scores_output,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
