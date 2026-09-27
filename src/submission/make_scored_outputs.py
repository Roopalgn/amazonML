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
import os
import sqlite3
import sys
import tempfile
from contextlib import ExitStack, closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKING_ROOT = REPO_ROOT / "src" / "blocking"
MATCHING_ROOT = REPO_ROOT / "src" / "matching"
for path in (BLOCKING_ROOT, MATCHING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_matcher import format_id_list, select_matches
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
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed source row in {path}")
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
        entity_id = row["entity_id"]
        if not entity_id.startswith("S1-") or entity_id != entity_id.strip() or entity_id in records:
            raise ValueError(f"Invalid or duplicate Source-1 ID: {entity_id!r}")
        records[row["entity_id"]] = make_features(
            row["entity_id"],
            row["business_name"],
            row["business_address"],
            row["country"],
        )
    if not records:
        raise ValueError("Source-1 roster is empty")
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


def require_target_ids(con: sqlite3.Connection, candidate_ids: set[str]) -> None:
    """Check unscored input IDs without loading or normalizing their text."""

    ordered = sorted(candidate_ids)
    for start in range(0, len(ordered), 800):
        chunk = ordered[start : start + 800]
        placeholders = ",".join("?" for _ in chunk)
        found = {row[0] for row in con.execute(
            f"SELECT entity_id FROM target_records WHERE entity_id IN ({placeholders})", chunk
        )}
        missing = set(chunk) - found
        if missing:
            raise ValueError(f"Unknown target IDs, e.g. {sorted(missing)[:5]}")


def iter_candidate_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError(f"Invalid candidate header in {path}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed candidate row in {path}")
            source1_id = row["source1_entity_id"]
            value = row["candidate_entity_ids"]
            ids = tuple(value.split(",")) if value else ()
            if len(ids) != len(set(ids)) or any(
                not entity_id.startswith(("S2-", "S3-")) or entity_id != entity_id.strip()
                for entity_id in ids
            ):
                raise ValueError(f"Invalid or duplicate candidate IDs for {source1_id!r}")
            yield source1_id, ids


def promote_outputs(staged: dict[Path, Path]) -> None:
    """Replace completed files, restoring prior outputs if publication raises.

    Renames are atomic per file, not across the entire group or a process crash.
    Backups stay beside their destinations so rollback uses same-volume renames.
    """

    backups: dict[Path, Path] = {}
    promoted: set[Path] = set()
    try:
        for destination, temporary in staged.items():
            if destination.exists():
                backup = temporary.with_suffix(".backup")
                destination.replace(backup)
                backups[destination] = backup
            temporary.replace(destination)
            promoted.add(destination)
    except BaseException as exc:
        failures = []
        for destination in reversed(staged):
            try:
                if destination in backups:
                    backups[destination].replace(destination)
                elif destination in promoted:
                    destination.unlink()
            except OSError:
                failures.append(str(backups.get(destination, destination)))
        if failures:
            raise RuntimeError(f"Publication rollback failed; recovery files retained: {failures}") from exc
        raise
    else:
        # Publication succeeded; a leftover backup must not turn it into failure.
        for backup in backups.values():
            with suppress(OSError):
                backup.unlink()


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
    score_candidate_limit: int | None,
    batch_size: int,
    rebuild_target_store: bool,
    scores_output: Path | None,
) -> dict[str, int | float | str]:
    select_matches((), threshold=threshold, max_matches=max_matches)
    if batch_size < 1 or (score_candidate_limit is not None and score_candidate_limit < 1):
        raise ValueError("batch_size and score_candidate_limit must be positive")
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    stats_path = output_dir / "person3_scored_outputs.stats.json"
    destinations = [matching_path, candidate_path, stats_path]
    if scores_output is not None:
        destinations.append(scores_output)
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Output paths must be distinct")
    protected = {path.resolve() for path in (source1, source2, source3, target_store)}
    if candidate_input.resolve() in protected:
        raise ValueError("Candidate input must be separate from source files and target store")
    for path in destinations:
        if path.resolve() in protected or (path.exists() and not path.is_file()):
            raise ValueError(f"Invalid output destination: {path}")
        if path.resolve() == candidate_input.resolve() and path != candidate_path:
            raise ValueError("Only candidate output may replace candidate input")

    staged: dict[Path, Path] = {}
    stats = {
        "rows": 0,
        "candidate_links": 0,
        "matched_links": 0,
        "rows_with_candidates": 0,
        "rows_with_matches": 0,
        "missing_source1_rows": 0,
        "missing_target_links": 0,
        "threshold": threshold,
        "score_candidate_limit": score_candidate_limit or 0,
        "candidate_input": str(candidate_input),
    }
    try:
        with closing(connect_store(target_store)) as con:
            ensure_target_store(con, source2, source3, rebuild_target_store)
            source1_features = load_source1_features(source1)
            with ExitStack() as stack:
                handles = {}
                for destination in destinations:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
                    staged[destination] = Path(name)
                    os.close(fd)
                    handles[destination] = stack.enter_context(staged[destination].open("w", encoding="utf-8", newline=""))
                match_writer = csv.writer(handles[matching_path], delimiter="\t", lineterminator="\n")
                candidate_writer = csv.writer(handles[candidate_path], delimiter="\t", lineterminator="\n")
                match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
                candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
                score_writer = None
                if scores_output is not None:
                    score_writer = csv.writer(handles[scores_output], delimiter="\t", lineterminator="\n")
                    score_writer.writerow(["source1_entity_id", "candidate_entity_id", "score"])
                seen: set[str] = set()
                batch: list[tuple[str, tuple[str, ...]]] = []
                rows = iter_candidate_rows(candidate_input)
                stack.callback(rows.close)
                for source1_id, candidates in rows:
                    if source1_id not in source1_features:
                        raise ValueError(f"Unknown Source-1 ID: {source1_id!r}")
                    if source1_id in seen:
                        raise ValueError(f"Duplicate Source-1 candidate row: {source1_id!r}")
                    seen.add(source1_id)
                    batch.append((source1_id, candidates))
                    if len(batch) >= batch_size:
                        process_batch(
                            batch,
                            source1_features,
                            con,
                            match_writer,
                            score_writer,
                            threshold,
                            max_matches,
                            score_candidate_limit,
                            stats,
                            candidate_writer,
                        )
                        batch.clear()
                        if int(stats["rows"]) % 10000 == 0:
                            print(f"  {stats['rows']:,} rows, {stats['matched_links']:,} matches", flush=True)
                if batch:
                    process_batch(
                        batch,
                        source1_features,
                        con,
                        match_writer,
                        score_writer,
                        threshold,
                        max_matches,
                        score_candidate_limit,
                        stats,
                        candidate_writer,
                    )
                if len(seen) != len(source1_features):
                    raise ValueError(f"Missing candidate rows for {len(source1_features) - len(seen)} Source-1 IDs")
                json.dump(stats, handles[stats_path], indent=2)
        promote_outputs(staged)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return stats


def process_batch(
    batch: list[tuple[str, tuple[str, ...]]],
    source1_features: dict[str, RecordFeatures],
    con: sqlite3.Connection,
    match_writer,
    score_writer,
    threshold: float,
    max_matches: int | None,
    score_candidate_limit: int | None,
    stats: dict[str, int | float | str],
    candidate_writer=None,
) -> None:
    # Validate even candidates beyond the scoring limit; never hide broken IDs.
    candidate_ids = {
        candidate_id
        for _, ids in batch
        for candidate_id in ids
    }
    scored_ids = {
        candidate_id
        for _, ids in batch
        for candidate_id in (ids[:score_candidate_limit] if score_candidate_limit is not None else ids)
    }
    require_target_ids(con, candidate_ids - scored_ids)
    target_features = fetch_targets(con, scored_ids)
    missing = scored_ids - target_features.keys()
    if missing:
        raise ValueError(f"Unknown target IDs, e.g. {sorted(missing)[:5]}")
    for source1_id, candidates in batch:
        source = source1_features.get(source1_id)
        if source is None:
            raise ValueError(f"Unknown Source-1 ID: {source1_id!r}")

        scored: dict[str, float] = {}
        scored_candidates = candidates[:score_candidate_limit] if score_candidate_limit is not None else candidates
        for candidate_id in scored_candidates:
            target = target_features[candidate_id]
            score = score_pair(source, target)
            scored[candidate_id] = score
            if score_writer is not None:
                score_writer.writerow([source1_id, candidate_id, repr(score)])

        selected = select_matches(scored_candidates, scored, threshold=threshold, max_matches=max_matches, policy="scored-threshold")

        match_writer.writerow([source1_id, format_id_list(selected)])
        if candidate_writer is not None:
            candidate_writer.writerow([source1_id, format_id_list(scored_candidates)])

        stats["rows"] = int(stats["rows"]) + 1
        stats["candidate_links"] = int(stats["candidate_links"]) + len(scored_candidates)
        stats["matched_links"] = int(stats["matched_links"]) + len(selected)
        stats["rows_with_candidates"] = int(stats["rows_with_candidates"]) + int(bool(scored_candidates))
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
    parser.add_argument(
        "--score-candidate-limit",
        type=int,
        default=None,
        help="Score and export only the first N candidate IDs per S1 row; input order must reflect the intended shortlist.",
    )
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
    if args.score_candidate_limit is not None and args.score_candidate_limit < 1:
        parser.error("--score-candidate-limit must be positive")

    stats = write_scored_outputs(
        candidate_input=args.candidate_input,
        source1=args.source1,
        source2=args.source2,
        source3=args.source3,
        target_store=args.target_store,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_matches=args.max_matches,
        score_candidate_limit=args.score_candidate_limit,
        batch_size=args.batch_size,
        rebuild_target_store=args.rebuild_target_store,
        scores_output=args.scores_output,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
