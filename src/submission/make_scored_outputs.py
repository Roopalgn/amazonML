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
import math
import shutil
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
    name_grams: frozenset[str]
    address_grams: frozenset[str]


PAIR_FEATURE_COLUMNS = (
    "name_jaccard",
    "name_overlap",
    "name_trigram_dice",
    "address_jaccard",
    "address_overlap",
    "address_trigram_dice",
    "address_number_overlap",
    "country_match",
    "name_exact",
    "address_exact",
    "target_address_missing",
    "target_is_source2",
)


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


def character_grams(value: str, size: int = 3) -> frozenset[str]:
    compact = "".join(normalize(value).split())
    if len(compact) < size:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


def make_features(entity_id: str, name: str, address: str, country: str) -> RecordFeatures:
    return RecordFeatures(
        entity_id=entity_id,
        name=normalize(name),
        address=normalize(address),
        country=normalize(country),
        name_tokens=useful_tokens(name, drop_legal=True),
        address_tokens=useful_tokens(address),
        address_numbers=number_tokens(address),
        name_grams=character_grams(name),
        address_grams=character_grams(address),
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def dice(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return 2.0 * len(left & right) / (len(left) + len(right))


def pair_features(source1: RecordFeatures, target: RecordFeatures) -> tuple[float, ...]:
    """Return pairwise similarities and simple source metadata for a matcher."""
    name_j = jaccard(source1.name_tokens, target.name_tokens)
    name_o = overlap(source1.name_tokens, target.name_tokens)
    name_d = dice(source1.name_grams, target.name_grams)
    address_j = jaccard(source1.address_tokens, target.address_tokens)
    address_o = overlap(source1.address_tokens, target.address_tokens)
    address_d = dice(source1.address_grams, target.address_grams)
    number_o = overlap(source1.address_numbers, target.address_numbers)
    country_match = 1.0 if source1.country and source1.country == target.country else 0.0
    return (
        name_j,
        name_o,
        name_d,
        address_j,
        address_o,
        address_d,
        number_o,
        country_match,
        float(bool(source1.name and source1.name == target.name)),
        float(bool(source1.address and source1.address == target.address)),
        float(not bool(target.address)),
        float(target.entity_id.startswith("S2-")),
    )


def score_pair(
    source1: RecordFeatures,
    target: RecordFeatures,
    features: tuple[float, ...] | None = None,
) -> float:
    """Return a precision-oriented heuristic score in [0, 1]."""

    (
        name_j,
        name_o,
        name_d,
        address_j,
        address_o,
        address_d,
        number_o,
        country_match,
        name_exact,
        address_exact,
        _target_address_missing,
        _target_is_source2,
    ) = features if features is not None else pair_features(source1, target)

    score = (
        0.24 * name_j
        + 0.18 * name_o
        + 0.20 * name_d
        + 0.12 * address_j
        + 0.08 * address_o
        + 0.12 * address_d
        + 0.03 * number_o
        + 0.03 * country_match
    )

    if name_exact:
        score = max(score, 0.92 if address_j >= 0.15 or number_o > 0 else 0.82)
    if address_exact and name_o >= 0.35:
        score = max(score, 0.90)
    if source1.country and target.country and source1.country != target.country:
        score *= 0.25
    if not target.address and name_o < 0.80:
        score *= 0.70
    return max(0.0, min(1.0, score))


def model_probability(features: tuple[float, ...], model: dict[str, object]) -> float:
    weights = model["weights"]
    logit = float(model["bias"]) + sum(float(value) * float(weight) for value, weight in zip(features, weights))
    if logit >= 0.0:
        exp_value = math.exp(-min(logit, 60.0))
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(max(logit, -60.0))
    return exp_value / (1.0 + exp_value)


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


def iter_source1_features(path: Path, source1_ids: set[str] | None = None):
    """Stream Source-1 features in file order.

    Person 2's candidate generator writes one row per Source-1 record in the
    same order as the source TSV used to generate candidates. ``source1_ids``
    optionally selects a fixed validation subset while preserving source-file
    order. Keeping that contract lets the scorer avoid retaining feature
    objects for the full dataset.
    """

    remaining_ids = set(source1_ids) if source1_ids is not None else None
    for row in iter_source(path):
        if remaining_ids is not None:
            if row["entity_id"] not in remaining_ids:
                continue
            remaining_ids.remove(row["entity_id"])
        yield make_features(
            row["entity_id"],
            row["business_name"],
            row["business_address"],
            row["country"],
        )
    if remaining_ids:
        raise ValueError(f"{len(remaining_ids)} requested Source-1 IDs were absent from {path}")


def read_source1_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        ids = {line.strip() for line in handle if line.strip()}
    if not ids:
        raise ValueError(f"No Source-1 IDs found in {path}")
    return ids


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
    score_candidate_limit: int | None,
    batch_size: int,
    rebuild_target_store: bool,
    scores_output: Path | None,
    source1_ids: set[str] | None,
    pair_model: dict[str, object] | None,
) -> dict[str, int | float | str]:
    con = connect_store(target_store)
    try:
        ensure_target_store(con, source2, source3, rebuild_target_store)
        output_dir.mkdir(parents=True, exist_ok=True)
        matching_path = output_dir / "matching_results.tsv"
        matching_partial_path = output_dir / "matching_results.tsv.partial"
        candidate_path = output_dir / "candidate_pairs.tsv"
        candidate_partial_path = output_dir / "candidate_pairs.tsv.partial"
        write_limited_candidates = score_candidate_limit is not None
        if not write_limited_candidates and candidate_input.resolve() != candidate_path.resolve():
            print(f"Copying candidate file to {candidate_path}", flush=True)
            shutil.copyfile(candidate_input, candidate_path)

        score_handle = None
        score_writer = None
        if scores_output is not None:
            scores_output.parent.mkdir(parents=True, exist_ok=True)
            score_handle = scores_output.open("w", encoding="utf-8", newline="")
            score_writer = csv.writer(score_handle, delimiter="\t", lineterminator="\n")
            score_writer.writerow(["source1_entity_id", "candidate_entity_id", "score", *PAIR_FEATURE_COLUMNS])

        stats = {
            "rows": 0,
            "candidate_links": 0,
            "input_candidate_links": 0,
            "matched_links": 0,
            "rows_with_candidates": 0,
            "rows_with_matches": 0,
            "missing_source1_rows": 0,
            "missing_target_links": 0,
            "threshold": threshold,
            "score_candidate_limit": score_candidate_limit or 0,
            "candidate_input": str(candidate_input),
            "scoring_mode": "logistic_model" if pair_model is not None else "heuristic",
        }

        try:
            source1_rows = iter_source1_features(source1, source1_ids)
            candidate_rows = iter_candidate_rows(candidate_input)
            with matching_partial_path.open("w", encoding="utf-8", newline="") as match_handle:
                match_writer = csv.writer(match_handle, delimiter="\t", lineterminator="\n")
                match_writer.writerow(["source1_entity_id", "matched_entity_ids"])

                candidate_handle = None
                candidate_writer = None
                if write_limited_candidates:
                    candidate_handle = candidate_partial_path.open("w", encoding="utf-8", newline="")
                    candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n")
                    candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

                batch: list[tuple[RecordFeatures, tuple[str, ...]]] = []
                try:
                    for candidate_source1_id, candidates in candidate_rows:
                        try:
                            source = next(source1_rows)
                        except StopIteration as exc:
                            raise ValueError("candidate file has more Source-1 rows than source1 TSV") from exc
                        if source.entity_id != candidate_source1_id:
                            raise ValueError(
                                "candidate/source1 order mismatch at row "
                                f"{int(stats['rows']) + len(batch) + 2}: "
                                f"candidate={candidate_source1_id}, source1={source.entity_id}"
                            )
                        if candidate_writer is not None:
                            retained = candidates[:score_candidate_limit]
                            candidate_writer.writerow([candidate_source1_id, format_id_list(retained)])
                        batch.append((source, candidates))
                        if len(batch) >= batch_size:
                            process_batch(
                                batch,
                                con,
                                match_writer,
                                score_writer,
                                threshold,
                                max_matches,
                                score_candidate_limit,
                                pair_model,
                                stats,
                            )
                            batch.clear()
                            if int(stats["rows"]) % 10000 == 0:
                                print(f"  {stats['rows']:,} rows, {stats['matched_links']:,} matches", flush=True)
                    if batch:
                        process_batch(
                            batch,
                            con,
                            match_writer,
                            score_writer,
                            threshold,
                            max_matches,
                            score_candidate_limit,
                            pair_model,
                            stats,
                        )
                finally:
                    if candidate_handle is not None:
                        candidate_handle.close()
                try:
                    extra_source = next(source1_rows)
                except StopIteration:
                    pass
                else:
                    raise ValueError(
                        "source1 TSV has more rows than the candidate file; "
                        f"first missing candidate row is {extra_source.entity_id}"
                    )
            if write_limited_candidates:
                candidate_partial_path.replace(candidate_path)
        finally:
            if score_handle is not None:
                score_handle.close()
        matching_partial_path.replace(matching_path)
    finally:
        con.close()

    (output_dir / "person3_scored_outputs.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def process_batch(
    batch: list[tuple[RecordFeatures, tuple[str, ...]]],
    con: sqlite3.Connection,
    match_writer,
    score_writer,
    threshold: float,
    max_matches: int | None,
    score_candidate_limit: int | None,
    pair_model: dict[str, object] | None,
    stats: dict[str, int | float | str],
) -> None:
    candidate_ids = {
        candidate_id
        for _, ids in batch
        for candidate_id in (ids[:score_candidate_limit] if score_candidate_limit is not None else ids)
    }
    target_features = fetch_targets(con, candidate_ids)
    for source, candidates in batch:
        source1_id = source.entity_id

        scored: list[tuple[str, float]] = []
        scored_candidates = candidates[:score_candidate_limit] if score_candidate_limit is not None else candidates
        for candidate_id in scored_candidates:
            target = target_features.get(candidate_id)
            if target is None:
                stats["missing_target_links"] = int(stats["missing_target_links"]) + 1
                continue
            features = pair_features(source, target) if score_writer is not None or pair_model is not None else None
            score = model_probability(features, pair_model) if pair_model is not None else score_pair(source, target, features)
            scored.append((candidate_id, score))
            if score_writer is not None:
                score_writer.writerow(
                    [source1_id, candidate_id, f"{score:.6f}", *(f"{value:.6f}" for value in features)]
                )

        selected = [candidate_id for candidate_id, score in sorted(scored, key=lambda item: (-item[1], item[0])) if score >= threshold]
        if pair_model is not None:
            minimum_query_probability = pair_model.get("min_query_probability")
            if minimum_query_probability is not None and (not scored or max(score for _, score in scored) < float(minimum_query_probability)):
                selected = []
        if max_matches is not None:
            selected = selected[:max_matches]

        match_writer.writerow([source1_id, format_id_list(selected)])

        stats["rows"] = int(stats["rows"]) + 1
        stats["candidate_links"] = int(stats["candidate_links"]) + len(scored_candidates)
        stats["input_candidate_links"] = int(stats["input_candidate_links"]) + len(candidates)
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
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument(
        "--score-candidate-limit",
        type=int,
        default=None,
        help="Only score the first N candidate IDs per S1 row and write that same subset to output/candidate_pairs.tsv.",
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--rebuild-target-store", action="store_true")
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument(
        "--source1-ids",
        type=Path,
        default=None,
        help="Optional newline-delimited Source-1 ID subset, used for fixed-split validation.",
    )
    parser.add_argument("--model-json", type=Path, default=None, help="Optional trained pair model JSON.")
    args = parser.parse_args()

    pair_model = json.loads(args.model_json.read_text(encoding="utf-8")) if args.model_json else None
    if pair_model is not None and pair_model.get("feature_columns") != list(PAIR_FEATURE_COLUMNS):
        parser.error("--model-json feature columns do not match this scorer")
    threshold = args.threshold if args.threshold is not None else float(pair_model.get("threshold", 0.82) if pair_model else 0.82)
    max_matches = args.max_matches if args.max_matches is not None else (pair_model.get("max_matches") if pair_model else None)
    if not 0.0 <= threshold <= 1.0:
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
        threshold=threshold,
        max_matches=max_matches,
        score_candidate_limit=args.score_candidate_limit,
        batch_size=args.batch_size,
        rebuild_target_store=args.rebuild_target_store,
        scores_output=args.scores_output,
        source1_ids=read_source1_ids(args.source1_ids),
        pair_model=pair_model,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
