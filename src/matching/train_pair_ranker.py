"""Train and compare a supervised candidate-pair ranker on the fixed split.

The input candidate file must be generated from train_source2/train_source3.
Rows whose Source 1 IDs occur in the fixed validation-ID file are held out;
all other candidate rows are used for training.  The script uses only the
challenge TSVs and labels.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKING_ROOT = REPO_ROOT / "src" / "blocking"
import sys

if str(BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCKING_ROOT))

from block_keys import LEGAL, normalize
from baseline_matcher import parse_id_list


DEFAULT_RESOURCE = REPO_ROOT / "dataset" / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
FEATURE_NAMES = (
    "legacy_score", "name_exact", "address_exact", "country_match",
    "name_jaccard", "name_overlap", "name_sequence", "name_char_jaccard",
    "name_first_token", "name_last_token", "address_jaccard", "address_overlap",
    "address_sequence", "address_char_jaccard", "number_overlap",
    "same_first_number", "name_length_ratio", "address_length_ratio",
    "candidate_rank", "candidate_count", "shared_name_tokens", "shared_address_tokens",
)


@dataclass(frozen=True)
class Record:
    entity_id: str
    name: str
    address: str
    country: str
    name_order: tuple[str, ...]
    name_tokens: frozenset[str]
    address_tokens: frozenset[str]
    numbers: frozenset[str]


def make_record(entity_id: str, name: str, address: str, country: str) -> Record:
    n = normalize(name)
    a = normalize(address)
    ordered_name_tokens = tuple(dict.fromkeys(t for t in n.split() if len(t) > 1 and t not in LEGAL))
    return Record(
        entity_id=entity_id,
        name=n,
        address=a,
        country=normalize(country),
        name_order=ordered_name_tokens,
        name_tokens=frozenset(ordered_name_tokens),
        address_tokens=frozenset(t for t in a.split() if len(t) > 1),
        numbers=frozenset(t for t in a.split() if t.isdigit()),
    )


def ratio(left: frozenset[str], right: frozenset[str], *, overlap_ratio: bool = False) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / (min(len(left), len(right)) if overlap_ratio else len(left | right))


def char_ngrams(value: str, n: int = 3) -> frozenset[str]:
    padded = f" {value} "
    return frozenset(padded[i : i + n] for i in range(max(0, len(padded) - n + 1)))


def safe_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def pair_features(source: Record, target: Record, rank: int, candidate_count: int) -> list[float]:
    name_j = ratio(source.name_tokens, target.name_tokens)
    name_o = ratio(source.name_tokens, target.name_tokens, overlap_ratio=True)
    address_j = ratio(source.address_tokens, target.address_tokens)
    address_o = ratio(source.address_tokens, target.address_tokens, overlap_ratio=True)
    number_o = ratio(source.numbers, target.numbers, overlap_ratio=True)
    country_match = float(bool(source.country) and source.country == target.country)
    name_char = ratio(char_ngrams(source.name), char_ngrams(target.name))
    address_char = ratio(char_ngrams(source.address), char_ngrams(target.address))
    name_sequence = difflib.SequenceMatcher(None, source.name, target.name, autojunk=False).ratio()
    address_sequence = difflib.SequenceMatcher(None, source.address, target.address, autojunk=False).ratio()
    legacy = 0.34 * name_j + 0.24 * name_o + 0.18 * address_j + 0.12 * address_o + 0.07 * number_o + 0.05 * country_match
    if source.name and source.name == target.name:
        legacy = max(legacy, 0.92 if address_j >= 0.15 or number_o > 0 else 0.82)
    if source.address and source.address == target.address and name_o >= 0.35:
        legacy = max(legacy, 0.90)
    if source.country and target.country and source.country != target.country:
        legacy *= 0.25
    if not target.address and name_o < 0.80:
        legacy *= 0.70
    shared_names = source.name_tokens & target.name_tokens
    shared_addresses = source.address_tokens & target.address_tokens
    return [
        legacy,
        float(bool(source.name) and source.name == target.name),
        float(bool(source.address) and source.address == target.address),
        country_match,
        name_j,
        name_o,
        name_sequence,
        name_char,
        float(bool(source.name_order and target.name_order) and source.name_order[0] == target.name_order[0]),
        float(bool(source.name_order and target.name_order) and source.name_order[-1] == target.name_order[-1]),
        address_j,
        address_o,
        address_sequence,
        address_char,
        number_o,
        float(bool(source.numbers & target.numbers)),
        safe_ratio(source.name, target.name),
        safe_ratio(source.address, target.address),
        math.log1p(rank),
        math.log1p(candidate_count),
        float(len(shared_names)),
        float(len(shared_addresses)),
    ]


def predict_pair_scores(model_bundle: dict, pairs: list[tuple[Record, Record, int, int]]) -> np.ndarray:
    """Score (source, target, rank, candidate_count) pairs with a saved bundle."""
    expected = tuple(model_bundle.get("feature_names", ()))
    if expected != FEATURE_NAMES:
        raise ValueError("Pair model feature schema does not match this code version")
    matrix = np.asarray(
        [pair_features(source, target, rank, count) for source, target, rank, count in pairs],
        dtype=np.float32,
    )
    if not pairs:
        return np.asarray([], dtype=np.float32)
    return model_bundle["model"].predict_proba(matrix)[:, 1]


def read_candidate_rows(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            rows.append((row["source1_entity_id"], parse_id_list(row.get("candidate_entity_ids"))))
    return rows


def read_id_file(path: Path) -> set[str]:
    return {line.strip() for line in path.open(encoding="utf-8") if line.strip()}


def read_source_rows(path: Path, wanted: set[str]) -> dict[str, Record]:
    records: dict[str, Record] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entity_id = row.get("entity_id", "")
            if entity_id in wanted:
                records[entity_id] = make_record(entity_id, row["business_name"], row["business_address"], row["country"])
                if len(records) == len(wanted):
                    break
    missing = wanted - records.keys()
    if missing:
        raise ValueError(f"Could not find {len(missing)} requested IDs in {path}; first: {next(iter(missing))}")
    return records


def read_truth(path: Path, wanted: set[str]) -> dict[str, frozenset[str]]:
    truth: dict[str, frozenset[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_id = row["source1_entity_id"]
            if source_id in wanted:
                truth[source_id] = frozenset(parse_id_list(row.get("matched_entity_ids")))
                if len(truth) == len(wanted):
                    break
    missing = wanted - truth.keys()
    if missing:
        raise ValueError(f"Could not find ground truth for {len(missing)} candidate rows")
    return truth


def read_target_rows(paths: Iterable[Path], wanted: set[str]) -> dict[str, Record]:
    records: dict[str, Record] = {}
    remaining = set(wanted)
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                entity_id = row.get("entity_id", "")
                if entity_id in remaining:
                    records[entity_id] = make_record(entity_id, row["business_name"], row["business_address"], row["country"])
                    remaining.remove(entity_id)
            print(f"Loaded {len(records):,} candidate targets after {path.name}", flush=True)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Could not find {len(remaining)} candidate IDs in target files; first: {next(iter(remaining))}")
    return records


def build_matrix(
    rows: list[tuple[str, tuple[str, ...]]],
    source_records: dict[str, Record],
    target_records: dict[str, Record],
    truth: dict[str, frozenset[str]],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[int]]]:
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    source_ids: list[str] = []
    ranges: dict[str, list[int]] = {}
    for source_id, candidates in rows:
        source = source_records[source_id]
        start = len(feature_rows)
        true_ids = truth.get(source_id, frozenset())
        for rank, target_id in enumerate(candidates):
            target = target_records.get(target_id)
            if target is None:
                continue
            feature_rows.append(pair_features(source, target, rank, len(candidates)))
            labels.append(int(target_id in true_ids))
            source_ids.append(source_id)
        ranges[source_id] = [start, len(feature_rows)]
        if len(ranges) % 1000 == 0:
            print(f"  built pair features for {len(ranges):,} queries", flush=True)
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.uint8), source_ids, ranges


def f05(truth: set[str] | frozenset[str], predicted: set[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(truth & predicted)
    if not tp:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def evaluate(
    row_ids: list[str],
    ranges: dict[str, list[int]],
    truth: dict[str, frozenset[str]],
    candidate_ids: list[tuple[str, ...]],
    scores: np.ndarray,
    threshold: float,
    candidate_limit: int | None = None,
) -> dict[str, float | int]:
    total = singleton_total = nonsingleton_total = 0.0
    singleton_count = nonsingleton_count = 0
    true_pairs = predicted_pairs = 0
    for row_number, source_id in enumerate(row_ids):
        start, stop = ranges[source_id]
        ids = candidate_ids[row_number]
        row_scores = scores[start:stop]
        if candidate_limit is not None:
            ids = ids[:candidate_limit]
            row_scores = row_scores[:candidate_limit]
        predicted = {target_id for target_id, score in zip(ids, row_scores) if score >= threshold}
        score = f05(truth[source_id], predicted)
        total += score
        true_pairs += len(truth[source_id])
        predicted_pairs += len(predicted)
        if truth[source_id]:
            nonsingleton_total += score
            nonsingleton_count += 1
        else:
            singleton_total += score
            singleton_count += 1
    count = len(row_ids)
    return {
        "macro_f0_5": total / count if count else 0.0,
        "singleton_macro_f0_5": singleton_total / singleton_count if singleton_count else 0.0,
        "non_singleton_macro_f0_5": nonsingleton_total / nonsingleton_count if nonsingleton_count else 0.0,
        "rows": count,
        "true_links": true_pairs,
        "predicted_links": predicted_pairs,
    }


def search_thresholds(row_ids, ranges, truth, candidate_ids, scores, candidate_limit=None) -> tuple[float, dict]:
    best_threshold = 0.0
    best_summary = None
    for threshold in np.arange(0.05, 1.001, 0.01):
        summary = evaluate(row_ids, ranges, truth, candidate_ids, scores, float(threshold), candidate_limit)
        if best_summary is None or summary["macro_f0_5"] > best_summary["macro_f0_5"]:
            best_threshold, best_summary = float(threshold), summary
    return best_threshold, best_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and validation-evaluate a supervised pair ranker.")
    parser.add_argument("--candidates", type=Path, default=REPO_ROOT / "data/candidates/person2/candidate_train_5k.tsv")
    parser.add_argument("--validation-ids", type=Path, default=REPO_ROOT / "data/processed/validation_source1_ids.txt")
    parser.add_argument("--train-source1", type=Path, default=DEFAULT_RESOURCE / "train/train_source1.tsv")
    parser.add_argument("--train-source2", type=Path, default=DEFAULT_RESOURCE / "train/train_source2.tsv")
    parser.add_argument("--train-source3", type=Path, default=DEFAULT_RESOURCE / "train/train_source3.tsv")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_RESOURCE / "train/train_ground_truth.tsv")
    parser.add_argument("--model-output", type=Path, default=REPO_ROOT / "data/features/person2_pair_ranker.joblib")
    parser.add_argument("--report-output", type=Path, default=REPO_ROOT / "data/features/person2_pair_ranker_validation.json")
    parser.add_argument("--matrix-cache", type=Path, default=REPO_ROOT / "data/features/person2_pair_ranker_sample.joblib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_cache.parent.mkdir(parents=True, exist_ok=True)
    all_rows = read_candidate_rows(args.candidates)
    validation_ids = read_id_file(args.validation_ids)
    train_rows = [(source_id, candidates) for source_id, candidates in all_rows if source_id not in validation_ids]
    validation_rows = [(source_id, candidates) for source_id, candidates in all_rows if source_id in validation_ids]
    if not train_rows or not validation_rows:
        raise ValueError("Candidate file must contain both training rows and rows from the fixed validation split")
    print(f"Candidate rows: {len(train_rows):,} train, {len(validation_rows):,} validation", flush=True)

    cache_key = hashlib.sha256()
    for path in (args.candidates, args.validation_ids, args.train_source1, args.train_source2, args.train_source3, args.ground_truth):
        cache_key.update(str(path.resolve()).encode("utf-8"))
        cache_key.update(str(path.stat().st_size).encode("ascii"))
    fingerprint = cache_key.hexdigest()
    if args.matrix_cache.exists():
        cached = joblib.load(args.matrix_cache)
    else:
        cached = {}
    if cached.get("fingerprint") == fingerprint:
        x_train, y_train = cached["x_train"], cached["y_train"]
        x_valid, y_valid = cached["x_valid"], cached["y_valid"]
        valid_ranges = cached["valid_ranges"]
        valid_truth = cached["valid_truth"]
        valid_candidates = cached["valid_candidates"]
        validation_row_ids = cached["validation_row_ids"]
        print("Reusing cached fixed-split pair features", flush=True)
    else:
        train_ids = {source_id for source_id, _ in train_rows}
        validation_source_ids = {source_id for source_id, _ in validation_rows}
        source_records = read_source_rows(args.train_source1, train_ids | validation_source_ids)
        truth = read_truth(args.ground_truth, train_ids | validation_source_ids)
        candidate_target_ids = {target_id for _, ids in all_rows for target_id in ids}
        target_records = read_target_rows((args.train_source2, args.train_source3), candidate_target_ids)

        x_train, y_train, _, _ = build_matrix(train_rows, source_records, target_records, truth)
        x_valid, y_valid, _, valid_ranges = build_matrix(validation_rows, source_records, target_records, truth)
        valid_truth = {source_id: truth[source_id] for source_id, _ in validation_rows}
        valid_candidates = [ids for _, ids in validation_rows]
        validation_row_ids = [source_id for source_id, _ in validation_rows]
        joblib.dump({
            "fingerprint": fingerprint,
            "x_train": x_train,
            "y_train": y_train,
            "x_valid": x_valid,
            "y_valid": y_valid,
            "valid_ranges": valid_ranges,
            "valid_truth": valid_truth,
            "valid_candidates": valid_candidates,
            "validation_row_ids": validation_row_ids,
        }, args.matrix_cache, compress=3)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training candidates must include positive and negative pairs")
    print(f"Pair rows: {len(y_train):,} train ({int(y_train.sum()):,} positive), {len(y_valid):,} validation ({int(y_valid.sum()):,} positive)", flush=True)

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        min_samples_leaf=80,
        subsample=0.8,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    best_threshold, model_summary = search_thresholds(
        validation_row_ids, valid_ranges,
        valid_truth, valid_candidates, probabilities,
    )

    legacy_scores = x_valid[:, 0]
    legacy_threshold, legacy_summary = search_thresholds(
        validation_row_ids, valid_ranges,
        valid_truth, valid_candidates, legacy_scores,
    )
    model_top50_threshold, model_top50_summary = search_thresholds(
        validation_row_ids, valid_ranges, valid_truth, valid_candidates, probabilities, candidate_limit=50,
    )
    legacy_top50_threshold, legacy_top50_summary = search_thresholds(
        validation_row_ids, valid_ranges, valid_truth, valid_candidates, legacy_scores, candidate_limit=50,
    )

    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "threshold": best_threshold}, args.model_output)
    report = {
        "candidate_file": str(args.candidates),
        "validation_split": "person1-v1",
        "candidate_rows_train": len(train_rows),
        "candidate_rows_validation": len(validation_rows),
        "train_pair_rows": len(y_train),
        "train_positive_pairs": int(y_train.sum()),
        "validation_pair_rows": len(y_valid),
        "validation_positive_pairs": int(y_valid.sum()),
        "legacy_heuristic": {"best_threshold": legacy_threshold, **legacy_summary},
        "supervised_ranker": {"best_threshold": best_threshold, **model_summary},
        "legacy_heuristic_first_50_candidates": {"best_threshold": legacy_top50_threshold, **legacy_top50_summary},
        "supervised_ranker_first_50_candidates": {"best_threshold": model_top50_threshold, **model_top50_summary},
        "feature_names": FEATURE_NAMES,
        "model_output": str(args.model_output),
        "candidate_sha256": hashlib.file_digest(args.candidates.open("rb"), "sha256").hexdigest(),
    }
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
