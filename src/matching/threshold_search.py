"""Threshold search for scored candidate files.

Expected scored candidate columns:

``source1_entity_id``, ``candidate_entity_id``, and ``score``.

This script is designed for Person 3's lane. It does not create candidates; it
only evaluates scores produced by the matching stage against the fixed validation
split owned by Person 1.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from baseline_matcher import parse_id_list, select_matches


def load_validation_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return {line.strip() for line in handle if line.strip()}


def load_truth(path: Path, validation_ids: set[str] | None) -> dict[str, tuple[str, ...]]:
    truth: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source1_id = row["source1_entity_id"]
            if validation_ids is not None and source1_id not in validation_ids:
                continue
            truth[source1_id] = parse_id_list(row.get("matched_entity_ids"))
    return truth


def load_scores(path: Path, allowed_source1_ids: set[str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_id", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required column(s): {sorted(missing)}")
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id not in allowed_source1_ids:
                continue
            candidate_id = row["candidate_entity_id"]
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            scores[source1_id][candidate_id] = score
    return scores


def f05_score(truth_ids: Iterable[str], predicted_ids: Iterable[str]) -> float:
    truth = set(truth_ids)
    predicted = set(predicted_ids)
    if not truth and not predicted:
        return 1.0
    if not truth or not predicted:
        return 0.0
    tp = len(truth & predicted)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def evaluate_threshold(
    truth: dict[str, tuple[str, ...]],
    scores_by_source1: dict[str, dict[str, float]],
    threshold: float,
    max_matches: int | None,
) -> tuple[float, float, float]:
    total = 0.0
    singleton_total = 0.0
    singleton_count = 0
    non_singleton_total = 0.0
    non_singleton_count = 0

    for source1_id, truth_ids in truth.items():
        scores = scores_by_source1.get(source1_id, {})
        predicted = select_matches(
            tuple(scores.keys()),
            scores,
            threshold=threshold,
            max_matches=max_matches,
            # The production scorer ranks by score before applying max_matches.
            policy="top-k",
        )
        score = f05_score(truth_ids, predicted)
        total += score
        if truth_ids:
            non_singleton_total += score
            non_singleton_count += 1
        else:
            singleton_total += score
            singleton_count += 1

    macro = total / len(truth) if truth else 0.0
    singleton_macro = singleton_total / singleton_count if singleton_count else 0.0
    non_singleton_macro = non_singleton_total / non_singleton_count if non_singleton_count else 0.0
    return macro, singleton_macro, non_singleton_macro


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Search a score threshold using macro F0.5.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--validation-ids", type=Path, default=None)
    parser.add_argument("--start", type=float, default=0.50)
    parser.add_argument("--stop", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--max-matches", type=int, default=None)
    args = parser.parse_args()

    validation_ids = load_validation_ids(args.validation_ids)
    truth = load_truth(args.ground_truth, validation_ids)
    scores = load_scores(args.scores, set(truth))

    best: tuple[float, float, float, float] | None = None
    print("threshold\tmacro_f0_5\tsingleton_macro\tnon_singleton_macro")
    for threshold in threshold_grid(args.start, args.stop, args.step):
        macro, singleton_macro, non_singleton_macro = evaluate_threshold(
            truth,
            scores,
            threshold,
            args.max_matches,
        )
        print(f"{threshold:.6f}\t{macro:.6f}\t{singleton_macro:.6f}\t{non_singleton_macro:.6f}")
        candidate = (macro, singleton_macro, non_singleton_macro, threshold)
        if best is None or candidate > best:
            best = candidate

    if best is not None:
        macro, singleton_macro, non_singleton_macro, threshold = best
        print()
        print(f"best_threshold={threshold:.6f}")
        print(f"best_macro_f0_5={macro:.6f}")
        print(f"best_singleton_macro={singleton_macro:.6f}")
        print(f"best_non_singleton_macro={non_singleton_macro:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
