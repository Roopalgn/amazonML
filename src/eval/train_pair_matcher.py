"""Fit and validate a lightweight logistic model on labeled candidate pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.io import parse_id_list, read_id_file
from src.submission.make_scored_outputs import BASE_PAIR_FEATURE_COLUMNS, PAIR_FEATURE_COLUMNS


def load_truth(path: Path, allowed_ids: set[str]) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id in allowed_ids:
                truth[source1_id] = parse_id_list(row.get("matched_entity_ids") or "")
    missing = allowed_ids - truth.keys()
    if missing:
        raise ValueError(f"Ground truth is missing {len(missing)} validation IDs")
    return truth


def feature_row(row: dict[str, str], feature_columns: tuple[str, ...]) -> np.ndarray:
    return np.fromiter((float(row[name]) for name in feature_columns), dtype=np.float32)


def is_holdout(source1_id: str) -> bool:
    return int(source1_id.rsplit("-", 1)[1]) % 5 == 0


def collect_reservoir(
    scores_path: Path,
    truth: dict[str, set[str]],
    *,
    heldout: bool,
    cap_per_class: int,
    seed: int,
    feature_columns: tuple[str, ...],
    hard_negative_rank: int,
    holdout_ids: set[str] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if hard_negative_rank < 0:
        raise ValueError("hard_negative_rank must be non-negative")
    rng = random.Random(seed)
    hard_capacity = min(cap_per_class // 2, cap_per_class) if hard_negative_rank else 0
    random_capacity = cap_per_class - hard_capacity
    arrays = {
        0: np.empty((random_capacity, len(feature_columns)), dtype=np.float32),
        1: np.empty((cap_per_class, len(feature_columns)), dtype=np.float32),
    }
    hard_negatives = np.empty((hard_capacity, len(feature_columns)), dtype=np.float32)
    seen = {0: 0, 1: 0}
    random_seen = {0: 0, 1: 0}
    used = {0: 0, 1: 0}
    hard_seen = 0
    hard_used = 0
    active_source1_id: str | None = None
    candidate_rank = 0
    with scores_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"source1_entity_id", "candidate_entity_id", *feature_columns}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{scores_path} missing feature column(s): {sorted(missing)}")
        for row in reader:
            source1_id = row["source1_entity_id"]
            in_holdout = source1_id in holdout_ids if holdout_ids is not None else is_holdout(source1_id)
            if in_holdout != heldout:
                continue
            if source1_id != active_source1_id:
                active_source1_id = source1_id
                candidate_rank = 0
            candidate_rank += 1
            label = int(row["candidate_entity_id"] in truth[source1_id])
            seen[label] += 1
            values = feature_row(row, feature_columns)
            if label == 0 and hard_capacity and candidate_rank <= hard_negative_rank:
                hard_seen += 1
                if hard_used < hard_capacity:
                    hard_negatives[hard_used] = values
                    hard_used += 1
                else:
                    replace_at = rng.randrange(hard_seen)
                    if replace_at < hard_capacity:
                        hard_negatives[replace_at] = values
            else:
                random_seen[label] += 1
                if used[label] < arrays[label].shape[0]:
                    arrays[label][used[label]] = values
                    used[label] += 1
                    continue
                replace_at = rng.randrange(random_seen[label])
                if replace_at < arrays[label].shape[0]:
                    arrays[label][replace_at] = values
    if not used[0] or not used[1]:
        raise ValueError(f"Need positive and negative examples; observed {seen}")
    negatives = np.concatenate((hard_negatives[:hard_used], arrays[0][: used[0]]), axis=0)
    positives = arrays[1][: used[1]]
    x = np.concatenate((negatives, positives), axis=0)
    y = np.concatenate((np.zeros(len(negatives), dtype=np.float32), np.ones(len(positives), dtype=np.float32)))
    return x, y, {
        "negative_pairs_seen": seen[0],
        "positive_pairs_seen": seen[1],
        "negative_pairs_sampled": len(negatives),
        "positive_pairs_sampled": len(positives),
        "hard_negative_pairs_seen": hard_seen,
        "hard_negative_pairs_sampled": hard_used,
    }


def available_feature_columns(scores_path: Path) -> tuple[str, ...]:
    with scores_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or ())
    if set(PAIR_FEATURE_COLUMNS).issubset(columns):
        return tuple(PAIR_FEATURE_COLUMNS)
    if set(BASE_PAIR_FEATURE_COLUMNS).issubset(columns):
        return tuple(BASE_PAIR_FEATURE_COLUMNS)
    missing = sorted(set(BASE_PAIR_FEATURE_COLUMNS) - columns)
    raise ValueError(f"{scores_path} is missing required feature columns: {missing}")


def fit_logistic(x: np.ndarray, y: np.ndarray, *, seed: int, epochs: int = 20) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    x64 = x.astype(np.float64, copy=False)
    batch_size = 16384
    learning_rate = 0.20
    l2 = 0.01
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            indices = order[start : start + batch_size]
            xb = x64[indices]
            yb = y[indices]
            logits = np.clip(xb @ weights + bias, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            error = probabilities - yb
            weights -= learning_rate * ((xb.T @ error) / len(indices) + l2 * weights)
            bias -= learning_rate * float(error.mean())
    return bias, weights.tolist()


def f05(truth: set[str], predicted: list[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = sum(entity_id in truth for entity_id in predicted)
    if not tp:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def sigmoid(logit: float) -> float:
    if logit >= 0:
        exp_value = np.exp(-min(logit, 60.0))
        return float(1.0 / (1.0 + exp_value))
    exp_value = np.exp(max(logit, -60.0))
    return float(exp_value / (1.0 + exp_value))


def tune_holdout(
    scores_path: Path,
    truth: dict[str, set[str]],
    bias: float,
    weights: list[float],
    *,
    feature_columns: tuple[str, ...],
    holdout_ids: set[str] | None,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[dict[str, float | int | None], list[dict[str, float | int | None]]]:
    max_grid: tuple[int | None, ...] = (None, 1, 2, 3, 5, 10)
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    query_gates: tuple[float | None, ...] = (None, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92)
    sums: dict[tuple[float, int | None, float | None], float] = defaultdict(float)
    counts: dict[tuple[float, int | None, float | None], int] = defaultdict(int)
    singleton_sums: dict[tuple[float, int | None, float | None], float] = defaultdict(float)
    non_singleton_sums: dict[tuple[float, int | None, float | None], float] = defaultdict(float)
    singleton_counts: dict[tuple[float, int | None, float | None], int] = defaultdict(int)
    non_singleton_counts: dict[tuple[float, int | None, float | None], int] = defaultdict(int)
    active_id: str | None = None
    ranked: list[tuple[str, float]] = []
    seen_source1_ids: set[str] = set()

    def evaluate_group(source1_id: str, items: list[tuple[str, float]]) -> None:
        items.sort(key=lambda item: (-item[1], item[0]))
        expected = truth[source1_id]
        maximum_probability = items[0][1] if items else 0.0
        for threshold in thresholds:
            eligible = [entity_id for entity_id, probability in items if probability >= threshold]
            for query_gate in query_gates:
                gated_eligible = eligible if query_gate is None or maximum_probability >= query_gate else []
                for limit in max_grid:
                    predicted = gated_eligible if limit is None else gated_eligible[:limit]
                    value = f05(expected, predicted)
                    key = (round(float(threshold), 6), limit, query_gate)
                    sums[key] += value
                    counts[key] += 1
                    if expected:
                        non_singleton_sums[key] += value
                        non_singleton_counts[key] += 1
                    else:
                        singleton_sums[key] += value
                        singleton_counts[key] += 1

    with scores_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source1_id = row["source1_entity_id"]
            in_holdout = source1_id in holdout_ids if holdout_ids is not None else is_holdout(source1_id)
            if not in_holdout:
                continue
            if active_id is not None and source1_id != active_id:
                evaluate_group(active_id, ranked)
                ranked = []
            active_id = source1_id
            seen_source1_ids.add(source1_id)
            x = feature_row(row, feature_columns)
            logit = bias + sum(float(value) * weight for value, weight in zip(x, weights))
            ranked.append((row["candidate_entity_id"], sigmoid(logit)))
    if active_id is not None:
        evaluate_group(active_id, ranked)
    for source1_id in truth:
        in_holdout = source1_id in holdout_ids if holdout_ids is not None else is_holdout(source1_id)
        if in_holdout and source1_id not in seen_source1_ids:
            evaluate_group(source1_id, [])

    results: list[dict[str, float | int | None]] = []
    for (threshold, limit, query_gate), total in sums.items():
        n = counts[(threshold, limit, query_gate)]
        results.append({
            "threshold": threshold,
            "max_matches": limit,
            "min_query_probability": query_gate,
            "macro_f0_5": total / n,
            "singleton_macro_f0_5": singleton_sums[(threshold, limit, query_gate)] / max(1, singleton_counts[(threshold, limit, query_gate)]),
            "non_singleton_macro_f0_5": non_singleton_sums[(threshold, limit, query_gate)] / max(1, non_singleton_counts[(threshold, limit, query_gate)]),
            "holdout_entities": n,
        })
    results.sort(key=lambda item: (float(item["macro_f0_5"]), -float(item["threshold"])), reverse=True)
    return results[0], results[:20]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and hold out a logistic pair matcher.")
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--validation-ids", required=True, type=Path)
    parser.add_argument(
        "--holdout-ids",
        type=Path,
        default=None,
        help="Optional explicit confirmation/holdout IDs; otherwise use numeric ID modulo 5.",
    )
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--sample-per-class", type=int, default=250000)
    parser.add_argument(
        "--hard-negative-rank",
        type=int,
        default=20,
        help="Reserve half the negative sample for the first N ranked candidates per query.",
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--threshold-min", type=float, default=0.60)
    parser.add_argument("--threshold-max", type=float, default=0.85)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    feature_columns = available_feature_columns(args.scores)

    validation_ids = read_id_file(args.validation_ids)
    holdout_ids = read_id_file(args.holdout_ids) if args.holdout_ids else None
    if holdout_ids is not None and not holdout_ids.issubset(validation_ids):
        raise ValueError("--holdout-ids must be a subset of --validation-ids")
    truth = load_truth(args.ground_truth, validation_ids)
    x_train, y_train, training_stats = collect_reservoir(
        args.scores, truth, heldout=False, cap_per_class=args.sample_per_class, seed=args.seed,
        feature_columns=feature_columns, hard_negative_rank=args.hard_negative_rank,
        holdout_ids=holdout_ids,
    )
    holdout_bias, holdout_weights = fit_logistic(x_train, y_train, seed=args.seed)
    best, top_results = tune_holdout(
        args.scores, truth, holdout_bias, holdout_weights,
        feature_columns=feature_columns,
        holdout_ids=holdout_ids,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )

    x_all, y_all, full_stats = collect_reservoir(
        args.scores, truth, heldout=False, cap_per_class=args.sample_per_class, seed=args.seed + 1,
        feature_columns=feature_columns, hard_negative_rank=args.hard_negative_rank,
        holdout_ids=holdout_ids,
    )
    # The reported model-selection score uses only held-out businesses. The
    # deployed model then uses sampled candidate pairs from both ID partitions.
    x_holdout, y_holdout, _ = collect_reservoir(
        args.scores, truth, heldout=True, cap_per_class=args.sample_per_class, seed=args.seed + 2,
        feature_columns=feature_columns, hard_negative_rank=args.hard_negative_rank,
        holdout_ids=holdout_ids,
    )
    x_all = np.concatenate((x_all, x_holdout), axis=0)
    y_all = np.concatenate((y_all, y_holdout), axis=0)
    final_bias, final_weights = fit_logistic(x_all, y_all, seed=args.seed + 3)

    result = {
        "model": "numpy_logistic_pair_matcher_v2_context_hardneg",
        "feature_columns": list(feature_columns),
        "bias": final_bias,
        "weights": final_weights,
        "threshold": best["threshold"],
        "max_matches": best["max_matches"],
        "min_query_probability": best["min_query_probability"],
        "heldout_split": str(args.holdout_ids) if args.holdout_ids else "source1 numeric ID modulo 5 equals zero",
        "heldout_best": best,
        "heldout_top20": top_results,
        "train_sample": training_stats,
        "all_sample": full_stats,
        "candidate_limit": 20,
        "hard_negative_rank": args.hard_negative_rank,
        "validation_ids_count": len(validation_ids),
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"heldout_best": best, "heldout_top20": top_results[:10], "model_output": str(args.model_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
