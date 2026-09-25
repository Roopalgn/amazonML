"""Official-style local macro F_0.5 scoring for list-based entity resolution."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set

from src.data.io import GROUND_TRUTH_COLUMNS, parse_id_list, read_id_file


PREDICTION_COLUMNS = ("source1_entity_id", "matched_entity_ids")
BETA_SQUARED = 0.25


@dataclass(frozen=True)
class ScoreSummary:
    """Macro score plus separate singleton/non-singleton diagnostics."""

    macro_f05: float
    entity_count: int
    singleton_count: int
    singleton_mean: float
    non_singleton_count: int
    non_singleton_mean: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "macro_f0_5": self.macro_f05,
            "entity_count": self.entity_count,
            "singleton_count": self.singleton_count,
            "singleton_mean_f0_5": self.singleton_mean,
            "non_singleton_count": self.non_singleton_count,
            "non_singleton_mean_f0_5": self.non_singleton_mean,
        }


def entity_f05(truth: Set[str], prediction: Set[str]) -> float:
    """Compute one entity's F_0.5, including the official singleton convention."""

    if not truth:
        return 1.0 if not prediction else 0.0
    if not prediction:
        return 0.0
    true_positives = len(truth & prediction)
    precision = true_positives / len(prediction)
    recall = true_positives / len(truth)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return (1.0 + BETA_SQUARED) * precision * recall / (BETA_SQUARED * precision + recall)


def load_list_tsv(path: str | Path, expected_columns: tuple[str, str]) -> Dict[str, Set[str]]:
    """Load a ground-truth or prediction list TSV with strict schema/duplicate checks."""

    result: Dict[str, Set[str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or tuple(reader.fieldnames) != expected_columns:
            raise ValueError(f"{path} must have exactly these columns: {expected_columns}")
        for line_number, row in enumerate(reader, start=2):
            entity_id = (row[expected_columns[0]] or "").strip()
            if not entity_id:
                raise ValueError(f"Blank Source-1 ID at line {line_number} in {path}")
            if entity_id in result:
                raise ValueError(f"Duplicate Source-1 ID {entity_id!r} in {path}")
            result[entity_id] = parse_id_list((row[expected_columns[1]] or "").strip())
    return result


def score(truth_by_id: Mapping[str, Set[str]], predictions_by_id: Mapping[str, Set[str]], evaluation_ids: Iterable[str] | None = None) -> ScoreSummary:
    """Score listed IDs. Missing prediction rows are deliberately treated as empty lists."""

    ids = set(evaluation_ids) if evaluation_ids is not None else set(truth_by_id)
    unknown = ids - set(truth_by_id)
    if unknown:
        example = sorted(unknown)[:5]
        raise ValueError(f"Evaluation IDs absent from ground truth, e.g. {example}")
    scores, singleton_scores, non_singleton_scores = [], [], []
    for entity_id in ids:
        truth = truth_by_id[entity_id]
        value = entity_f05(truth, predictions_by_id.get(entity_id, set()))
        scores.append(value)
        (singleton_scores if not truth else non_singleton_scores).append(value)
    if not scores:
        raise ValueError("No entities selected for scoring")
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return ScoreSummary(mean(scores), len(scores), len(singleton_scores), mean(singleton_scores), len(non_singleton_scores), mean(non_singleton_scores))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score list-based entity-resolution predictions with macro F_0.5.")
    parser.add_argument("--truth", required=True, help="Path to train_ground_truth.tsv")
    parser.add_argument("--predictions", required=True, help="Path to a matching_results.tsv-style file")
    parser.add_argument("--ids", help="Optional newline-delimited fixed validation Source-1 IDs")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    summary = score(load_list_tsv(args.truth, GROUND_TRUTH_COLUMNS), load_list_tsv(args.predictions, PREDICTION_COLUMNS), read_id_file(args.ids) if args.ids else None)
    if args.json:
        print(json.dumps(summary.as_dict(), sort_keys=True))
    else:
        for label, value in summary.as_dict().items():
            print(f"{label}: {value:.6f}" if isinstance(value, float) else f"{label}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
