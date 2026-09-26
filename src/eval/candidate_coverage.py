"""Measure the recall ceiling imposed by a list-format candidate file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.data.io import parse_id_list, read_id_file
from src.eval.macro_f05 import entity_f05


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = DEFAULT_ROOT / "dataset" / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"


def load_truth(path: Path, validation_ids: set[str]) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id in validation_ids:
                truth[source1_id] = parse_id_list(row.get("matched_entity_ids") or "")
    missing = validation_ids - truth.keys()
    if missing:
        raise ValueError(f"Ground truth is missing {len(missing)} validation IDs")
    return truth


def load_countries(path: Path, validation_ids: set[str]) -> dict[str, str]:
    countries: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source1_id = row["entity_id"]
            if source1_id in validation_ids:
                countries[source1_id] = row.get("country", "")
    return countries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_DATASET / "train" / "train_ground_truth.tsv")
    parser.add_argument("--source1", type=Path, default=DEFAULT_DATASET / "train" / "train_source1.tsv")
    parser.add_argument("--validation-ids", type=Path, default=DEFAULT_ROOT / "data" / "processed" / "validation_source1_ids.txt")
    args = parser.parse_args()

    validation_ids = read_id_file(args.validation_ids)
    truth = load_truth(args.ground_truth, validation_ids)
    countries = load_countries(args.source1, validation_ids)
    if countries.keys() != validation_ids:
        raise ValueError(f"Source-1 country rows do not match validation ID set: {len(countries)} found")

    cutoffs: tuple[int | None, ...] = (1, 5, 10, 20, 50, 100, 200, None)
    match_caps: tuple[int | None, ...] = (None, 3, 5, 10)
    score_sums: dict[tuple[str, int | None], float] = defaultdict(float)
    score_counts: Counter[tuple[str, int | None]] = Counter()
    capped_oracle_sums: dict[tuple[str, int | None, int | None], float] = defaultdict(float)
    cardinality_sums: dict[tuple[str, int | None], float] = defaultdict(float)
    cardinality_capped_sums: dict[tuple[str, int | None, int | None], float] = defaultdict(float)
    cardinality_counts: Counter[str] = Counter()
    rows_seen: set[str] = set()
    candidate_link_count = 0
    rows_with_candidates = 0
    rows_with_any_truth_candidate = 0
    true_link_count = sum(len(ids) for ids in truth.values())
    true_links_found: Counter[str] = Counter()
    true_links_found_by_country: dict[str, Counter[str]] = defaultdict(Counter)
    true_link_rank_histogram: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    country_rows: dict[str, dict[str, int]] = defaultdict(lambda: {"queries": 0, "truth_links": 0, "rows_with_candidate": 0, "rows_with_true_candidate": 0})

    with args.candidates.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ("source1_entity_id", "candidate_entity_ids")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected candidate schema {reader.fieldnames}; expected {expected}")
        for line_number, row in enumerate(reader, start=2):
            source1_id = row["source1_entity_id"]
            if source1_id not in validation_ids:
                raise ValueError(f"Unexpected Source-1 ID {source1_id} at line {line_number}")
            if source1_id in rows_seen:
                raise ValueError(f"Duplicate Source-1 ID {source1_id} at line {line_number}")
            rows_seen.add(source1_id)
            candidate_ids = parse_id_list(row.get("candidate_entity_ids") or "")
            # Candidate order is meaningful: Person 2 places higher-ranked candidates first.
            ordered = row.get("candidate_entity_ids", "").split(",") if row.get("candidate_entity_ids") else []
            ordered = [entity_id.strip() for entity_id in ordered if entity_id.strip()]
            if len(ordered) != len(candidate_ids):
                raise ValueError(f"Duplicate candidate IDs for {source1_id}")
            truth_ids = truth[source1_id]
            cardinality = (
                "singleton" if not truth_ids else
                "1-2" if len(truth_ids) <= 2 else
                "3-5" if len(truth_ids) <= 5 else "6+"
            )
            cardinality_counts[cardinality] += 1
            country = countries[source1_id] or "(blank)"
            source_counts[country] += 1
            group = country_rows[country]
            group["queries"] += 1
            group["truth_links"] += len(truth_ids)
            candidate_link_count += len(candidate_ids)
            rows_with_candidates += int(bool(candidate_ids))
            group["rows_with_candidate"] += int(bool(candidate_ids))
            positions = {candidate_id: index + 1 for index, candidate_id in enumerate(ordered)}
            hits = truth_ids.intersection(candidate_ids)
            rows_with_any_truth_candidate += int(bool(hits))
            group["rows_with_true_candidate"] += int(bool(hits))
            for entity_id in hits:
                rank = positions[entity_id]
                true_link_rank_histogram[str(rank if rank <= 20 else "21+")] += 1
                true_links_found["all"] += 1
                true_links_found_by_country[country]["all"] += 1
                for cutoff in (1, 5, 10, 20, 50, 100, 200):
                    if rank <= cutoff:
                        true_links_found[str(cutoff)] += 1
                        true_links_found_by_country[country][str(cutoff)] += 1

            for cutoff in cutoffs:
                prefix = ordered if cutoff is None else ordered[:cutoff]
                prediction = truth_ids.intersection(prefix)
                key = (country, cutoff)
                score_sums[key] += entity_f05(truth_ids, prediction)
                score_counts[key] += 1
                for match_cap in match_caps:
                    capped_prediction = prediction if match_cap is None else set(list(prediction)[:match_cap])
                    capped_oracle_sums[(country, cutoff, match_cap)] += entity_f05(truth_ids, capped_prediction)
                cardinality_sums[(cardinality, cutoff)] += entity_f05(truth_ids, prediction)
                for match_cap in match_caps:
                    capped_prediction = prediction if match_cap is None else set(list(prediction)[:match_cap])
                    cardinality_capped_sums[(cardinality, cutoff, match_cap)] += entity_f05(truth_ids, capped_prediction)

    missing_rows = validation_ids - rows_seen
    if missing_rows:
        raise ValueError(f"Candidate file is missing {len(missing_rows)} validation IDs")

    overall = {}
    capped_oracle_overall = {}
    for cutoff in cutoffs:
        total = sum(score_sums[(country, cutoff)] for country in source_counts)
        overall["all" if cutoff is None else str(cutoff)] = total / len(validation_ids)
        cap_name = "all" if cutoff is None else str(cutoff)
        capped_oracle_overall[cap_name] = {}
        for match_cap in match_caps:
            total_capped = sum(capped_oracle_sums[(country, cutoff, match_cap)] for country in source_counts)
            capped_oracle_overall[cap_name]["all" if match_cap is None else str(match_cap)] = total_capped / len(validation_ids)

    by_country = {}
    for country, counts in country_rows.items():
        by_country[country] = {
            **counts,
            "singleton_oracle_f0_5": 1.0,
            "candidate_oracle_macro_by_prefix": {
                "all" if cutoff is None else str(cutoff): score_sums[(country, cutoff)] / max(1, score_counts[(country, cutoff)])
                for cutoff in cutoffs
            },
            "candidate_oracle_macro_by_prefix_and_match_cap": {
                "all" if cutoff is None else str(cutoff): {
                    "all" if match_cap is None else str(match_cap): capped_oracle_sums[(country, cutoff, match_cap)] / max(1, score_counts[(country, cutoff)])
                    for match_cap in match_caps
                }
                for cutoff in cutoffs
            },
            "true_link_recall_by_prefix": {
                "all" if cutoff is None else str(cutoff): true_links_found_by_country[country]["all" if cutoff is None else str(cutoff)] / max(1, counts["truth_links"])
                for cutoff in cutoffs
            },
        }

    cardinality_breakdown = {}
    for cardinality, count in sorted(cardinality_counts.items()):
        cardinality_breakdown[cardinality] = {
            "queries": count,
            "oracle_macro_f0_5_by_prefix": {
                "all" if cutoff is None else str(cutoff): cardinality_sums[(cardinality, cutoff)] / max(1, count)
                for cutoff in cutoffs
            },
            "oracle_macro_f0_5_by_prefix_and_match_cap": {
                "all" if cutoff is None else str(cutoff): {
                    "all" if match_cap is None else str(match_cap): cardinality_capped_sums[(cardinality, cutoff, match_cap)] / max(1, count)
                    for match_cap in match_caps
                }
                for cutoff in cutoffs
            },
        }

    report = {
        "validation_queries": len(validation_ids),
        "candidate_rows": len(rows_seen),
        "candidate_links": candidate_link_count,
        "rows_with_candidates": rows_with_candidates,
        "truth_links": true_link_count,
        "rows_with_at_least_one_candidate_true_link": rows_with_any_truth_candidate,
        "candidate_link_recall": {
            "all": true_links_found["all"] / max(1, true_link_count),
            **{str(cutoff): true_links_found[str(cutoff)] / max(1, true_link_count) for cutoff in (1, 5, 10, 20, 50, 100, 200)},
        },
        "candidate_oracle_macro_f0_5_by_prefix": overall,
        "candidate_oracle_macro_f0_5_by_prefix_and_match_cap": capped_oracle_overall,
        "true_link_rank_histogram": dict(true_link_rank_histogram),
        "source1_country_breakdown": by_country,
        "truth_cardinality_breakdown": cardinality_breakdown,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
