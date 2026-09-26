"""Break fixed-validation candidate recall and oracle F0.5 by truth size."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.data.io import parse_id_list, read_id_file
from src.eval.macro_f05 import entity_f05


def bucket(size: int) -> str:
    if size == 0:
        return "singleton_no_match"
    if size <= 2:
        return "1-2 matches"
    if size <= 5:
        return "3-5 matches"
    return "6+ matches"


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    dataset = root / "dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=dataset / "train_ground_truth.tsv")
    parser.add_argument("--source1", type=Path, default=dataset / "train_source1.tsv")
    parser.add_argument("--validation-ids", type=Path, default=root / "data/processed/validation_source1_ids.txt")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    validation_ids = read_id_file(args.validation_ids)
    truth: dict[str, set[str]] = {}
    with args.ground_truth.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qid = row["source1_entity_id"]
            if qid in validation_ids:
                truth[qid] = parse_id_list(row.get("matched_entity_ids") or "")
    if truth.keys() != validation_ids:
        raise ValueError(f"Ground truth missing {len(validation_ids - truth.keys())} fixed IDs")
    countries: dict[str, str] = {}
    with args.source1.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qid = row["entity_id"]
            if qid in validation_ids:
                countries[qid] = row.get("country") or "(blank)"
    if countries.keys() != validation_ids:
        raise ValueError(f"Source 1 missing {len(validation_ids - countries.keys())} fixed IDs")

    cutoffs: tuple[int | None, ...] = (20, 50, 100, 200, 500, 1000, None)
    groups: dict[str, dict[str, object]] = defaultdict(lambda: {"queries": 0, "truth_links": 0, "rows_with_at_least_one_true_candidate": 0, "hits": Counter(), "oracle_sum": Counter()})
    target_source = defaultdict(lambda: {"truth_links": 0, "hits": Counter()})
    seen: set[str] = set()
    with args.candidates.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_columns = ("source1_entity_id", "candidate_entity_ids")
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"Unexpected schema {reader.fieldnames}")
        for line, row in enumerate(reader, start=2):
            qid = row["source1_entity_id"]
            if qid not in validation_ids or qid in seen:
                raise ValueError(f"Unexpected or duplicate validation ID at line {line}: {qid}")
            seen.add(qid)
            raw = row.get("candidate_entity_ids") or ""
            ordered = raw.split(",") if raw else []
            if len(ordered) != len(set(ordered)):
                raise ValueError(f"Duplicate candidate IDs at line {line}")
            if any(not item.startswith(("S2-", "S3-")) for item in ordered):
                raise ValueError(f"Invalid candidate target prefix at line {line}")
            truth_ids = truth[qid]
            keys = ("all", f"country:{countries[qid]}", f"truth:{bucket(len(truth_ids))}")
            for key in keys:
                group = groups[key]
                group["queries"] += 1
                group["truth_links"] += len(truth_ids)
                group["rows_with_at_least_one_true_candidate"] += int(bool(truth_ids.intersection(ordered)))
                for cutoff in cutoffs:
                    predictions = ordered if cutoff is None else ordered[:cutoff]
                    true_found = truth_ids.intersection(predictions)
                    group["hits"]["all" if cutoff is None else str(cutoff)] += len(true_found)
                    group["oracle_sum"]["all" if cutoff is None else str(cutoff)] += entity_f05(truth_ids, true_found)
            for prefix in ("S2-", "S3-"):
                source_truth = {entity_id for entity_id in truth_ids if entity_id.startswith(prefix)}
                target_source[prefix]["truth_links"] += len(source_truth)
                for cutoff in cutoffs:
                    predictions = ordered if cutoff is None else ordered[:cutoff]
                    target_source[prefix]["hits"]["all" if cutoff is None else str(cutoff)] += len(source_truth.intersection(predictions))
    if seen != validation_ids:
        raise ValueError(f"Candidate file missing {len(validation_ids - seen)} fixed IDs")

    report: dict[str, object] = {"candidate_file": str(args.candidates.resolve()), "validation_queries": len(validation_ids), "cutoffs": ["20", "50", "100", "200", "500", "1000", "all"], "groups": {}, "true_link_recall_by_target_source": {}}
    for prefix, values in target_source.items():
        report["true_link_recall_by_target_source"][prefix.rstrip("-")] = {
            "truth_links": values["truth_links"],
            "pair_recall_by_prefix": {cutoff: (values["hits"][cutoff] / values["truth_links"] if values["truth_links"] else 1.0) for cutoff in ("20", "50", "100", "200", "500", "1000", "all")},
        }
    for key, group in groups.items():
        query_count = group["queries"]
        links = group["truth_links"]
        report["groups"][key] = {
            "queries": query_count,
            "truth_links": links,
            "rows_with_at_least_one_true_candidate": group["rows_with_at_least_one_true_candidate"],
            "pair_recall_by_prefix": {cutoff: (group["hits"][cutoff] / links if links else 1.0) for cutoff in ("20", "50", "100", "200", "500", "1000", "all")},
            "oracle_macro_f0_5_by_prefix": {cutoff: (group["oracle_sum"][cutoff] / query_count if query_count else 0.0) for cutoff in ("20", "50", "100", "200", "500", "1000", "all")},
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
