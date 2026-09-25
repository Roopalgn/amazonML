"""Create official challenge output files for Person 3.

The script can create a validator-safe all-singleton baseline when no candidate
input exists yet. Once Person 2 publishes candidate lists, pass them with
``--candidate-input``; once scored candidates exist, pass them with ``--scores``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHING_ROOT = REPO_ROOT / "src" / "matching"
if str(MATCHING_ROOT) not in sys.path:
    sys.path.insert(0, str(MATCHING_ROOT))

from baseline_matcher import decide_matches, format_id_list, parse_id_list


DEFAULT_STUDENT_RESOURCE = REPO_ROOT / "dataset" / "6ab10eb3b23ba_student_resource" / "student_resource"
DEFAULT_TEST_SOURCE1 = DEFAULT_STUDENT_RESOURCE / "dataset" / "test" / "test_source1.tsv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


def iter_source1_ids(test_source1: Path):
    with test_source1.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "entity_id" not in (reader.fieldnames or []):
            raise ValueError(f"{test_source1} must contain entity_id")
        for row in reader:
            entity_id = row.get("entity_id", "").strip()
            if entity_id:
                yield entity_id


def load_candidate_lists(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    candidates: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required column(s): {sorted(missing)}")
        for row in reader:
            candidates[row["source1_entity_id"]] = parse_id_list(row.get("candidate_entity_ids"))
    return candidates


def load_scores(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_id", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required column(s): {sorted(missing)}")
        for row in reader:
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            scores[row["source1_entity_id"]][row["candidate_entity_id"]] = score
    return scores


def write_outputs(
    *,
    test_source1: Path,
    output_dir: Path,
    candidate_input: Path | None,
    scores_input: Path | None,
    threshold: float,
    max_matches: int | None,
    policy: str,
) -> tuple[int, int, int]:
    candidate_lists = load_candidate_lists(candidate_input)
    scores_by_source1 = load_scores(scores_input)

    if scores_input is not None and policy == "empty":
        policy = "scored-threshold"

    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    rows = 0
    non_empty_candidates = 0
    non_empty_matches = 0

    with matching_path.open("w", encoding="utf-8", newline="") as matching_handle, candidate_path.open(
        "w", encoding="utf-8", newline=""
    ) as candidate_handle:
        matching_writer = csv.writer(matching_handle, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for source1_id in iter_source1_ids(test_source1):
            candidates = candidate_lists.get(source1_id, ())
            scores = scores_by_source1.get(source1_id)
            decision = decide_matches(
                source1_id,
                candidates,
                scores,
                threshold=threshold,
                max_matches=max_matches,
                policy=policy,
            )

            candidate_writer.writerow([source1_id, format_id_list(decision.candidate_entity_ids)])
            matching_writer.writerow([source1_id, format_id_list(decision.matched_entity_ids)])

            rows += 1
            if decision.candidate_entity_ids:
                non_empty_candidates += 1
            if decision.matched_entity_ids:
                non_empty_matches += 1

    return rows, non_empty_candidates, non_empty_matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Create official matching and candidate TSV outputs.")
    parser.add_argument("--test-source1", type=Path, default=DEFAULT_TEST_SOURCE1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-input", type=Path, default=None)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument(
        "--policy",
        choices=["empty", "pass-through", "scored-threshold", "top-k"],
        default="empty",
        help="Default is the safe all-singleton stub. Scores imply scored-threshold unless another policy is set.",
    )
    args = parser.parse_args()

    rows, non_empty_candidates, non_empty_matches = write_outputs(
        test_source1=args.test_source1,
        output_dir=args.output_dir,
        candidate_input=args.candidate_input,
        scores_input=args.scores,
        threshold=args.threshold,
        max_matches=args.max_matches,
        policy=args.policy,
    )
    print(f"wrote_rows={rows}")
    print(f"non_empty_candidate_rows={non_empty_candidates}")
    print(f"non_empty_match_rows={non_empty_matches}")
    print(f"matching={args.output_dir / 'matching_results.tsv'}")
    print(f"candidate={args.output_dir / 'candidate_pairs.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
