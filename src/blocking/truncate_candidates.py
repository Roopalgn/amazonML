"""Write a stable top-K prefix from an already ranked candidate TSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"queries": 0, "with_candidates": 0, "total_candidates": 0, "capped_queries": 0}
    with args.input.open(encoding="utf-8", newline="") as source, args.output.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source, delimiter="\t")
        expected = ("source1_entity_id", "candidate_entity_ids")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected candidate columns: {reader.fieldnames}")
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(expected)
        for line, row in enumerate(reader, start=2):
            raw = row.get("candidate_entity_ids") or ""
            ids = raw.split(",") if raw else []
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate candidate IDs at line {line}")
            chosen = ids[:args.max_candidates]
            stats["queries"] += 1
            stats["with_candidates"] += int(bool(chosen))
            stats["total_candidates"] += len(chosen)
            stats["capped_queries"] += int(len(ids) > args.max_candidates)
            writer.writerow((row["source1_entity_id"], ",".join(chosen)))
    stats.update({"input": str(args.input.resolve()), "output": str(args.output.resolve()), "max_candidates": args.max_candidates})
    args.output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
