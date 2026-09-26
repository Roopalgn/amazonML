"""Select the fixed Person 1 validation IDs from the supplied train Source 1 TSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.data.io import read_id_file


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, default=root / "dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv")
    parser.add_argument("--validation-ids", type=Path, default=root / "data/processed/validation_source1_ids.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected_ids = read_id_file(args.validation_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with args.source1.open(encoding="utf-8", newline="") as source, args.output.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames or "entity_id" not in reader.fieldnames:
            raise ValueError("Source 1 TSV has no entity_id column")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            entity_id = row["entity_id"]
            if entity_id in selected_ids:
                if entity_id in seen:
                    raise ValueError(f"Duplicate fixed validation ID in Source 1: {entity_id}")
                writer.writerow(row)
                seen.add(entity_id)
    missing = selected_ids - seen
    if missing:
        raise ValueError(f"Source 1 is missing {len(missing)} fixed validation IDs")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"rows": len(seen), "validation_ids": len(selected_ids), "all_present": seen == selected_ids, "output": str(args.output.resolve()), "bytes": args.output.stat().st_size, "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
