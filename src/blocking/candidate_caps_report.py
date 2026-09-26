"""Structural size report for candidate-list prefixes from a ranked TSV."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path

from src.data.io import parse_id_list, read_id_file


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, default=root / "data/processed/validation_source1_ids.txt")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[200, 500, 1000])
    args = parser.parse_args()
    expected_ids = read_id_file(args.validation_ids)
    caps = sorted(set(args.caps))
    header = b"source1_entity_id\tcandidate_entity_ids\n"
    reports = {cap: {"candidate_links": 0, "s2_candidate_links": 0, "s3_candidate_links": 0, "tsv_bytes": len(header), "maximum_candidates_per_row": 0, "rows_at_cap": 0} for cap in caps}
    seen: set[str] = set()
    with args.candidates.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("source1_entity_id", "candidate_entity_ids"):
            raise ValueError(f"Unexpected candidate schema: {reader.fieldnames}")
        for line, row in enumerate(reader, start=2):
            qid = row["source1_entity_id"]
            if qid not in expected_ids or qid in seen:
                raise ValueError(f"Unexpected or duplicate Source-1 ID at line {line}: {qid}")
            seen.add(qid)
            raw = row.get("candidate_entity_ids") or ""
            ordered = raw.split(",") if raw else []
            if len(ordered) != len(set(ordered)):
                raise ValueError(f"Duplicate candidates for {qid}")
            if any(not candidate.startswith(("S2-", "S3-")) for candidate in ordered):
                raise ValueError(f"Unexpected target ID prefix for {qid}")
            for cap, report in reports.items():
                prefix = ordered[:cap]
                count = len(prefix)
                encoded = ",".join(prefix)
                report["candidate_links"] += count
                report["s2_candidate_links"] += sum(candidate.startswith("S2-") for candidate in prefix)
                report["s3_candidate_links"] += sum(candidate.startswith("S3-") for candidate in prefix)
                report["tsv_bytes"] += len(qid.encode("utf-8")) + 1 + len(encoded.encode("utf-8")) + 1
                report["maximum_candidates_per_row"] = max(report["maximum_candidates_per_row"], count)
                report["rows_at_cap"] += int(count == cap)
    missing = expected_ids - seen
    if missing:
        raise ValueError(f"Candidate file is missing {len(missing)} fixed validation IDs")
    digest = hashlib.sha256()
    with args.candidates.open("rb") as candidate_file:
        for chunk in iter(lambda: candidate_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    for values in reports.values():
        links = values["candidate_links"]
        values["s2_share"] = values["s2_candidate_links"] / links if links else 0.0
        values["s3_share"] = values["s3_candidate_links"] / links if links else 0.0
    result = {"candidate_file": str(args.candidates.resolve()), "candidate_file_bytes": args.candidates.stat().st_size, "candidate_file_sha256": digest.hexdigest(), "validation_ids": len(expected_ids), "rows": len(seen), "all_fixed_validation_ids_present": seen == expected_ids, "candidate_prefix_structural_metrics": {str(cap): values for cap, values in reports.items()}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
