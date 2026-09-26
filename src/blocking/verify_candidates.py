"""Validate a candidate TSV structurally against its official Source 1 TSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    source_rows = csv.DictReader(args.source1.open(encoding="utf-8", newline=""), delimiter="\t")
    candidate_handle = args.candidates.open(encoding="utf-8", newline="")
    candidate_rows = csv.DictReader(candidate_handle, delimiter="\t")
    expected = ("source1_entity_id", "candidate_entity_ids")
    if tuple(candidate_rows.fieldnames or ()) != expected:
        raise ValueError(f"Unexpected candidate columns: {candidate_rows.fieldnames}")
    count = links = max_list = france_rows = france_with_candidates = 0
    s2_links = s3_links = 0
    for line, (source, candidate) in enumerate(zip(source_rows, candidate_rows, strict=True), start=2):
        qid = source["entity_id"]
        if not qid.startswith("S1-") or candidate["source1_entity_id"] != qid:
            raise ValueError(f"Source 1 order or ID mismatch at row {line}: {qid} != {candidate['source1_entity_id']}")
        raw = candidate.get("candidate_entity_ids") or ""
        ids = raw.split(",") if raw else []
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate candidate IDs at line {line}")
        if any(not target.startswith(("S2-", "S3-")) for target in ids):
            raise ValueError(f"Unexpected candidate ID prefix at line {line}")
        count += 1
        links += len(ids)
        max_list = max(max_list, len(ids))
        s2_links += sum(target.startswith("S2-") for target in ids)
        s3_links += sum(target.startswith("S3-") for target in ids)
        if (source.get("country") or "").strip().casefold() == "france":
            france_rows += 1
            france_with_candidates += bool(ids)
    candidate_handle.close()
    digest = hashlib.sha256()
    with args.candidates.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "source1": str(args.source1.resolve()),
        "candidate_file": str(args.candidates.resolve()),
        "candidate_file_bytes": args.candidates.stat().st_size,
        "candidate_file_sha256": digest.hexdigest(),
        "queries": count,
        "candidate_links": links,
        "maximum_candidates_per_row": max_list,
        "s2_candidate_links": s2_links,
        "s3_candidate_links": s3_links,
        "france_source1_rows": france_rows,
        "france_rows_with_candidates": france_with_candidates,
        "source1_order_and_row_count_match": True,
        "only_s2_s3_candidate_ids": True,
        "no_duplicate_candidates_per_row": True,
    }
    serialized = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
