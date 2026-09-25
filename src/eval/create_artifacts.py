"""Create Person 1's fixed split, small dev set, and streaming dataset profile.

Run once at the beginning of the project.  Re-running with the same arguments is
safe and reproduces the exact ID files.  Do not change the seed or split fraction
after teammates begin reporting validation scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from src.data.io import DatasetPaths, find_dataset_root, iter_ground_truth, iter_source, parse_id_list, write_id_file


SPLIT_VERSION = "person1-v1"


def stable_bucket(entity_id: str, seed: str) -> int:
    """Return a stable value in [0, 9_999] independent of Python hash randomization."""

    digest = hashlib.sha256(f"{seed}\0{entity_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def empty_field_stats() -> Dict[str, Any]:
    return {"count": 0, "empty": 0, "chars": 0, "non_ascii": 0, "contains_digit": 0}


def update_field_stats(stats: Dict[str, Any], value: str) -> None:
    stats["count"] += 1
    if not value:
        stats["empty"] += 1
        return
    stats["chars"] += len(value)
    stats["non_ascii"] += int(any(ord(char) > 127 for char in value))
    stats["contains_digit"] += int(any(char.isdigit() for char in value))


def summarize_field(stats: Dict[str, Any]) -> Dict[str, float | int]:
    present = stats["count"] - stats["empty"]
    return {
        "rows": stats["count"],
        "empty": stats["empty"],
        "empty_pct": round(100 * stats["empty"] / stats["count"], 4) if stats["count"] else 0.0,
        "mean_chars_when_present": round(stats["chars"] / present, 3) if present else 0.0,
        "non_ascii_rows": stats["non_ascii"],
        "non_ascii_pct": round(100 * stats["non_ascii"] / stats["count"], 4) if stats["count"] else 0.0,
        "contains_digit_rows": stats["contains_digit"],
        "contains_digit_pct": round(100 * stats["contains_digit"] / stats["count"], 4) if stats["count"] else 0.0,
    }


def profile_source(paths: DatasetPaths, split: str, source: str) -> Dict[str, Any]:
    countries: Counter[str] = Counter()
    fields = {"business_name": empty_field_stats(), "business_address": empty_field_stats()}
    country_fields: dict[str, dict[str, Dict[str, Any]]] = defaultdict(
        lambda: {"business_name": empty_field_stats(), "business_address": empty_field_stats()}
    )
    for row in iter_source(paths, split, source):
        country = row["country"]
        countries[country] += 1
        for field, stats in fields.items():
            value = row[field]
            update_field_stats(stats, value)
            update_field_stats(country_fields[country][field], value)
    return {
        "rows": sum(countries.values()),
        "countries": dict(sorted(countries.items())),
        "fields": {field: summarize_field(stats) for field, stats in fields.items()},
        "country_fields": {
            country: {field: summarize_field(stats) for field, stats in field_map.items()}
            for country, field_map in sorted(country_fields.items())
        },
    }


def profile_ground_truth(paths: DatasetPaths) -> Dict[str, Any]:
    matches_per_s1: Counter[int] = Counter()
    target_sources: Counter[str] = Counter()
    rows = 0
    for row in iter_ground_truth(paths):
        matched_ids = parse_id_list(row["matched_entity_ids"])
        rows += 1
        matches_per_s1[len(matched_ids)] += 1
        for entity_id in matched_ids:
            target_sources[entity_id[:2]] += 1
    return {"rows": rows, "match_count_distribution": dict(sorted(matches_per_s1.items())), "singleton_rows": matches_per_s1[0], "target_source_links": dict(sorted(target_sources.items()))}


def select_ids(paths: DatasetPaths, validation_fraction: float, seed: str, dev_per_country: int) -> tuple[set[str], set[str], Dict[str, Dict[str, int]]]:
    cutoff = round(validation_fraction * 10_000)
    validation_ids: set[str] = set()
    dev_ranked: dict[str, list[tuple[int, str]]] = defaultdict(list)
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "validation": 0})
    for row in iter_source(paths, "train", "source1"):
        entity_id, country = row["entity_id"], row["country"]
        counts[country]["total"] += 1
        bucket = stable_bucket(entity_id, seed)
        if bucket < cutoff:
            validation_ids.add(entity_id)
            counts[country]["validation"] += 1
        ranked = dev_ranked[country]
        ranked.append((bucket, entity_id))
        # Keep a bounded deterministic sample without retaining all Source-1 IDs.
        if len(ranked) > dev_per_country * 4:
            ranked.sort()
            del ranked[dev_per_country:]
    development_ids = {entity_id for ranked in dev_ranked.values() for _, entity_id in sorted(ranked)[:dev_per_country]}
    return validation_ids, development_ids, dict(sorted(counts.items()))


def markdown_profile(source_stats: Dict[str, Dict[str, Any]], truth_stats: Dict[str, Any], split_stats: Dict[str, Dict[str, int]], paths: DatasetPaths, seed: str, validation_fraction: float, dev_per_country: int) -> str:
    lines = [
        "# Dataset profile and fixed evaluation contract",
        "",
        "Generated by `python -m src.eval.create_artifacts` from the supplied official TSVs. All counts below are streaming scans of the complete files; no external data was used.",
        "",
        "## Schema and source sizes",
        "",
        "Every source file has `entity_id`, `business_name`, `business_address`, and `country`; `train_ground_truth.tsv` has `source1_entity_id` and comma-separated `matched_entity_ids`. Files are UTF-8 TSVs.",
        "",
        "| Split / source | Rows | Country counts |",
        "| --- | ---: | --- |",
    ]
    for name, stats in source_stats.items():
        countries = ", ".join(f"{country}: {count:,}" for country, count in stats["countries"].items())
        lines.append(f"| {name} | {stats['rows']:,} | {countries} |")
    lines += ["", "## Name and address quality by source", "", "Percentages are per row. Non-ASCII is a useful indicator of script/transliteration variation, not an error.", "", "| Source | Field | Empty | Mean characters when present | Non-ASCII rows | Rows containing digits |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, stats in source_stats.items():
        for field, field_stats in stats["fields"].items():
            lines.append(f"| {name} | {field} | {field_stats['empty']:,} ({field_stats['empty_pct']:.2f}%) | {field_stats['mean_chars_when_present']:.2f} | {field_stats['non_ascii_rows']:,} ({field_stats['non_ascii_pct']:.2f}%) | {field_stats['contains_digit_rows']:,} ({field_stats['contains_digit_pct']:.2f}%) |")
    lines += [
        "",
        "## Country-level noise profile",
        "",
        "This table is the basis for country-aware, but open-set, normalization: use country only to compare records within the same supplied label. France has test records but no training labels, so no France-specific learned rule is justified.",
        "",
        "| Source | Country | Field | Empty | Mean characters when present | Non-ASCII rows | Rows containing digits |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in source_stats.items():
        for country, field_map in stats["country_fields"].items():
            for field, field_stats in field_map.items():
                lines.append(f"| {name} | {country} | {field} | {field_stats['empty']:,} ({field_stats['empty_pct']:.2f}%) | {field_stats['mean_chars_when_present']:.2f} | {field_stats['non_ascii_rows']:,} ({field_stats['non_ascii_pct']:.2f}%) | {field_stats['contains_digit_rows']:,} ({field_stats['contains_digit_pct']:.2f}%) |")
    distribution = ", ".join(f"{count}: {rows:,}" for count, rows in truth_stats["match_count_distribution"].items())
    lines += [
        "",
        "## Training labels",
        "",
        f"`train_ground_truth.tsv` has {truth_stats['rows']:,} rows; {truth_stats['singleton_rows']:,} ({100 * truth_stats['singleton_rows'] / truth_stats['rows']:.2f}%) are singletons. Match-count distribution: `{distribution}`.",
        "",
        f"Target links by source: {', '.join(f'{source}: {count:,}' for source, count in truth_stats['target_source_links'].items())}.",
        "",
        "## Fixed validation and development IDs",
        "",
        f"Split version: `{SPLIT_VERSION}`. Validation uses the deterministic SHA-256 bucket rule with seed `{seed}` and target fraction {validation_fraction:.0%}. Keep `data/processed/validation_source1_ids.txt` unchanged once experiment comparison starts.",
        "",
        "| Country | Train Source-1 rows | Validation rows | Validation fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for country, counts in split_stats.items():
        lines.append(f"| {country} | {counts['total']:,} | {counts['validation']:,} | {counts['validation'] / counts['total']:.2%} |")
    lines += [
        "",
        f"`dev_sample_source1_ids.txt` contains the {dev_per_country:,} lowest-hash Source-1 IDs per observed training country for quick, country-balanced smoke tests.",
        "",
        "## Generalization and scoring notes",
        "",
        "- Training countries are not a closed vocabulary: France appears in test but has no labels. Preserve the string country field, use it for within-country blocking only, and do not create France-specific rules from external information.",
        "- Address fields can be blank in Sources 2 and 3. Candidate generation and matching must have a name-led fallback for those rows.",
        "- The official metric is macro F_0.5. `src/eval/macro_f05.py` scores empty/empty singleton predictions as 1.0, and reports singleton and non-singleton means separately. Optimize precision: false merges are more costly than missed links.",
        "- Use only the supplied challenge data. External business lookup, geocoding, APIs, and data augmentation are prohibited.",
        "",
        "Data root used: the official challenge dataset directory, resolved locally at runtime.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Person 1 fixed split, dev IDs, and complete streaming profile.")
    parser.add_argument("--data-root", help="Directory containing train/ and test/ (defaults to the bundled resource)")
    parser.add_argument("--output-root", default=".", help="Repository root for docs/ and data/ outputs")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", default="amazon-ml-2026-person-1-v1")
    parser.add_argument("--dev-per-country", type=int, default=500)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.dev_per_country < 1:
        parser.error("--dev-per-country must be positive")
    paths = find_dataset_root(args.data_root)
    output_root = Path(args.output_root).resolve()
    validation_ids, development_ids, split_stats = select_ids(paths, args.validation_fraction, args.seed, args.dev_per_country)
    source_stats = {f"{split}_{source}": profile_source(paths, split, source) for split in ("train", "test") for source in ("source1", "source2", "source3")}
    truth_stats = profile_ground_truth(paths)
    write_id_file(output_root / "data" / "processed" / "validation_source1_ids.txt", validation_ids)
    write_id_file(output_root / "data" / "processed" / "dev_sample_source1_ids.txt", development_ids)
    (output_root / "docs").mkdir(parents=True, exist_ok=True)
    (output_root / "docs" / "dataset_profile.md").write_text(markdown_profile(source_stats, truth_stats, split_stats, paths, args.seed, args.validation_fraction, args.dev_per_country), encoding="utf-8")
    print(json.dumps({"validation_ids": len(validation_ids), "dev_ids": len(development_ids), "profile": str(output_root / 'docs' / 'dataset_profile.md')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
