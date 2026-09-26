"""Derive tuning and confirmation subsets from the frozen validation IDs.

The master validation file remains the evaluation contract.  This utility only
creates deterministic sub-splits for model selection and one final confirmation
run; it never regenerates or changes the master split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.data.io import read_id_file, write_id_file


def bucket(seed: str, entity_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{entity_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--tuning-output", type=Path, required=True)
    parser.add_argument("--confirmation-output", type=Path, required=True)
    parser.add_argument("--seed", default="amazon-ml-2026-person-1-confirmation-v1")
    parser.add_argument("--confirmation-modulus", type=int, default=4)
    parser.add_argument("--confirmation-remainder", type=int, default=0)
    args = parser.parse_args()
    if args.confirmation_modulus < 2 or not 0 <= args.confirmation_remainder < args.confirmation_modulus:
        parser.error("confirmation modulus must be >= 2 and remainder must be in its range")

    ids = read_id_file(args.validation_ids)
    confirmation = {
        entity_id for entity_id in ids
        if bucket(args.seed, entity_id) % args.confirmation_modulus == args.confirmation_remainder
    }
    tuning = ids - confirmation
    if not tuning or not confirmation:
        raise ValueError("Sub-split produced an empty partition")

    write_id_file(args.tuning_output, tuning)
    write_id_file(args.confirmation_output, confirmation)
    report = {
        "master_validation_ids": str(args.validation_ids),
        "master_count": len(ids),
        "tuning_count": len(tuning),
        "confirmation_count": len(confirmation),
        "confirmation_fraction": len(confirmation) / len(ids),
        "seed": args.seed,
        "confirmation_modulus": args.confirmation_modulus,
        "confirmation_remainder": args.confirmation_remainder,
        "tuning_sha256": sha256(args.tuning_output),
        "confirmation_sha256": sha256(args.confirmation_output),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
