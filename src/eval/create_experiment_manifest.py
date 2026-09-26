"""Create a reproducibility manifest for a candidate/matcher experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        portable_path = str(path.relative_to(Path.cwd()))
    except ValueError:
        portable_path = str(path)
    return {"path": portable_path, "bytes": path.stat().st_size, "sha256": sha256(path)}


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--matching-version", required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--candidate-cap", type=int)
    parser.add_argument("--block-settings", default="")
    parser.add_argument("--threshold-policy", default="")
    parser.add_argument("--local-score", type=float)
    parser.add_argument("--public-score", type=float)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    manifest = {
        "candidate_version": args.candidate_version,
        "matching_version": args.matching_version,
        "validation_ids": file_record(args.validation_ids),
        "candidate_artifact": file_record(args.candidate),
        "model_artifact": file_record(args.model),
        "candidate_cap": args.candidate_cap,
        "block_settings": args.block_settings,
        "threshold_policy": args.threshold_policy,
        "local_macro_f0_5": args.local_score,
        "public_score": args.public_score,
        "notes": args.notes,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
