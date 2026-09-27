"""Upload challenge dataset and code to S3 for SageMaker training.

Usage:
    python aws/upload_data.py --bucket YOUR_BUCKET_NAME [--prefix amazonml]

Creates:
    s3://YOUR_BUCKET/amazonml/dataset/train/...
    s3://YOUR_BUCKET/amazonml/dataset/test/...
    s3://YOUR_BUCKET/amazonml/code/source.tar.gz
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def upload_file(local: Path, s3_uri: str) -> None:
    print(f"  {local.name}  ->  {s3_uri}")
    run([sys.executable, "-m", "awscli", "s3", "cp", str(local), s3_uri, "--no-progress"])


def upload_dataset(bucket: str, prefix: str) -> dict[str, str]:
    """Upload all 7 official TSV files. Returns S3 URIs."""
    dataset_root = ROOT / "dataset" / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
    uris = {}
    for split in ("train", "test"):
        split_dir = dataset_root / split
        s3_base = f"s3://{bucket}/{prefix}/dataset/{split}"
        print(f"\nUploading {split} split ({split_dir})...")
        for f in sorted(split_dir.iterdir()):
            if f.suffix == ".tsv":
                dest = f"{s3_base}/{f.name}"
                upload_file(f, dest)
                uris[f.stem] = dest
    return uris


def build_code_tarball(dest: Path) -> None:
    """Bundle src/, requirements-matching.txt, and data/processed/ into a tarball."""
    print(f"\nBuilding code tarball -> {dest}")
    with tarfile.open(dest, "w:gz") as tar:
        for path in (ROOT / "src").rglob("*.py"):
            tar.add(path, arcname=str(path.relative_to(ROOT)))
        tar.add(ROOT / "requirements-matching.txt", arcname="requirements-matching.txt")
        # Include validation split IDs (small, needed for training query selection)
        for f in (ROOT / "data" / "processed").iterdir():
            tar.add(f, arcname=str(f.relative_to(ROOT)))
        # Include the training entrypoint
        tar.add(ROOT / "aws" / "train_entrypoint.py", arcname="aws/train_entrypoint.py")
    size_mb = dest.stat().st_size / 1e6
    print(f"  tarball size: {size_mb:.1f} MB")


def upload_code(bucket: str, prefix: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "source.tar.gz"
        build_code_tarball(tarball)
        s3_uri = f"s3://{bucket}/{prefix}/code/source.tar.gz"
        print(f"\nUploading code tarball...")
        upload_file(tarball, s3_uri)
        return s3_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="S3 bucket name (must exist)")
    parser.add_argument("--prefix", default="amazonml", help="S3 key prefix (default: amazonml)")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset upload (already uploaded)")
    parser.add_argument("--skip-code", action="store_true", help="Skip code upload")
    args = parser.parse_args()

    print(f"Target: s3://{args.bucket}/{args.prefix}/")

    if not args.skip_dataset:
        upload_dataset(args.bucket, args.prefix)

    if not args.skip_code:
        upload_code(args.bucket, args.prefix)

    print(f"\nDone. S3 prefix: s3://{args.bucket}/{args.prefix}/")
    print(f"\nNext step:")
    print(f"  python aws/launch_training_job.py --bucket {args.bucket} --prefix {args.prefix}")


if __name__ == "__main__":
    main()
