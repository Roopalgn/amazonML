"""Zip a candidate TSV with Zip64 and write/verify its SHA-256 sidecar."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--sha256", type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"candidate TSV does not exist: {args.input}")
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        archive.write(args.input, arcname=args.input.name)
    with args.input.open("rb") as source:
        source_hash = sha256_stream(source)
    with zipfile.ZipFile(args.zip, "r") as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != args.input.name or members[0].file_size != args.input.stat().st_size:
            raise ValueError(f"Unexpected ZIP contents: {members}")
        with archive.open(members[0]) as extracted:
            archive_hash = sha256_stream(extracted)
    if archive_hash != source_hash:
        raise ValueError("Candidate TSV checksum does not match the archived file")
    sidecar = args.sha256 or args.zip.with_name(args.zip.name + ".sha256")
    zip_digest = hashlib.sha256()
    with args.zip.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            zip_digest.update(chunk)
    sidecar.write_text(f"{zip_digest.hexdigest()}  {args.zip.name}\n", encoding="ascii")
    print(f"TSV SHA256  {source_hash}  {args.input.resolve()}")
    print(f"ZIP SHA256  {zip_digest.hexdigest()}  {args.zip.resolve()}")
    print(f"ZIP bytes   {args.zip.stat().st_size}")
    print(f"SHA file    {sidecar.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
