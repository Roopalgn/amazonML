"""Small end-to-end test for block keys, IDs, country handling and singleton rows."""

import csv
from pathlib import Path

from generate_candidates import build_index, connect, generate


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["entity_id", "business_name", "business_address", "country"])
        writer.writerows(rows)


def main():
    scratch = Path("data/candidates/person2/smoke")
    scratch.mkdir(parents=True, exist_ok=True)
    s1, s2, s3 = (scratch / name for name in ("source1.tsv", "source2.tsv", "source3.tsv"))
    write(s1, [
        ("S1-1", "Acme Robotics Inc", "2621 Cotten Road, Tyler, TX", "US"),
        ("S1-2", "Brahma Infosoft Pvt Ltd", "Coimatore Colony, Mysore, Karnataka", "India"),
        ("S1-3", "Marina Ecole France Sarl", "63 Rue de Dieppe, Lille", "France"),
        ("S1-4", "Unique Business", "Somewhere", "US"),
    ])
    write(s2, [
        ("S2-1", "ACME ROBOTICS", "2621 Cotten Rd, Tyler", "US"),
        ("S2-2", "Brahma Infosoft", "", "India"),
        ("S2-3", "Marina Ecole France", "63 R. DE DIEPPE, LILLE", "France"),
    ])
    write(s3, [
        ("S3-1", "Acme Robotics Corporation", "Different Address", "US"),
        ("S3-2", "Other Business", "2621 Cotten Road, Tyler", "US"),
    ])
    db = scratch / "index.sqlite"
    for suffix in ("", "-wal", "-shm"):
        (scratch / ("index.sqlite" + suffix)).unlink(missing_ok=True)
    con = connect(db)
    try:
        build_index(con, (s2, s3))
        output = scratch / "candidate_pairs.tsv"
        stats = generate(con, s1, output, 100, 100)
    finally:
        con.close()
    with open(output, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 4
    by_id = {row["source1_entity_id"]: set(filter(None, row["candidate_entity_ids"].split(","))) for row in rows}
    assert {"S2-1", "S3-1"}.issubset(by_id["S1-1"])
    assert "S2-2" in by_id["S1-2"]
    assert "S2-3" in by_id["S1-3"]
    assert by_id["S1-4"] == set()
    assert stats["queries"] == 4
    print("Blocking smoke test passed")


if __name__ == "__main__":
    main()
