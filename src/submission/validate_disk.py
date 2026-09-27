"""Read-only submission validation using a bounded, disk-backed DuckDB workspace.

Run with --matching, --candidate, --test-dir, and --work-dir. The work directory
needs room for the database and spilled joins/aggregations, potentially several
times the TSV size. No challenge-sized Python ID collections are constructed.
The JSON report goes to stdout; exit status is zero only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


MEMORY_LIMIT = "768MB"
THREADS = 2
SAMPLE_LIMIT = 5
SOURCE_HEADER = ("entity_id", "business_name", "business_address", "country")
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def check_header(path: Path, columns: tuple[str, ...]) -> None:
    # Bound the read even if a malformed file has no newline.
    with path.open("rb") as handle:
        header = handle.readline(4096)
    expected = "\t".join(columns).encode("utf-8")
    if header not in (expected + b"\n", expected + b"\r\n", expected):
        raise ValueError(f"{path}: expected exact TSV header {columns!r}")


def csv_query(columns: tuple[str, ...]) -> str:
    # Column names come only from the constants above. Paths use bound parameters.
    schema = ", ".join(f"'{column}': 'VARCHAR'" for column in columns)
    return (
        "read_csv(?, header=true, delim='\\t', quote='', escape='', "
        f"columns={{{schema}}}, auto_detect=false, nullstr='', "
        "null_padding=false, strict_mode=true, ignore_errors=false, parallel=false)"
    )


def record_check(con, report: dict, code: str, query: str) -> None:
    count = con.execute(f"SELECT count(*) FROM ({query}) AS failures").fetchone()[0]
    if count:
        cursor = con.execute(f"SELECT * FROM ({query}) AS failures LIMIT {SAMPLE_LIMIT}")
        names = [column[0] for column in cursor.description]
        samples = [dict(zip(names, row)) for row in cursor.fetchall()]
        report["errors"].append({"code": code, "count": count, "examples": samples})


def validate_tables(con, report: dict) -> None:
    counts = report["counts"]
    for source, prefix in (("source1", "S1-"), ("source2", "S2-"), ("source3", "S3-")):
        counts[f"{source}_rows"] = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
        if not counts[f"{source}_rows"] and source == "source1":
            report["errors"].append({"code": "source1.empty", "count": 1, "examples": []})
        record_check(con, report, f"{source}.invalid_id", f"""
            SELECT entity_id FROM {source}
            WHERE entity_id IS NULL OR NOT starts_with(entity_id, '{prefix}')
                OR length(entity_id) <= 3 OR regexp_matches(entity_id, '[[:space:],]')
        """)
        record_check(con, report, f"{source}.duplicate_id", f"""
            SELECT entity_id, count(*) AS occurrences FROM {source}
            GROUP BY entity_id HAVING count(*) > 1
        """)

    con.execute("CREATE TABLE targets AS SELECT entity_id FROM source2 UNION ALL SELECT entity_id FROM source3")
    for label in ("matching", "candidate"):
        rows = f"{label}_rows"
        links = f"{label}_links"
        counts[f"{label}_rows"] = con.execute(f"SELECT count(*) FROM {rows}").fetchone()[0]
        counts[f"{label}_empty_rows"] = con.execute(f"SELECT count(*) FROM {rows} WHERE ids = ''").fetchone()[0]
        record_check(con, report, f"{label}.invalid_source1_id", f"""
            SELECT row_no, source1_entity_id FROM {rows}
            WHERE source1_entity_id IS NULL OR NOT starts_with(source1_entity_id, 'S1-')
                OR length(source1_entity_id) <= 3 OR regexp_matches(source1_entity_id, '[[:space:],]')
        """)
        record_check(con, report, f"{label}.duplicate_source1_id", f"""
            SELECT source1_entity_id, count(*) AS occurrences FROM {rows}
            GROUP BY source1_entity_id HAVING count(*) > 1
        """)
        record_check(con, report, f"{label}.unknown_source1_id", f"""
            SELECT r.row_no, r.source1_entity_id FROM {rows} r
            ANTI JOIN source1 s ON r.source1_entity_id = s.entity_id
        """)
        record_check(con, report, f"{label}.missing_source1_id", f"""
            SELECT s.entity_id AS source1_entity_id FROM source1 s
            ANTI JOIN {rows} r ON s.entity_id = r.source1_entity_id
        """)

        # Persistent link tables allow subsequent joins and GROUP BYs to spill.
        # Empty lists produce zero links; empty elements in nonempty lists fail.
        con.execute(f"""
            CREATE TABLE {links} AS
            SELECT row_no, source1_entity_id, unnest(string_split(ids, ',')) AS target_id
            FROM {rows} WHERE ids <> ''
        """)
        counts[f"{label}_links"] = con.execute(f"SELECT count(*) FROM {links}").fetchone()[0]
        record_check(con, report, f"{label}.invalid_target_id", f"""
            SELECT row_no, source1_entity_id, target_id FROM {links}
            WHERE NOT (starts_with(target_id, 'S2-') OR starts_with(target_id, 'S3-'))
                OR length(target_id) <= 3 OR regexp_matches(target_id, '[[:space:]]')
        """)
        record_check(con, report, f"{label}.duplicate_target_id", f"""
            SELECT row_no, target_id, count(*) AS occurrences FROM {links}
            GROUP BY row_no, target_id HAVING count(*) > 1
        """)
        record_check(con, report, f"{label}.unknown_target_id", f"""
            SELECT l.row_no, l.source1_entity_id, l.target_id FROM {links} l
            ANTI JOIN targets t ON l.target_id = t.entity_id
        """)
        con.execute("CHECKPOINT")

    record_check(con, report, "matching.not_in_candidates", """
        SELECT m.source1_entity_id, m.target_id FROM matching_links m
        ANTI JOIN candidate_links c
            ON m.source1_entity_id = c.source1_entity_id AND m.target_id = c.target_id
    """)


def validate_submission(*, matching: Path, candidate: Path, test_dir: Path, work_dir: Path) -> dict:
    report = {
        "valid": False,
        "inputs": {"matching": str(matching.resolve()), "candidate": str(candidate.resolve()),
                   "test_dir": str(test_dir.resolve())},
        "settings": {"memory_limit": MEMORY_LIMIT, "threads": THREADS, "disk_backed": True},
        "counts": {},
        "errors": [],
    }
    inputs = [("matching", matching, MATCHING_HEADER), ("candidate", candidate, CANDIDATE_HEADER)]
    inputs += [(f"source{number}", test_dir / f"test_source{number}.tsv", SOURCE_HEADER) for number in (1, 2, 3)]
    snapshots = {}
    for label, path, header in inputs:
        try:
            check_header(path, header)
            stat = path.stat()
            snapshots[label] = (stat.st_size, stat.st_mtime_ns)
        except (OSError, ValueError) as exc:
            report["errors"].append({"code": f"{label}.input", "message": str(exc)})
    if report["errors"]:
        return report

    try:
        import duckdb

        report["duckdb_version"] = duckdb.__version__
        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="validate-", dir=work_dir.resolve()) as workspace:
            workspace = Path(workspace)
            spill_dir = workspace / "spill"
            spill_dir.mkdir()
            con = duckdb.connect(str(workspace / "validation.duckdb"), config={
                "memory_limit": MEMORY_LIMIT,
                "threads": THREADS,
                "temp_directory": str(spill_dir),
                "preserve_insertion_order": False,
            })
            try:
                # Scan once into the working database; all ID comparisons stay in SQL.
                for label, path, header in inputs:
                    try:
                        reader = csv_query(header)
                        if label.startswith("source"):
                            con.execute(f"CREATE TABLE {label} AS SELECT entity_id FROM {reader}", [str(path.resolve())])
                        else:
                            con.execute(f"""
                                CREATE TABLE {label}_rows AS
                                SELECT row_number() OVER () AS row_no, source1_entity_id,
                                    coalesce({header[1]}, '') AS ids FROM {reader}
                            """, [str(path.resolve())])
                    except duckdb.Error as exc:
                        report["errors"].append({"code": f"{label}.parse", "message": str(exc)[:3000]})
                        return report
                validate_tables(con, report)
            finally:
                con.close()
        for label, path, _ in inputs:
            stat = path.stat()
            if snapshots[label] != (stat.st_size, stat.st_mtime_ns):
                report["errors"].append({"code": f"{label}.changed_during_validation"})
    except Exception as exc:
        report["errors"].append({"code": "validator.runtime", "message": str(exc)[:3000]})
    report["valid"] = not report["errors"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = validate_submission(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
