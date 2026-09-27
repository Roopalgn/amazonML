"""Resolve multi-owner target conflicts from prediction scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.matching.supervised_pipeline import file_sha256, log, sql_path


def resolve(*, accepted: Path, source1: Path, output: Path, margin: float) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"""
        CREATE TABLE accepted AS
        SELECT qid, tid, cast(score AS DOUBLE) score
        FROM read_csv({sql_path(accepted)}, delim='\t', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE TABLE source1 AS
        SELECT entity_id qid
        FROM read_csv({sql_path(source1)}, delim='\t', header=true, all_varchar=true)
    """)
    before = con.execute("SELECT count(*) FROM accepted").fetchone()[0]
    conflicted_targets = con.execute("""
        SELECT count(*) FROM (
            SELECT tid FROM accepted GROUP BY tid HAVING count(DISTINCT qid)>1
        )
    """).fetchone()[0]
    con.execute(f"""
        CREATE TABLE kept AS
        WITH ranked AS (
            SELECT qid, tid, score,
                ln(greatest(score, 1e-12) / greatest(1 - score, 1e-12)) logit,
                count(*) OVER (PARTITION BY tid) owners,
                row_number() OVER (PARTITION BY tid ORDER BY score DESC, qid) rn,
                lead(ln(greatest(score, 1e-12) / greatest(1 - score, 1e-12)), 1, -1e100)
                    OVER (PARTITION BY tid ORDER BY score DESC, qid) next_logit
            FROM accepted
        )
        SELECT qid, tid, score FROM ranked
        WHERE owners=1 OR (rn=1 AND logit-next_logit >= {margin})
    """)
    after = con.execute("SELECT count(*) FROM kept").fetchone()[0]
    con.execute("""
        CREATE TABLE rows AS
        SELECT s.qid source1_entity_id,
            coalesce(string_agg(k.tid, ',' ORDER BY k.score DESC, k.tid), '') matched_entity_ids
        FROM source1 s LEFT JOIN kept k ON k.qid=s.qid
        GROUP BY s.qid
    """)
    partial = output.with_suffix(output.suffix + ".partial")
    con.execute(f"COPY rows TO {sql_path(partial)} (DELIMITER '\t', HEADER TRUE)")
    partial.replace(output)
    con.close()
    return {
        "input_links": before,
        "output_links": after,
        "removed_links": before - after,
        "conflicted_targets": conflicted_targets,
        "sha256": file_sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--source1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()
    report = resolve(**vars(args))
    log(str(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
