"""Batched DuckDB candidate retrieval from the disk-backed SQLite block index.

Requires ``duckdb==1.5.5`` and DuckDB's SQLite extension. Only challenge data is
queried; the extension is a software dependency, not an entity lookup service.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import duckdb

from block_keys import block_keys
from generate_candidates import KEY_WEIGHTS, build_index, connect, records


def ensure_index(index, source2, source3, rebuild=False):
    index.parent.mkdir(parents=True, exist_ok=True)
    con = connect(index)
    try:
        ready = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()
        if rebuild or not ready:
            build_index(con, (source2, source3))
        actual = con.execute("SELECT source_signatures FROM metadata").fetchone()[0]
        expected = json.dumps([str(path.resolve()) + ":" + str(path.stat().st_size) for path in (source2, source3)])
        if actual != expected:
            raise ValueError("Index belongs to different target files; use --rebuild-index or a separate index")
    finally:
        con.close()


def make_batch_files(rows, keys_path, ids_path):
    with open(keys_path, "w", encoding="utf-8", newline="") as keys_file, open(ids_path, "w", encoding="utf-8", newline="") as ids_file:
        key_writer = csv.writer(keys_file, delimiter="\t", lineterminator="\n")
        id_writer = csv.writer(ids_file, delimiter="\t", lineterminator="\n")
        key_writer.writerow(["qid", "key", "weight"])
        id_writer.writerow(["ord", "qid"])
        for position, row in enumerate(rows):
            qid = row["entity_id"]
            if not qid.startswith("S1-"):
                raise ValueError(f"Unexpected query ID {qid}")
            id_writer.writerow([position, qid])
            for key in block_keys(row["business_name"], row["business_address"], row["country"]):
                key_writer.writerow([qid, key, KEY_WEIGHTS[key.split("|", 2)[1]]])


def process_batch(con, rows, keys_path, ids_path, out, max_block_size, max_candidates):
    started = time.time()
    make_batch_files(rows, keys_path, ids_path)
    con.execute(f"CREATE OR REPLACE TEMP TABLE qkeys AS SELECT qid, key, CAST(weight AS INTEGER) AS weight FROM read_csv('{keys_path.as_posix()}', delim='\t', header=true, all_varchar=true)")
    con.execute(f"CREATE OR REPLACE TEMP TABLE qids AS SELECT CAST(ord AS INTEGER) AS ord, qid FROM read_csv('{ids_path.as_posix()}', delim='\t', header=true, all_varchar=true)")
    con.execute("CREATE OR REPLACE TEMP TABLE selected AS SELECT b.key, b.entity_id FROM idx.blocks b SEMI JOIN (SELECT DISTINCT key FROM qkeys) q USING(key)")
    print(f"  selected blocks: {time.time()-started:.0f}s", flush=True)
    con.execute(f"CREATE OR REPLACE TEMP TABLE eligible AS SELECT key FROM selected GROUP BY key HAVING COUNT(*) <= {max_block_size}")
    skipped = con.execute(f"SELECT COUNT(*) FROM qkeys q JOIN (SELECT key FROM selected GROUP BY key HAVING COUNT(*) > {max_block_size}) large USING(key)").fetchone()[0]
    con.execute("CREATE OR REPLACE TEMP TABLE scores AS SELECT q.qid, s.entity_id, SUM(q.weight) AS score FROM qkeys q JOIN eligible e USING(key) JOIN selected s USING(key) GROUP BY q.qid,s.entity_id")
    print(f"  scored candidates: {time.time()-started:.0f}s", flush=True)
    capped = con.execute(f"SELECT COUNT(*) FROM (SELECT qid FROM scores GROUP BY qid HAVING COUNT(*) > {max_candidates})").fetchone()[0]
    con.execute("CREATE OR REPLACE TEMP TABLE ranked AS SELECT qid, entity_id, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY score DESC, entity_id DESC) AS rn FROM scores")
    print(f"  ranked candidates: {time.time()-started:.0f}s", flush=True)
    result = con.execute(f"""
        SELECT q.qid, COALESCE(string_agg(r.entity_id, ',' ORDER BY r.rn), '') AS candidates
        FROM qids q LEFT JOIN ranked r ON q.qid = r.qid AND r.rn <= {max_candidates}
        GROUP BY q.ord, q.qid ORDER BY q.ord
    """)
    with_candidates = pairs = 0
    while batch := result.fetchmany(1000):
        for qid, candidates in batch:
            out.writerow([qid, candidates])
            with_candidates += bool(candidates)
            pairs += 0 if not candidates else candidates.count(",") + 1
    print(f"  wrote candidate lists: {time.time()-started:.0f}s", flush=True)
    return {"queries": len(rows), "with_candidates": with_candidates, "total_candidates": pairs, "skipped_large_blocks": skipped, "capped_queries": capped}


def main():
    p = argparse.ArgumentParser(description="Generate candidates with a batched full-index join")
    p.add_argument("--source1", type=Path, required=True)
    p.add_argument("--source2", type=Path, required=True)
    p.add_argument("--source3", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--batch-size", type=int, default=100000)
    p.add_argument("--max-block-size", type=int, default=1000)
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--max-queries", type=int, help="Development only; never use for final output")
    args = p.parse_args()
    if min(args.batch_size, args.max_block_size, args.max_candidates) < 1:
        p.error("batch size, block size and candidate limit must be positive")
    ensure_index(args.index, args.source2, args.source3, args.rebuild_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workspace = args.output.parent / ".duckdb_work"
    workspace.mkdir(exist_ok=True)
    con = duckdb.connect(config={"extension_directory": str(workspace / "extensions"), "temp_directory": str(workspace / "temp"), "memory_limit": "4GB"})
    con.execute("INSTALL sqlite; LOAD sqlite")
    con.execute(f"ATTACH '{args.index.resolve().as_posix()}' AS idx (TYPE sqlite)")
    stats = {"queries": 0, "with_candidates": 0, "total_candidates": 0, "skipped_large_blocks": 0, "capped_queries": 0}
    start = time.time()
    keys_path, ids_path = workspace / "qkeys.tsv", workspace / "qids.tsv"
    with open(args.output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        batch = []
        for row in records(args.source1):
            if args.max_queries is not None and stats["queries"] + len(batch) >= args.max_queries:
                break
            batch.append(row)
            if len(batch) == args.batch_size:
                outcome = process_batch(con, batch, keys_path, ids_path, writer, args.max_block_size, args.max_candidates)
                for key, value in outcome.items():
                    stats[key] += value
                print(f"{stats['queries']:,} queries; {stats['total_candidates']:,} candidates; {time.time()-start:.0f}s", flush=True)
                batch.clear()
        if batch:
            outcome = process_batch(con, batch, keys_path, ids_path, writer, args.max_block_size, args.max_candidates)
            for key, value in outcome.items():
                stats[key] += value
            print(f"{stats['queries']:,} queries; {stats['total_candidates']:,} candidates; {time.time()-start:.0f}s", flush=True)
    con.close()
    stats["elapsed_seconds"] = round(time.time() - start, 3)
    stats["settings"] = {
        "batch_size": args.batch_size,
        "max_block_size": args.max_block_size,
        "max_candidates": args.max_candidates,
        "index": str(args.index.resolve()),
        "output": str(args.output.resolve()),
    }
    stats["exact_generator_command"] = [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    args.output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
