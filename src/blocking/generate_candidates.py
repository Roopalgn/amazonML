"""Disk-backed first-pass candidate generation for the official TSV files.

The index is built once per split, then reused. No external data or API is used.
"""

import argparse
import csv
import heapq
import json
import sqlite3
import time
from pathlib import Path

from block_keys import block_keys, normalize


def records(path):
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {"entity_id", "business_name", "business_address", "country"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {expected - set(reader.fieldnames or [])}")
        yield from reader


def connect(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    # CREATE INDEX sorts tens of millions of keys; keep its temp B-tree on disk.
    con.execute("PRAGMA temp_store=FILE")
    con.execute("PRAGMA cache_size=-131072")
    return con


def build_index(con, target_paths):
    con.executescript("""
        DROP TABLE IF EXISTS blocks;
        DROP TABLE IF EXISTS targets;
        DROP TABLE IF EXISTS metadata;
        CREATE TABLE targets (entity_id TEXT PRIMARY KEY, name TEXT, address TEXT);
        CREATE TABLE blocks (key TEXT NOT NULL, entity_id TEXT NOT NULL);
    """)
    count = 0
    for path in target_paths:
        print(f"Indexing {path}", flush=True)
        for row in records(path):
            entity_id = row["entity_id"]
            if not entity_id.startswith(("S2-", "S3-")):
                raise ValueError(f"Unexpected target ID {entity_id}")
            name = normalize(row["business_name"])
            address = normalize(row["business_address"])
            con.execute("INSERT INTO targets VALUES (?, ?, ?)", (entity_id, name, address))
            con.executemany("INSERT INTO blocks VALUES (?, ?)", ((key, entity_id) for key in block_keys(row["business_name"], row["business_address"], row["country"])))
            count += 1
            if count % 100000 == 0:
                con.commit()
                print(f"  {count:,} target rows", flush=True)
    con.commit()
    finish_index(con, target_paths, count)


def finish_index(con, target_paths, count):
    signatures = [str(path.resolve()) + ":" + str(path.stat().st_size) for path in target_paths]
    print("Creating block index on disk; this can take time on the full dataset", flush=True)
    con.execute("CREATE INDEX IF NOT EXISTS blocks_key_idx ON blocks(key)")
    con.execute("CREATE TABLE IF NOT EXISTS metadata (target_count INTEGER, source_signatures TEXT)")
    con.execute("DELETE FROM metadata")
    con.execute("INSERT INTO metadata VALUES (?, ?)", (count, json.dumps(signatures)))
    con.commit()
    print(f"Indexed {count:,} target rows", flush=True)


KEY_WEIGHTS = {"name": 5, "address": 5, "pair": 3, "postal": 2, "number": 2, "locality": 2, "addressnum": 2, "prefixaddr": 1}


def generate(con, query_path, output_path, max_block_size, max_candidates, max_queries=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"queries": 0, "with_candidates": 0, "total_candidates": 0, "skipped_large_blocks": 0, "capped_queries": 0}
    started = time.time()
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for query in records(query_path):
            if max_queries is not None and stats["queries"] >= max_queries:
                break
            qid = query["entity_id"]
            if not qid.startswith("S1-"):
                raise ValueError(f"Unexpected query ID {qid}")
            hits = {}
            for key in block_keys(query["business_name"], query["business_address"], query["country"]):
                rows = con.execute("SELECT entity_id FROM blocks WHERE key=? LIMIT ?", (key, max_block_size + 1)).fetchall()
                if len(rows) > max_block_size:
                    stats["skipped_large_blocks"] += 1
                    continue
                weight = KEY_WEIGHTS[key.split("|", 2)[1]]
                for (target_id,) in rows:
                    hits[target_id] = hits.get(target_id, 0) + weight
            if len(hits) > max_candidates:
                stats["capped_queries"] += 1
                selected = [item[0] for item in heapq.nlargest(max_candidates, hits.items(), key=lambda item: (item[1], item[0]))]
            else:
                selected = list(hits)
            selected.sort()
            writer.writerow([qid, ",".join(selected)])
            stats["queries"] += 1
            stats["with_candidates"] += bool(selected)
            stats["total_candidates"] += len(selected)
            if stats["queries"] % 100000 == 0:
                print(f"  {stats['queries']:,} queries, {stats['total_candidates']:,} pairs, {time.time()-started:.0f}s", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate official candidate list TSV using disk-backed blocking")
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="SQLite cache, one per train/test split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--resume-index", action="store_true", help="Finish indexing already-loaded targets after an interrupted CREATE INDEX")
    parser.add_argument("--max-block-size", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-queries", type=int, help="Development only; never use for final submission")
    args = parser.parse_args()
    if args.max_block_size < 1 or args.max_candidates < 1:
        parser.error("block and candidate limits must be positive")
    args.index.parent.mkdir(parents=True, exist_ok=True)
    con = connect(args.index)
    try:
        ready = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()
        if args.resume_index:
            if ready or not con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='blocks'").fetchone():
                raise ValueError("--resume-index requires loaded blocks and no metadata table")
            count = con.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            finish_index(con, (args.source2, args.source3), count)
        elif args.rebuild_index or not ready:
            build_index(con, (args.source2, args.source3))
        else:
            actual = con.execute("SELECT source_signatures FROM metadata").fetchone()[0]
            expected = json.dumps([str(path.resolve()) + ":" + str(path.stat().st_size) for path in (args.source2, args.source3)])
            if actual != expected:
                raise ValueError("Index was built from different target files; use --rebuild-index or a different --index path")
        stats = generate(con, args.source1, args.output, args.max_block_size, args.max_candidates, args.max_queries)
    finally:
        con.close()
    stats["index"] = str(args.index)
    stats["output"] = str(args.output)
    args.output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
