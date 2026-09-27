"""Tiny ordering regressions; DuckDB uses an in-memory index without extensions."""

import csv
import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "blocking"))

from generate_candidates import generate
from generate_candidates_fast import process_batch
import duckdb


class BlockingOrderTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.rows = [
            {"entity_id": "S1-20", "business_name": "Acme Robotics", "business_address": "", "country": "US"},
            {"entity_id": "S1-3", "business_name": "Unknown Business", "business_address": "", "country": "US"},
            {"entity_id": "S1-10", "business_name": "Acme Robotics", "business_address": "", "country": "US"},
        ]
        name_key = "us|name|acme robotics"
        pair_key = "us|pair|acme|robotics"
        self.postings = [
            (pair_key, "S3-99"),
            (name_key, "S2-99"),
            (name_key, "S2-10"),
            (pair_key, "S2-10"),
            (name_key, "S3-20"),
            (pair_key, "S3-20"),
        ]
        # Scores are 8, 8, 5, 3; the score-8 tie is resolved by descending ID.
        self.ranked = ["S3-20", "S2-10", "S2-99", "S3-99"]

    def run_sqlite(self, cap):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        con.execute("CREATE TEMP TABLE blocks (key TEXT, entity_id TEXT)")
        con.executemany("INSERT INTO blocks VALUES (?, ?)", self.postings)
        queries = self.root / "queries.tsv"
        output = self.root / "candidates.tsv"
        with queries.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)
        stats = generate(con, queries, output, max_block_size=1000, max_candidates=cap)
        with output.open(encoding="utf-8", newline="") as stream:
            result = list(csv.reader(stream, delimiter="\t"))
        self.assertEqual(result[0], ["source1_entity_id", "candidate_entity_ids"])
        return result[1:], stats

    def run_duckdb(self, cap, reverse_storage=False):
        con = duckdb.connect(":memory:", config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
        })
        self.addCleanup(con.close)
        con.execute("CREATE SCHEMA idx")
        con.execute("CREATE TABLE idx.blocks (key VARCHAR, entity_id VARCHAR)")
        con.executemany("INSERT INTO idx.blocks VALUES (?, ?)", self.postings)
        connection = con
        if reverse_storage:
            def execute(query):
                result = con.execute(query)
                if query.startswith("CREATE OR REPLACE TEMP TABLE ranked AS"):
                    # SQL table storage order must not determine list order.
                    con.execute("CREATE OR REPLACE TEMP TABLE ranked AS SELECT * FROM ranked ORDER BY rn DESC, qid")
                return result

            connection = Mock()
            connection.execute.side_effect = execute
        output = io.StringIO()
        with redirect_stdout(io.StringIO()):
            stats = process_batch(
                connection, self.rows, self.root / "keys.tsv", self.root / "ids.tsv",
                csv.writer(output, delimiter="\t", lineterminator="\n"),
                max_block_size=1000, max_candidates=cap,
            )
        return list(csv.reader(io.StringIO(output.getvalue()), delimiter="\t")), stats

    def assert_ranked_output(self, result, stats, cap):
        candidates = ",".join(self.ranked[:cap])
        self.assertEqual(result, [
            ["S1-20", candidates],
            ["S1-3", ""],
            ["S1-10", candidates],
        ])
        self.assertEqual(stats, {
            "queries": 3,
            "with_candidates": 2,
            "total_candidates": 2 * min(cap, len(self.ranked)),
            "skipped_large_blocks": 0,
            "capped_queries": 2 if cap < len(self.ranked) else 0,
        })

    def test_sqlite_order_with_and_without_cap(self):
        for cap in (1, 3, 4, 200):
            with self.subTest(cap=cap):
                self.assert_ranked_output(*self.run_sqlite(cap), cap)

    def test_duckdb_order_with_and_without_cap(self):
        for cap in (1, 3, 4, 200):
            with self.subTest(cap=cap):
                self.assert_ranked_output(*self.run_duckdb(cap), cap)

    def test_duckdb_order_ignores_ranked_table_storage_order(self):
        for cap in (3, 200):
            with self.subTest(cap=cap):
                self.assert_ranked_output(*self.run_duckdb(cap, reverse_storage=True), cap)


if __name__ == "__main__":
    unittest.main()
