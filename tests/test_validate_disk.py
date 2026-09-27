"""Tiny fixtures for the independent disk-backed submission validator."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.submission.validate_disk import (
    CANDIDATE_HEADER, MATCHING_HEADER, SOURCE_HEADER, main, validate_submission,
)


def write_tsv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class DiskValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.test_dir = self.root / "test"
        self.test_dir.mkdir()
        self.work_dir = self.root / "work"
        self.matching = self.root / "matching.tsv"
        self.candidate = self.root / "candidate.tsv"
        self.source1 = self.test_dir / "test_source1.tsv"
        write_tsv(self.source1, SOURCE_HEADER, [
            ("S1-a", "Alpha", "12 Road, Town", "US"),
            ("S1-b", "Beta", "", "India"),
            ("S1-c", "Gamma", "7 Rue", "France"),
        ])
        write_tsv(self.test_dir / "test_source2.tsv", SOURCE_HEADER, [
            ("S2-a", "Alpha", "12 Road", "US"), ("S2-b", "Alpha LLC", "", "US"),
        ])
        write_tsv(self.test_dir / "test_source3.tsv", SOURCE_HEADER, [("S3-a", "Alpha", "12 Road", "US")])
        self.set_outputs()

    def set_outputs(self):
        write_tsv(self.matching, MATCHING_HEADER, [("S1-c", ""), ("S1-a", "S2-a,S3-a"), ("S1-b", "")])
        write_tsv(self.candidate, CANDIDATE_HEADER, [("S1-a", "S3-a,S2-b,S2-a"), ("S1-b", ""), ("S1-c", "")])

    def validate(self):
        paths = [self.matching, self.candidate, *self.test_dir.glob("*.tsv")]
        before = {path: path.read_bytes() for path in paths if path.exists()}
        result = validate_submission(matching=self.matching, candidate=self.candidate,
                                     test_dir=self.test_dir, work_dir=self.work_dir)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        if self.work_dir.exists():
            self.assertEqual(list(self.work_dir.iterdir()), [])
        return result

    def assert_error(self, report, code):
        self.assertFalse(report["valid"], report)
        self.assertIn(code, [error["code"] for error in report["errors"]], report)

    def test_valid_multi_match_and_empty_lists(self):
        report = self.validate()
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["counts"]["matching_links"], 2)
        self.assertEqual(report["counts"]["candidate_links"], 3)
        self.assertEqual(report["counts"]["matching_empty_rows"], 2)
        self.assertEqual(report["settings"]["memory_limit"], "768MB")
        self.assertEqual(report["settings"]["threads"], 2)

    def test_output_source1_coverage_and_uniqueness(self):
        for label, path, header in (("matching", self.matching, MATCHING_HEADER),
                                    ("candidate", self.candidate, CANDIDATE_HEADER)):
            for rows, suffix in (([("S1-a", ""), ("S1-b", "")], "missing_source1_id"),
                                 ([("S1-a", ""), ("S1-b", ""), ("S1-c", ""), ("S1-a", "")], "duplicate_source1_id"),
                                 ([("S1-a", ""), ("S1-b", ""), ("S1-unknown", "")], "unknown_source1_id"),
                                 ([("", ""), ("S1-b", ""), ("S1-c", "")], "invalid_source1_id")):
                with self.subTest(label=label, suffix=suffix):
                    self.set_outputs()
                    write_tsv(path, header, rows)
                    self.assert_error(self.validate(), f"{label}.{suffix}")

    def test_invalid_unknown_and_duplicate_targets_in_both_files(self):
        for label, path, header in (("matching", self.matching, MATCHING_HEADER),
                                    ("candidate", self.candidate, CANDIDATE_HEADER)):
            for ids, suffix in (("S2-missing", "unknown_target_id"), ("S1-a", "invalid_target_id"),
                                ("S2-a,S2-a", "duplicate_target_id"), ("S2-a,", "invalid_target_id"),
                                (",S2-a", "invalid_target_id"), ("S2-a,,S3-a", "invalid_target_id"),
                                (" S2-a", "invalid_target_id"), (" ", "invalid_target_id")):
                with self.subTest(label=label, ids=ids):
                    self.set_outputs()
                    write_tsv(path, header, [("S1-a", ids), ("S1-b", ""), ("S1-c", "")])
                    self.assert_error(self.validate(), f"{label}.{suffix}")

    def test_subset_is_per_source1_entity(self):
        write_tsv(self.candidate, CANDIDATE_HEADER, [("S1-a", "S2-a"), ("S1-b", "S3-a"), ("S1-c", "")])
        report = self.validate()
        self.assert_error(report, "matching.not_in_candidates")
        issue = next(error for error in report["errors"] if error["code"] == "matching.not_in_candidates")
        self.assertEqual(issue["count"], 1)

    def test_exact_headers_and_missing_file(self):
        for header in ("matched_entity_ids\tsource1_entity_id\n", "source1_entity_id,matched_entity_ids\n",
                       "\ufeffsource1_entity_id\tmatched_entity_ids\n", "source1_entity_id\tmatched_entity_ids\textra\n"):
            with self.subTest(header=header):
                self.matching.write_text(header, encoding="utf-8")
                self.assert_error(self.validate(), "matching.input")
        self.set_outputs()
        self.candidate.unlink()
        self.assert_error(self.validate(), "candidate.input")

    def test_bad_column_counts_and_invalid_utf8(self):
        for tail in (b"S1-a\n", b"S1-a\tS2-a\textra\n", b"S1-a\t\xff\n"):
            with self.subTest(tail=tail):
                self.matching.write_bytes(b"source1_entity_id\tmatched_entity_ids\n" + tail)
                self.assert_error(self.validate(), "matching.parse")

    def test_crlf_and_final_empty_list_without_newline(self):
        self.matching.write_bytes(b"source1_entity_id\tmatched_entity_ids\r\nS1-a\tS2-a,S3-a\r\nS1-b\t\r\nS1-c\t")
        self.assertTrue(self.validate()["valid"])

    def test_bad_rosters_fail(self):
        write_tsv(self.source1, SOURCE_HEADER, [("S1-a", "A", "", "US"), ("S1-a", "B", "", "US")])
        self.assert_error(self.validate(), "source1.duplicate_id")
        write_tsv(self.source1, SOURCE_HEADER, [])
        self.assert_error(self.validate(), "source1.empty")

    def test_target_roster_prefixes_and_duplicates_fail(self):
        source2 = self.test_dir / "test_source2.tsv"
        write_tsv(source2, SOURCE_HEADER, [("S2-a", "A", "", "US"), ("S2-a", "B", "", "US")])
        self.assert_error(self.validate(), "source2.duplicate_id")
        write_tsv(source2, SOURCE_HEADER, [("S3-wrong-source", "A", "", "US")])
        self.assert_error(self.validate(), "source2.invalid_id")

    def test_blank_source1_row_is_not_an_empty_match_list(self):
        with self.matching.open("ab") as handle:
            handle.write(b"\t\n")
        self.assert_error(self.validate(), "matching.invalid_source1_id")

    def test_paths_with_apostrophes_use_bound_parameters(self):
        destination = self.root / "matching's.tsv"
        self.matching.rename(destination)
        self.matching = destination
        self.work_dir = self.root / "validator's work"
        self.assertTrue(self.validate()["valid"])

    def test_workspace_error_returns_failure_report(self):
        self.work_dir.write_text("not a directory", encoding="utf-8")
        report = validate_submission(matching=self.matching, candidate=self.candidate,
                                     test_dir=self.test_dir, work_dir=self.work_dir)
        self.assert_error(report, "validator.runtime")
        self.assertEqual(self.work_dir.read_text(), "not a directory")

    def test_empty_target_sources_and_singletons_are_valid(self):
        for number in (2, 3):
            write_tsv(self.test_dir / f"test_source{number}.tsv", SOURCE_HEADER, [])
        for path, header in ((self.matching, MATCHING_HEADER), (self.candidate, CANDIDATE_HEADER)):
            write_tsv(path, header, [("S1-a", ""), ("S1-b", ""), ("S1-c", "")])
        self.assertTrue(self.validate()["valid"])

    def test_report_samples_bounded(self):
        write_tsv(self.matching, MATCHING_HEADER, [(f"S1-unknown{i}", "") for i in range(12)])
        report = self.validate()
        issue = next(error for error in report["errors"] if error["code"] == "matching.unknown_source1_id")
        self.assertEqual(issue["count"], 12)
        self.assertEqual(len(issue["examples"]), 5)

    def test_cli_json_and_exit_codes(self):
        args = ["--matching", str(self.matching), "--candidate", str(self.candidate),
                "--test-dir", str(self.test_dir), "--work-dir", str(self.work_dir)]
        for expected in (0, 1):
            if expected:
                self.candidate.unlink()
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = main(args)
            self.assertEqual(code, expected)
            self.assertEqual(json.loads(stream.getvalue())["valid"], expected == 0)


if __name__ == "__main__":
    unittest.main()
