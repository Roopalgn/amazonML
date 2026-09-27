"""Small stdlib regression tests; no challenge data or external dependencies."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.matching.baseline_matcher import select_matches
from src.matching.threshold_search import (
    evaluate_threshold,
    f05_score,
    load_scores,
    load_truth,
    load_validation_ids,
    threshold_grid,
)
from src.submission import make_scored_outputs as output


def write_tsv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class SelectionTests(unittest.TestCase):
    def test_audited_cap_and_threshold_search_agree(self):
        scores = {"S2-wrong": 0.83, "S3-right": 0.99}
        for policy in ("scored-threshold", "top-k"):
            self.assertEqual(
                select_matches(tuple(scores), scores, threshold=0.82, max_matches=1, policy=policy),
                ("S3-right",),
            )
        self.assertEqual(evaluate_threshold({"S1-a": ("S3-right",)}, {"S1-a": scores}, 0.82, 1), (1, 0, 1))

    def test_multiple_matches_ties_and_inclusive_boundary(self):
        scores = {"S3-c": 0.9, "S2-b": 0.9, "S2-a": 0.9, "S3-low": 0.89}
        self.assertEqual(
            select_matches(tuple(scores), scores, threshold=0.9, policy="scored-threshold"),
            ("S2-a", "S2-b", "S3-c"),
        )
        self.assertEqual(select_matches(("S2-a",), {}, policy="scored-threshold"), ())

    def test_nonfinite_scores_and_bad_controls_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_matches(("S2-a",), {"S2-a": value}, policy="scored-threshold")
        with self.assertRaises(ValueError):
            select_matches((), max_matches=0)
        with self.assertRaises(ValueError):
            threshold_grid(0.5, 0.9, 0)
        with self.assertRaises(ValueError):
            evaluate_threshold({}, {}, 0.8, None)
        with self.assertRaises(ValueError):
            evaluate_threshold({"S1-a": ()}, {"S1-unknown": {}}, 0.8, None)

    def test_audited_metric_examples(self):
        self.assertAlmostEqual(f05_score(("S2-a", "S3-b"), ("S2-a", "S2-c", "S3-b")), 5 / 7)
        self.assertEqual(f05_score((), ()), 1)
        self.assertEqual(f05_score(("a", "b", "c", "d"), ("a",)), 0.625)

    def test_heuristic_weights_and_overrides_unchanged(self):
        source = output.make_features("S1-a", "Atlas Trading", "12 Oak Road 10001", "US")
        conflict = output.make_features("S2-a", "Atlas Trading", "99 Pine Street 90002", "US")
        missing = output.make_features("S2-b", "Atlas Trading LLC", "", "US")
        self.assertAlmostEqual(output.score_pair(source, conflict), 0.82)
        self.assertAlmostEqual(output.score_pair(source, missing), 0.63)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source1 = self.root / "source1.tsv"
        self.source2 = self.root / "source2.tsv"
        self.source3 = self.root / "source3.tsv"
        self.candidates = self.root / "input.tsv"
        self.out = self.root / "output"
        self.out.mkdir()
        self.score_path = self.root / "scores.tsv"
        header = ["entity_id", "business_name", "business_address", "country"]
        write_tsv(self.source1, header, [
            ("S1-a", "Atlas Trading", "12 Oak Road 10001", "US"),
            ("S1-b", "Other Business", "42 Elm Road", "France"),
        ])
        write_tsv(self.source2, header, [
            ("S2-wrong", "Atlas Trading", "99 Pine Street 90002", "US"),
            ("S2-extra", "Atlas Trading LLC", "", "US"),
        ])
        write_tsv(self.source3, header, [("S3-right", "Atlas Trading", "12 Oak Road 10001", "US")])
        self.set_candidates([("S1-a", "S2-wrong,S3-right,S2-extra"), ("S1-b", "")])
        self.destinations = [self.out / "matching_results.tsv", self.out / "candidate_pairs.tsv",
                             self.out / "person3_scored_outputs.stats.json", self.score_path]
        for path in self.destinations:
            path.write_bytes(b"previous " + path.name.encode("ascii"))
        self.previous = {path: path.read_bytes() for path in self.destinations}

    def set_candidates(self, rows):
        write_tsv(self.candidates, ["source1_entity_id", "candidate_entity_ids"], rows)

    def run_writer(self, **overrides):
        args = dict(candidate_input=self.candidates, source1=self.source1, source2=self.source2,
                    source3=self.source3, target_store=self.root / "targets.sqlite", output_dir=self.out,
                    threshold=0.82, max_matches=None, score_candidate_limit=None, batch_size=1,
                    rebuild_target_store=False, scores_output=self.score_path)
        args.update(overrides)
        with redirect_stdout(io.StringIO()):
            return output.write_scored_outputs(**args)

    def assert_preserved(self):
        for path, previous in self.previous.items():
            self.assertEqual(path.read_bytes(), previous, path.name)
        self.assertEqual(list(self.root.rglob("*.tmp")), [])
        self.assertEqual(list(self.root.rglob("*.backup")), [])

    def test_scored_subset_matches_export_and_tuning(self):
        scores = {"S2-wrong": 0.83, "S3-right": 0.99, "S2-extra": 0.95}
        with patch.object(output, "score_pair", side_effect=lambda source, target: scores[target.entity_id]) as scorer:
            stats = self.run_writer(score_candidate_limit=2, max_matches=1)
        self.assertEqual(scorer.call_count, 2)
        self.assertEqual(read_tsv(self.destinations[1]), [
            {"source1_entity_id": "S1-a", "candidate_entity_ids": "S2-wrong,S3-right"},
            {"source1_entity_id": "S1-b", "candidate_entity_ids": ""},
        ])
        self.assertEqual(read_tsv(self.destinations[0]), [
            {"source1_entity_id": "S1-a", "matched_entity_ids": "S3-right"},
            {"source1_entity_id": "S1-b", "matched_entity_ids": ""},
        ])
        exported = load_scores(self.score_path, {"S1-a", "S1-b"})
        self.assertEqual(evaluate_threshold({"S1-a": ("S3-right",), "S1-b": ()}, exported, 0.82, 1), (1, 1, 1))
        self.assertEqual(stats["candidate_links"], 2)
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(json.loads(self.destinations[2].read_text()), stats)

    def test_no_limit_retains_multiple_matches_and_roundtrips_scores(self):
        values = {"S2-wrong": 0.8199999, "S3-right": 0.90000001, "S2-extra": 0.95}
        with patch.object(output, "score_pair", side_effect=lambda source, target: values[target.entity_id]):
            self.run_writer()
        exported = load_scores(self.score_path, {"S1-a", "S1-b"})
        self.assertEqual(exported["S1-a"], values)
        selected = select_matches(tuple(values), exported["S1-a"], threshold=0.82, policy="scored-threshold")
        self.assertEqual(selected, ("S2-extra", "S3-right"))
        self.assertEqual(read_tsv(self.destinations[0])[0]["matched_entity_ids"], ",".join(selected))

    def test_candidate_input_can_be_canonical_output(self):
        self.destinations[1].write_bytes(self.candidates.read_bytes())
        self.run_writer(candidate_input=self.destinations[1], score_candidate_limit=1, scores_output=None)
        self.assertEqual(read_tsv(self.destinations[1])[0]["candidate_entity_ids"], "S2-wrong")
        self.assertEqual(self.score_path.read_bytes(), self.previous[self.score_path])

    def test_bad_candidates_fail_closed_and_preserve_all_outputs(self):
        cases = {
            "missing": [("S1-a", "S3-right")],
            "duplicate": [("S1-a", "S3-right"), ("S1-a", "S2-wrong"), ("S1-b", "")],
            "unknown_source": [("S1-a", "S3-right"), ("S1-unknown", "")],
            "unknown_target": [("S1-a", "S3-right"), ("S1-b", "S2-unknown")],
            "unknown_after_limit": [("S1-a", "S3-right,S2-unknown"), ("S1-b", "")],
            "bad_prefix": [("S1-a", "S1-b"), ("S1-b", "")],
            "duplicate_target": [("S1-a", "S3-right,S3-right"), ("S1-b", "")],
            "empty_element": [("S1-a", "S3-right,"), ("S1-b", "")],
            "missing_column": [("S1-a",)],
            "extra_column": [("S1-a", "S3-right", "extra")],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                self.set_candidates(rows)
                with self.assertRaises(ValueError):
                    self.run_writer(score_candidate_limit=1)
                self.assert_preserved()

    def test_duplicate_or_empty_source_roster_rejected(self):
        header = ["entity_id", "business_name", "business_address", "country"]
        row = ("S1-a", "Atlas", "12 Road", "US")
        for rows in ([row, row], []):
            write_tsv(self.source1, header, rows)
            with self.assertRaises(ValueError):
                self.run_writer()
            self.assert_preserved()

    def test_scoring_exception_preserves_outputs(self):
        with patch.object(output, "score_pair", side_effect=[0.99, RuntimeError("scorer failed")]):
            with self.assertRaisesRegex(RuntimeError, "scorer failed"):
                self.run_writer()
        self.assert_preserved()

    def test_unscored_candidates_do_not_build_features(self):
        original_features = output.make_features
        self.run_writer()
        with patch.object(output, "make_features", wraps=original_features) as features:
            self.run_writer(score_candidate_limit=1)
        target_ids = [call.args[0] for call in features.call_args_list if call.args[0].startswith(("S2-", "S3-"))]
        self.assertEqual(target_ids, ["S2-wrong"])

    def test_failed_backup_cleanup_does_not_report_failed_publication(self):
        original_unlink = Path.unlink

        def fail_backup(path, *args, **kwargs):
            if path.suffix == ".backup":
                raise PermissionError("backup busy")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_backup):
            stats = self.run_writer()
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(len(read_tsv(self.destinations[0])), 2)
        for backup in self.root.rglob("*.backup"):
            self.assertIn(backup.read_bytes(), self.previous.values())

    def test_promotion_error_rolls_back_already_replaced_files(self):
        original_replace = Path.replace

        def fail_stats(temporary, destination):
            if temporary.suffix == ".tmp" and destination == self.destinations[2]:
                raise OSError("publication failed")
            return original_replace(temporary, destination)

        with patch.object(Path, "replace", fail_stats):
            with self.assertRaisesRegex(OSError, "publication failed"):
                self.run_writer()
        self.assert_preserved()

    def test_promotion_error_removes_new_outputs(self):
        for path in self.destinations:
            path.unlink()
        original_replace = Path.replace

        def fail_scores(temporary, destination):
            if temporary.suffix == ".tmp" and destination == self.score_path:
                raise OSError("publication failed")
            return original_replace(temporary, destination)

        with patch.object(Path, "replace", fail_scores):
            with self.assertRaises(OSError):
                self.run_writer()
        self.assertTrue(all(not path.exists() for path in self.destinations))
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_output_collisions_rejected_without_changes(self):
        for path in (self.destinations[0], self.source1, self.candidates):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run_writer(scores_output=path)
            self.assert_preserved()

    def test_validation_truth_must_cover_requested_ids(self):
        truth_path = self.root / "truth.tsv"
        write_tsv(truth_path, ["source1_entity_id", "matched_entity_ids"], [("S1-a", "S3-right")])
        with self.assertRaisesRegex(ValueError, "absent"):
            load_truth(truth_path, {"S1-a", "S1-b"})
        write_tsv(truth_path, ["source1_entity_id", "matched_entity_ids"], [])
        with self.assertRaises(ValueError):
            load_truth(truth_path, None)
        ids = self.root / "ids.txt"
        for contents in ("", "S1-a\nS1-a\n"):
            ids.write_text(contents)
            with self.assertRaises(ValueError):
                load_validation_ids(ids)


if __name__ == "__main__":
    unittest.main()
