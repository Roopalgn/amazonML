"""Small correctness tests for the disk-backed supervised path."""

import csv
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np

from src.eval.macro_f05 import entity_f05
from src.matching.pair_features import FEATURE_NAMES, FAST_FEATURE_INDICES, fast_features, pair_features
from src.matching.supervised_pipeline import metric_arrays, prepare, retrieve


class FeatureTests(unittest.TestCase):
    def test_features_are_finite_and_missing_is_explicit(self):
        rows = [("S1-a", "S2-a", "acme ltd", "12 main road", "acme ltd", "12 main road"),
                ("S1-b", "S3-b", "ecole marina", "", "ecole marina", "")]
        features = pair_features(rows)
        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[0, FEATURE_NAMES.index("name_exact")], 1)
        self.assertEqual(features[1, FEATURE_NAMES.index("address_missing")], 1)
        self.assertEqual(features[1, FEATURE_NAMES.index("address_exact")], 0)
        np.testing.assert_allclose(fast_features(rows),features[:,FAST_FEATURE_INDICES])

    def test_macro_metric_includes_missing_candidates_and_singletons(self):
        truth = [{"a", "b"}, set(), {"c"}, {"d"}]
        predictions = [{"a", "wrong"}, set(), {"c"}, set()]
        expected = sum(entity_f05(t, p) for t, p in zip(truth, predictions)) / 4
        value, _, _ = metric_arrays(np.array([1, 0, 1]), np.array([.9, .8, .9]),
            np.array([0, 0, 2]), np.array([2, 0, 1, 1]), np.ones(4, dtype=bool), .5)
        self.assertAlmostEqual(value, expected)


class RetrievalTests(unittest.TestCase):
    def test_retrieval_is_country_restricted_and_preserves_short_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            sources = [
                [("S1-a", "Acme Ltd", "12 Main Road, Exampletown", "US"),
                 ("S1-b", "I 4", "81 Longstreet Samplecity", "France")],
                [("S2-a", "ACME LIMITED", "12 Main Road, Exampletown", "US"),
                 ("S2-wrong", "Acme Ltd", "12 Main Road, Exampletown", "France")],
                [("S3-b", "I 4", "81 Longstreet Samplecity", "France")],
            ]
            for source, rows in enumerate(sources, 1):
                with (root / "train" / f"train_source{source}.tsv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, delimiter="\t")
                    writer.writerow(["entity_id", "business_name", "business_address", "country"])
                    writer.writerows(rows)
            con = duckdb.connect()
            try:
                prepare(con, root, "train", root)
                con.execute("CREATE TABLE queries AS SELECT *, 'fit' AS role FROM qfull")
                retrieve(con, root, "train")
                pairs = set(con.execute("SELECT * FROM pairs").fetchall())
                self.assertIn(("S1-a", "S2-a"), pairs)
                self.assertIn(("S1-b", "S3-b"), pairs)
                self.assertNotIn(("S1-a", "S2-wrong"), pairs)
            finally:
                con.close()


if __name__ == "__main__":
    unittest.main()
