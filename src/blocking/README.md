# Member 2: blocking baseline

This code generates candidate lists from the **provided TSV data only**. It uses a disk-backed SQLite index so that the 10-million-row target corpus does not need to fit in RAM. It preserves IDs as strings and treats `country` as an open string label, including France.

## Run

From the repository root, using the official local resource package:

```powershell
$resource = 'dataset/6ab10eb3b23ba_student_resource/student_resource/dataset'
python src/blocking/generate_candidates.py `
  --source1 "$resource/test/test_source1.tsv" `
  --source2 "$resource/test/test_source2.tsv" `
  --source3 "$resource/test/test_source3.tsv" `
  --index 'data/candidates/person2/test_index.sqlite' `
  --output 'data/candidates/person2/candidate_pairs_v01.tsv'
```

The first run builds the index; later runs reuse it. Keep separate indexes for train and test. Use `--max-queries 5000` only for quick development, never for a file passed to Person 3 as the full candidate set.

Run `python src/blocking/smoke_test.py` for a small functional test.

## Contract for Person 3

The output is tab-separated with exactly `source1_entity_id` and `candidate_entity_ids`, one row per Source 1 record and comma-separated S2/S3 IDs. It is a **candidate set**, so Person 3 must score or filter it for `matching_results.tsv`. Person 3 copies the selected full version to canonical `output/candidate_pairs.tsv`; Member 2 does not write to `output/`.

`--max-block-size` skips extremely broad block keys and `--max-candidates` caps candidate lists. Defaults are 1000 and 200. The sidecar `.stats.json` records skip and cap counts. Those defaults need validation on the **full index** because small-sample blocks understate their real sizes.

## Current evidence and limitations

On a development sample of 5,000 labeled Source 1 rows and 58,477 Source 2/3 rows (all known positives for those queries plus a small background sample), known-positive pair recall improved from **80.7% to 94.8%** after adding address blocks and preserving non-Latin text. At least one true match was found for **99.4%** of non-singleton rows. This sample enriches the index with positives and has fewer competing records than the full corpus, so these figures **are not full-dataset recall estimates**. Full-index candidate recall and candidate volume remain the next gate.

The strongest missed-pair pattern was a business alias or transliterated name with a similar address. Address-only blocks recover many such cases. Further work should examine false negatives by country, large-block skips, and missing addresses before adding expensive fuzzy retrieval.
