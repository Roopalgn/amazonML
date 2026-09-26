# Member 2: blocking baseline

This code generates candidate lists from the **provided TSV data only**. It uses a disk-backed SQLite index so that the 10-million-row target corpus does not need to fit in RAM. The fast generator uses DuckDB to join queries in batches against that index. It preserves IDs as strings and treats `country` as an open string label, including France.

## Run

From the repository root, using the official local resource package:

In PyCharm, select the local interpreter at `.venv/Scripts/python.exe`. This checkout's local `.venv` has Python 3.13 and DuckDB 1.5.5; it is ignored by Git. Teammates can create their own environment using `requirements.txt`.

```powershell
$resource = 'dataset/6ab10eb3b23ba_student_resource/student_resource/dataset'
python -m pip install -r src/blocking/requirements.txt
python src/blocking/generate_candidates_fast.py `
  --source1 "$resource/test/test_source1.tsv" `
  --source2 "$resource/test/test_source2.tsv" `
  --source3 "$resource/test/test_source3.tsv" `
  --index 'data/candidates/person2/test_index.sqlite' `
  --output 'data/candidates/person2/candidate_pairs_v01.tsv'
```

The first run builds the index; later runs reuse it. Keep separate indexes for train and test. DuckDB downloads its SQLite extension once as a software dependency into `data/candidates/person2/.duckdb_work/extensions/`. Use `--max-queries 5000` only for quick development, never for a file passed to Person 3 as the full candidate set.

If an index build is interrupted after loading all target rows but before the lookup index is ready, rerun the same command with `--resume-index` to finish sorting without rereading the source files. This is valid only when both source files were completely loaded.

Run `python src/blocking/smoke_test.py` for a small functional test.

## Contract for Person 3

The output is tab-separated with exactly `source1_entity_id` and `candidate_entity_ids`, one row per Source 1 record and comma-separated S2/S3 IDs. It is a **candidate set**, so Person 3 must score or filter it for `matching_results.tsv`. Person 3 copies the selected full version to canonical `output/candidate_pairs.tsv`; Member 2 does not write to `output/`.

`--max-block-size` skips extremely broad block keys and `--max-candidates` caps candidate lists. Defaults are 1000 and 200. The sidecar `.stats.json` records skip and cap counts. Those defaults need validation on the **full index** because small-sample blocks understate their real sizes.

## Version 02 fixed-validation experiment

Version 02 adds country-scoped address-only postal/locality/long-token blocks and bounded leading-zero variants for short numeric house tokens. It retains name-only keys when addresses are absent. Both generators now rank candidates by summed key weight, then descending target ID; serialization preserves that same order, so a stable prefix is a valid top-K truncation. The fast SQL rank order uses the same tie-break.

To reproduce the fixed-split experiment from the repository root (using the project Python environment):

```powershell
$resource = 'dataset/6ab10eb3b23ba_student_resource/student_resource/dataset'
python src/blocking/select_validation_source1.py --output 'data/candidates/person2/validation/validation_source1.tsv'
python src/blocking/generate_candidates_fast.py `
  --source1 'data/candidates/person2/validation/validation_source1.tsv' `
  --source2 "$resource/train/train_source2.tsv" `
  --source3 "$resource/train/train_source3.tsv" `
  --index 'data/candidates/person2/train_index_v02.sqlite' `
  --output 'data/candidates/person2/validation/candidate_pairs_train_validation_v02/candidate_pairs_train_validation_v02.tsv' `
  --batch-size 10000 --max-block-size 1000 --max-candidates 1000
python src/blocking/truncate_candidates.py `
  --input 'data/candidates/person2/validation/candidate_pairs_train_validation_v02/candidate_pairs_train_validation_v02.tsv' `
  --max-candidates 500 `
  --output 'data/candidates/person2/validation/candidate_pairs_train_validation_v02/candidate_pairs_train_validation_v02_cap500.tsv'
```

For the published names after those two commands, move the raw 1,000-cap TSV aside as `candidate_pairs_train_validation_v02_cap1000.tsv`, then move the cap-500 output to `candidate_pairs_train_validation_v02.tsv`.

The fixed validation ID list SHA-256 is `DBF52E20D318F152AFCA76CE77202C75208D46BDB7FFE07D207E72C6ADF51CBC`. The recommended v02 validation artifact has 441,287 rows, 161,214,669 candidate links, a maximum of 500 links per query, and SHA-256 `4d0d7db0b5e459c7fd59cd55150ecf7734af37719ed68723cd42cb8a8d7f8d9f`. Its ZIP SHA-256 is `15c0281fe8d41b525259751fe9221dfc480937a8e78fc6e480c7645190a3136a`. Full recall/oracle comparisons and known limitations are in `docs/person2_v02_report.md`.

## Current evidence and limitations

On a development sample of 5,000 labeled Source 1 rows and 58,477 Source 2/3 rows (all known positives for those queries plus a small background sample), known-positive pair recall improved from **80.7% to 94.8%** after adding address blocks and preserving non-Latin text. At least one true match was found for **99.4%** of non-singleton rows. This sample enriches the index with positives and has fewer competing records than the full corpus, so these figures **are not full-dataset recall estimates**. Full-index candidate recall and candidate volume remain the next gate.

The strongest missed-pair pattern was a business alias or transliterated name with a similar address. Address-only blocks recover many such cases. Further work should examine false negatives by country, large-block skips, and missing addresses before adding expensive fuzzy retrieval.

## Full test run, version 01

The fast generator completed all **1,732,544** test Source 1 rows, producing **253,317,953** candidate links in `data/candidates/person2/candidate_pairs_v01.tsv` (3,287,320,004 bytes). A streaming structural check confirmed one row per test Source 1 ID in source order, no duplicate IDs within lists, S2/S3 prefixes only, and a maximum of 200 candidates per row. It included all **259,452 France** rows. The file is local and ignored by Git; share it privately with Person 3, then Person 3 copies the chosen version into canonical `output/candidate_pairs.tsv`.

The 200-candidate cap affected **1,037,239** queries (59.9%). This is a material recall risk. The current version is a valid candidate-generation baseline; the separate training-index check below quantifies its recall on labeled rows.

## Full training-index recall check

A separate index of all **10,320,219** training Source 2/3 records was used to query the first 5,000 ground-truth Source 1 IDs (2,982 US and 2,018 India). This sample is the first 5,000 ground-truth rows, not a randomized validation split; Person 1's fixed split remains the evaluation authority.

| Setting | Pair recall overall | US | India | Candidate links per 5,000 queries |
| --- | ---: | ---: | ---: | ---: |
| block size 1,000; cap 200 (v01) | 91.94% | 94.99% | 87.52% | 693,752 |
| block size 1,000; cap 500 | 93.41% | 96.26% | 89.29% | 1,338,069 |
| block size 1,000; cap 1,000 | 94.18% | 96.73% | 90.48% | 1,889,443 |
| block size 3,000; cap 200 | 91.72% | 94.53% | 87.65% | 716,001 |

At cap 200, at least one true match was retrieved for 99.05% of non-singleton rows, but all true matches were retrieved for only 78.40%. Raising the block-size limit while keeping the cap at 200 slightly reduced recall; it admitted more weak candidates that displaced stronger ones.

The next Member 2 experiment should add carefully sized **address-only locality pairs** and **leading-zero-normalized house-number keys**, then measure full-index block sizes and recall on Person 1's fixed split. In a diagnostic of the 1,004 true pairs missed even with cap 1,000, 716 shared at least one prospective new key. This is only a potential gain: common blocks and candidate caps may prevent retrieval. Avoid rebuilding the test index until a training-index experiment shows a clear improvement.

The existing v01 test file is a valid first handoff for Person 3. It can be shared as `data/candidates/person2/candidate_pairs_v01.zip` (1,383,094,551 bytes), which contains `candidate_pairs.tsv`; the ZIP and TSV stay outside Git. Verify the ZIP SHA-256 against `candidate_pairs_v01.zip.sha256` after transfer.
