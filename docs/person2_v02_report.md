# Person 2 v02 blocking experiment

## Recommendation

Recommend the v02 **cap-500 fixed-validation candidate file** for Person 3's validation and threshold/model experiments. It materially improves the candidate oracle over v01 and improves India recall while using substantially fewer links than cap 1,000. This is a retrieval recommendation only; it does not establish a leaderboard score or guarantee 0.99+.

## Changes

- Added country-scoped address-only postal, locality, and long-token blocking keys.
- Added bounded leading-zero variants for short numeric house/address tokens.
- Preserved name-based routes when one or both addresses are blank.
- Made candidate ordering deterministic: descending accumulated key weight, then descending target ID. The selected top-K and serialized list now share the same order in the SQLite and DuckDB implementations.
- Added smoke coverage for blank names, leading-zero variants, equal-score order, and country scoping.

All keys use only supplied challenge fields. The country component remains open-set; the implementation does not hard-code a country allowlist.

## Fixed validation results

The fixed ID file has 441,287 Source-1 IDs (SHA-256 `DBF52E20D318F152AFCA76CE77202C75208D46BDB7FFE07D207E72C6ADF51CBC`). Validation candidates use the train Source-1/Source-2/Source-3 files and the exact fixed IDs. No new split was created.

| Candidate setting | Links | Max list | Pair recall | Oracle macro F0.5 |
| --- | ---: | ---: | ---: | ---: |
| v01, cap 200 | 75,450,093 | 200 | 0.918594 | 0.968053 |
| v02, cap 500 | 161,214,669 | 500 | 0.951704 | 0.982187 |
| v02, cap 1,000 | 251,282,906 | 1,000 | 0.960616 | 0.985621 |

The ordered top-50 prefix is the same for the cap-500 and cap-1,000 files: pair recall is **0.897650** and oracle macro F0.5 is **0.956907**, compared with **0.872411 / 0.942269** for v01. At cap 500, India pair recall is **0.917845** and oracle macro F0.5 is **0.967828**; US pair recall is **0.974305** and oracle macro F0.5 is **0.991749**. At cap 1,000, India is **0.930021 / 0.972952** and US is **0.981039 / 0.994058** for pair recall / oracle macro F0.5.

The full cap-1,000 ranked-list evaluation also measured these list-prefix results (pair recall / candidate-oracle macro F0.5):

| Prefix | Pair recall | Oracle macro F0.5 |
| ---: | ---: | ---: |
| 20 | 0.877406 | 0.947272 |
| 50 | 0.897650 | 0.956907 |
| 100 | 0.922870 | 0.970372 |
| 200 | 0.937498 | 0.976560 |
| Full 1,000 | 0.960616 | 0.985621 |

Country oracle macro F0.5 at prefixes 20 / 50 / 100 / 200 / full 1,000 was **India: 0.907239 / 0.923008 / 0.951691 / 0.959874 / 0.972952**, and **US: 0.973929 / 0.979480 / 0.982811 / 0.987670 / 0.994058**. Pair recall at prefix 50 / full 1,000 was **India: 0.833727 / 0.930021**, and **US: 0.940319 / 0.981039**. Across all 441,287 validation queries, 414,800 had at least one retrieved true target.

Cap 500 contains 51,381,645 S2 and 109,833,024 S3 links (31.87% / 68.13%); cap 1,000 contains 105,448,254 S2 and 145,834,652 S3 links (41.96% / 58.04%). At cap 500, true-link recall is 0.945594 for S2 truths and 0.957415 for S3 truths, so S3's larger candidate volume did not crowd out S2 truth retrieval. The cap-500 file is about 2.08 GB uncompressed versus 3.24 GB at cap 1,000. Its companion ZIP is about 934 MB.

The structural verification found one row for every fixed ID, no missing or duplicate query IDs, no duplicate candidates per row, only S2/S3 target prefixes, and candidate order preserved from the ranked cap-1,000 source. The fixed train validation subset has no France source rows; France handling remains open-set in the code, and the separate v01 test artifact retains all 259,452 France test rows.

## Artifacts and reproducibility

- Recommended validation TSV SHA-256: `4d0d7db0b5e459c7fd59cd55150ecf7734af37719ed68723cd42cb8a8d7f8d9f`.
- Recommended validation ZIP SHA-256: `15c0281fe8d41b525259751fe9221dfc480937a8e78fc6e480c7645190a3136a`.
- Exact generator and truncation commands, source files, settings, ID checksum, and artifact hashes are recorded in `person2_v02_manifest.json`.
- The generated validation TSV/ZIP and indexes are local ignored artifacts; they are not part of the Git branch.
- Full test-set v02 candidates were not generated during this validation run. The existing full test v01 candidate file remains available; use v02 code and a separate full test index to create a test artifact before replacing it.

## Pair-scoring diagnostic

I trained a scratch LightGBM pair scorer on v02's top 50 candidates per query, using the existing matching features and only labels from the supplied fixed-split training data. The split is deterministic by Source-1 numeric ID: train on IDs `% 5 != 0`, tune on `% 10 == 0`, and report on `% 10 == 5`. A 300,000-positive/300,000-negative model (450 trees, 47 leaves) scored **0.866282** on the 43,977-query fixed holdout at threshold 0.95 and max 10 predictions. The v01-candidate LightGBM result under the same procedure was **0.857645**; the v01 logistic comparator was **0.762945**.

I then selected a higher-capacity model using tuning IDs only: 800,000 sampled pairs per class, 700 trees, 63 leaves, and learning rate 0.035. The refined tuning sweep selected threshold **0.957** and max 10 predictions, with tuning macro F0.5 **0.869156** and fixed-holdout macro F0.5 **0.868436**. This is **0.002154** above the v02 default model and **0.010791** above the v01-candidate LightGBM under the same procedure. A 1,000,000-per-class sample and adding the heuristic score as another feature did not improve the tuning score. These are offline fixed-split diagnostics, not public leaderboard scores and not directly comparable to the teammate's reported 0.895 if its evaluation procedure differs. The model artifacts and experiments remain local; no Person 3 production matching/submission logic was changed.

## Validation and limits

`src/blocking/smoke_test.py` passes, and all `src/blocking/` modules compile. Full validation candidate structure and checksums pass. Candidate-oracle F0.5 assumes an oracle that knows which retrieved targets are true; it is a ceiling, not a model score. The top-50 oracle ceiling of 0.956907 means this candidate list alone cannot support a 0.998 top-50 validation score. No public leaderboard result is inferred from the offline split.
