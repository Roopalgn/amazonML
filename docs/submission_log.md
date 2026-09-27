# Submission Log

Person 3 owns this log. Add one row for every local validation run that may lead to a leaderboard upload, and every actual leaderboard upload.

## Daily Budget

The team has 5 leaderboard submissions per day. Keep at least one slot reserved until late in the day.

| Slot | Intended Use | Status |
| --- | --- | --- |
| 1 | First real validator-passing baseline | submitted: user reports 0.554, rank 750/1100 |
| 2 | Blocking improvement | unused |
| 3 | Matching threshold/model improvement | unused |
| 4 | Conservative precision-focused variant | unused |
| 5 | Reserved final slot | unused |

## Runs

| Timestamp IST | Submission # | Branch | Commit | Candidate Version | Matching Version | Local Macro F0.5 | Public Score | Change Summary | Decision Next |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| 2026-09-25 12:24:51 +05:30 | local-only | `main` | `e48449d` | all-empty stub | all-singleton stub | TBD | N/A | Generated `output/candidate_pairs.tsv` and `output/matching_results.tsv` with one empty row per test Source 1 entity; official validator passed. | Replace with Person 2 candidates before first real submission. |
| 2026-09-25 15:40:47 +05:30 | local-only | `main` | `fee8b5a` | Person 2 smoke candidates | heuristic scorer smoke run | N/A | N/A | Verified `src/submission/make_scored_outputs.py` can score a Person 2 list-format candidate handoff and produce official output files. | Wait for full `candidate_pairs_v01.tsv` or zip, then run streaming scorer on the full test handoff. |
| 2026-09-25 18:26:12 +05:30 | ready-to-upload | `main` | `3c7d98d` | `candidate_pairs_v01.tsv` | top1 first-candidate baseline | TBD | TBD | Generated fast first real submission by taking the first candidate ID per Source 1 row. `matching_results.tsv` has 1,732,544 rows, 1,732,417 non-empty rows, and passed upload-file validation. SHA-256: `B5BCB68AFD2569A634A6962FE552830130AC67A18D8FAE49DA0C4FF72C2FE89D`. | Upload this as the first real baseline, then use the public score to decide whether to tune candidate count/threshold. |
| 2026-09-25, upload time not supplied | 1, reported by user | `main` | `d16235d` | `candidate_pairs_v01.tsv` | top1 first-candidate baseline | not measured | 0.554 | User reports rank 750/1100 and leader score 0.986955. These leaderboard figures have not been independently verified. | Replace top1 with locally validated multi-match classification and singleton abstention. Do not spend another slot on an unmeasured candidate-count change. |
| 2026-09-26, upload time not supplied | reported by user | `main` | uncommitted supervised pipeline | supervised-v1 retrieval | LightGBM, threshold 0.91 | 0.927182 gated holdout | 0.895 | User reports rank in the 1900s. Uploaded artifact is preserved in `output/ec2_supervised/`. | Audit retrieval losses, false merges and the France domain shift. |
| 2026-09-26 | ready, not confirmed submitted | `main` | uncommitted | unchanged supervised-v1 candidates | ownership-v1, log-odds margin 0.5 | 0.930701 holdout; 0.938873 tune | pending | Full-corpus competition check; 1,071,892 links removed. `output/ownership_v1/matching_results.tsv` passed full validation, zero errors. | Obtain actual public score while evaluating corrected multilingual retrieval and corpus-aware features. |
