# Submission Log

Person 3 owns this log. Add one row for every local validation run that may lead to a leaderboard upload, and every actual leaderboard upload.

## Daily Budget

The team has 5 leaderboard submissions per day. Keep at least one slot reserved until late in the day.

| Slot | Intended Use | Status |
| --- | --- | --- |
| 1 | First real validator-passing baseline | unused |
| 2 | Blocking improvement | unused |
| 3 | Matching threshold/model improvement | unused |
| 4 | Conservative precision-focused variant | unused |
| 5 | Reserved final slot | unused |

## Runs

| Timestamp IST | Submission # | Branch | Commit | Candidate Version | Matching Version | Local Macro F0.5 | Public Score | Change Summary | Decision Next |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| 2026-09-25 12:24:51 +05:30 | local-only | `main` | `e48449d` | all-empty stub | all-singleton stub | TBD | N/A | Generated `output/candidate_pairs.tsv` and `output/matching_results.tsv` with one empty row per test Source 1 entity; official validator passed. | Replace with Person 2 candidates before first real submission. |
| 2026-09-25 15:40:47 +05:30 | local-only | `main` | `fee8b5a` | Person 2 smoke candidates | heuristic scorer smoke run | N/A | N/A | Verified `src/submission/make_scored_outputs.py` can score a Person 2 list-format candidate handoff and produce official output files. | Wait for full `candidate_pairs_v01.tsv` or zip, then run streaming scorer on the full test handoff. |
