# Integration checklist

Use this checklist before merging a teammate branch or uploading a submission.

## Person 2 handoff

- [ ] Candidate generator code is in `src/blocking/` and has a branch/commit.
- [ ] Candidate version is new; v01 was not overwritten.
- [ ] Validation candidate file contains exactly the fixed validation IDs.
- [ ] Candidate order is deterministic and matches ranking order.
- [ ] Stats include candidate links, caps, block skips, recall, and oracle macro F0.5.
- [ ] India and US results are reported separately.
- [ ] Candidate TSV/ZIP and checksum were transferred through Drive.
- [ ] No dataset, index, or large generated artifact was pushed to GitHub.

## Person 3 handoff

- [ ] Matcher was evaluated against a named candidate version and checksum.
- [ ] Threshold and maximum-match policy are recorded.
- [ ] Tuning and confirmation scores are both reported.
- [ ] Singleton and non-singleton scores are reported.
- [ ] `output/matching_results.tsv` has one row per test Source-1 ID.
- [ ] `output/candidate_pairs.tsv` is exactly the candidate list scored.
- [ ] Every predicted match is present in the candidate file.
- [ ] Official validator passes, preferably with `--check-ids`.
- [ ] Submission log records the commit, hashes, local score, and public score.

## Final upload

Only `output/matching_results.tsv` is uploaded to the leaderboard. Preserve
the candidate file, model, statistics, manifest, and checksum for audit and
the final reproducibility package.
