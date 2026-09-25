# Person 3 Runbook: Matching And Submission

This runbook covers the Person 3 lane: matching, singleton decisions, output generation, validation, and submission logging.

## Owned Files

Person 3 owns:

```text
src/matching/
src/submission/
output/
docs/submission_log.md
docs/person3_runbook.md
```

Do not edit Person 1's validation split/scorer or Person 2's blocking code to test a matching idea. Consume their outputs through files.

## Generate Emergency All-Singleton Outputs

Use this only as a validator-safe stub or emergency baseline.

```bash
python src/submission/make_outputs.py
```

This writes:

```text
output/candidate_pairs.tsv
output/matching_results.tsv
```

Both files contain one row for every test Source 1 entity and empty ID lists.

## Generate Outputs From Person 2 Candidates

Person 2 should provide official list-format candidates:

```text
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	
```

Run:

```bash
python src/submission/make_outputs.py \
  --candidate-input data/candidates/person2/candidate_pairs_v01.tsv
```

Without scores, the default policy remains precision-safe and predicts empty matches. Use this to validate candidate formatting before matching scores exist.

## Generate Outputs From Scored Candidates

Scored candidate file contract:

```text
source1_entity_id	candidate_entity_id	score
S1-00001	S2-00047	0.982
S1-00001	S3-00812	0.941
```

Run:

```bash
python src/submission/make_outputs.py \
  --candidate-input data/candidates/person2/candidate_pairs_v01.tsv \
  --scores data/features/scored_candidates_v01.tsv \
  --threshold 0.90 \
  --policy scored-threshold
```

Useful options:

- `--policy empty`: predict no matches.
- `--policy pass-through`: write all candidates as matches; debugging only.
- `--policy scored-threshold`: keep candidates at or above threshold.
- `--policy top-k --max-matches N`: keep highest scoring candidates above threshold.

## Search Thresholds

Once Person 1 publishes a fixed validation split and a scored candidate file exists:

```bash
python src/matching/threshold_search.py \
  --ground-truth dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv \
  --scores data/features/scored_candidates_validation_v01.tsv \
  --validation-ids data/processed/validation_source1_ids.txt \
  --start 0.50 \
  --stop 0.99 \
  --step 0.01
```

Use the best validation threshold as a candidate for leaderboard submission, but keep a precision-focused variant for the reserved daily slot.

## Generate Scored Outputs Directly From Person 2 Candidates

For the full handoff file from Person 2, prefer the streaming scorer. It builds a local SQLite cache of Source 2/3 records, streams `candidate_pairs.tsv` in batches, and writes both official output files.

```bash
python src/submission/make_scored_outputs.py \
  --candidate-input data/candidates/person2/candidate_pairs_v01.tsv \
  --threshold 0.82 \
  --score-candidate-limit 50 \
  --output-dir output
```

Optional useful flags:

- `--rebuild-target-store`: rebuild the Source 2/3 cache if the dataset changes.
- `--score-candidate-limit N`: score only the first N candidates per Source 1 while preserving the full candidate list in `output/candidate_pairs.tsv`. Use this for rapid leaderboard iteration on the huge v01 file.
- `--max-matches N`: cap final matches per Source 1 after scoring.
- `--scores-output data/features/scored_candidates_v01.tsv`: also write per-candidate scores. This can be very large on the full candidate file.
- `--batch-size N`: tune memory/runtime tradeoff.

The first full run is expected to take time because it indexes the full test Source 2/3 files.

## Validate Before Upload

Run from the repo root:

```bash
python dataset/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

The validator must pass before any leaderboard upload.

## Submission Discipline

- The team has 5 submissions per day.
- Log every local validation run that could lead to an upload.
- Log every actual upload.
- Archive the exact output files and commit hash for any leaderboard submission.
- Do not use a leaderboard slot for format debugging.
