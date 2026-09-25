# Initial Three-Person Team Plan

This plan is based on the local challenge package in `dataset/6ab10eb3b23ba_student_resource/student_resource/`. The dataset is large, the leaderboard allows only 5 submissions per day, and the three teammates will work on separate GitHub branches. The first phase should prioritize a valid end-to-end pipeline, controlled leaderboard iteration, and strict separation of ownership.

## Current Dataset Facts

| File | Rows | Countries | Notes |
| --- | ---: | --- | --- |
| `dataset/train/train_source1.tsv` | 2,206,821 | India, US | Reference source, no missing fields observed |
| `dataset/train/train_source2.tsv` | 5,034,616 | India, US | 168,967 missing `business_address` values |
| `dataset/train/train_source3.tsv` | 5,285,603 | India, US | 175,916 missing `business_address` values |
| `dataset/test/test_source1.tsv` | 1,732,544 | France, India, US | Every row must appear in `matching_results.tsv` |
| `dataset/test/test_source2.tsv` | 4,887,273 | France, India, US | 129,408 missing `business_address` values |
| `dataset/test/test_source3.tsv` | 5,082,316 | France, India, US | 136,098 missing `business_address` values |
| `dataset/train/train_ground_truth.tsv` | 2,206,821 | India, US | 123,247 singletons, about 5.58% |

Ground-truth match count distribution:

```text
0: 123247
1: 119157
2: 375212
3: 530841
4: 484115
5: 321957
6: 164868
7: 63968
8: 18680
9: 4205
10: 534
11: 37
```

Important implications:

- Do not create all pairwise comparisons. The full Cartesian space is impossible.
- Use `country` as a blocking feature, but keep the logic open-set because test includes `France`, which is unseen in training.
- Address handling must tolerate missing addresses in Source 2 and Source 3.
- The official output files are list-based, one row per Source 1 entity, not one row per candidate pair.
- The public leaderboard is only directional; final ranking depends on the private leaderboard.

## Phase 1 Objective

Within the first working block, produce a reproducible baseline that:

- Reads the official TSV files with `sep="\t"`.
- Builds candidate lists for all test Source 1 entities.
- Generates `output/candidate_pairs.tsv`.
- Generates `output/matching_results.tsv`.
- Passes the official validator.
- Uses no external lookups, APIs, geocoding, or third-party business databases.
- Logs local validation and leaderboard results for every submission.

## Branch Ownership

Each person works on one branch. Nobody should edit another person's owned files without agreement.

| Person | Branch | Main Output | Primary Role |
| --- | --- | --- | --- |
| Person 1 | `eda-validation-contract` | validation split, scorer, dataset notes | Data profiling and local evaluation |
| Person 2 | `blocking-candidate-generation` | candidate generation code and candidate lists | Blocking and retrieval |
| Person 3 | `matching-submission-pipeline` | scoring, thresholding, final submission files | Matching and leaderboard submissions |

`main` should stay stable. Merge only code that runs, has a clear owner, and does not break the agreed command-line interfaces.

## File Ownership Rules

To prevent one person's work from affecting another's task, use owned directories.

| Path | Owner | Rule |
| --- | --- | --- |
| `src/data/` | Person 1 | Shared loading/schema utilities only |
| `src/eval/` | Person 1 | Local scorer and validation split logic |
| `docs/dataset_profile.md` | Person 1 | Dataset observations and schema notes |
| `src/blocking/` | Person 2 | Candidate generation and blocking experiments |
| `data/candidates/person2/` | Person 2 | Candidate experiment outputs |
| `src/matching/` | Person 3 | Candidate scoring, thresholds, singleton decisions |
| `src/submission/` | Person 3 | Output formatting and submission packaging |
| `output/` | Person 3 | Canonical files used for validation/submission |
| `docs/submission_log.md` | Person 3 | Every local and leaderboard attempt |
| `notebooks/<person-name>/` | Individual | Scratch only, never the production pipeline |

Canonical files in `output/` are updated only by Person 3 after pulling the latest candidate file from Person 2.

## Official Output Contracts

### `output/matching_results.tsv`

This is the only file uploaded to the leaderboard.

```text
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	
```

Rules:

- Tab-separated.
- Exact header: `source1_entity_id`, `matched_entity_ids`.
- One row for every test Source 1 entity.
- Empty `matched_entity_ids` means predicted singleton.
- Matched IDs must only be from test Source 2 or Source 3.
- No duplicate IDs inside a row.

### `output/candidate_pairs.tsv`

This is included in the final zip and audited. It should be the candidate set actually fed into the matching model.

```text
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	
```

Rules:

- Tab-separated.
- Exact header: `source1_entity_id`, `candidate_entity_ids`.
- One row for every test Source 1 entity.
- Final matches should be a subset of candidates.
- This file is not the internal feature table. If Person 2 or Person 3 needs row-per-pair features, save them separately under `data/candidates/person2/` or `data/features/`.

## Required Validation Command

Run from `dataset/6ab10eb3b23ba_student_resource/student_resource/`:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

For deeper diagnostics, use `--check-ids`, but expect higher memory use on the full test set.

## Submission Budget Policy

The team has 5 leaderboard submissions per day. Use them deliberately but early enough to learn from feedback.

Daily submission budget:

| Slot | Purpose | Rule |
| --- | --- | --- |
| 1 | First valid baseline | Submit as soon as validator passes |
| 2 | Blocking improvement | Submit only if candidate recall/local score improves |
| 3 | Matching threshold/model change | Submit only if local validation improves or error analysis justifies it |
| 4 | Conservative robust candidate | Submit a precision-focused version for private leaderboard safety |
| 5 | Reserved final slot | Keep unused until late day unless a major improvement appears |

Submission rules:

- Person 3 is the only uploader unless the team explicitly delegates.
- Every submission must have a row in `docs/submission_log.md`.
- Never submit a file that fails the validator.
- Never spend a submission on formatting checks; the local validator exists for that.
- Do not chase tiny public leaderboard gains if local validation suggests overfitting.
- Keep the best public submission file archived with the commit hash that produced it.

Submission log fields:

```text
timestamp_ist
submission_number_for_day
branch
commit_hash
candidate_version
matching_version
local_macro_f0_5
public_score
change_summary
decision_next
```

## Person 1: Data, Validation, And Scoring

Goal: make the team compare experiments consistently.

Tasks:

- Confirm schema and row counts from the official files.
- Create a fixed validation split from `train_ground_truth.tsv`.
- Preserve country distribution in validation if feasible.
- Include France-aware notes even though France has no training labels.
- Implement or document a local macro F0.5 scorer.
- Track singleton performance separately from non-singleton performance.
- Profile name/address noise by country.
- Create a small sampled development set for fast debugging.

Deliverables:

- `docs/dataset_profile.md`
- `data/processed/validation_source1_ids.txt`
- `data/processed/dev_sample_source1_ids.txt`
- `src/eval/macro_f05.py`
- `src/data/io.py`

Do not:

- Change Person 2's blocking logic.
- Change Person 3's submission thresholds.
- Create a different validation split after the team starts comparing results.

## Person 2: Blocking And Candidate Generation

Goal: generate high-recall candidates without exploding memory or runtime.

Tasks:

- Build normalization for `business_name` and `business_address`.
- Use country-aware blocking while allowing unseen country labels.
- Handle missing Source 2/3 addresses with name-heavy fallback blocks.
- Start with cheap blocks:
  - normalized country + strong name token
  - normalized country + address numeric tokens
  - normalized country + postal/PIN-like tokens where present
  - normalized country + first significant name token + locality token
- Add fuzzy/TF-IDF retrieval only after the first valid candidate file exists.
- Report candidate coverage on training validation.
- Export the official list-format `candidate_pairs.tsv`.
- Save internal pair/feature files separately.

Deliverables:

- `src/blocking/generate_candidates.py`
- `src/blocking/block_keys.py`
- `data/candidates/person2/candidate_pairs_<version>.tsv`
- `data/candidates/person2/candidate_stats_<version>.md`

Do not:

- Write directly to canonical `output/candidate_pairs.tsv`.
- Filter candidates so tightly that matching cannot recover recall.
- Use France-specific assumptions learned from outside the dataset.

## Person 3: Matching, Singleton Logic, And Submission

Goal: turn Person 2's candidates into validated leaderboard files.

Tasks:

- Consume only the agreed candidate list/feature contracts.
- Build a first precision-focused baseline quickly.
- Optimize thresholds on Person 1's fixed validation split.
- Treat singleton prediction as a first-class decision.
- Generate both official files under `output/`.
- Run the validator before every upload.
- Maintain the submission log and archive submitted files.

Deliverables:

- `src/matching/baseline_matcher.py`
- `src/matching/threshold_search.py`
- `src/submission/make_outputs.py`
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `docs/submission_log.md`

Do not:

- Modify Person 1's validation split.
- Modify Person 2's candidate generation code just to test a model idea.
- Upload without logging the exact commit and file versions.

## Merge And Integration Flow

Use this flow to keep the three tracks independent:

1. Person 1 merges the data contract and scorer first.
2. Person 2 builds candidates against Person 1's loader/schema contract.
3. Person 3 builds a stub matcher that can run even with a tiny candidate sample.
4. Person 2 publishes versioned candidates under `data/candidates/person2/`.
5. Person 3 copies the selected candidate version into `output/candidate_pairs.tsv`, creates `output/matching_results.tsv`, validates, and submits.
6. Only stable, reusable code is merged into `main`.

No teammate should depend on unmerged notebook state from another branch.

## First 6-Hour Execution Plan

1. Person 1 creates the fixed validation split and local scorer.
2. Person 2 creates an exact/near-exact blocking baseline using normalized names and address tokens.
3. Person 3 creates a stub matcher that predicts empty lists for all test Source 1 rows, then a simple thresholded matcher once candidates arrive.
4. Person 3 validates the all-singleton/stub output locally but does not submit it unless needed as an emergency baseline.
5. Person 2 exports the first real candidate version.
6. Person 3 generates the first real `matching_results.tsv`.
7. Team spends Submission Slot 1 on the first validator-passing real baseline.
8. Team reviews public score, local validation score, singleton rate, and false-positive risk before spending Slot 2.

## First Phase Success Criteria

- The official validator passes.
- One real leaderboard submission has been made and logged.
- The local scorer exists and is shared.
- Candidate generation is reproducible.
- Final matches are a subset of candidates.
- Every person's branch has clear owned files and no cross-branch overwrite risk.
- The team still has at least one reserved submission slot for the day.
