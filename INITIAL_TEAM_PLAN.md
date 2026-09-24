# Initial Team Plan

This plan is for a three-person team working separately on three GitHub branches during the first phase of the Amazon ML Challenge 2026. The goal of this phase is to produce a clean baseline quickly, agree on shared interfaces, and avoid duplicate or conflicting work.

## First Phase Objective

Build an end-to-end baseline for business entity resolution:

- Read the provided tab-separated challenge data.
- Normalize business names and addresses.
- Generate candidate pairs through blocking.
- Score candidate pairs with a simple baseline.
- Produce `matching_results.tsv` in the required format.
- Save `candidate_pairs.tsv` for audit and final packaging.
- Keep the pipeline reproducible from raw data to submission.

## Branch Ownership

Each person works on a separate branch and owns one track.

| Person | Branch | Primary Ownership |
| --- | --- | --- |
| Person 1 | `eda-data-contract` | Data inspection, schema notes, validation split |
| Person 2 | `blocking-candidates` | Normalization, blocking, `candidate_pairs.tsv` |
| Person 3 | `baseline-submission` | Match scoring, singleton handling, `matching_results.tsv` |

Use small, frequent commits. Open pull requests early, even if the work is still rough, so the rest of the team can see the direction.

## Shared Folder Structure

Use this structure unless the team agrees to change it before coding:

```text
data/
  raw/                  # untouched challenge files, not committed if large
  processed/            # cleaned/intermediate files
  candidates/           # candidate pair files from blocking
  submissions/          # leaderboard-ready matching_results.tsv files
docs/
  notes/                # EDA notes, assumptions, experiment logs
src/
  data/                 # loading and schema utilities
  features/             # normalization and similarity features
  blocking/             # blocking logic
  modeling/             # scoring, thresholds, model code
  submission/           # output formatting and validation helpers
notebooks/
  scratch/              # personal notebooks only
```

Do not commit large raw data files unless the challenge explicitly requires it and the team agrees.

## Shared File Contracts

Agree to these interfaces before building real logic. This lets the three branches work in parallel.

### `candidate_pairs.tsv`

Produced by Person 2 and consumed by Person 3.

Required columns:

```text
source_1_id
candidate_source
candidate_id
block_key
name_similarity
address_similarity
candidate_score
```

Rules:

- All files must be tab-separated.
- All IDs must be read and written as strings.
- `candidate_source` must identify whether the candidate came from Source 2 or Source 3.
- `candidate_score` may start as a simple heuristic score and improve later.
- Keep one row per candidate pair.
- Include candidates broadly enough to protect recall; the matching stage will filter.

### `matching_results.tsv`

Produced by Person 3.

Expected shape:

```text
source_1_id
matches
```

Rules:

- One row per Source 1 entity.
- `matches` should contain the predicted matching Source 2 / Source 3 IDs in the required comma-separated format.
- Use an empty value for predicted singletons.
- Output must be validated before any leaderboard upload.

## Person 1: Data And EDA

Main responsibility: understand the data and prevent schema surprises.

Tasks:

- Inspect all provided files and column names.
- Confirm the exact ID columns for Source 1, Source 2, and Source 3.
- Confirm whether names and addresses are split or raw strings.
- Check missing values, duplicates, text noise, file sizes, and source-wise counts.
- Create a shared validation split from the training labels.
- Write short notes on common name/address patterns and likely singleton behavior.
- Build or document a simple local scoring approach that mirrors macro F0.5 as closely as possible.

Deliverables:

- `docs/notes/data_overview.md`
- `data/processed/validation_source_1_ids.txt`
- Initial schema notes for all teammates

## Person 2: Blocking And Candidate Generation

Main responsibility: generate high-recall candidate pairs.

Tasks:

- Build reusable normalization utilities for business names and addresses.
- Create simple blocking keys from normalized name and address text.
- Generate a first candidate file using exact or near-exact normalized name/address keys.
- Add broader blocking strategies after the first baseline works.
- Track candidate counts per Source 1 entity.
- Save every major candidate-generation version with a meaningful filename.

Deliverables:

- `src/features/normalization.py`
- `src/blocking/generate_candidates.py`
- `data/candidates/candidate_pairs.tsv`
- Notes explaining blocking keys and candidate counts

Guiding principle:

- Blocking should favor recall. Missing a true candidate here means the matching stage can never recover it.

## Person 3: Baseline, Matching, And Submission

Main responsibility: turn candidate pairs into valid predictions.

Tasks:

- Build a baseline matcher against the agreed `candidate_pairs.tsv` contract.
- Start with a simple threshold/rule-based scorer.
- Add explicit singleton handling instead of relying only on thresholds.
- Produce `matching_results.tsv` in the required format.
- Run the provided validation script before any submission.
- Keep a log of every leaderboard submission and local validation score.

Deliverables:

- `src/modeling/baseline_matcher.py`
- `src/submission/make_submission.py`
- `data/submissions/matching_results.tsv`
- `docs/notes/submission_log.md`

Guiding principle:

- Matching should favor precision. The metric is macro F0.5, so false matches hurt more than missed matches.

## Team Rules

### Data Rules

- Use only the provided challenge data.
- Do not use external databases, APIs, geocoders, maps, search engines, or lookup services.
- Treat IDs as strings everywhere to avoid leading-zero bugs.
- Read files with an explicit tab separator.
- Keep raw data unchanged.

### Git Rules

- Work only on your assigned branch.
- Pull from the main branch before starting each work session.
- Commit small, reviewable changes.
- Do not edit another person's core files without discussing it.
- Do not commit generated caches, temporary files, or large raw data.
- Move reusable notebook code into `.py` files before merging.

### Notebook Rules

- Notebooks are for personal exploration only.
- Do not build the main pipeline as one shared notebook.
- Keep reusable logic in `src/`.
- If a notebook result matters, summarize it in `docs/notes/`.

### Output Rules

- Never overwrite another teammate's output file.
- Use versioned output names while experimenting, for example:

```text
data/candidates/candidate_pairs_person2_v01.tsv
data/submissions/matching_results_person3_v01.tsv
```

- Copy the current best version to the canonical filename only after the team agrees:

```text
data/candidates/candidate_pairs.tsv
data/submissions/matching_results.tsv
```

### Validation Rules

- Use one shared validation split.
- Use one shared scoring script or scoring notebook.
- Do not compare model changes using different validation splits.
- Run the challenge validation script before upload.
- Log every leaderboard submission with:
  - timestamp
  - branch/commit
  - local validation score
  - public leaderboard score
  - short note on what changed

### Submission Rules

- Only one person should upload to the leaderboard at a time.
- Do not submit experiments that have not passed local validation.
- Keep the best robust submission, not just the best public leaderboard spike.
- Before the deadline, confirm the latest accepted `matching_results.tsv` is the intended final version.

## First 6-Hour Plan

1. Person 1 confirms schema, file sizes, and label/submission format.
2. Person 2 creates the first normalization functions and a simple candidate-pair generator.
3. Person 3 creates a stub pipeline that can read `candidate_pairs.tsv` and write `matching_results.tsv`.
4. Team agrees on the exact columns and ID formatting.
5. Person 2 produces the first real `candidate_pairs.tsv`.
6. Person 3 produces the first valid baseline `matching_results.tsv`.
7. Team validates the output and records the first baseline score.

## First Phase Success Criteria

- The data schema is documented.
- A shared validation split exists.
- `candidate_pairs.tsv` is generated reproducibly.
- `matching_results.tsv` is generated reproducibly.
- Singleton predictions are handled deliberately.
- At least one valid baseline submission is ready.
- The methodology document has been started before deeper experimentation.
