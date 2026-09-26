# Fast-track plan for the next entity-resolution iteration

**Target:** improve the public score from 0.720829 toward 0.98 without violating the challenge rules. This is an aggressive target, not a promised outcome. The current candidate pipeline cannot reach it on the fixed validation set, even with a perfect matcher; candidate recall must improve first.

## 1. Current baseline and what it means

Freeze these as the comparison baseline; do not overwrite them during experiments:

- Public score reported by the team: **0.720829**.
- Fixed validation: `data/processed/validation_source1_ids.txt` (441,287 Source-1 IDs); keep this split unchanged.
- Current Person-2 validation candidates: `data/candidates/person2/validation/candidate_pairs_train_validation_v01/candidate_pairs_train_validation_v01.tsv`.
- Candidate recall on that validation file: **84.8% in the top 20**, **91.9% across all 200**.
- Idealized oracle macro F0.5 on that same set: **0.930 for top 20**, **0.968 for all 200**. An oracle knows the labels and returns only true matches, so a real model must score lower.
- Country-specific all-candidate oracle: **US 0.983; India 0.946**. India is the primary blocking-recall gap.
- Current matcher held-out result: **0.765** on its Source-1 ID holdout. This is not the public score and is not directly comparable to the candidate oracle.
- Test-only distribution shift: France is 259,452 / 1,732,544 (**15.0%**) of test Source-1 rows and has no labeled training rows. Test France names/addresses also have much more non-ASCII text than US training data.

**Implication:** threshold tuning or a larger model alone will not get this baseline to 0.98. First improve the candidates. For a 0.98 result to be plausible, target an oracle ceiling of at least **0.995 overall**, with no country substantially below it; this is a go/no-go target, not a guarantee.

## 2. Team coordination: parallel work, controlled integration

All three people can work at the same time, but not by editing a single live OneDrive-synced checkout. Each person should use their own Git branch/checkout and their own local copy of the challenge dataset at the same relative path. The large dataset, generated candidate TSVs, caches, indexes, and model binaries stay out of GitHub.

Keep the existing ownership boundaries from `INITIAL_TEAM_PLAN.md`:

| Owner | Workstream | Owns / delivers |
|---|---|---|
| Person 1 | Evaluation and experiment control | Preserve validation IDs; candidate-oracle and official macro-F0.5 reports; country/cardinality breakdowns; experiment ledger; integration checklist. Do not change blocker or submission model behavior. |
| Person 2 | Candidate retrieval | New blocking keys/ranking/caps; deterministic candidate order; versioned validation candidate TSV + stats + SHA-256; evidence that recall/oracle improved. Do not write canonical `output/` files. |
| Person 3 | Pair matching and submission | Train/compare matchers against a fixed candidate version; score the exact candidates supplied to inference; versioned model/output; run official validator. Person 3 remains the only uploader unless the team explicitly changes that rule. |

Use the team’s existing branch names and submit small PRs. Keep `main` stable until a result passes its gate. Each experiment writes to a versioned location (for example `data/candidates/person2/validation/v02/` or `output/experiments/v02/`), never over the baseline. Merge code in this order: Person 1’s evaluation contract, Person 2’s approved candidate version, then Person 3’s matcher/output changes.

## 3. What to share—and what not to share

Do **not** upload the whole local folder to Drive. It duplicates the >2 GB challenge data, can include large temporary files or local caches, and a live synced folder can expose incomplete or conflicting artifacts.

Share via GitHub branches/PRs:

- Source code, this plan, small reports, command lines, and experiment metadata.
- No challenge dataset, credentials, local environment, DuckDB/SQLite indexes, or full generated test outputs.

Use Drive only when an artifact is genuinely expensive for another teammate to regenerate. Share a **closed, versioned snapshot** (not a working file) with its `.sha256` and a small stats/manifest file. The useful handoff is normally the fixed-validation candidate TSV for one new version—not the whole repository or raw dataset. Since everyone already has the dataset locally, provide exact reproduction commands so each teammate can regenerate where practical.

Every experiment manifest must record:

```text
candidate_version and SHA-256
validation ID file SHA-256
code branch and commit
command and dependency versions
candidate cap / block settings / ranking version
candidate count, link recall, oracle macro F0.5
matcher feature/model version and threshold policy
held-out macro F0.5 overall, by country, and by truth cardinality
```

## 4. Work plan and pass/fail gates

### Workstream A — Person 1: trustworthy evaluation (parallel, start now)

1. Freeze and hash `data/processed/validation_source1_ids.txt`; all candidate and model comparisons use this exact ID set.
2. Maintain one machine-readable or Markdown experiment ledger. Record candidate version and hash alongside every score; never compare models run on different candidates as if only the model changed.
3. For each candidate version, report:
   - Pair recall and fraction of queries with at least one true candidate.
   - Oracle macro F0.5 at top 20/50/100/200 and at realistic prediction caps (5/10/no cap).
   - The same metrics for India and US, plus truth match-count buckets (singleton, 1–2, 3–5, 6+ links).
4. Keep a final untouched portion of validation for confirmation after feature/threshold selection. Tune on the other portion; do not repeatedly optimize against the final report slice.
5. Verify every final result with `src/eval/macro_f05.py` and the official submission validator. Macro F0.5 weights precision more than recall; report singleton and non-singleton behavior separately.

**Gate A:** the report is reproducible from a command and identifies the exact candidate/model versions. If candidate hashes differ, do not treat the scores as a controlled model comparison.

### Workstream B — Person 2: raise candidate recall (parallel, highest priority)

Run these in order on the fixed validation queries, not on a small enriched smoke sample:

1. **Determinism and ranking:** make aggregation explicitly preserve rank (`ORDER BY` inside aggregation); ties must have a stable, meaningful tie-break. Confirm the emitted candidate order is the order used for top-K selection.
2. **Recover address-led matches:** test address-only locality/token pairs and normalized house-number keys, including leading-zero variants. Keep keys country-scoped and add them as recall-oriented routes; measure large/common blocks and cap effects.
3. **Improve rank/cap behavior:** compare caps 200, 500, and 1,000; test whether block-size skipping or one source monopolizing a cap hides true links. Consider balanced S2/S3 quotas only if validation shows a gain. Keep each variant separately versioned.
4. **Target India:** inspect missed true pairs by name/address availability, shared tokens, country, source prefix (S2 vs S3), and whether they shared any skipped/uncapped block. Prioritize changes that recover India links without flooding candidates.
5. **Evaluate the oracle before sending a file to Person 3.** Publish the validation candidate TSV (or a Drive snapshot), checksum, stats, and exact generation command only for versions that beat v01.

**Gate B (minimum):** new candidate version raises oracle macro F0.5 materially above 0.968, with a clear India gain and manageable candidate volume. **Target for pursuing 0.98:** oracle at least 0.995 overall and approximately 0.99+ for both training countries. If it does not clear this gate, a 0.98 prediction is not supported by validation evidence; continue retrieval work rather than spending a leaderboard upload.

### Workstream C — Person 3: improve candidate-conditioned matching (parallel on v01, then new best candidate version)

1. Start on frozen v01 while Person 2 works; later rerun the best matcher on Person 2’s best candidate version. This isolates matcher progress and keeps everyone productive.
2. Score the exact list that will be recorded as `candidate_pairs.tsv`. If increasing candidate count, do not silently score only 20 while claiming the larger list was considered.
3. Add features that the current 12-feature logistic matcher lacks:
   - Candidate retrieval rank, retrieval/block score, and which independent key families retrieved the pair.
   - Name token/character similarity and address token/character similarity, with separate missing-field indicators.
   - Address number, postal/locality overlap and normalized forms; safe Unicode normalization that preserves non-Latin combining marks.
   - Source-aware calibration (S2 vs S3) and query-level signals such as score gap, candidate count, and best score.
4. Train on hard negatives: plausible high-ranked nonmatches, not only a uniform sample of easy candidates. Use the fixed group/Source-1 holdout to tune threshold, singleton decision, and match-count cap. Because F0.5 is precision-weighted, choose thresholds on entity-level macro F0.5, not pair accuracy.
5. Compare a regularized linear baseline with a nonlinear model only if the dependency and model license satisfy the challenge package rules. Keep the model within the stated parameter/license limit; no external business data or pretrained business lookup model.
6. Do not force one-to-one matching: the rules allow zero, one, or many S2/S3 records per S1. Tune caps from labels; do not assume a universal cap of five.

**Gate C:** improve on the held-out macro F0.5 for the *same candidate hash*, without a large precision collapse in singleton rows or India. Then re-evaluate on the best candidate version and the untouched confirmation slice.

### Workstream D — test-only France risk (Person 1 coordinates; Persons 2/3 implement within ownership)

- France has no labels, so report it as an explicit unknown-generalization risk; do not claim a France score based on US/India validation.
- Use only provided challenge records. Test-time text may be normalized for matching, but do not use external databases, APIs, geocoding, or third-party business lookups.
- Preserve open-set country handling; never drop France rows or assume the only countries are US/India.
- Test accent/Unicode variants and missing addresses in generic, language-safe logic using the labeled validation countries. Do not invent France-specific business rules from outside information.
- Avoid pseudo-labeling/self-training until the organizers’ rules clearly permit it; it adds risk and is not the first route to a reliable gain.

## 5. Integration and leaderboard policy

1. Agree on one candidate version/hash and one matcher version before integration. Candidate improvements and matcher improvements are separate PRs/experiments.
2. Run full local evaluation first. Require no validator errors, exactly one output row per test S1, no duplicate IDs, valid S2/S3 IDs, and every prediction contained in the exact candidate file.
3. Run `dataset/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py` against both output files and the test directory. Keep `matching_results.tsv` as the only leaderboard upload; keep `candidate_pairs.tsv` for the required package/audit.
4. Person 3 uploads only when local evidence justifies it. Log timestamp, day’s submission slot, commit, candidate/model version, local score, public score, and next decision in `docs/submission_log.md`. The rules in the existing plan allow five uploads/day; reserve at least one slot.
5. Archive the exact best-scoring output and its commit/version; never overwrite the only copy of a known-good submission.

## 6. Fastest practical sequence

Work in parallel now: Person 1 establishes the experiment ledger/holdout; Person 2 tests blocker recall changes; Person 3 improves model features on v01. Person 2 then hands off only the best validation candidate snapshot and manifest. Person 3 reruns the leading model on that version; Person 1 independently verifies the score and validator. Integrate only the winning code and retain the baseline.

Do not spend time moving the full local folder to Drive or launching AWS/Kaggle/Colab first. This is currently a **candidate-recall and model-quality** problem, not a lack-of-GPU problem. The default Python runtime checked for this plan did not have DuckDB; use a reproducible project environment with the pinned `src/blocking/requirements.txt` dependency before running the DuckDB blocker. Cloud compute is optional only if profiling later shows local indexing/runtime is the bottleneck.

## 7. Stop/go rule for the 0.98 target

Treat 0.98 as a hypothesis. Continue toward it only if new candidate versions push the fixed-validation oracle near/above 0.995 and the held-out matcher keeps pace. If oracle recall plateaus well below that, report the ceiling honestly and optimize the best supported score instead. Public leaderboard feedback is useful but cannot replace the fixed local evaluation, especially with France absent from labeled training data.
