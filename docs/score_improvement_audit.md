# Score improvement audit

This records measured failures in the submission reported at **0.895** on the
public leaderboard. The earlier top-candidate baseline scored 0.554. Public scores
and ranks are user-reported; local scores below are computed from supplied labels.
The objective is 0.99, not a claim that any current artifact achieves it.

## 1. Contradictory ownership, concentrated in France

The complete training ground truth contains 7,638,365 distinct target records.
Every one belongs to exactly one Source 1 entity. A Source 1 entity can still have
many targets: this is a many-targets-to-one-query constraint, not one-to-one matching.

The 0.895 submission had 6,312,857 predicted links to 5,267,122 distinct targets.
190,588 targets were assigned to multiple Source 1 records, creating 1,045,735
assignments beyond one per target. The largest group had 262 proposed owners.
Applying the training ownership structure to test is an explicit assumption.

| Country | Links involving conflicted targets | Affected queries | Conflicted targets |
| --- | ---: | ---: | ---: |
| France | 1,178,228 | 114,700 | 165,754 |
| India | 50,309 | 33,418 | 21,198 |
| US | 7,786 | 6,088 | 3,636 |

France represents approximately 95% of the conflicting links. It has no training
labels, so the previous India/US validation did not measure this failure directly.

### Implemented correction

`src/matching/resolve_conflicts.py` compares proposed owners by model confidence.
It keeps the best owner only if its log-odds advantage meets a tuned margin;
exact ties abstain instead of choosing an entity by ID order. Uncontested links
remain eligible. Source 1 rows may retain multiple distinct target IDs.

For local validation, accepted targets are compared against the **entire**
2,206,821-row training Source 1 catalog, not just the sampled validation queries.
Competitors are retrieved without using their labels. The margin was chosen on
3,000 tuning queries and then applied to the separate 3,000-query holdout.

| Slice | Original macro F0.5 | After ownership resolution | True links removed | False links removed |
| --- | ---: | ---: | ---: | ---: |
| Tune, all | 0.933029 | 0.938873 | 8 | 83 |
| Holdout, all | 0.927182 | 0.930701 | 5 | 52 |
| Holdout, India | 0.901295 | 0.906756 | 2 | 29 |
| Holdout, US | 0.943919 | 0.946183 | 3 | 23 |

The selected log-odds margin is 0.5. This is a measured local improvement, not a
forecast of the French or public leaderboard score. Reusing this holdout for later
development also means it should not be described as a fresh final test.

### Submission artifact

- Upload: `output/ownership_v1/matching_results.tsv`.
- Candidate audit file remains `output/ec2_supervised/candidate_pairs.tsv`.
- Matching SHA-256: `dc5485ba3853be361ed281508376d234163f9c57c1d48d3722396d6b92f33e54`.
- Required Source 1 rows: 1,732,544, all present.
- Predicted links: 5,240,965; empty predictions: 122,624.
- Removed links: 1,071,892, including abstentions on ambiguous best owners.
- Full disk-backed validator: valid, zero errors, against all three test sources
  and the original candidate set.

Reproduce with:

```powershell
.venv\Scripts\python.exe -m src.matching.resolve_conflicts validate
.venv\Scripts\python.exe -m src.matching.resolve_conflicts predict
```

## 2. Retrieval imposes a score ceiling below the objective

The original holdout retrieval oracle is 0.982259 macro F0.5. Even a perfect
classifier cannot reach 0.99 using those candidates. Across tuning and holdout,
956 of 20,910 positive links never reach the classifier: 675 Indian links and
281 US links. These are separate from the 1,946 retrieved positives rejected by
the classifier at its 0.91 threshold.

Concrete missed cases include three-letter business tokens, leading zeros in
street numbers, house numbers that moved after an inserted prefix, concatenated
names, translated names and shortened addresses. Candidate frequency caps also
discard common keys entirely.

The v2 retrieval union adds canonical and compact names, rare name/address words,
normalized address numbers paired with name or address tokens, and broader name
token pairs. Existing candidates remain in the union. Candidate membership is
never obtained from validation labels.

## 3. Destructive normalization of Indian scripts

The old preprocessing calls `strip_accents` before transliteration. Inspection of
the supplied raw records confirmed that it deletes meaningful Indian-script vowel
marks. A supplied name transliterates to `blu bijnes` before stripping, but only
`bl bjns` afterward. This affects both retrieval and classification.

`src/matching/resolution_features.py` transliterates the original text first. It
also learns recurring token variants from fitting-query positive pairs only.
Tuning and holdout labels do not enter this dictionary. Tests check this boundary.
The original feature representation is retained as additional evidence rather
than replacing every signal with a single aggressive normalization.

## 4. Generic overlap and unseen-country vocabulary

The original model relies on unweighted string similarities. Agreement on generic
company words, common address words and frequent cities can resemble agreement on
a distinctive business or street name. The original model was also fitted on only
12,000 Source 1 entities out of more than 2.2 million available.

The v2 classifier adds weighted and fuzzy token agreement, full-catalog name and
address frequencies, canonical numeric agreement, and corrected multilingual
evidence. Inference word frequencies are calculated from the supplied unlabeled
test catalog, so common French words are not incorrectly assigned rare-word weight
just because they were unseen in the India/US training vocabulary.

No external business lookup, geocoding, business database, or external entity
labels are used. IDs identify rows and splits but are not classifier features.

## Experiment policy

Keep the 0.895 output and all newer artifacts in separate directories. Record each
actual public score in `docs/submission_log.md`; do not treat a generated file as
an actual submission. Tune on the tuning partition, report country and singleton
slices, and check the retrieval oracle before expanding classifier complexity.
Only publish complete output files, and validate them against all official IDs.
Preserve score caches so confidence experiments do not require another full run.

The v2 implementation is an experiment until its measured results and final
submission validation are recorded. It must not be described as a 0.99 model based
on architecture, increased training size, or a leaderboard target alone.
