"""Precision-first baseline matching utilities.

This module intentionally knows nothing about how candidates are generated. It only
receives candidate IDs, optional per-candidate scores, and returns the matched IDs
that should be written to ``matching_results.tsv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


VALID_MATCH_PREFIXES = ("S2-", "S3-")


@dataclass(frozen=True)
class MatchDecision:
    """Decision for one Source 1 entity."""

    source1_entity_id: str
    candidate_entity_ids: tuple[str, ...]
    matched_entity_ids: tuple[str, ...]


def parse_id_list(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated challenge ID list while preserving order."""

    if not value:
        return ()
    return dedupe_preserve_order(part.strip() for part in value.split(",") if part.strip())


def format_id_list(ids: Iterable[str]) -> str:
    """Format IDs for the official list-style TSV files."""

    return ",".join(dedupe_preserve_order(ids))


def dedupe_preserve_order(ids: Iterable[str]) -> tuple[str, ...]:
    """Remove duplicate IDs without changing their first-seen order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for entity_id in ids:
        if entity_id and entity_id not in seen:
            seen.add(entity_id)
            ordered.append(entity_id)
    return tuple(ordered)


def keep_valid_match_ids(ids: Iterable[str]) -> tuple[str, ...]:
    """Keep only Source 2 / Source 3 IDs."""

    return tuple(entity_id for entity_id in dedupe_preserve_order(ids) if entity_id.startswith(VALID_MATCH_PREFIXES))


def select_matches(
    candidate_entity_ids: Sequence[str],
    scores: Mapping[str, float] | None = None,
    *,
    threshold: float = 0.90,
    max_matches: int | None = None,
    policy: str = "empty",
) -> tuple[str, ...]:
    """Select final matches from candidates.

    Policies:
    - ``empty``: predict singleton for every entity. This is the safest stub.
    - ``pass-through``: return all candidates. Useful only for debugging.
    - ``scored-threshold``: keep candidates with ``score >= threshold``.
    - ``top-k``: keep the highest scoring candidates above ``threshold``.
    """

    candidates = keep_valid_match_ids(candidate_entity_ids)
    if policy == "empty":
        return ()
    if policy == "pass-through":
        return candidates[:max_matches] if max_matches is not None else candidates
    if scores is None:
        return ()

    scored = [(entity_id, scores.get(entity_id)) for entity_id in candidates]
    above_threshold = [(entity_id, score) for entity_id, score in scored if score is not None and score >= threshold]

    if policy == "scored-threshold":
        selected = [entity_id for entity_id, _ in above_threshold]
    elif policy == "top-k":
        ranked = sorted(above_threshold, key=lambda item: (-item[1], item[0]))
        selected = [entity_id for entity_id, _ in ranked]
    else:
        raise ValueError(f"Unknown match policy: {policy}")

    if max_matches is not None:
        selected = selected[:max_matches]
    return tuple(selected)


def decide_matches(
    source1_entity_id: str,
    candidate_entity_ids: Sequence[str],
    scores: Mapping[str, float] | None = None,
    *,
    threshold: float = 0.90,
    max_matches: int | None = None,
    policy: str = "empty",
) -> MatchDecision:
    """Build a complete match decision for one Source 1 entity."""

    candidates = keep_valid_match_ids(candidate_entity_ids)
    matches = select_matches(
        candidates,
        scores,
        threshold=threshold,
        max_matches=max_matches,
        policy=policy,
    )
    return MatchDecision(
        source1_entity_id=source1_entity_id,
        candidate_entity_ids=candidates,
        matched_entity_ids=matches,
    )
