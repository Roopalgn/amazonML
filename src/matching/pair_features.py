"""Batched, country-independent evidence for supervised entity matching."""

from __future__ import annotations

from functools import lru_cache
import re

from anyascii import anyascii
import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler


FEATURE_VERSION = "pair-evidence-v1"
FEATURE_NAMES = [
    "name_ratio", "name_sorted", "name_set", "name_partial",
    "address_ratio", "address_sorted", "address_set", "address_partial",
    "name_jaro", "address_jaro", "ascii_name", "ascii_address",
    "compact_name", "numbers_ratio", "name_exact", "address_exact",
    "house_equal", "house_conflict", "numbers_equal", "address_missing",
    "name_length_ratio", "address_length_ratio", "name_length",
    "target_name_length", "address_length", "target_address_length",
    "name_overlap", "address_overlap", "number_overlap", "name_jaccard",
    "address_jaccard", "number_jaccard",
]


@lru_cache(maxsize=30000)
def ascii_text(value: str) -> str:
    if value.isascii():
        return value
    return re.sub(r"\s+", " ", anyascii(value)).strip()


FAST_FEATURE_INDICES = [0, 2, 4, 6, 19]


def fast_features(rows: list[tuple]) -> np.ndarray:
    result = []
    for first, second in ((2, 4), (3, 5)):
        left, right = [r[first] or "" for r in rows], [r[second] or "" for r in rows]
        for scorer in (fuzz.ratio, fuzz.token_set_ratio):
            result.append(process.cpdist(left, right, scorer=scorer, dtype=np.float32, workers=2) / 100)
    result.append(np.array([not r[3] or not r[5] for r in rows], dtype=np.float32))
    return np.column_stack(result)


def pair_features(rows: list[tuple]) -> np.ndarray:
    """Rows: query ID, target ID, qname, qaddress, tname, taddress."""
    n, a, tn, ta = ([row[i] or "" for row in rows] for i in range(2, 6))
    count = len(rows)
    if not count:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    result = []
    for left, right in ((n, tn), (a, ta)):
        for scorer in (fuzz.ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio):
            result.append(process.cpdist(left, right, scorer=scorer, dtype=np.float32, workers=2) / 100)
    for left, right in ((n, tn), (a, ta)):
        result.append(process.cpdist(left, right, scorer=JaroWinkler.normalized_similarity, dtype=np.float32, workers=2))
    for left, right in ((n, tn), (a, ta)):
        result.append(process.cpdist([ascii_text(v) for v in left], [ascii_text(v) for v in right], scorer=fuzz.token_set_ratio, dtype=np.float32, workers=2) / 100)
    result.append(process.cpdist([v.replace(" ", "") for v in n], [v.replace(" ", "") for v in tn], scorer=fuzz.ratio, dtype=np.float32, workers=2) / 100)
    nums = [re.findall(r"\d+", v) for v in a]
    tnums = [re.findall(r"\d+", v) for v in ta]
    canonical = lambda values: [str(int(v)) for v in values]
    nums, tnums = [canonical(v) for v in nums], [canonical(v) for v in tnums]
    result.append(process.cpdist([" ".join(sorted(v)) for v in nums], [" ".join(sorted(v)) for v in tnums], scorer=fuzz.ratio, dtype=np.float32, workers=2) / 100)
    extra = np.empty((count, 18), dtype=np.float32)
    for i, (name, address, other_name, other_address) in enumerate(zip(n, a, tn, ta)):
        number, other_number = nums[i], tnums[i]
        sets = [(set(name.split()), set(other_name.split())),
                (set(address.split()), set(other_address.split())),
                (set(number), set(other_number))]
        overlaps = [len(x & y) / max(1, min(len(x), len(y))) for x, y in sets]
        jaccards = [len(x & y) / max(1, len(x | y)) for x, y in sets]
        extra[i] = [
            bool(name) and name == other_name, bool(address) and address == other_address,
            bool(number and other_number) and number[0] == other_number[0],
            bool(number and other_number) and number[0] != other_number[0],
            bool(number) and set(number) == set(other_number), not address or not other_address,
            min(len(name), len(other_name)) / max(1, len(name), len(other_name)),
            min(len(address), len(other_address)) / max(1, len(address), len(other_address)),
            len(name), len(other_name), len(address), len(other_address),
            *overlaps, *jaccards,
        ]
    return np.column_stack([*result, extra]).astype(np.float32, copy=False)
