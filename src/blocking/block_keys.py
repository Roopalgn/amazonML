"""Deterministic, country-aware blocking keys using only challenge fields."""

import unicodedata

LEGAL = {"inc", "incorporated", "corp", "corporation", "co", "company", "llc", "ltd", "limited", "pvt", "private", "plc", "gmbh", "sarl", "sas", "sa", "the"}
ADDRESS_WORDS = {"road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr", "building", "bldg", "floor", "fl", "unit", "suite", "ste", "near", "opp", "opposite", "district", "dist", "state"}


def normalize(value):
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = value.replace("&", " and ")
    # Preserve Indic letters and combining vowel marks; ASCII-only regexes lose
    # many true matches in the India split.
    value = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in value)
    return " ".join(value.split())


def parts(name, address, country):
    country = normalize(country)
    name_tokens = [t for t in normalize(name).split() if t not in LEGAL and len(t) > 1]
    address_tokens = [t for t in normalize(address).split() if t not in ADDRESS_WORDS]
    numbers = [t for t in address_tokens if t.isdigit() and len(t) <= 5]
    postal = [t for t in address_tokens if t.isdigit() and len(t) in (5, 6)]
    locality = [t for t in address_tokens if t.isalpha() and len(t) >= 4]
    return country, name_tokens, address_tokens, numbers, postal, locality


def number_forms(value):
    """Return raw and leading-zero-normalized forms for a short house number."""
    forms = {value}
    if value.isdigit() and len(value) <= 5:
        # Treat 00012 and 12 as a possible match while retaining the raw form.
        # Do not normalize longer digit strings (often account/route numbers).
        forms.add(value.lstrip("0") or "0")
    return forms


def block_keys(name, address, country):
    country, nt, at, numbers, postal, locality = parts(name, address, country)
    if not country:
        return ()
    keys = set()
    if nt:
        name_sorted = sorted(set(nt))
        keys.add(f"{country}|name|{' '.join(name_sorted)}")
        for i in range(min(3, len(nt))):
            for j in range(i + 1, min(4, len(nt))):
                keys.add(f"{country}|pair|{min(nt[i], nt[j])}|{max(nt[i], nt[j])}")
        first = nt[0]
        for code in postal[:2]:
            keys.add(f"{country}|postal|{first}|{code}")
        for number in numbers[:2]:
            keys.add(f"{country}|number|{first}|{number}")
        for token in locality[:2]:
            keys.add(f"{country}|locality|{first}|{token}")
    if at:
        keys.add(f"{country}|address|{' '.join(sorted(set(at)))}")
    # Address-only routes allow retrieval when either record has a blank or
    # severely corrupted name. Common locality/postal blocks are filtered by
    # the generator's max-block-size guard.
    for code in postal[:2]:
        keys.add(f"{country}|addr_postal|{code}")
    for token in locality[:3]:
        keys.add(f"{country}|addr_locality|{token}")
    # Additional long address tokens cover records whose locality is encoded
    # as a district, industrial area, or market name rather than a city token.
    for token in [token for token in at if token.isalpha() and len(token) >= 6][:3]:
        keys.add(f"{country}|addr_token|{token}")
    for number in numbers[:3]:
        for token in locality[:3]:
            for form in number_forms(number):
                keys.add(f"{country}|addressnum|{form}|{token}")
    if nt and len(nt[0]) >= 4 and at:
        first = nt[0]
        keys.add(f"{country}|prefixaddr|{first[:4]}|{at[0][:4]}")
    return tuple(sorted(keys))
