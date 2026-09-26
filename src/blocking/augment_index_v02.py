"""Add v02-only address blocks to a copied v01 training SQLite index.

The base index must have been built from the supplied train Source 2 and 3
TSVs with v01 block_keys. This only appends v02 keys and never changes targets
or the original v01 candidate artifact.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from block_keys import block_keys, number_forms, parts
from generate_candidates import records


def additional_keys(row: dict[str, str]) -> set[str]:
    country, name_tokens, address_tokens, numbers, postal, locality = parts(
        row["business_name"], row["business_address"], row["country"]
    )
    if not country:
        return set()
    if not name_tokens:
        # v01 returned no keys at all for name-empty rows. Add all the v02
        # address keys so these targets can be reached by address-only queries.
        return set(block_keys(row["business_name"], row["business_address"], row["country"]))

    keys = {f"{country}|addr_postal|{code}" for code in postal[:2]}
    keys.update(f"{country}|addr_locality|{token}" for token in locality[:3])
    long_tokens = [token for token in address_tokens if token.isalpha() and len(token) >= 6]
    keys.update(f"{country}|addr_token|{token}" for token in long_tokens[:3])
    for number in numbers[:3]:
        for token in locality[:3]:
            for form in number_forms(number) - {number}:
                keys.add(f"{country}|addressnum|{form}|{token}")
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    args = parser.parse_args()
    signatures = [str(path.resolve()) + ":" + str(path.stat().st_size) for path in (args.source2, args.source3)]
    con = sqlite3.connect(args.index)
    try:
        metadata = con.execute("SELECT target_count FROM metadata").fetchone()
        if not metadata:
            raise ValueError("Index has no v01 metadata")
        has_v02_metadata = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='person2_v02_metadata'").fetchone()
        if has_v02_metadata:
            columns = [row[1] for row in con.execute("PRAGMA table_info(person2_v02_metadata)")]
            if columns == ["source_signatures", "inserted_keys"]:
                # Backward compatibility with a completed v02 run before the
                # progress checkpoint was added.
                applied = con.execute("SELECT source_signatures FROM person2_v02_metadata").fetchone()
                if applied and applied[0] == json.dumps(signatures):
                    print("v02 address keys already exist; no changes made")
                    return 0
                raise ValueError("An older interrupted v02 augmentation needs manual recovery")
            if columns != ["source_signatures", "inserted_keys", "targets_processed"]:
                raise ValueError("Unexpected v02 augmentation metadata schema")
            row = con.execute("SELECT source_signatures, inserted_keys, targets_processed FROM person2_v02_metadata").fetchone()
            if row and row[0] != json.dumps(signatures):
                raise ValueError("The existing v02 augmentation belongs to different target files")
            inserted = row[1] if row else 0
            processed = row[2] if row else 0
        else:
            con.execute("CREATE TABLE person2_v02_metadata (source_signatures TEXT NOT NULL, inserted_keys INTEGER NOT NULL, targets_processed INTEGER NOT NULL)")
            con.execute("INSERT INTO person2_v02_metadata VALUES (?, 0, 0)", (json.dumps(signatures),))
            con.commit()
            inserted = processed = 0

        existing_signatures = json.loads(con.execute("SELECT source_signatures FROM metadata").fetchone()[0])
        if len(existing_signatures) != 2 or [item.rsplit(":", 1)[-1] for item in existing_signatures] != [str(p.stat().st_size) for p in (args.source2, args.source3)]:
            raise ValueError("The v01 index does not match the supplied Source 2/3 file sizes")
        existing_targets = metadata[0]
        total_rows = 0
        started = time.time()
        for path in (args.source2, args.source3):
            print(f"Adding v02 blocks from {path}", flush=True)
            for row in records(path):
                if total_rows < processed:
                    total_rows += 1
                    continue
                target_id = row["entity_id"]
                keys = additional_keys(row)
                if keys:
                    con.executemany("INSERT INTO blocks(key, entity_id) VALUES (?, ?)", ((key, target_id) for key in keys))
                    inserted += len(keys)
                total_rows += 1
                if total_rows % 100_000 == 0:
                    con.execute("UPDATE person2_v02_metadata SET inserted_keys=?, targets_processed=?", (inserted, total_rows))
                    con.commit()
                    print(f"  {total_rows:,} targets; {inserted:,} added block rows; {time.time()-started:.0f}s", flush=True)
        if total_rows != existing_targets:
            raise ValueError(f"Index has {existing_targets:,} targets but supplied files contain {total_rows:,}")
        con.execute("UPDATE person2_v02_metadata SET inserted_keys=?, targets_processed=?", (inserted, total_rows))
        # Repoint signatures to this clean workspace so the normal generator's
        # safety check can verify and reuse the copied index.
        con.execute("UPDATE metadata SET source_signatures=?", (json.dumps(signatures),))
        con.commit()
        print(json.dumps({"targets": total_rows, "inserted_block_rows": inserted, "seconds": round(time.time()-started, 2)}, indent=2))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
