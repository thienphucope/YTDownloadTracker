#!/usr/bin/env python3
"""Populate ``count`` values in an existing metadata.json file.

This is a companion utility for the downloader; it can be run against older
metadata files that pre‑date the ``count`` feature.

Each video entry is given a sequential integer starting at 1 in the order it
appears in the JSON (i.e. the order ``dict.items()`` returns).  The script
updates the file in place only if any counts were missing or incorrect.  A
convenience message is printed to standard output.

Usage:

    python add_count.py path/to/metadata.json

Example:

    > python add_count.py downloads/solus/metadata.json
    updated counts for 214 entries

If the file already has correct counts the script exits silently with
"no changes needed".
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python add_count.py <metadata.json>")
        sys.exit(1)

    metadata_path = Path(sys.argv[1])
    if not metadata_path.is_file():
        print(f"error: {metadata_path} does not exist or is not a file")
        sys.exit(1)

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: failed to parse JSON: {e}")
        sys.exit(1)

    if not isinstance(data, dict):
        print("error: metadata file must contain a top-level object")
        sys.exit(1)

    changed = False
    for idx, (video_id, entry) in enumerate(data.items(), start=1):
        if not isinstance(entry, dict):
            continue
        if entry.get("count") != idx:
            entry["count"] = idx
            changed = True

    if changed:
        metadata_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"updated counts for {len(data)} entries")
    else:
        print("no changes needed")


if __name__ == "__main__":
    main()
