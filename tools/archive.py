"""Read-only verification of named original inputs in a separately supplied archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_check import ROOT, digest, safe_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root of the original research archive")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error("The archive root must be an existing directory.")
    manifest = json.loads((ROOT / "provenance/archive_inputs.json").read_text())
    missing, changed = [], []
    for name, item in manifest["files"].items():
        path = safe_path(root, name)
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            changed.append(name)
    result = {"status": "failed" if missing or changed else "passed",
              "files_checked": len(manifest["files"]), "missing": missing, "changed": changed,
              "scope": "Named original sources supporting this release; not a full raw-data pipeline rebuild."}
    print(json.dumps(result, indent=2))
    if missing or changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
