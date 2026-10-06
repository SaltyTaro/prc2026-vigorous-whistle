"""Lever 5 (fork E) node check: every own-points feature table under data/external/adsblol/features_v19own must have the SHA-256 recorded in
docs/dig/ctx_ownpoints_features_manifest.json (tunnel 21's manifest: the tables the shipped own_all member was fitted on)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "docs/dig/ctx_ownpoints_features_manifest.json").read_text())["files"]
    folder = ROOT / "data/external/adsblol/features_v19own"
    bad = {name: "missing" for name in manifest if not (folder / name).exists()}
    for name, entry in manifest.items():
        path = folder / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            bad[name] = "hash differs"
    print(json.dumps({"own_tables": len(manifest), "own_tables_equal_manifest": not bad, "problems": bad}))
    if bad:
        sys.exit("own-points tables differ from the manifest")


if __name__ == "__main__":
    main()
