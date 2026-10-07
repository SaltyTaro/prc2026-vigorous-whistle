"""Lever 9 (fork J) node check: the shipped own-points tables (data/external/adsblol/features_v19own) must equal
docs/dig/ctx_ownpoints_features_manifest.json, and the OPDI network tables (data/external/opdi/features_v19ownopdi) must equal
docs/ctx_opdi/features_manifest.json (the committed copy of the tables' manifest written by tools/ctx_opdi_features.py on arcw4)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(manifest_path: Path, folder: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())["files"]
    bad = {name: "missing" for name in manifest if not (folder / name).exists()}
    for name, entry in manifest.items():
        path = folder / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            bad[name] = "hash differs"
    return {"tables": len(manifest), "problems": bad}


def main():
    own = check(ROOT / "docs/dig/ctx_ownpoints_features_manifest.json", ROOT / "data/external/adsblol/features_v19own")
    opdi = check(ROOT / "docs/ctx_opdi/features_manifest.json", ROOT / "data/external/opdi/features_v19ownopdi")
    print(json.dumps({"own": own, "opdi": opdi}))
    if own["problems"] or opdi["problems"]:
        sys.exit("feature tables differ from their manifests")


if __name__ == "__main__":
    main()
