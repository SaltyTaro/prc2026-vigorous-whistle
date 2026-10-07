"""Lever 12 (fork N) node check: the shipped own-points tables (data/external/adsblol/features_v19own) must equal
docs/dig/ctx_ownpoints_features_manifest.json (tools/ctx_opdi_check_features.py without the OPDI tables)."""
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
    print(json.dumps({"own": own}))
    if own["problems"]:
        sys.exit("feature tables differ from their manifests")


if __name__ == "__main__":
    main()
