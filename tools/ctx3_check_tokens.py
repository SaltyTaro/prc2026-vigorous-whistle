"""GPU sprint node check: the node's token build (data/ctx/tokens_v1/manifest.json) must have the same per-file content hashes as the local
build (local_token_manifests/tokens_v1.json, mounted from the workstation); the ADS-B feature cache hashes are printed for the record."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    local = {k: v["content_sha256"] for k, v in json.loads((ROOT / "local_token_manifests/tokens_v1.json").read_text())["files"].items()}
    node = {k: v["content_sha256"] for k, v in json.loads((ROOT / "data/ctx/tokens_v1/manifest.json").read_text())["files"].items()}
    features = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((ROOT / "runs/adsb_nn/features").glob("*.parquet"))}
    print(json.dumps({"tokens_equal_local": local == node, "token_files": len(node), "adsb_feature_files": len(features), "adsb_features": features}))
    if local != node:
        sys.exit("token content hashes differ from the local build")


if __name__ == "__main__":
    main()
