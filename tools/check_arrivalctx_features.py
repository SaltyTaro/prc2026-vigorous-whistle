"""Reconstruct arrival features through independent direct event scans."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file
from prc2026.features import ID
from prc2026.validation import identity_hash
from prc2026_arrivalctx.features import EXTRA_NAMES, read_context
from audit_arrival_context import direct_check


def verify_feature_samples(root, base, manifest, original):
    previous = root / manifest["previous_bundle"]
    old = read_json(previous / "manifest.json")
    inherited_ids = pd.concat([pd.read_parquet(previous / "features" / name, columns=[ID]) for name in old["rows"]], ignore_index=True)
    saved = pd.read_parquet(base / "arrival_features.parquet")
    if len(saved) != manifest["complete_rows"] or identity_hash(saved[ID]) != manifest["complete_ids_sha256"] or not np.array_equal(saved[ID], inherited_ids[ID]):
        raise ValueError("The entire arrival feature population differs from its parent.")
    if list(saved.columns) != [ID, *EXTRA_NAMES] or np.isinf(saved[EXTRA_NAMES].to_numpy(dtype=float)).any():
        raise ValueError("Invalid saved additional feature values.")
    queries, arrivals = read_context(root, original, manifest["descriptor"]["config"])
    indexes = np.unique(np.r_[np.linspace(0, len(queries) - 1, 96, dtype=int),
                             queries.groupby("airport", sort=True).head(2).index])
    sample = queries.iloc[indexes].reset_index(drop=True)
    aligned = sample[[ID]].merge(saved, on=ID, validate="one_to_one", how="left")
    # The direct oracle applies raw event masks and sums without prefix-sum helpers.
    manual = direct_check(sample, arrivals, aligned)
    audit_report = read_json(root / "docs/next_lever/arrival_context_audit.json")
    audit_file = root / audit_report["feature_artifact"]
    if sha256_file(audit_file) != audit_report["feature_artifact_sha256"]:
        raise ValueError("The prior fitting-only audit feature table changed.")
    audit_features = pd.read_parquet(audit_file)
    replay = audit_features[[ID]].merge(saved, on=ID, validate="one_to_one", how="left")
    pd.testing.assert_frame_equal(replay, audit_features, check_exact=True)
    # Verify the full inherited feature files through the already validated parent receipts.
    parent_files = [item for item in old["artifacts"] if item["path"].startswith("features/")]
    for item in parent_files:
        if sha256_file(previous / item["path"]) != item["sha256"]:
            raise ValueError("An immutable parent feature file changed.")
    return {"verified_parent_rows": len(saved), "verified_parent_feature_files": len(parent_files),
            "independent_direct_scan": manual, "prior_audit_reproduction_rows": len(audit_features),
            "prior_audit_reproduction_feature_values": len(audit_features) * len(EXTRA_NAMES),
            "prior_audit_reproduction_maximum_difference": 0.,
            "audit_code_sha256": sha256_file(root / "tools/audit_arrival_context.py"),
            "audit_feature_artifact_sha256": audit_report["feature_artifact_sha256"],
            "arrival_context_rows": len(arrivals), "departure_context_labels_read": False, "audit_used": False}
