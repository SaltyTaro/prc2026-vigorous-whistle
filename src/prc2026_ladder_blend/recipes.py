"""Fixed convex cross-campaign recipes; all parent models remain unchanged."""
import numpy as np
from prc2026.artifacts import object_hash, sha256_file
from prc2026_ladder_scale.training import model_code_receipt as parent_code


def model_code_receipt(root):
    files = dict(parent_code(root)["files"])
    for relative in ["src/prc2026_ladder_blend/__init__.py", "src/prc2026_ladder_blend/recipes.py", "tools/evaluate_ladder_blend.py", "requirements-ladder-blend.txt"]:
        files[relative] = sha256_file(root/relative)
    return {"files": files, "sha256": object_hash(files)}


def convex(left, right, left_weight):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Complete aligned finite parent vectors required.")
    if left_weight not in [.25, .5]:
        raise ValueError("Undeclared cross-campaign weight.")
    return left_weight*left+(1-left_weight)*right


def validate_config(config):
    if config["threads"] != 16 or config["refit_months"] != [f"2025-{m:02}" for m in range(1,12)]:
        raise ValueError("Retain the authorized worker and protected refit months.")
    if config["audit_used"] or config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("No audit, filtering or clipping.")
    expected = {(a,b,w) for a in ["missing_ground_v6_50","missing_ground_v6_75"]
                for b in ["clock300_missing300_v6_25","clock300_missing300_v6_50"] for w in [.25,.5]}
    actual = {(row["ladder_candidate"],row["scale_candidate"],row["ladder_weight"])
              for row in config["candidates"].values()}
    if len(config["candidates"]) != 8 or actual != expected:
        raise ValueError("All eight separately declared combinations are required.")
    if config["minimum_improvement_fraction_each_fold"] != .005 or config["development_folds"] != ["seasonal","forward"]:
        raise ValueError("Both complete original folds and the fixed margin are required.")
