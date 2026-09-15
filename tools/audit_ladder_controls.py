"""Check that the longer CatBoost comparison preserves v6's data and model settings."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, verify_receipts, write_json


def audit(root, runs, previous_runs):
    config = read_json(root/"configs/ladder_v1.json")
    selection_path = root/config["reference_selection"]
    if sha256_file(selection_path) != config["reference_selection_sha256"]:
        raise ValueError("The original v6 selection changed.")
    previous = read_json(selection_path)
    output = {}
    for fold in config["development_folds"]:
        prior = previous["development_runs"]["cat_residual"][fold]
        source = previous_runs/prior["run_id"]
        if sha256_file(source/"run.json") != prior["run_sha256"]:
            raise ValueError("An original v6 development run changed.")
        matches = [p.parent for p in runs.glob("*/run.json")
                   if read_json(p)["stage"] == "development"
                   and read_json(p)["variant"] == "cat_long"
                   and read_json(p)["fold"] == fold]
        if len(matches) != 1:
            raise ValueError("Exactly one completed longer fit is required per fold.")
        run = matches[0]
        for directory in [source, run]:
            verify_receipts(directory, read_json(directory/"run.json")["artifacts"])
        old, new = read_json(source/"nodes.json")["global"], read_json(run/"node.json")
        old_settings, new_settings = deepcopy(old["settings"]), deepcopy(new["settings"])
        old_cap = old_settings["catboost"].pop("rounds")
        new_cap = new_settings["catboost"].pop("rounds")
        if old_settings != new_settings or old_cap != 6000 or new_cap != 10000:
            raise ValueError("Something other than the declared cap changed in CatBoost settings.")
        for key in ["fit_rows", "fit_ids_sha256", "early_stop_rows", "early_stop_ids_sha256", "target_filter"]:
            if old[key] != new[key]:
                raise ValueError("The longer comparison changed a protected data population: "+key)
        for key in ["kind", "features", "route", "target"]:
            if old["spec"][key] != new["spec"][key]:
                raise ValueError("The longer comparison changed a native feature/target contract.")
        old_encoder, new_encoder = read_json(source/old["encoder_path"]), read_json(run/new["encoder_path"])
        for key in ["kind", "feature_names", "categorical_columns", "vocabularies"]:
            if old_encoder[key] != new_encoder[key]:
                raise ValueError("The longer comparison changed the fitted encoder.")
        old_history, new_history = read_json(source/old["history_path"]), read_json(run/new["history_path"])
        curves = {}
        for partition, losses in old_history.items():
            curves[partition] = {}
            for loss, values in losses.items():
                a = np.asarray(values, dtype=float)
                b = np.asarray(new_history[partition][loss], dtype=float)
                if b.size < a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("A native learning curve is incomplete.")
                delta = float(np.max(np.abs(a-b[:len(a)])))
                curves[partition][loss] = {"old_evaluated_iterations": len(a), "new_evaluated_iterations": len(b),
                                           "maximum_prefix_difference": delta,
                                           "prefix_matches_to_1e_minus_8": delta <= 1e-8}
        output[fold] = {
            "previous_run_sha256": sha256_file(source/"run.json"),
            "longer_run_sha256": sha256_file(run/"run.json"),
            "old_cap": old_cap, "new_cap": new_cap,
            "old_october_selected_iterations": old["iterations"],
            "new_october_selected_iterations": new["iterations"],
            "identical_fitting_and_stopping_populations": True,
            "identical_encoder_values": True, "only_round_cap_changed_in_settings": True,
            "learning_curve_prefix": curves,
        }
    return {
        "created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
        "audit_used": False, "new_labels_read": False, "new_models_fitted": 0,
        "folds": output,
        "interpretation": "An experimental-control check only; complete original-fold scores and the declared selection rule determine promotion.",
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/ladder"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--output", type=Path, default=Path("docs/ladder/control_comparison.json"))
    args = parser.parse_args()
    destination = root/args.output
    if destination.exists():
        raise ValueError("Preserve the original control comparison.")
    report = audit(root, root/args.runs, root/args.previous_runs)
    write_json(destination, report)
    print(report["folds"])


if __name__ == "__main__":
    main()

