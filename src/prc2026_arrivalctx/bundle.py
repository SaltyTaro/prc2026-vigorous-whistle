"""Content-addressed arrival features stored beside a hash-bound frozen parent."""
from functools import lru_cache
from pathlib import Path
import argparse
import tempfile

import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, environment, object_hash, read_json, sha256_file, utc_now, verify_receipts, versions, write_json
from prc2026.features import ID
from prc2026.phase1 import validate_inputs, write_parquet
from prc2026.validation import identity_hash
from prc2026_nmclock.bundle import load_bundle as load_previous
from prc2026_phase3.bundle import load_partition as parent_partition
from .features import POLICY, QUERY_COLUMNS, ARRIVAL_COLUMNS, EXTRA_NAMES, FEATURE_NAMES, SPECS, read_context, context

DEFAULT_PREVIOUS = Path("data/nmclock/72e86725e23367ede5b5")


def feature_code_receipt(root):
    paths = [root / "src/prc2026_arrivalctx" / name for name in ["__init__.py", "features.py", "bundle.py"]]
    paths += [root / "tools/prepare_arrivalctx.py", root / "requirements-arrivalctx.txt"]
    files = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return {"files": files, "sha256": object_hash(files)}


def validate_config(root, config, previous):
    if (config["feature_policy"] != POLICY or config["previous_bundle_id"] != previous["bundle_id"]
            or config["query_columns"] != QUERY_COLUMNS or config["arrival_only_columns"] != ARRIVAL_COLUMNS
            or config["new_feature_names"] != EXTRA_NAMES or object_hash(config["statistic_specifications"]) != object_hash(SPECS)):
        raise ValueError("Arrival context feature declaration changed.")
    if config["audit_used"] or config["departure_labels_used_in_shared_features"]:
        raise ValueError("Departure targets and the reserved audit cannot enter shared features.")
    if config["training_months"] != [f"2025-{month:02}" for month in range(1, 12)] or config["ranking_months"] != ["2026-01", "2026-07"]:
        raise ValueError("Protected context months changed.")
    for key in ["field_use_review", "prior_fitting_audit"]:
        if sha256_file(root / config[key]["path"]) != config[key]["sha256"]:
            raise ValueError("Feature availability or prior evidence changed.")


def prepare(root, previous, config_path, output):
    config = read_json(config_path)
    with ResourceMonitor() as monitor:
        old, parent, original, catalog = load_previous(root, previous)
        validate_config(root, config, old)
        descriptor = {"previous_bundle_id": old["bundle_id"], "previous_manifest_sha256": sha256_file(previous / "manifest.json"),
                      "input_files": validate_inputs(root), "config": config, "code_sha256": feature_code_receipt(root)["sha256"],
                      "transform_packages": {key: value for key, value in versions().items() if key in {"numpy", "pandas", "pyarrow", "tzdata"}}}
        bundle_id = object_hash(descriptor)
        destination = output / bundle_id[:20]
        if destination.exists():
            raise ValueError("Preserve the existing arrival feature bundle.")
        output.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=bundle_id[:20] + ".incomplete-", dir=output))
        queries, arrivals = read_context(root, original, config)
        inherited_ids = pd.concat([pd.read_parquet(previous / "features" / name, columns=[ID]) for name in old["rows"]], ignore_index=True)
        if len(queries) != len(inherited_ids) or identity_hash(queries[ID]) != identity_hash(inherited_ids[ID]):
            raise ValueError("New raw queries differ from the complete inherited populations.")
        queries = inherited_ids.merge(queries, on=ID, validate="one_to_one", how="left", sort=False)
        extra = context(queries, arrivals)
        if not np.array_equal(extra[ID], inherited_ids[ID]) or list(extra.columns) != [ID, *EXTRA_NAMES]:
            raise ValueError("Arrival feature identities or columns changed.")
        artifacts = []
        write_parquet(temporary, "arrival_features.parquet", extra, artifacts)
        manifest = {"created_at_utc": utc_now(), "bundle_id": bundle_id, "descriptor": descriptor, "code": feature_code_receipt(root),
                    "previous_bundle": previous.relative_to(root).as_posix(), "parent_bundle": parent.relative_to(root).as_posix(),
                    "feature_names": FEATURE_NAMES, "new_features": EXTRA_NAMES, "rows": old["rows"], "source_ids_sha256": old["source_ids_sha256"],
                    "complete_rows": len(extra), "complete_ids_sha256": identity_hash(extra[ID]), "arrival_context_rows": len(arrivals),
                    "audit_used": False, "frozen_folds": original["folds"], "ranking_ids_sha256": original["ranking_ids_sha256"],
                    "artifacts": artifacts, "storage": "Immutable parent feature files plus a separately hash-checked ID-aligned table of fourteen new features."}
    manifest.update(environment=environment(root), resources=monitor.stats)
    write_json(temporary / "manifest.json", manifest)
    if not temporary.resolve().is_relative_to(output.resolve()) or not destination.resolve().is_relative_to(output.resolve()):
        raise ValueError("Bundle publication escaped its intended directory.")
    temporary.rename(destination)
    pointer = output / "latest.json"
    if pointer.exists():
        raise ValueError("Preserve the previous arrival bundle pointer.")
    write_json(pointer, {"bundle": destination.relative_to(root).as_posix(), "bundle_id": bundle_id})
    print({"bundle": str(destination), "complete_rows": len(extra), "resources": monitor.stats})
    return destination


def load_bundle(root, base):
    manifest = read_json(base / "manifest.json")
    if manifest["audit_used"] or manifest["bundle_id"] != object_hash(manifest["descriptor"]) or manifest["code"] != feature_code_receipt(root):
        raise ValueError("Arrival bundle identity or feature code changed.")
    previous = root / manifest["previous_bundle"]
    old, parent, original, catalog = load_previous(root, previous)
    validate_config(root, manifest["descriptor"]["config"], old)
    if sha256_file(previous / "manifest.json") != manifest["descriptor"]["previous_manifest_sha256"]:
        raise ValueError("The frozen parent manifest changed.")
    if manifest["frozen_folds"] != original["folds"] or manifest["ranking_ids_sha256"] != original["ranking_ids_sha256"] or manifest["rows"] != old["rows"] or manifest["source_ids_sha256"] != old["source_ids_sha256"]:
        raise ValueError("Protected inherited feature populations changed.")
    if manifest["feature_names"] != FEATURE_NAMES or manifest["new_features"] != EXTRA_NAMES:
        raise ValueError("The stored feature contract changed.")
    verify_receipts(base, manifest["artifacts"])
    return manifest, parent, original, catalog


@lru_cache(maxsize=2)
def extra_table(path):
    frame = pd.read_parquet(path)
    if frame[ID].duplicated().any() or list(frame.columns) != [ID, *EXTRA_NAMES]:
        raise ValueError("Invalid additional feature identities or columns.")
    return frame.set_index(ID, verify_integrity=True)


def load_partition(base, parent, catalog, assignment, role, names):
    root = Path(__file__).resolve().parents[2]
    manifest = read_json(base / "manifest.json")
    old_names = [name for name in names if name not in EXTRA_NAMES]
    features, truth = parent_partition(root / manifest["previous_bundle"], parent, catalog, assignment, role, old_names)
    new_names = [name for name in names if name in EXTRA_NAMES]
    if new_names:
        extra = extra_table(base / "arrival_features.parquet").loc[features[ID], new_names].reset_index(drop=True)
        if len(extra) != len(features):
            raise ValueError("Missing inherited feature identities.")
        features = pd.concat([features.reset_index(drop=True), extra], axis=1)
    return features[[ID, *names]], truth


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--config", type=Path, default=Path("configs/features_arrivalctx_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/arrivalctx"))
    args = parser.parse_args()
    prepare(root, root / args.previous, root / args.config, root / args.output)
