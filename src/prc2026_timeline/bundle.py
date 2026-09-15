"""Fresh timeline feature bundle with the complete NM parent preserved."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, environment, object_hash, read_json, sha256_file, utc_now, verify_receipts, versions, write_json
from prc2026.features import ID, TIME
from prc2026.phase1 import emit, validate_inputs, write_parquet
from prc2026.validation import identity_hash
from prc2026_nmclock.bundle import load_bundle as load_previous
from .features import FEATURE_NAMES, EXTRA_NAMES, INPUT_COLUMNS, POLICY, PREVIOUS_NAMES, timeline_features

DEFAULT_PREVIOUS = Path("data/nmclock/72e86725e23367ede5b5")


def feature_code_receipt(root):
    paths = [root/"src/prc2026_timeline"/name for name in ["__init__.py", "features.py", "bundle.py"]]
    paths += [root/"tools/prepare_timeline.py", root/"requirements-timeline.txt"]
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def validate_config(root, config, previous):
    if (config["feature_policy"] != POLICY or config["previous_bundle_id"] != previous["bundle_id"]
            or config["input_columns"] != INPUT_COLUMNS or config["new_feature_names"] != EXTRA_NAMES):
        raise ValueError("Timeline feature contract changed.")
    if config["training_months"] != [f"2025-{m:02}" for m in range(1, 12)] or config["ranking_months"] != ["2026-01", "2026-07"]:
        raise ValueError("Protected movement periods changed.")
    if config["audit_used"] or config["departure_labels_used_in_shared_features"]:
        raise ValueError("Shared timeline features must not use protected labels.")
    review = config["field_use_review"]
    if sha256_file(root/review["path"]) != review["sha256"]:
        raise ValueError("Field-availability review changed.")


def read_context(root, original, config):
    frames = []
    for source in original["sources"]:
        name = source["source"]
        raw = pd.read_parquet(root/"data/raw"/name, columns=INPUT_COLUMNS)
        allowed = config["ranking_months"] if name == "ranking.parquet" else config["training_months"]
        frames.append(raw.loc[raw[TIME].dt.strftime("%Y-%m").isin(allowed)])
    return pd.concat(frames, ignore_index=True)


def prepare(root, previous, config_path, output):
    config = read_json(config_path)
    with ResourceMonitor() as monitor:
        old, parent, original, catalog = load_previous(root, previous)
        validate_config(root, config, old)
        descriptor = {"previous_bundle_id": old["bundle_id"], "previous_manifest_sha256": sha256_file(previous/"manifest.json"),
                      "input_files": validate_inputs(root), "config": config, "code_sha256": feature_code_receipt(root)["sha256"],
                      "transform_packages": {k: v for k, v in versions().items() if k in {"numpy", "pandas", "pyarrow", "tzdata"}}}
        bundle_id = object_hash(descriptor)
        destination = output/bundle_id[:20]
        if (destination/"manifest.json").exists():
            manifest = read_json(destination/"manifest.json")
            if manifest["descriptor"] != descriptor or manifest["bundle_id"] != bundle_id:
                raise ValueError("Existing timeline bundle identity differs.")
            verify_receipts(destination, manifest["artifacts"])
            return destination
        if shutil.disk_usage(root).free < 1_800_000_000:
            raise ValueError("Timeline preparation requires at least 1.8 GB free.")
        output.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=bundle_id[:20]+".incomplete-", dir=output))
        context = read_context(root, original, config)
        sources = list(old["rows"])
        query_ids = pd.concat([pd.read_parquet(previous/"features"/source, columns=[ID]) for source in sources], ignore_index=True)[ID]
        emit("timeline_context_preparing", movements=len(context), queries=len(query_ids), extra_predictors=len(EXTRA_NAMES))
        extra = timeline_features(context, query_ids)
        context_rows = len(context)
        del context, query_ids
        gc.collect()
        artifacts, rows, ids, pos = [], {}, {}, 0
        for source in sources:
            inherited = pd.read_parquet(previous/"features"/source, columns=[ID, *PREVIOUS_NAMES])
            part = extra.iloc[pos:pos+len(inherited)].reset_index(drop=True)
            pos += len(inherited)
            if not np.array_equal(inherited[ID], part[ID]) or identity_hash(part[ID]) != old["source_ids_sha256"][source]:
                raise ValueError("Timeline query order or population changed.")
            combined = pd.concat([inherited, part.drop(columns=ID)], axis=1)[[ID, *FEATURE_NAMES]]
            pd.testing.assert_frame_equal(combined[[ID, *PREVIOUS_NAMES]], inherited, check_exact=True)
            write_parquet(temporary, "features/"+source, combined, artifacts)
            rows[source], ids[source] = len(combined), identity_hash(combined[ID])
            emit("timeline_features_prepared", source=source, rows=len(combined))
        if pos != len(extra):
            raise ValueError("Some timeline queries were not exported.")
        manifest = {"created_at_utc": utc_now(), "bundle_id": bundle_id, "descriptor": descriptor,
                    "code": feature_code_receipt(root), "previous_bundle": previous.relative_to(root).as_posix(),
                    "parent_bundle": parent.relative_to(root).as_posix(), "feature_names": FEATURE_NAMES,
                    "new_features": EXTRA_NAMES, "rows": rows, "source_ids_sha256": ids,
                    "context_rows": context_rows, "audit_used": False, "frozen_folds": original["folds"],
                    "ranking_ids_sha256": original["ranking_ids_sha256"], "artifacts": artifacts}
    manifest.update(environment=environment(root), resources=monitor.stats)
    write_json(temporary/"manifest.json", manifest)
    temporary.rename(destination)
    write_json(output/"latest.json", {"bundle": destination.relative_to(root).as_posix(), "bundle_id": bundle_id})
    emit("timeline_bundle_complete", bundle=str(destination), predictors=len(FEATURE_NAMES), resources=monitor.stats)
    return destination


def load_bundle(root, base):
    manifest = read_json(base/"manifest.json")
    if manifest["audit_used"] or manifest["bundle_id"] != object_hash(manifest["descriptor"]):
        raise ValueError("Timeline bundle identity changed.")
    if manifest["code"] != feature_code_receipt(root):
        raise ValueError("Timeline feature code changed; prepare a fresh bundle.")
    previous = root/manifest["previous_bundle"]
    old, parent, original, catalog = load_previous(root, previous)
    validate_config(root, manifest["descriptor"]["config"], old)
    if sha256_file(previous/"manifest.json") != manifest["descriptor"]["previous_manifest_sha256"]:
        raise ValueError("Frozen parent manifest changed.")
    if manifest["frozen_folds"] != original["folds"] or manifest["ranking_ids_sha256"] != original["ranking_ids_sha256"]:
        raise ValueError("Protected fold identities changed.")
    verify_receipts(base, manifest["artifacts"])
    return manifest, parent, original, catalog


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--config", type=Path, default=Path("configs/features_timeline_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/timeline"))
    args = parser.parse_args()
    prepare(root, root/args.previous, root/args.config, root/args.output)
