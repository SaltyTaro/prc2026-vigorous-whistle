"""A new immutable feature bundle preserving every original scoring identity."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from prc2026.artifacts import (ResourceMonitor, environment, object_hash, read_json,
                               sha256_file, utc_now, verify_receipts, versions, write_json)
from prc2026.features import ID, TIME
from prc2026.phase1 import emit, validate_inputs, write_parquet
from prc2026.validation import identity_hash
from prc2026_phase4.bundle import load_bundle as load_previous
from .features import CORE_NAMES, FEATURE_NAMES, GROUND_NAMES, INPUT_COLUMNS, POLICY, ground_features

DEFAULT_PREVIOUS = Path("data/phase4/2ceb3449c26767a97311")


def feature_code_receipt(root: Path) -> dict:
    paths = [root / "src/prc2026_phase5r" / n for n in ["__init__.py", "features.py", "bundle.py"]]
    paths += [root / "tools/prepare_phase5r.py", root / "requirements-phase5r.txt"]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def validate_feature_config(config: dict, old: dict) -> None:
    if config["feature_policy"] != POLICY or config["previous_bundle_id"] != old["bundle_id"]:
        raise ValueError("Feature policy or previous bundle changed.")
    if config["training_months"] != [f"2025-{m:02}" for m in range(1, 12)] or config["ranking_months"] != ["2026-01", "2026-07"]:
        raise ValueError("Protected development/ranking months changed.")
    if config["departure_labels_used_in_shared_features"] or config["audit_used"] or config["external_data"]:
        raise ValueError("Shared features cannot consume targets, audit data or undeclared external inputs.")
    if config["input_columns"] != INPUT_COLUMNS:
        raise ValueError("Declared feature projection changed.")


def prepare(root: Path, previous: Path, config_path: Path, output: Path) -> Path:
    config = read_json(config_path)
    with ResourceMonitor() as monitor:
        old, parent, original, catalog = load_previous(root, previous)
        validate_feature_config(config, old)
        code, inputs = feature_code_receipt(root), validate_inputs(root)
        descriptor = {"previous_bundle_id": old["bundle_id"], "previous_manifest_sha256": sha256_file(previous / "manifest.json"),
                      "input_files": inputs, "config": config, "code_sha256": code["sha256"],
                      "transform_packages": {k: v for k, v in versions().items() if k in {"numpy", "pandas", "pyarrow", "tzdata"}}}
        bundle_id = object_hash(descriptor)
        destination = output / bundle_id[:20]
        if (destination / "manifest.json").is_file():
            manifest = read_json(destination / "manifest.json")
            if manifest["descriptor"] != descriptor or manifest["bundle_id"] != bundle_id:
                raise ValueError("Existing bundle descriptor mismatch.")
            verify_receipts(destination, manifest["artifacts"])
        else:
            output.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=bundle_id[:20]+".incomplete-", dir=output))
            # Read an explicit projection even from the December file, retaining
            # only November movements stored across that file boundary.
            contexts = []
            for source in original["sources"]:
                name = source["source"]
                raw = pd.read_parquet(root / "data/raw" / name, columns=INPUT_COLUMNS)
                months = config["ranking_months"] if name == "ranking.parquet" else config["training_months"]
                contexts.append(raw.loc[raw[TIME].dt.strftime("%Y-%m").isin(months)])
            context = pd.concat(contexts, ignore_index=True)
            del contexts, raw
            allowed = catalog[TIME].dt.strftime("%Y-%m").isin(config["training_months"])
            sources = sorted(catalog.loc[allowed, "source_file"].unique()) + ["ranking.parquet"]
            query_parts = [pd.read_parquet(previous / "features" / source, columns=[ID]) for source in sources]
            query_ids = pd.concat(query_parts, ignore_index=True)[ID]
            emit("phase5r_context_preparing", movements=len(context), queries=len(query_ids), extra_predictors=len(GROUND_NAMES))
            extra = ground_features(context, query_ids)
            context_rows = len(context)
            del context, query_parts, query_ids
            gc.collect()
            artifacts, counts, source_ids, pos = [], {}, {}, 0
            for source in sources:
                inherited = pd.read_parquet(previous / "features" / source, columns=[ID, *CORE_NAMES])
                part = extra.iloc[pos:pos+len(inherited)].reset_index(drop=True)
                pos += len(inherited)
                if not np.array_equal(inherited[ID].to_numpy(), part[ID].to_numpy()):
                    raise ValueError("New context features changed row order.")
                expected = original["ranking_ids_sha256"] if source == "ranking.parquet" else identity_hash(
                    catalog.loc[allowed & catalog["source_file"].eq(source), ID])
                if identity_hash(inherited[ID]) != expected:
                    raise ValueError("Protected feature IDs changed.")
                combined = pd.concat([inherited, part.drop(columns=ID)], axis=1)[[ID, *FEATURE_NAMES]]
                pd.testing.assert_frame_equal(combined[[ID, *CORE_NAMES]], inherited, check_exact=True)
                write_parquet(temporary, "features/"+source, combined, artifacts)
                counts[source], source_ids[source] = len(combined), expected
                emit("phase5r_features_prepared", source=source, rows=len(combined))
            if pos != len(extra):
                raise ValueError("Incomplete feature export.")
            manifest = {"created_at_utc": utc_now(), "bundle_id": bundle_id, "descriptor": descriptor, "code": code,
                        "previous_bundle": previous.relative_to(root).as_posix(), "parent_bundle": parent.relative_to(root).as_posix(),
                        "feature_names": FEATURE_NAMES, "new_features": GROUND_NAMES, "rows": counts,
                        "source_ids_sha256": source_ids, "context_rows": context_rows, "audit_used": False,
                        "frozen_folds": original["folds"], "ranking_ids_sha256": original["ranking_ids_sha256"],
                        "artifacts": artifacts}
    if not (destination / "manifest.json").is_file():
        manifest.update(environment=environment(root), resources=monitor.stats)
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    write_json(output / "latest.json", {"bundle": destination.relative_to(root).as_posix(), "bundle_id": bundle_id})
    emit("phase5r_bundle_complete", bundle=str(destination), predictors=len(FEATURE_NAMES), resources=monitor.stats)
    return destination


def load_bundle(root: Path, base: Path):
    manifest = read_json(base / "manifest.json")
    if manifest["audit_used"] or manifest["bundle_id"] != object_hash(manifest["descriptor"]):
        raise ValueError("New feature identity/audit policy changed.")
    if manifest["code"]["sha256"] != feature_code_receipt(root)["sha256"]:
        raise ValueError("Feature code changed; a fresh bundle is required.")
    previous = root / manifest["previous_bundle"]
    old, parent, original, catalog = load_previous(root, previous)
    validate_feature_config(manifest["descriptor"]["config"], old)
    if sha256_file(previous / "manifest.json") != manifest["descriptor"]["previous_manifest_sha256"]:
        raise ValueError("Previous manifest changed.")
    verify_receipts(base, manifest["artifacts"])
    return manifest, parent, original, catalog


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--config", type=Path, default=Path("configs/features_phase5r_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/phase5r"))
    args = parser.parse_args()
    prepare(root, root/args.previous, root/args.config, root/args.output)
