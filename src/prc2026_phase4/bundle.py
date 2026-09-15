"""A fresh feature bundle with portable, fully checked parent identities."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from prc2026.artifacts import (ResourceMonitor, environment, object_hash, read_json,
                               sha256_file, utc_now, verify_receipts, versions, write_json)
from prc2026.features import ID, TIME
from prc2026.phase1 import emit, validate_inputs, write_parquet
from prc2026.validation import identity_hash
from prc2026_phase3.bundle import (DEFAULT_PARENT, feature_code_receipt as prior_code,
                                   load_parent, validate_feature_config as validate_prior_config)
from .features import (CATEGORIES, FEATURE_NAMES, INPUT_COLUMNS, NUMERIC, POLICY,
                       PREVIOUS_NAMES, quality_features)

DEFAULT_PREVIOUS = Path("data/phase3/410850c6cdc70bd5b987")
NONSCIENTIFIC_PARENT_FIELDS = ["created_at_utc", "preparation", "sources[*].wall_seconds"]


def scientific_parent(manifest: dict) -> dict:
    result = deepcopy(manifest)
    for name in ["created_at_utc", "preparation"]:
        result.pop(name, None)
    for source in result["sources"]:
        source.pop("wall_seconds", None)
    return result


def feature_code_receipt(root: Path) -> dict:
    paths = [root / "src/prc2026_phase4" / n for n in ["__init__.py", "features.py", "bundle.py"]]
    paths += [root / "tools/prepare_phase4.py", root / "requirements-phase4.txt"]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def load_previous(root: Path, previous: Path, parent: Path, original: dict) -> dict:
    old = read_json(previous / "manifest.json")
    if old["audit_used"] or old["bundle_id"] != object_hash(old["descriptor"]):
        raise ValueError("Previous feature descriptor or audit policy changed.")
    if old["code"]["sha256"] != prior_code(root)["sha256"]:
        raise ValueError("Frozen Phase-3 feature code changed.")
    validate_prior_config(old["descriptor"]["config"], original)
    expected = old["descriptor"]["parent_manifest_sha256"]
    if sha256_file(parent / "manifest.json") != expected:
        archive = root / "docs/phase3/parent_manifest_azure.json"
        if not archive.is_file() or sha256_file(archive) != expected:
            raise ValueError("The producing parent manifest is needed to prove portability.")
        if scientific_parent(read_json(archive)) != scientific_parent(original):
            raise ValueError("Parent data/code/split identity differs from the producing machine.")
    verify_receipts(previous, old["artifacts"])
    return old


def validate_feature_config(config: dict, original: dict, previous: dict) -> None:
    if config["feature_policy"] != POLICY or config["parent_bundle_id"] != original["bundle_id"]:
        raise ValueError("Feature policy or original parent changed.")
    if config["previous_bundle_id"] != previous["bundle_id"]:
        raise ValueError("Phase-3 parent identity changed.")
    if config["training_months"] != [f"2025-{m:02}" for m in range(1, 12)]:
        raise ValueError("Development features must exclude the reserved audit period.")
    if config["ranking_months"] != ["2026-01", "2026-07"]:
        raise ValueError("Ranking population changed.")
    if config["departure_labels_used_in_features"] or config["audit_features_materialized"] or config["external_data"]:
        raise ValueError("Unimplemented target/audit/external-data feature access.")
    if config["parent_identity_excludes_only"] != NONSCIENTIFIC_PARENT_FIELDS:
        raise ValueError("Only measured time/host parent metadata may be excluded from identity.")


def prepare(root: Path, parent: Path, previous: Path, config_path: Path, output: Path) -> Path:
    config = read_json(config_path)
    with ResourceMonitor() as monitor:
        original, catalog = load_parent(root, parent)
        old = load_previous(root, previous, parent, original)
        validate_feature_config(config, original, old)
        code, inputs = feature_code_receipt(root), validate_inputs(root)
        descriptor = {
            "parent_bundle_id": original["bundle_id"],
            "parent_scientific_sha256": object_hash(scientific_parent(original)),
            "previous_bundle_id": old["bundle_id"],
            "previous_manifest_sha256": sha256_file(previous / "manifest.json"),
            "input_files": inputs, "config": config, "code_sha256": code["sha256"],
            "transform_packages": {k: v for k, v in versions().items() if k in {"numpy", "pandas", "pyarrow", "tzdata"}},
        }
        bundle_id = object_hash(descriptor)
        destination = output / bundle_id[:20]
        if (destination / "manifest.json").is_file():
            existing = read_json(destination / "manifest.json")
            if existing["bundle_id"] != bundle_id or existing["descriptor"] != descriptor:
                raise ValueError("Existing bundle identity mismatch.")
            verify_receipts(destination, existing["artifacts"])
        else:
            output.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=bundle_id[:20] + ".incomplete-", dir=output))
            allowed = catalog[TIME].dt.strftime("%Y-%m").isin(config["training_months"])
            sources = sorted(catalog.loc[allowed, "source_file"].unique()) + ["ranking.parquet"]
            artifacts, counts, source_ids, gate_counts = [], {}, {}, {}
            for source in sources:
                raw = pd.read_parquet(root / "data/raw" / source, columns=INPUT_COLUMNS)
                months = config["ranking_months"] if source == "ranking.parquet" else config["training_months"]
                raw = raw.loc[raw[TIME].dt.strftime("%Y-%m").isin(months)]
                extra = quality_features(raw)
                inherited = pd.read_parquet(previous / "features" / source, columns=[ID, *PREVIOUS_NAMES])
                if not np.array_equal(extra[ID].to_numpy(), inherited[ID].to_numpy()):
                    raise ValueError("Previous/new feature rows are misaligned.")
                expected_ids = (original["ranking_ids_sha256"] if source == "ranking.parquet" else
                                identity_hash(catalog.loc[allowed & catalog["source_file"].eq(source), ID]))
                if identity_hash(extra[ID]) != expected_ids:
                    raise ValueError("Protected feature population changed.")
                combined = pd.concat([inherited, extra.drop(columns=ID)], axis=1)[[ID, *FEATURE_NAMES]]
                pd.testing.assert_frame_equal(combined[[ID, *PREVIOUS_NAMES]], inherited, check_exact=True)
                write_parquet(temporary, "features/" + source, combined, artifacts)
                counts[source], source_ids[source] = len(combined), expected_ids
                gate_counts[source] = int(extra["lirf_missing_record"].sum())
                emit("phase4_features_prepared", source=source, rows=len(combined), gated_rows=gate_counts[source])
            manifest = {
                "created_at_utc": utc_now(), "bundle_id": bundle_id, "descriptor": descriptor, "code": code,
                "parent_bundle": parent.relative_to(root).as_posix(), "previous_bundle": previous.relative_to(root).as_posix(),
                "producing_parent_manifest_sha256": sha256_file(parent / "manifest.json"),
                "feature_names": FEATURE_NAMES, "new_features": CATEGORIES + NUMERIC,
                "rows": counts, "source_ids_sha256": source_ids, "gate_rows": gate_counts,
                "audit_used": False, "frozen_folds": original["folds"],
                "ranking_ids_sha256": original["ranking_ids_sha256"], "artifacts": artifacts,
            }
    if not (destination / "manifest.json").is_file():
        manifest.update(environment=environment(root), resources=monitor.stats)
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    write_json(output / "latest.json", {"bundle": destination.relative_to(root).as_posix(), "bundle_id": bundle_id})
    emit("phase4_bundle_complete", bundle=str(destination), predictors=len(FEATURE_NAMES), resources=monitor.stats)
    return destination


def load_bundle(root: Path, base: Path) -> tuple[dict, Path, dict, pd.DataFrame]:
    manifest = read_json(base / "manifest.json")
    if manifest["audit_used"] or manifest["bundle_id"] != object_hash(manifest["descriptor"]):
        raise ValueError("Feature descriptor/audit mismatch.")
    if manifest["code"]["sha256"] != feature_code_receipt(root)["sha256"]:
        raise ValueError("Feature code changed; build a fresh bundle.")
    parent, previous = root / manifest["parent_bundle"], root / manifest["previous_bundle"]
    original, catalog = load_parent(root, parent)
    old = load_previous(root, previous, parent, original)
    validate_feature_config(manifest["descriptor"]["config"], original, old)
    if object_hash(scientific_parent(original)) != manifest["descriptor"]["parent_scientific_sha256"]:
        raise ValueError("Original parent scientific identity changed.")
    if sha256_file(previous / "manifest.json") != manifest["descriptor"]["previous_manifest_sha256"]:
        raise ValueError("Previous parent manifest changed.")
    verify_receipts(base, manifest["artifacts"])
    return manifest, parent, original, catalog


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--config", type=Path, default=Path("configs/features_phase4_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/phase4"))
    args = parser.parse_args()
    prepare(root, root / args.parent, root / args.previous, root / args.config, root / args.output)
