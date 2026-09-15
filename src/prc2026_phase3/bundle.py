"""Immutable phase-3 features layered on the original validation populations."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from prc2026.artifacts import (ResourceMonitor, code_receipt, environment, object_hash,
                               read_json, sha256_file, utc_now, verify_receipts, versions, write_json)
from prc2026.features import FEATURE_NAMES, ID, PHASE, TARGET, TIME
from prc2026.phase1 import emit, validate_inputs, write_parquet
from prc2026.validation import identity_hash, validate_protocol
from .features import (ALL_FEATURE_NAMES, CONTEXT_COLUMNS, FAMILIES, INPUT_COLUMNS,
                       POLICY, row_features, traffic_features)

DEFAULT_PARENT = Path("data/phase1/58d621b245de3f5e12dd")


def feature_code_receipt(root: Path) -> dict:
    paths = [root / "src/prc2026_phase3" / name for name in ["__init__.py", "features.py", "bundle.py"]]
    paths += [root / "tools/prepare_phase3.py", root / "requirements-phase3.txt"]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def load_parent(root: Path, parent: Path) -> tuple[dict, pd.DataFrame]:
    manifest = read_json(parent / "manifest.json")
    validate_protocol(manifest["descriptor"]["protocol"])
    if code_receipt(root)["sha256"] != manifest["code"]["sha256"]:
        raise ValueError("The original feature/split code changed.")
    verify_receipts(parent, manifest["artifacts"])
    return manifest, pd.read_parquet(parent / "train_catalog.parquet")


def validate_feature_config(config: dict, parent_manifest: dict) -> None:
    protocol = parent_manifest["descriptor"]["protocol"]
    if config["feature_policy"] != POLICY or config["parent_bundle_id"] != parent_manifest["bundle_id"]:
        raise ValueError("Feature policy or parent identity mismatch.")
    if set(config["training_months"]) & set(protocol["reserved_audit_months"]):
        raise ValueError("Feature development cannot consume the reserved audit period.")
    if config["training_months"] != [f"2025-{m:02}" for m in range(1, 12)]:
        raise ValueError("Phase 3 must preserve the January-November development population.")
    if config["ranking_months"] != ["2026-01", "2026-07"]:
        raise ValueError("Ranking months differ from the audited data.")
    if config["departure_labels_used_in_features"] or config["audit_features_materialized"]:
        raise ValueError("Unimplemented target/audit feature access.")
    if config["last_event_max_gap_seconds"] != 21600 or config["external_data"]:
        raise ValueError("Unimplemented feature configuration.")


def prepare(root: Path, parent: Path, config_path: Path, output: Path) -> Path:
    config = read_json(config_path)
    with ResourceMonitor() as monitor:
        original, catalog = load_parent(root, parent)
        validate_feature_config(config, original)
        inputs = validate_inputs(root)
        code = feature_code_receipt(root)
        descriptor = {
            "parent_bundle_id": original["bundle_id"],
            "parent_manifest_sha256": sha256_file(parent / "manifest.json"),
            "input_files": inputs, "config": config, "code_sha256": code["sha256"],
            "transform_packages": {k: v for k, v in versions().items()
                                   if k in {"numpy", "pandas", "pyarrow", "tzdata"}},
        }
        bundle_id = object_hash(descriptor)
        destination = output / bundle_id[:20]
        if (destination / "manifest.json").exists():
            existing = read_json(destination / "manifest.json")
            if existing["descriptor"] != descriptor or existing["bundle_id"] != bundle_id:
                raise ValueError("Existing feature bundle descriptor mismatch.")
            verify_receipts(destination, existing["artifacts"])
            write_json(output / "latest.json", {"bundle": destination.relative_to(root).as_posix(), "bundle_id": bundle_id})
            emit("verified_phase3_bundle", bundle=str(destination))
            return destination
        output.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=bundle_id[:20] + ".incomplete-", dir=output))
        allowed = catalog[TIME].dt.strftime("%Y-%m").isin(config["training_months"])
        sources = sorted(catalog.loc[allowed, "source_file"].unique()) + ["ranking.parquet"]
        context_parts = []
        # Calendar filtering also handles the single November arrival stored in
        # the December file. Only explicitly named arrival context is read.
        for receipt in original["artifacts"]:
            if not receipt["path"].startswith("context/"):
                continue
            frame = pd.read_parquet(parent / receipt["path"], columns=CONTEXT_COLUMNS)
            months = config["ranking_months"] if receipt["path"] == "context/ranking.parquet" else config["training_months"]
            context_parts.append(frame.loc[frame[TIME].dt.strftime("%Y-%m").isin(months)])
        context = pd.concat(context_parts, ignore_index=True)
        del context_parts, frame
        row_parts, query_parts, row_counts, source_ids = [], [], {}, {}
        for source in sources:
            raw = pd.read_parquet(root / "data/raw" / source, columns=INPUT_COLUMNS)
            months = config["ranking_months"] if source == "ranking.parquet" else config["training_months"]
            raw = raw.loc[raw[TIME].dt.strftime("%Y-%m").isin(months)]
            features = row_features(raw)
            if source == "ranking.parquet":
                expected_ids = original["ranking_ids_sha256"]
            else:
                expected_ids = identity_hash(catalog.loc[allowed & catalog["source_file"].eq(source), ID])
            if identity_hash(features[ID]) != expected_ids:
                raise ValueError("Feature preparation changed the protected ID population.")
            # Existing predictors must remain byte-for-value equivalent; this
            # makes ablations comparable to phase 2 and catches row misalignment.
            baseline = pd.read_parquet(parent / "features" / source)
            baseline = baseline.loc[baseline[ID].isin(features[ID])].reset_index(drop=True)
            pd.testing.assert_frame_equal(features[[ID, *FEATURE_NAMES]], baseline, check_exact=True)
            row_parts.append(features)
            query_parts.append(raw.loc[raw[PHASE].eq("DEP"), [ID, TIME, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]])
            row_counts[source] = len(features)
            source_ids[source] = expected_ids
            emit("phase3_rows_prepared", source=source, rows=len(features))
        queries = pd.concat(query_parts, ignore_index=True)
        del query_parts, raw, features, baseline
        gc.collect()
        emit("phase3_traffic_preparing", movements=len(context), departures=len(queries))
        traffic = traffic_features(context, queries)
        context_rows = len(context)
        del context, queries
        gc.collect()
        artifacts, position = [], 0
        for source, features in zip(sources, row_parts):
            stop = position + len(features)
            part = traffic.iloc[position:stop].reset_index(drop=True)
            if not np.array_equal(features[ID].to_numpy(), part[ID].to_numpy()):
                raise ValueError("Traffic feature row order changed.")
            features = pd.concat([features, part.drop(columns=ID)], axis=1)[[ID, *ALL_FEATURE_NAMES]]
            write_parquet(temporary, "features/" + source, features, artifacts)
            position = stop
        if position != len(traffic):
            raise ValueError("Incomplete traffic feature export.")
    manifest = {
        "created_at_utc": utc_now(), "bundle_id": bundle_id, "descriptor": descriptor, "code": code,
        "parent_bundle": parent.relative_to(root).as_posix(), "feature_names": ALL_FEATURE_NAMES,
        "families": FAMILIES, "rows": row_counts, "source_ids_sha256": source_ids,
        "context_rows": context_rows, "audit_used": False,
        "frozen_folds": original["folds"], "ranking_ids_sha256": original["ranking_ids_sha256"],
        "artifacts": artifacts, "environment": environment(root), "resources": monitor.stats,
    }
    write_json(temporary / "manifest.json", manifest)
    temporary.rename(destination)
    write_json(output / "latest.json", {"bundle": destination.relative_to(root).as_posix(), "bundle_id": bundle_id})
    emit("phase3_features_complete", bundle=str(destination), features=len(ALL_FEATURE_NAMES), resources=monitor.stats)
    return destination


def load_bundle(root: Path, base: Path) -> tuple[dict, Path, dict, pd.DataFrame]:
    manifest = read_json(base / "manifest.json")
    if manifest["audit_used"] or manifest["code"]["sha256"] != feature_code_receipt(root)["sha256"]:
        raise ValueError("Feature code changed or audit data was consumed; build a fresh bundle.")
    if manifest["bundle_id"] != object_hash(manifest["descriptor"]):
        raise ValueError("Feature descriptor identity mismatch.")
    parent = root / manifest["parent_bundle"]
    original, catalog = load_parent(root, parent)
    if sha256_file(parent / "manifest.json") != manifest["descriptor"]["parent_manifest_sha256"]:
        raise ValueError("The parent feature/split manifest changed.")
    validate_feature_config(manifest["descriptor"]["config"], original)
    verify_receipts(base, manifest["artifacts"])
    return manifest, parent, original, catalog


def load_partition(base: Path, parent: Path, catalog: pd.DataFrame, assignment: pd.DataFrame,
                   role: str, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = assignment.loc[assignment["partition"].eq(role), ID]
    metadata = catalog.loc[catalog[ID].isin(ids)]
    if metadata.empty or len(metadata) != len(ids) or metadata[TIME].dt.strftime("%Y-%m").eq("2025-12").any():
        raise ValueError("Missing, empty, or reserved audit partition.")
    feature_parts, label_parts = [], []
    for source, meta in metadata.groupby("source_file", sort=True):
        features = pd.read_parquet(base / "features" / source, columns=[ID, *names])
        labels = pd.read_parquet(parent / "labels" / source)
        feature_parts.append(features.loc[features[ID].isin(meta[ID])])
        label_parts.append(labels.loc[labels[ID].isin(meta[ID])])
    features = pd.concat(feature_parts, ignore_index=True)
    labels = pd.concat(label_parts, ignore_index=True)
    truth = metadata.merge(labels, on=ID, how="left", validate="one_to_one")
    if identity_hash(features[ID]) != identity_hash(ids) or len(features) != len(ids):
        raise ValueError("Feature partition ID mismatch.")
    if not np.isfinite(truth[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite or unaligned labels.")
    return features, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--config", type=Path, default=Path("configs/features_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/phase3"))
    args = parser.parse_args()
    root = args.root.resolve()
    prepare(root, root / args.parent, root / args.config, root / args.output)
