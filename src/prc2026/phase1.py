"""Prepare immutable feature/split bundles and exercise the evaluation pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .artifacts import (ResourceMonitor, code_receipt, environment, object_hash,
                        read_json, sha256_file, utc_now, verify_receipts, versions,
                        write_json)
from .features import (BLOCK, CATEGORY_SOURCES, EXCLUDED_PREDICTORS, FEATURE_NAMES,
                       ID, NUMERIC_FEATURES, PHASE, POLICY, TARGET, TIME, TIMEZONES,
                       departure_features, movement_context, ranking_view,
                       validate_movements)
from .validation import (assign_fold, evaluate, identity_hash,
                         require_development_fold, validate_protocol)


def emit(event: str, **details) -> None:
    print(json.dumps({"event": event, **details}, allow_nan=False), flush=True)


def write_parquet(base: Path, relative: str, frame: pd.DataFrame, receipts: list) -> None:
    path = base / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    receipts.append({"path": relative, "rows": len(frame),
                     "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})


def validate_inputs(root: Path) -> list[dict]:
    manifest = read_json(root / "docs/competition/download_manifest.json")
    if manifest["endpoint"] != "https://s3.opensky-network.org":
        raise ValueError("Unexpected data provenance endpoint.")
    records = sorted(manifest["objects"], key=lambda x: x["key"])
    expected = {"ranking.parquet", "submitting.parquet"}
    expected.update(f"training_2025-{month:02d}-01_"
                    f"{2026 if month == 12 else 2025}-{month % 12 + 1:02d}-01.parquet"
                    for month in range(1, 13))
    if len(records) != 14 or {x["key"] for x in records} != expected:
        raise ValueError("This protocol expects the 14 audited PRC input files.")
    receipts = []
    for record in records:
        if record["bucket"] != "prc-2026-datasets":
            raise ValueError("Unexpected input bucket.")
        receipts.append({"path": record["key"], "sha256": record["sha256"],
                         "size_bytes": record["size_bytes"]})
    verify_receipts(root / "data/raw", receipts)
    return receipts


def catalog_part(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    dep = frame.loc[frame[PHASE].eq("DEP")]
    out = dep[[ID, "FLIGHT_ID_mvt", TIME]].copy()
    out["airport"] = dep["ADEP_mvt"].astype("string")
    out["source_file"] = source
    out["nm_missing"] = dep["ADEP_flt"].isna()
    return out.reset_index(drop=True)


def _prepare(root: Path, config: Path, output: Path) -> tuple[Path, dict, bool]:
    protocol = read_json(config)
    validate_protocol(protocol)
    if protocol["feature_policy"] != POLICY:
        raise ValueError("Protocol and implemented feature policy disagree.")
    inputs = validate_inputs(root)
    code = code_receipt(root)
    descriptor = {"input_files": inputs, "protocol": protocol, "code_sha256": code["sha256"],
                  "transform_packages": {k: v for k, v in versions().items()
                                         if k in {"numpy", "pandas", "pyarrow", "tzdata"}}}
    bundle_id = object_hash(descriptor)
    base = output / bundle_id[:20]
    if (base / "manifest.json").exists():
        manifest = read_json(base / "manifest.json")
        if manifest["bundle_id"] != bundle_id or manifest["descriptor"] != descriptor:
            raise ValueError("Bundle descriptor mismatch.")
        verify_receipts(base, manifest["artifacts"])
        emit("verified_existing_bundle", bundle=str(base))
        return base, manifest, True
    output.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix=bundle_id[:20] + ".incomplete-", dir=output))
    # Interrupted builds remain separate; only a completed bundle is published.
    receipts, sources, train_parts, ranking_parts = [], [], [], []
    for record in inputs:
        source = record["path"]
        if source == "submitting.parquet":
            continue
        started = time.perf_counter()
        frame = pd.read_parquet(root / "data/raw" / source)
        validate_movements(frame)
        dep = frame[PHASE].eq("DEP")
        is_training = source.startswith("training_")
        if is_training:
            if not np.isfinite(frame.loc[dep, TARGET].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite training departure labels in {source}.")
            exact_label = (frame.loc[dep, TIME] - frame.loc[dep, BLOCK]).dt.total_seconds()
            if not np.array_equal(exact_label.to_numpy(), frame.loc[dep, TARGET].to_numpy()):
                raise ValueError("Training label no longer matches its documented definition.")
            write_parquet(base, f"labels/{source}", frame.loc[dep, [ID, TARGET]], receipts)
        elif frame.loc[dep, [TARGET, BLOCK]].notna().any().any():
            raise ValueError("Ranking departure labels/block times are unexpectedly available.")
        view = ranking_view(frame)
        features = departure_features(view)
        if set(features) != {ID, *FEATURE_NAMES} or set(FEATURE_NAMES) & EXCLUDED_PREDICTORS:
            raise ValueError("Feature contract violation.")
        context = movement_context(view)
        write_parquet(base, f"features/{source}", features, receipts)
        write_parquet(base, f"context/{source}", context, receipts)
        catalog = catalog_part(frame, source)
        (train_parts if is_training else ranking_parts).append(catalog)
        source_month = re.search(r"training_(\d{4}-\d{2})-", source)
        spill = {}
        if source_month:
            mismatched = frame[TIME].dt.strftime("%Y-%m").ne(source_month.group(1))
            spill = {str(k): int(v) for k, v in frame.loc[mismatched, PHASE].value_counts().items()}
        report = {
            "source": source, "movements": len(frame), "departures": len(features),
            "movement_time_min_utc": frame[TIME].min().isoformat(),
            "movement_time_max_utc": frame[TIME].max().isoformat(),
            "rows_outside_filename_month_by_phase": spill,
            "wall_seconds": time.perf_counter() - started,
        }
        sources.append(report)
        emit("prepared_source", **report)
        del frame, view, features, context, catalog
        gc.collect()
    train = pd.concat(train_parts, ignore_index=True)
    ranking = pd.concat(ranking_parts, ignore_index=True)
    del train_parts, ranking_parts
    if train[ID].duplicated().any() or ranking[ID].duplicated().any():
        raise ValueError("Global departure ID collision.")
    if train[ID].isin(ranking[ID]).any():
        raise ValueError("Training/ranking departure ID overlap.")
    template = pd.read_parquet(root / "data/raw/submitting.parquet")
    if list(template) != [ID, TARGET] or template[TARGET].notna().any():
        raise ValueError("Unexpected submission template.")
    if not np.array_equal(template[ID].to_numpy(), ranking[ID].to_numpy()):
        raise ValueError("Ranking features do not preserve exact template ID order.")
    template_schema = pq.read_schema(root / "data/raw/submitting.parquet")
    if str(template_schema.field(ID).type) != "double" or str(template_schema.field(TARGET).type) != "int32":
        raise ValueError("Submission template types have changed.")
    write_parquet(base, "train_catalog.parquet", train, receipts)
    write_parquet(base, "ranking_catalog.parquet", ranking, receipts)
    folds = {}
    for fold in protocol["folds"]:
        assignment, report = assign_fold(train, protocol, fold)
        write_parquet(base, f"splits/{fold}.parquet", assignment[[ID, "partition"]], receipts)
        report["movement_time_ranges_utc"] = {
            str(role): {"min": part[TIME].min().isoformat(), "max": part[TIME].max().isoformat()}
            for role, part in assignment.groupby("partition")
        }
        folds[fold] = report
        emit("frozen_fold", fold=fold, counts=report["counts"],
             group_overlap_counts=report["group_overlap_counts"])
        del assignment
        gc.collect()
    month_counts = ranking[TIME].dt.month.value_counts().sort_index()
    if set(month_counts.index) != {1, 7}:
        raise ValueError("Ranking seasonal month support has changed.")
    manifest = {
        "bundle_id": bundle_id, "created_at_utc": utc_now(), "descriptor": descriptor,
        "code": code, "sources": sources, "artifacts": receipts, "folds": folds,
        "feature_contract": {
            "policy": POLICY, "predictors": FEATURE_NAMES,
            "categorical_sources": CATEGORY_SOURCES, "numeric_dtype": "float32",
            "categorical_missing_value": "__MISSING__", "timezones": TIMEZONES,
            "excluded_predictors": sorted(EXCLUDED_PREDICTORS),
            "departure_targets_and_block_times_masked_before_transform": True,
            "arrival_context_saved_separately": True,
        },
        "training_departures": len(train), "ranking_departures": len(ranking),
        "training_ids_sha256": identity_hash(train[ID]),
        "ranking_ids_sha256": identity_hash(ranking[ID]),
        "ranking_template_order_matches": True,
        "ranking_month_counts": {str(k): int(v) for k, v in month_counts.items()},
        "ranking_month_weights": {str(k): float(v / len(ranking)) for k, v in month_counts.items()},
    }
    return base, manifest, False


def prepare(root: Path, config: Path, output: Path, label: str) -> Path:
    with ResourceMonitor() as monitor:
        base, manifest, reused = _prepare(root, config, output)
    if not reused:
        manifest["preparation"] = {"label": label, "environment": environment(root),
                                   "resources": monitor.stats}
        write_json(base / "manifest.json", manifest)  # completion marker, written last
        completed = output / manifest["bundle_id"][:20]
        base.rename(completed)
        base = completed
    emit("preparation_complete", bundle=str(base), reused=reused, resources=monitor.stats)
    return base


def load_partition(base: Path, catalog: pd.DataFrame, assignment: pd.DataFrame,
                   role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = assignment.loc[assignment["partition"].eq(role), ID]
    metadata = catalog.loc[catalog[ID].isin(ids)]
    if metadata.empty or len(metadata) != len(ids):
        raise ValueError(f"Missing or empty {role} partition.")
    feature_parts, label_parts = [], []
    for source, meta in metadata.groupby("source_file", sort=True):
        feature = pd.read_parquet(base / "features" / source)
        labels = pd.read_parquet(base / "labels" / source)
        if list(feature) != [ID, *FEATURE_NAMES] or TARGET in feature or BLOCK in feature:
            raise ValueError("Invalid materialized feature schema.")
        feature_parts.append(feature.loc[feature[ID].isin(meta[ID])])
        label_parts.append(labels.loc[labels[ID].isin(meta[ID])])
    features = pd.concat(feature_parts, ignore_index=True)
    labels = pd.concat(label_parts, ignore_index=True)
    if len(features) != len(ids) or len(labels) != len(ids):
        raise ValueError("Incomplete materialized partition.")
    truth = metadata.merge(labels, on=ID, how="left", validate="one_to_one")
    if not np.isfinite(truth[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite materialized target.")
    if identity_hash(features[ID]) != identity_hash(ids):
        raise ValueError("Feature partition ID mismatch.")
    return features, truth


def benchmark(root: Path, base: Path, fold: str, output: Path, label: str) -> Path:
    with ResourceMonitor() as monitor:
        manifest = read_json(base / "manifest.json")
        protocol = manifest["descriptor"]["protocol"]
        validate_protocol(protocol)
        require_development_fold(protocol, fold)
        if manifest["code"]["sha256"] != code_receipt(root)["sha256"]:
            raise ValueError("Code changed after preparation; build a fresh bundle.")
        verify_receipts(base, manifest["artifacts"])
        catalog = pd.read_parquet(base / "train_catalog.parquet")
        assignment = pd.read_parquet(base / "splits" / f"{fold}.parquet")
        fit_features, fit_truth = load_partition(base, catalog, assignment, "fit")
        fit = fit_features[[ID, "airport"]].merge(fit_truth[[ID, TARGET]], on=ID, validate="one_to_one")
        means = {str(k): float(v) for k, v in fit.groupby("airport")[TARGET].mean().items()}
        global_mean = float(fit[TARGET].mean())
        model = {"kind": "airport_mean", "fit_rows": len(fit), "airport_means_seconds": means,
                 "global_fallback_seconds": global_mean, "target_filter": "none",
                 "fit_ids_sha256": identity_hash(fit[ID])}
        materialized_bytes = int(fit_features.memory_usage(index=True, deep=True).sum())
        del fit_features, fit_truth, fit
        gc.collect()
        features, truth = load_partition(base, catalog, assignment, "score")
        predictions = pd.DataFrame({ID: features[ID],
                                    "prediction": features["airport"].map(means).fillna(global_mean).astype(float)})
        if identity_hash(truth[ID]) != manifest["folds"][fold]["identity_sha256"]["score"]:
            raise ValueError("Score population differs from its frozen identity.")
        metrics = evaluate(truth, predictions,
                           manifest["ranking_month_weights"] if fold == "seasonal" else None)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{fold}_airport_mean"
        run = output / run_id
        run.mkdir(parents=True, exist_ok=False)
        predictions.to_parquet(run / "predictions.parquet", index=False, compression="zstd")
        write_json(run / "model.json", model)
        write_json(run / "metrics.json", metrics)
        files = [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in [run / "predictions.parquet", run / "model.json", run / "metrics.json"]]
    report = {
        "run_id": run_id, "created_at_utc": utc_now(), "label": label,
        "purpose": "Phase-1 plumbing and capacity check; not a tuned competition baseline",
        "bundle_id": manifest["bundle_id"], "protocol_id": protocol["protocol_id"],
        "fold": fold, "parameters": {"model": "airport_mean", "seed": None,
                                    "early_stopping": "unused by an arithmetic group mean"},
        "environment": environment(root), "resources": monitor.stats,
        "materialized_fit_feature_bytes": materialized_bytes,
        "fit_rows": model["fit_rows"], "score_rows": len(predictions),
        "code": manifest["code"], "artifacts": files,
    }
    write_json(run / "run.json", report)
    emit("benchmark_complete", run=str(run), fold=fold, metrics=metrics["overall"],
         resources=monitor.stats)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, default=Path("configs/validation_v1.json"))
    prep.add_argument("--output", type=Path, default=Path("data/phase1"))
    prep.add_argument("--label", default="local")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--bundle", type=Path, required=True)
    bench.add_argument("--fold", choices=["seasonal", "forward", "final_audit"], required=True)
    bench.add_argument("--output", type=Path, default=Path("runs/phase1"))
    bench.add_argument("--label", default="local")
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        prepare(root, root / args.config, root / args.output, args.label)
    elif args.command == "benchmark":
        benchmark(root, root / args.bundle, args.fold, root / args.output, args.label)
    else:
        base = root / args.bundle
        manifest = read_json(base / "manifest.json")
        verify_receipts(base, manifest["artifacts"])
        emit("bundle_verified", bundle=str(base), artifacts=len(manifest["artifacts"]),
             bundle_id=manifest["bundle_id"])
