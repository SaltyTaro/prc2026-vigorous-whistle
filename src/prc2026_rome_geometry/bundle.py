"""Small immutable geometry extension bound to the full frozen timeline bundle."""

from pathlib import Path
import argparse
import gc
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prc2026.artifacts import ResourceMonitor, environment, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TIME, PHASE
from prc2026.phase1 import emit, validate_inputs, write_parquet
from prc2026.validation import identity_hash
from prc2026_timeline.bundle import load_bundle as load_previous
from prc2026_source_geometry.geometry_review import REVIEW_CONFIG
from prc2026_source_geometry.endpoint_review import explicit_controls
from .common import configurations, FEATURE_CONFIG
from .features import INPUT_COLUMNS, EXTRA_NAMES, build_map, make_features


def feature_code_receipt(root):
    paths = [root / "src/prc2026_rome_geometry" / name for name in ["__init__.py", "features.py", "common.py", "bundle.py"]]
    paths += [root / "src/prc2026_source_geometry" / name for name in ["geometry_audit.py", "geometry_review.py", "endpoint_review.py"]]
    paths += [root / "tools/prepare_rome_geometry.py", root / "requirements-rome-geometry.txt"]
    files = {path.relative_to(root).as_posix():sha256_file(path) for path in paths}
    return {"files":files, "sha256":object_hash(files)}


def load_maps(root, config):
    review = read_json(root / REVIEW_CONFIG)
    audit = read_json(root / "docs/source_geometry/geometry_endpoint_review.json")
    gate = audit["airports"]["LIRF"]
    if not gate["endpoint_inference_permitted"] or gate["control_distinct_ways"] < 30 or gate["last_node_control_fraction"] < .99:
        raise ValueError("LIRF endpoint controls must pass the frozen audit.")
    maps, sources = {}, {}
    for stamp in sorted(set(config["snapshot_assignment"].values())):
        receipt = audit["source_receipts"]["LIRF/" + stamp]
        path = root / receipt["path"]
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError("Historical map bytes changed.")
        elements = read_json(path)["elements"]
        if any(item.get("timestamp", "") > stamp for item in elements):
            raise ValueError("A historical map contains a later edit.")
        controls = explicit_controls(elements, 1.)
        if any(not item["last_node_convention_passes"] for item in controls):
            raise ValueError("A LIRF historical endpoint contradicts its control.")
        maps[stamp] = build_map(elements, review, config)
        sources[stamp] = receipt
    return maps, sources


def read_context(root, config, old):
    frames, ids_by_source = [], {}
    for source in old["rows"]:
        raw = pq.read_table(root / "data/raw" / source, columns=INPUT_COLUMNS,
            filters=[("ADEP_mvt", "=", "LIRF"), (PHASE, "=", "DEP")]).to_pandas()
        allowed = config["ranking_months"] if source == "ranking.parquet" else config["training_months"]
        if not raw[TIME].dt.strftime("%Y-%m").isin(allowed).all():
            raise ValueError("A reserved movement entered geometry context.")
        frames.append(raw)
        parent_ids = pd.read_parquet(root / config["previous_bundle"] / "features" / source, columns=[ID])[ID]
        ids_by_source[source] = parent_ids.loc[parent_ids.isin(raw[ID])].reset_index(drop=True)
    return pd.concat(frames, ignore_index=True), ids_by_source


def prepare(root):
    config, _ = configurations(root)
    previous = root / config["previous_bundle"]
    with ResourceMonitor() as monitor:
        old, original_base, original, catalog = load_previous(root, previous)
        if old["bundle_id"] != config["previous_bundle_id"] or sha256_file(previous / "manifest.json") != config["previous_manifest_sha256"]:
            raise ValueError("Frozen parent bundle changed.")
        descriptor = {"config":config, "code_sha256":feature_code_receipt(root)["sha256"],
            "previous_manifest_sha256":sha256_file(previous / "manifest.json"), "input_files":validate_inputs(root)}
        bundle_id = object_hash(descriptor)
        output = root / "data/rome_geometry"
        destination = output / bundle_id[:20]
        if (destination / "manifest.json").exists():
            saved = read_json(destination / "manifest.json")
            if saved["descriptor"] != descriptor:
                raise ValueError("Existing bundle descriptor differs.")
            verify_receipts(destination, saved["artifacts"])
            return destination
        if destination.exists() or shutil.disk_usage(root).free < 850_000_000:
            raise ValueError("Preserve existing destination and maintain 850 MB free before preparation.")
        maps, map_sources = load_maps(root, config)
        raw, source_ids = read_context(root, config, old)
        query_ids = pd.concat(list(source_ids.values()), ignore_index=True)
        emit("rome_geometry_preparing", context_rows=len(raw), focal_rows=len(query_ids), added_predictors=len(EXTRA_NAMES))
        extra = make_features(raw, query_ids, maps, config)
        context_rows = len(raw)
        del raw
        gc.collect()
        output.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=bundle_id[:20] + ".incomplete-", dir=output))
        artifacts, rows, hashes, coverage, pos = [], {}, {}, {}, 0
        for source, ids in source_ids.items():
            part = extra.iloc[pos:pos + len(ids)].reset_index(drop=True)
            pos += len(ids)
            if not np.array_equal(part[ID], ids):
                raise ValueError("Feature extension order changed.")
            write_parquet(temporary, "features/" + source, part, artifacts)
            rows[source], hashes[source] = len(part), identity_hash(part[ID])
            coverage[source] = {"rows":len(part), "parking_rows":int(part.geo_stand_available.sum()), "route_rows":int(part.geo_route_available.sum())}
        if pos != len(extra):
            raise ValueError("Feature export omitted rows.")
        route_records = {stamp:{ref + "|" + runway:{"features":item["features"], "nodes":item["nodes"], "edges":item["edges"]}
            for (ref, runway), item in mapped["routes"].items()} for stamp, mapped in maps.items()}
        write_json(temporary / "routes.json", route_records)
        artifacts.append({"path":"routes.json", "sha256":sha256_file(temporary / "routes.json"), "size_bytes":(temporary / "routes.json").stat().st_size})
        manifest = {"created_at_utc":utc_now(), "bundle_id":bundle_id, "descriptor":descriptor,
            "code":feature_code_receipt(root), "previous_bundle":config["previous_bundle"],
            "parent_bundle":original_base.relative_to(root).as_posix(), "feature_names":EXTRA_NAMES,
            "rows":rows, "source_ids_sha256":hashes, "parent_rows_unchanged":old["rows"],
            "frozen_folds":original["folds"], "ranking_ids_sha256":original["ranking_ids_sha256"],
            "coverage":coverage, "map_sources":map_sources, "context_rows":context_rows,
            "artifacts":artifacts, "audit_used":False, "departure_labels_used":False}
    manifest.update(resources=monitor.stats, environment=environment(root))
    write_json(temporary / "manifest.json", manifest)
    temporary.rename(destination)
    latest = output / "latest.json"
    if latest.exists():
        raise ValueError("Preserve the existing geometry bundle pointer.")
    write_json(latest, {"bundle":destination.relative_to(root).as_posix(), "bundle_id":bundle_id})
    emit("rome_geometry_bundle_complete", bundle=str(destination), coverage=coverage, resources=monitor.stats)
    return destination


def load_bundle(root, base=None):
    config, _ = configurations(root)
    base = base or root / read_json(root / "data/rome_geometry/latest.json")["bundle"]
    manifest = read_json(base / "manifest.json")
    if manifest["bundle_id"] != object_hash(manifest["descriptor"]) or manifest["code"] != feature_code_receipt(root) or manifest["audit_used"]:
        raise ValueError("Geometry bundle identity or frozen code changed.")
    if manifest["descriptor"]["config"] != config:
        raise ValueError("Geometry feature declaration changed.")
    previous = root / manifest["previous_bundle"]
    old, original_base, original, catalog = load_previous(root, previous)
    if sha256_file(previous / "manifest.json") != manifest["descriptor"]["previous_manifest_sha256"]:
        raise ValueError("Parent features changed.")
    if manifest["frozen_folds"] != original["folds"] or manifest["ranking_ids_sha256"] != original["ranking_ids_sha256"]:
        raise ValueError("Protected populations changed.")
    verify_receipts(base, manifest["artifacts"])
    extra = pd.concat([pd.read_parquet(base / "features" / source) for source in manifest["rows"]], ignore_index=True)
    if extra[ID].duplicated().any():
        raise ValueError("Duplicate IDs in the feature extension.")
    return base, manifest, previous, original_base, original, catalog, extra


def main():
    root = Path(__file__).resolve().parents[2]
    prepare(root)
