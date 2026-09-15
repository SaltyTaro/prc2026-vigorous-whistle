"""Prepare masked predictors separately from labels and original routing metadata."""
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prc2026.artifacts import (read_json, write_json, sha256_file, object_hash,
                              verify_receipts, ResourceMonitor, utc_now)
from prc2026.features import ID, TIME, PHASE
from prc2026.validation import identity_hash
from prc2026_phase5r.features import GROUND_NAMES
from .features import (AIRPORT_INPUTS, FEATURE_NAMES, UNCHANGED_NAMES,
                       masked_row_features, masked_ground_features, eligible_route)
from .common import configurations, code_receipt


def prepare(root):
    feature, _ = configurations(root)
    for path, digest in feature["input_receipts_sha256"].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"Frozen feature input receipt changed: {path}")
    previous = root / feature["weather_bundle"]
    route_base = root / feature["routing_bundle"]
    original_base = root / feature["original_bundle"]
    old, route_manifest, original = [read_json(p / "manifest.json") for p in [previous, route_base, original_base]]
    for base, manifest in [(previous, old), (route_base, route_manifest), (original_base, original)]:
        verify_receipts(base, manifest["artifacts"])
    originals = read_json(root / "docs/next_lever/preservation_final.json")["original_input_hashes"]
    for name, digest in originals.items():
        if sha256_file(root / "data/raw" / name) != digest:
            raise ValueError("An original input changed.")
    code = code_receipt(root, feature=True)
    descriptor = {"config": feature, "code_sha256": code["sha256"], "original_input_sha256": originals,
                  "weather_bundle_id": old["bundle_id"], "routing_bundle_id": route_manifest["bundle_id"],
                  "original_bundle_id": original["bundle_id"]}
    identifier = object_hash(descriptor)
    output = root / "data/nm_mask" / identifier[:20]
    if output.exists():
        raise ValueError("A masking bundle already exists; do not replace it.")
    (output / "features").mkdir(parents=True)
    (output / "routing").mkdir()
    artifacts, row_parts, contexts, query_parts, coverage = [], {}, [], [], {}
    def save(name, frame):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
        artifacts.append({"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "rows": len(frame)})
    with ResourceMonitor() as monitor:
        sources = sorted(originals)
        for source in sources:
            if source == "submitting.parquet":
                continue
            raw = pd.read_parquet(root / "data/raw" / source, columns=AIRPORT_INPUTS)
            months = feature["ranking_months"] if source == "ranking.parquet" else feature["training_months"]
            raw = raw.loc[raw[TIME].dt.strftime("%Y-%m").isin(months)].reset_index(drop=True)
            contexts.append(raw)
            source_path = previous / "features" / source
            if not source_path.exists():
                if raw[PHASE].eq("DEP").any():
                    raise ValueError("A non-reserved departure has no original feature identity.")
                continue
            inherited = pd.read_parquet(source_path, columns=[ID, *UNCHANGED_NAMES, "conditional_schedule_offset_s"])
            rows = masked_row_features(raw)
            if not np.array_equal(rows[ID], inherited[ID]):
                raise ValueError("The masked feature population differs from the original weather bundle.")
            routing = pd.read_parquet(route_base / "features" / source, columns=[ID, "nm_missing", "nmclock_available"])
            if not np.array_equal(rows[ID], routing[ID]):
                raise ValueError("Original routing metadata does not align.")
            natural = eligible_route(routing)
            if not np.array_equal(rows.conditional_schedule_offset_s.to_numpy()[natural],
                                  inherited.conditional_schedule_offset_s.to_numpy()[natural]):
                raise ValueError("The existing conditional schedule rule changed on eligible rows.")
            frame = rows.merge(inherited[[ID, *UNCHANGED_NAMES]], on=ID, validate="one_to_one", sort=False)
            row_parts[source] = frame
            query_parts.append(frame[[ID]])
            save("routing/" + source, routing)
            coverage[source] = {"rows": len(frame), "natural_missing_rows": int(natural.sum())}
            print(f"masked_rows_prepared source={source} rows={len(frame)} natural={natural.sum()}", flush=True)
        context = pd.concat(contexts, ignore_index=True)
        del contexts, raw, frame, rows, routing, inherited
        gc.collect()
        save("inputs/airport_context.parquet", context)
        queries = pd.concat(query_parts, ignore_index=True)[ID]
        ground = masked_ground_features(context, queries)
        del context, query_parts, queries
        gc.collect()
        position = 0
        for source, rows in row_parts.items():
            part = ground.iloc[position:position + len(rows)].reset_index(drop=True)
            if not np.array_equal(rows[ID], part[ID]):
                raise ValueError("Masked ground features lost their query identities.")
            frame = pd.concat([rows, part[GROUND_NAMES]], axis=1)[[ID, *FEATURE_NAMES]]
            save("features/" + source, frame)
            position += len(rows)
        if position != len(ground):
            raise ValueError("Incomplete masked ground output.")
    manifest = {"created_at_utc": utc_now(), "bundle_id": identifier, "descriptor": descriptor, "code": code,
        "feature_names": FEATURE_NAMES, "original_bundle": feature["original_bundle"],
        "weather_bundle": feature["weather_bundle"], "routing_bundle": feature["routing_bundle"],
        "coverage": coverage, "artifacts": artifacts, "rows": sum(v["rows"] for v in coverage.values()),
        "frozen_folds": original["folds"], "ranking_ids_sha256": original["ranking_ids_sha256"],
        "audit_used": False, "departure_labels_used": False, "resources": monitor.stats}
    write_json(output / "manifest.json", manifest)
    write_json(root / "data/nm_mask/latest.json", {"bundle": output.relative_to(root).as_posix(), "bundle_id": identifier})
    print(f"masked_features_complete bundle={output.relative_to(root)} rows={manifest['rows']}", flush=True)
    return output


def load_bundle(root):
    pointer = read_json(root / "data/nm_mask/latest.json")
    base = root / pointer["bundle"]
    manifest = read_json(base / "manifest.json")
    if manifest["bundle_id"] != object_hash(manifest["descriptor"]) or manifest["code"]["sha256"] != code_receipt(root, feature=True)["sha256"]:
        raise ValueError("Masked feature identity changed.")
    if manifest["audit_used"] or manifest["departure_labels_used"]:
        raise ValueError("The feature bundle includes prohibited outcomes.")
    verify_receipts(base, manifest["artifacts"])
    return base, manifest
