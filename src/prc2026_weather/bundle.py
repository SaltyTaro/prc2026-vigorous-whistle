"""Hash-bound weather inputs layered onto unchanged Phase-5R departure features."""
from __future__ import annotations
import argparse
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, environment, object_hash, read_json, sha256_file, utc_now, verify_receipts, versions, write_json
from prc2026.features import ID, TIME
from prc2026.phase1 import emit, validate_inputs, write_parquet
from prc2026.validation import identity_hash
from prc2026_phase5r.bundle import load_bundle as load_previous
from .features import FEATURE_NAMES, INPUT_COLUMNS, POLICY, PREVIOUS_NAMES, WEATHER_NAMES, load_observations, runway_geometry, verify_external, weather_features

DEFAULT_PREVIOUS = Path("data/phase5r/d4decaaa6d8e7915d821")


def feature_code_receipt(root: Path) -> dict:
    paths = [root/"src/prc2026_weather"/n for n in ["__init__.py", "features.py", "bundle.py"]]
    paths += [root/"tools/prepare_weather.py", root/"requirements-weather.txt"]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def validate_config(config: dict, old: dict):
    if config["feature_policy"] != POLICY or config["previous_bundle_id"] != old["bundle_id"]:
        raise ValueError("Feature declaration or parent bundle changed.")
    if config["training_months"] != [f"2025-{m:02}" for m in range(1, 12)] or config["ranking_months"] != ["2026-01", "2026-07"]:
        raise ValueError("Protected periods changed.")
    if config["audit_used"] or config["departure_labels_used_in_shared_features"] or config["input_columns"] != INPUT_COLUMNS:
        raise ValueError("Protected label-free weather contract changed.")


def prepare(root: Path, previous: Path, config_path: Path, output: Path) -> Path:
    config = read_json(config_path)
    with ResourceMonitor() as monitor:
        old, parent, original, catalog = load_previous(root, previous)
        validate_config(config, old)
        external, inputs, code = verify_external(root, config), validate_inputs(root), feature_code_receipt(root)
        descriptor = {"previous_bundle_id": old["bundle_id"], "previous_manifest_sha256": sha256_file(previous/"manifest.json"),
                      "input_files": inputs, "config": config, "code_sha256": code["sha256"],
                      "transform_packages": {k: v for k, v in versions().items() if k in {"numpy", "pandas", "pyarrow", "tzdata"}}}
        bundle_id = object_hash(descriptor)
        destination = output/bundle_id[:20]
        if (destination/"manifest.json").is_file():
            manifest = read_json(destination/"manifest.json")
            if manifest["descriptor"] != descriptor or manifest["bundle_id"] != bundle_id:
                raise ValueError("Existing weather bundle changed.")
            verify_receipts(destination, manifest["artifacts"])
        else:
            output.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=bundle_id[:20]+".incomplete-", dir=output))
            observations = load_observations(root, external["weather"])
            runway_path = next(x["path"] for x in external["runways"]["objects"] if x["path"].endswith("runways.csv"))
            geometry = runway_geometry(pd.read_csv(root/runway_path))
            artifacts, rows, identities, coverage = [], {}, {}, {}
            for source in old["rows"]:
                inherited = pd.read_parquet(previous/"features"/source, columns=[ID, *PREVIOUS_NAMES])
                raw = pd.read_parquet(root/"data/raw"/source, columns=INPUT_COLUMNS)
                months = config["ranking_months"] if source == "ranking.parquet" else config["training_months"]
                raw = raw.loc[raw[TIME].dt.strftime("%Y-%m").isin(months)]
                queries = inherited[[ID]].merge(raw, on=ID, how="left", validate="one_to_one", sort=False)
                extra = weather_features(queries, observations, geometry)
                if not np.array_equal(inherited[ID], extra[ID]) or identity_hash(extra[ID]) != old["source_ids_sha256"][source]:
                    raise ValueError("Weather feature row identity changed.")
                combined = pd.concat([inherited, extra.drop(columns=ID)], axis=1)[[ID, *FEATURE_NAMES]]
                pd.testing.assert_frame_equal(combined[[ID, *PREVIOUS_NAMES]], inherited, check_exact=True)
                write_parquet(temporary, "features/"+source, combined, artifacts)
                rows[source], identities[source] = len(combined), identity_hash(combined[ID])
                coverage[source] = {"rows": len(combined), "weather_missing_rows": int(extra["wx_missing"].sum()),
                                    "external_runway_bearing_rows": int(extra["wx_runway_external_bearing"].sum())}
                emit("weather_features_prepared", source=source, **coverage[source])
            manifest = {"created_at_utc": utc_now(), "bundle_id": bundle_id, "descriptor": descriptor, "code": code,
                        "previous_bundle": previous.relative_to(root).as_posix(), "parent_bundle": parent.relative_to(root).as_posix(),
                        "feature_names": FEATURE_NAMES, "new_features": WEATHER_NAMES, "rows": rows, "coverage": coverage,
                        "source_ids_sha256": identities, "audit_used": False, "frozen_folds": original["folds"],
                        "ranking_ids_sha256": original["ranking_ids_sha256"], "artifacts": artifacts}
    if not (destination/"manifest.json").is_file():
        manifest.update(environment=environment(root), resources=monitor.stats)
        write_json(temporary/"manifest.json", manifest)
        temporary.rename(destination)
    write_json(output/"latest.json", {"bundle": destination.relative_to(root).as_posix(), "bundle_id": bundle_id})
    emit("weather_bundle_complete", bundle=str(destination), predictors=len(FEATURE_NAMES), resources=monitor.stats)
    return destination


def load_bundle(root: Path, base: Path):
    manifest = read_json(base/"manifest.json")
    if manifest["audit_used"] or manifest["bundle_id"] != object_hash(manifest["descriptor"]):
        raise ValueError("Weather identity/audit declaration changed.")
    if manifest["code"]["sha256"] != feature_code_receipt(root)["sha256"]:
        raise ValueError("Weather feature code changed; prepare a fresh bundle.")
    previous = root/manifest["previous_bundle"]
    old, parent, original, catalog = load_previous(root, previous)
    validate_config(manifest["descriptor"]["config"], old)
    verify_external(root, manifest["descriptor"]["config"])
    if sha256_file(previous/"manifest.json") != manifest["descriptor"]["previous_manifest_sha256"]:
        raise ValueError("Previous feature manifest changed.")
    if manifest["frozen_folds"] != original["folds"] or manifest["ranking_ids_sha256"] != original["ranking_ids_sha256"]:
        raise ValueError("Protected split identity changed.")
    verify_receipts(base, manifest["artifacts"])
    return manifest, parent, original, catalog


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--config", type=Path, default=Path("configs/features_weather_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/weather"))
    args = parser.parse_args()
    prepare(root, root/args.previous, root/args.config, root/args.output)
