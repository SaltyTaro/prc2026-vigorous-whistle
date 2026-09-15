"""Protected populations and provenance for the post-v7 diagnostic study."""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prc2026.artifacts import read_json, sha256_file
from prc2026.features import ID, TIME, PHASE, TARGET, BLOCK
from prc2026.validation import identity_hash

CONFIG = "configs/source_geometry_audit_v1.json"
COVARIATES = [ID, TIME, PHASE, "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
SOURCE_COLUMNS = COVARIATES + ["FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt",
    "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt", "ARVT_1_flt", "AOBT_3_flt", "LOBT_flt", TARGET]


def checked_config(root):
    config = read_json(root / CONFIG)
    if config["audit_used"] or config["native_models_to_fit"] or config["prediction_changes"]:
        raise ValueError("This declaration is for diagnostics only.")
    for name, expected in config["input_receipts_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError("Changed declared parent: " + name)
    return config


def common_fitting(root, config):
    base = root / config["split_source"]
    manifest = read_json(base / "manifest.json")
    expected = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    paths = ["train_catalog.parquet", "splits/seasonal.parquet", "splits/forward.parquet"]
    for name in paths:
        if sha256_file(base / name) != expected[name]:
            raise ValueError("Frozen split/catalog changed: " + name)
    folds = [pd.read_parquet(base / f"splits/{name}.parquet") for name in ["seasonal", "forward"]]
    fitting = set.intersection(*[set(frame.loc[frame.partition.eq("fit"), ID]) for frame in folds])
    protected = set.union(*[set(frame.loc[~frame.partition.eq("fit"), ID]) for frame in folds])
    if fitting & protected:
        raise ValueError("Protected IDs entered the fitting intersection.")
    digest = identity_hash(sorted(fitting))
    if len(fitting) != config["expected_all_airport_common_fitting_rows"] or digest != config["expected_all_airport_common_fitting_ids_sha256"]:
        raise ValueError("The declared fitting population changed.")
    catalog = pd.read_parquet(base / "train_catalog.parquet")
    selected = catalog.loc[catalog[ID].isin(fitting) & catalog.airport.isin(config["airports"])].copy()
    if not selected[TIME].dt.strftime("%Y-%m").isin(config["common_fitting_months"]).all():
        raise ValueError("A non-fitting month entered the audit.")
    receipt = {"all_airport_fitting_rows": len(fitting), "all_airport_fitting_ids_sha256": digest,
               "focal_airport_fitting_rows": len(selected), "focal_fitting_ids_sha256": identity_hash(selected[ID]),
               "split_sha256": {name: sha256_file(base / name) for name in paths}}
    return selected, receipt


def validate_fitting_frame(frame, expected_ids, allowed_months):
    if frame[ID].isna().any() or frame[ID].duplicated().any() or set(frame[ID]) != set(expected_ids):
        raise ValueError("Fitting IDs were changed or duplicated.")
    if not frame[PHASE].eq("DEP").all() or not frame[TIME].dt.strftime("%Y-%m").isin(allowed_months).all():
        raise ValueError("Only actual-UTC fitting departures are allowed.")
    if TARGET in frame and not np.isfinite(frame[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("All finite fitting labels must be retained.")


def load_fitting(root, config, *, with_labels):
    selected, receipt = common_fitting(root, config)
    columns = SOURCE_COLUMNS if with_labels else COVARIATES
    frames, hashes = [], {}
    for source, rows in selected.groupby("source_file", sort=True):
        path = root / "data/raw" / source
        ids = rows[ID].tolist()
        frame = pq.read_table(path, columns=columns, filters=[(ID, "in", ids)]).to_pandas()
        validate_fitting_frame(frame, ids, config["common_fitting_months"])
        frames.append(frame)
        hashes[source] = sha256_file(path)
    result = pd.concat(frames, ignore_index=True)
    validate_fitting_frame(result, selected[ID], config["common_fitting_months"])
    receipt.update(raw_input_sha256=hashes, columns_read=columns, labels_read=with_labels)
    return result, receipt


def read_geometry_covariates(root, config):
    if TARGET in COVARIATES or BLOCK in COVARIATES:
        raise ValueError("Geometry reads must not contain departure labels/block times.")
    fitting, receipt = load_fitting(root, config, with_labels=False)
    populations = {"common_fitting": fitting}
    sources = ["ranking.parquet", "training_2025-01-01_2025-02-01.parquet", "training_2025-07-01_2025-08-01.parquet"]
    for source in sources:
        path = root / "data/raw" / source
        raw = pq.read_table(path, columns=COVARIATES,
                            filters=[("ADEP_mvt", "in", config["airports"]), (PHASE, "=", "DEP")]).to_pandas()
        months = raw[TIME].dt.strftime("%Y-%m")
        allowed = {"2026-01", "2026-07"} if source == "ranking.parquet" else {source[9:16]}
        if not months.isin(allowed).all():
            raise ValueError("Unexpected actual movement month in geometry population.")
        for month, part in raw.assign(month=months).groupby("month", sort=True):
            populations[str(month)] = part.drop(columns="month").reset_index(drop=True)
        receipt["raw_input_sha256"][source] = sha256_file(path)
    return populations, receipt
