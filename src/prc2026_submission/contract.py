"""Exact template IDs, integer-second predictions, and export receipts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prc2026.artifacts import sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash

TEAM = "vigorous-whistle"
FILENAME = re.compile(r"vigorous-whistle_v([1-9]\d*)\.parquet")


def filename_version(name: str) -> int:
    match = FILENAME.fullmatch(name)
    if not match:
        raise ValueError("Submission must be named vigorous-whistle_vN.parquet with N >= 1.")
    return int(match.group(1))


def rounded_predictions(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Predictions must be a finite one-dimensional array.")
    rounded = np.rint(values)
    limits = np.iinfo(np.int32)
    if ((rounded < limits.min) | (rounded > limits.max)).any():
        raise ValueError("Rounded predictions exceed the int32 template range.")
    return rounded.astype(np.int32)


def validate_submission(path: Path, template_path: Path) -> dict:
    version = filename_version(path.name)
    table, template = pq.read_table(path), pq.read_table(template_path)
    if table.column_names != [ID, TARGET] or template.column_names != [ID, TARGET]:
        raise ValueError("Expected exactly the two template columns in template order.")
    if table.num_rows == 0 or table.num_rows != template.num_rows:
        raise ValueError("Submission row count differs from the complete template.")
    for name, expected_type in [(ID, pa.float64()), (TARGET, pa.int32())]:
        if table.schema.field(name).type != expected_type or template.schema.field(name).type != expected_type:
            raise ValueError(f"Wrong submission/template type for {name}.")
        if table[name].null_count:
            raise ValueError("Submission contains null IDs or predictions.")
    ids = table[ID].to_numpy()
    if not np.array_equal(ids, template[ID].to_numpy()):
        raise ValueError("Submission IDs must match exact template order.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate movement IDs.")
    id_hash = identity_hash(ids)
    values = table[TARGET].to_numpy()
    return {
        "validated_at_utc": utc_now(), "filename": path.name, "version": version,
        "rows": table.num_rows, "columns": table.column_names,
        "types": {ID: "double", TARGET: "int32"},
        "template_sha256": sha256_file(template_path),
        "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        "ids_sha256": id_hash, "exact_template_order": True,
        "minimum_prediction_seconds": int(values.min()),
        "maximum_prediction_seconds": int(values.max()),
        "negative_predictions": int((values < 0).sum()),
    }


def export_submission(predictions: pd.DataFrame, template_path: Path, output: Path,
                      provenance: dict) -> dict:
    filename_version(output.name)
    if output.exists():
        raise ValueError("Submission output already exists; versions are immutable.")
    if list(predictions) != [ID, "prediction"] or predictions.empty:
        raise ValueError("Ranking predictions must have ID and prediction columns.")
    if predictions[ID].isna().any() or predictions[ID].duplicated().any():
        raise ValueError("Null or duplicate prediction IDs.")
    template = pq.read_table(template_path)
    if template.column_names != [ID, TARGET] or template[TARGET].null_count != template.num_rows:
        raise ValueError("Unexpected original submission template.")
    template_ids = template[ID].to_numpy()
    if len(predictions) != len(template_ids) or set(predictions[ID]) != set(template_ids):
        raise ValueError("Ranking predictions must cover the complete template exactly.")
    aligned = pd.DataFrame({ID: template_ids}).merge(predictions, on=ID, how="left", validate="one_to_one")
    values = rounded_predictions(aligned["prediction"])
    target_field = template.schema.field(TARGET)
    table = template.set_column(template.schema.get_field_index(TARGET), target_field, pa.array(values, type=pa.int32()))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.part")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)
    receipt = validate_submission(output, template_path)
    receipt.update({"rounding": "numpy.rint, nearest integer with ties to even; no clipping",
                    "maximum_rounding_change_seconds": float(np.max(np.abs(values.astype(float) - aligned["prediction"]))),
                    "provenance": provenance})
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt
