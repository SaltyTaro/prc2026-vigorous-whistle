from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prc2026.features import ID, TARGET
from prc2026_submission.contract import (FINAL_NAME, export_submission, rounded_predictions, validate_final_submission,
                                        validate_submission)
from prc2026_submission.remote import check_final_upload, check_upload_window


@pytest.fixture
def template(tmp_path):
    path = tmp_path / "submitting.parquet"
    pq.write_table(pa.table({ID: pa.array([3., 1., 2.], type=pa.float64()),
                             TARGET: pa.array([None] * 3, type=pa.int32())}), path)
    return path


def test_export_preserves_template_order_types_and_rounds_without_clipping(template, tmp_path):
    predictions = pd.DataFrame({ID: [1., 2., 3.], "prediction": [-2.4, 20000.6, 100.5]})
    output = tmp_path / "vigorous-whistle_v1.parquet"
    report = export_submission(predictions, template, output, {"audit_used": False})
    table = pq.read_table(output)
    assert table[ID].to_pylist() == [3., 1., 2.]
    assert table[TARGET].to_pylist() == [100, -2, 20001]
    assert table[TARGET].type == pa.int32()
    assert report["rows"] == 3 and report["maximum_rounding_change_seconds"] <= 0.5
    with pytest.raises(ValueError, match="already exists"):
        export_submission(predictions, template, output, {})


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "wrong_id", "nan", "inf"])
def test_bad_prediction_populations_are_rejected(template, tmp_path, case):
    frame = pd.DataFrame({ID: [1., 2., 3.], "prediction": [10., 20., 30.]})
    if case == "missing":
        frame = frame.iloc[:-1]
    elif case == "extra":
        frame.loc[3] = [4., 40.]
    elif case == "duplicate":
        frame.loc[0, ID] = 2.
    elif case == "wrong_id":
        frame.loc[0, ID] = 4.
    else:
        frame.loc[0, "prediction"] = np.nan if case == "nan" else np.inf
    with pytest.raises(ValueError):
        export_submission(frame, template, tmp_path / "vigorous-whistle_v1.parquet", {})


@pytest.mark.parametrize("value", [np.inf, np.nan, 2**31, -(2**31) - 1])
def test_rounding_rejects_nonfinite_and_overflow(value):
    with pytest.raises(ValueError):
        rounded_predictions([value])


def test_validator_rejects_reordered_submission(template, tmp_path):
    path = tmp_path / "vigorous-whistle_v1.parquet"
    pq.write_table(pa.table({ID: pa.array([1., 2., 3.]), TARGET: pa.array([1, 2, 3], type=pa.int32())}), path)
    with pytest.raises(ValueError, match="exact template order"):
        validate_submission(path, template)


def test_validator_rejects_wrong_target_type(template, tmp_path):
    path = tmp_path / "vigorous-whistle_v1.parquet"
    pq.write_table(pa.table({ID: pa.array([3., 1., 2.]), TARGET: pa.array([1., 2., 3.])}), path)
    with pytest.raises(ValueError, match="type"):
        validate_submission(path, template)


def test_upload_limits_prevent_overwrites_stale_versions_and_daily_excess():
    now = datetime.now(timezone.utc)
    objects = [{"Key": f"vigorous-whistle_v{i}.parquet", "Size": 100,
                "LastModified": now - timedelta(hours=1)} for i in range(1, 6)]
    with pytest.raises(ValueError, match="overwrite"):
        check_upload_window(objects, "vigorous-whistle_v1.parquet", 100, now)
    with pytest.raises(ValueError, match="Five submissions"):
        check_upload_window(objects, "vigorous-whistle_v6.parquet", 100, now)
    with pytest.raises(ValueError, match="increase"):
        check_upload_window(objects[2:], "vigorous-whistle_v2.parquet", 100, now)
    for obj in objects:
        obj["LastModified"] = now - timedelta(days=2)
    assert check_upload_window(objects, "vigorous-whistle_v6.parquet", 100, now)["submissions_in_previous_24h"] == 0


def test_upload_limit_counts_all_bucket_objects():
    objects = [{"Key": "large-existing-result.json", "Size": 999_999_950,
                "LastModified": datetime.now(timezone.utc)}]
    with pytest.raises(ValueError, match="1 GB"):
        check_upload_window(objects, "vigorous-whistle_v1.parquet", 100)


def test_final_file_needs_the_final_name_and_the_same_table_contract(template, tmp_path):
    table = pa.table({ID: pa.array([3., 1., 2.]), TARGET: pa.array([5, -2, 7], type=pa.int32())})
    versioned = tmp_path / "vigorous-whistle_v9.parquet"
    pq.write_table(table, versioned)
    with pytest.raises(ValueError, match="final"):
        validate_final_submission(versioned, template)
    final = tmp_path / FINAL_NAME
    pq.write_table(table, final)
    report = validate_final_submission(final, template)
    assert FINAL_NAME == "vigorous-whistle_final.parquet"
    assert report["version"] == "final" and report["rows"] == 3 and report["minimum_prediction_seconds"] == -2
    pq.write_table(pa.table({ID: pa.array([1., 2., 3.]), TARGET: pa.array([5, -2, 7], type=pa.int32())}), final)
    with pytest.raises(ValueError, match="exact template order"):
        validate_final_submission(final, template)
    with pytest.raises(ValueError, match="vN"):
        validate_submission(final, template)


def test_final_upload_is_single_and_respects_the_bucket_allowance():
    now = datetime.now(timezone.utc)
    objects = [{"Key": f"vigorous-whistle_v{i}.parquet", "Size": 100, "LastModified": now} for i in range(1, 6)]
    # Five ranked versions in the window do not block the unscored final file.
    assert check_final_upload(objects, 100)["existing_submission_versions"] == 5
    present = objects + [{"Key": FINAL_NAME, "Size": 100, "LastModified": now}]
    with pytest.raises(ValueError, match="--replace-sha256"):
        check_final_upload(present, 100)
    with pytest.raises(ValueError, match="1 GB"):
        check_final_upload([{"Key": "big.json", "Size": 999_999_950, "LastModified": now}], 100)
    with pytest.raises(ValueError, match="Empty"):
        check_final_upload(objects, 0)


def test_final_replacement_is_explicit_and_frees_the_old_size():
    # A re-upload overwrites the previous final file: allowed only when asked for, never by default.
    now = datetime.now(timezone.utc)
    objects = [{"Key": "big.json", "Size": 999_999_800, "LastModified": now}, {"Key": FINAL_NAME, "Size": 150, "LastModified": now}]
    report = check_final_upload(objects, 180, replacing=True)
    assert report["final_object_present"] and report["bucket_bytes_after_upload"] == 999_999_980
    with pytest.raises(ValueError, match="1 GB"):
        check_final_upload(objects, 260, replacing=True)
    with pytest.raises(ValueError, match="nothing to replace"):
        check_final_upload(objects[:1], 100, replacing=True)
