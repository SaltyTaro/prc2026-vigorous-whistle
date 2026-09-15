import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TARGET, TIME
from prc2026.validation import (assign_fold, evaluate, identity_hash,
                               require_development_fold, validate_protocol)


def test_actual_clock_and_group_purging_protect_complete_score_set(protocol):
    catalog = pd.DataFrame({
        ID: np.arange(1, 10, dtype=float),
        "FLIGHT_ID_mvt": [20, 20, 30, 30, 30, 40, 40, np.nan, np.nan],
        TIME: pd.to_datetime(["2025-01-10", "2025-02-10", "2025-10-10", "2025-03-10",
                              "2025-07-31T23:59:58", "2025-12-10", "2025-04-10",
                              "2025-05-10", "2025-07-10"], format="mixed", utc=True),
        "airport": ["EDDF"] * 9,
        "source_file": ["misleading_filename.parquet"] * 9,
    })
    assignment, report = assign_fold(catalog, protocol, "seasonal")
    roles = assignment.set_index(ID)["partition"].to_dict()
    assert roles == {1: "score", 2: "purged_fit", 3: "purged_early_stop",
                     4: "purged_fit", 5: "score", 6: "reserved_audit",
                     7: "purged_fit", 8: "fit", 9: "score"}
    assert report["counts"]["score"] == 3
    assert all(count == 0 for count in report["group_overlap_counts"].values())
    assert report["identity_sha256"]["score"] == identity_hash([9, 1, 5])


def test_early_stopping_groups_are_purged_from_fit(protocol):
    catalog = pd.DataFrame({ID: [1., 2., 3.], "FLIGHT_ID_mvt": [42., 42., 43.],
                            TIME: pd.to_datetime(["2025-05-01", "2025-10-01", "2025-11-01"], utc=True),
                            "airport": ["EDDF"] * 3, "source_file": ["source"] * 3})
    assigned, _ = assign_fold(catalog, protocol, "forward")
    assert assigned["partition"].tolist() == ["purged_fit", "early_stop", "score"]


def test_final_audit_is_sealed(protocol):
    require_development_fold(protocol, "seasonal")
    with pytest.raises(ValueError, match="sealed"):
        require_development_fold(protocol, "final_audit")


def test_reserved_audit_cannot_be_used_for_model_selection(protocol):
    protocol["folds"]["forward"]["fit_months"].append("2025-12")
    with pytest.raises(ValueError, match="reserved audit"):
        validate_protocol(protocol)


def test_forward_fold_must_be_chronological(protocol):
    protocol["folds"]["forward"]["score_months"] = ["2024-11"]
    with pytest.raises(ValueError, match="chronological"):
        validate_protocol(protocol)


@pytest.fixture
def truth():
    return pd.DataFrame({ID: [1., 2., 3., 4.], TARGET: [-2., 0., 100., 20000.],
                         TIME: pd.to_datetime(["2025-01-01", "2025-01-02",
                                               "2025-07-01", "2025-07-01"], utc=True),
                         "airport": ["EDDF", "EGLL", "EDDF", "LIRF"],
                         "nm_missing": [False, True, False, False]})


def test_metrics_align_ids_and_keep_negative_zero_and_extreme_labels(truth):
    predictions = pd.DataFrame({ID: [4., 2., 1., 3.], "prediction": [0., 0., 0., 0.]})
    report = evaluate(truth, predictions, {"1": 0.25, "7": 0.75})
    assert report["overall"]["rows"] == 4
    assert report["overall"]["sse"] == 4 + 10000 + 400000000
    assert report["overall"]["rmse_seconds"] == pytest.approx(np.sqrt((4 + 10000 + 400000000) / 4))
    assert report["ranking_month_mix_rmse_seconds"] == pytest.approx(np.sqrt(0.25 * 2 + 0.75 * 200005000))
    assert report["by_target_band"]["negative"]["rows"] == 1
    assert report["by_target_band"]["zero"]["rows"] == 1
    assert report["by_target_band"]["over_3h"]["sse"] == 400000000


def test_shuffled_predictions_are_joined_by_id(truth):
    predictions = pd.DataFrame({ID: [4., 2., 1., 3.], "prediction": [20000., 0., -2., 100.]})
    assert evaluate(truth, predictions)["overall"]["rmse_seconds"] == 0


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "wrong_id", "nan", "inf"])
def test_incomplete_or_invalid_predictions_fail(truth, case):
    predictions = pd.DataFrame({ID: [1., 2., 3., 4.], "prediction": [10.] * 4})
    if case == "missing":
        predictions = predictions.iloc[:-1]
    elif case == "extra":
        predictions.loc[4] = [5., 10.]
    elif case == "duplicate":
        predictions.loc[3, ID] = 3.
    elif case == "wrong_id":
        predictions.loc[3, ID] = 5.
    else:
        predictions.loc[3, "prediction"] = np.nan if case == "nan" else np.inf
    with pytest.raises(ValueError):
        evaluate(truth, predictions)


def test_nonfinite_labels_cannot_be_silently_dropped(truth):
    truth.loc[1, TARGET] = np.nan
    predictions = pd.DataFrame({ID: truth[ID], "prediction": [10.] * 4})
    with pytest.raises(ValueError, match="Nonfinite evaluation label"):
        evaluate(truth, predictions)
