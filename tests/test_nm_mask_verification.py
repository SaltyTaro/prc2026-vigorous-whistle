"""Receipt serialization must not hide changed metrics or colliding keys."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_nm_mask import nested_equal


@pytest.fixture
def receipt_pair():
    actual = {"folds": {"seasonal": {"ranking_month_weights": {1: .4, 7: .6},
                                    "overall": {"rmse_seconds": 322.5}}}}
    return json.loads(json.dumps(actual)), actual


def test_saved_json_month_keys_compare(receipt_pair):
    nested_equal(*receipt_pair)


def test_changed_metric_is_rejected(receipt_pair):
    expected, actual = receipt_pair
    changed = deepcopy(actual)
    changed["folds"]["seasonal"]["overall"]["rmse_seconds"] = 300.
    with pytest.raises(ValueError, match="numeric receipt differs"):
        nested_equal(expected, changed)


def test_colliding_month_keys_are_rejected(receipt_pair):
    expected, actual = receipt_pair
    changed = deepcopy(actual)
    changed["folds"]["seasonal"]["ranking_month_weights"]["1"] = .4
    with pytest.raises(ValueError, match="receipt keys differ"):
        nested_equal(expected, changed)


def test_missing_metric_is_rejected(receipt_pair):
    expected, actual = receipt_pair
    changed = deepcopy(actual)
    del changed["folds"]["seasonal"]["overall"]
    with pytest.raises(ValueError, match="receipt keys differ"):
        nested_equal(expected, changed)
