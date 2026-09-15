"""Keep the new experiment within the original validation and selection rules."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from prc2026_timeline.training import validate_config


@pytest.mark.parametrize("change", ["margin", "december", "route"])
def test_timeline_rejects_selection_or_protected_population_changes(change):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"configs/timeline_v1.json").read_text())
    protocol = json.loads((root/"configs/validation_v1.json").read_text())
    validate_config(config, protocol)
    altered = deepcopy(config)
    if change == "margin":
        altered["minimum_improvement_fraction_each_fold"] = .004
    elif change == "december":
        altered["refit_months"].append("2025-12")
    else:
        altered["variants"]["lgb_flown"]["route"] = "all_departures"
    with pytest.raises(ValueError):
        validate_config(altered, protocol)
