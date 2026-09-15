"""Cross-campaign mixtures retain signed values and the fixed complete-fold contract."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from prc2026_ladder_blend.recipes import convex,validate_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("weight",[.25,.5])
def test_cross_campaign_mean_keeps_signed_and_extreme_values(weight):
    a=np.array([-100.,120000.,350.])
    b=np.array([-20.,90000.,1800.])
    result=convex(a,b,weight)
    np.testing.assert_array_equal(result,weight*a+(1-weight)*b)
    assert result[0]<0 and result[1]>86400


def test_cross_campaign_mean_rejects_misalignment_nonfinite_and_new_weights():
    for other in [np.array([1.]),np.array([np.nan,2.])]:
        with pytest.raises(ValueError):
            convex(np.array([1.,2.]),other,.25)
    with pytest.raises(ValueError):
        convex(np.array([1.,2.]),np.array([1.,2.]),.4)


def test_cross_campaign_declaration_is_exactly_the_eight_common_mixtures():
    original=json.loads((ROOT/"configs/ladder_blend_v1.json").read_text())
    validate_config(original)
    for field,value in [("audit_used",True),("minimum_improvement_fraction_each_fold",.002),
                        ("threads",32),("refit_months",original["refit_months"]+["2025-12"])]:
        altered=deepcopy(original);altered[field]=value
        with pytest.raises(ValueError):
            validate_config(altered)
    altered=deepcopy(original);altered["candidates"].pop(next(iter(altered["candidates"])))
    with pytest.raises(ValueError):
        validate_config(altered)

