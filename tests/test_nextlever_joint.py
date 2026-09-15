"""Complementary routes and predeclared weights must remain exact."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nextlever_joint import combine_routes


def test_joint_routes_select_the_correct_expert_and_common_weight():
    base = np.array([100., 200.])
    timeline, arrival, available = np.array([80., 200.]), np.array([100., 160.]), np.array([True, False])
    np.testing.assert_array_equal(combine_routes(base, timeline, arrival, available, 1.), [80., 160.])
    np.testing.assert_array_equal(combine_routes(base, timeline, arrival, available, .5), [90., 180.])


def test_joint_rejects_changed_fallbacks_and_unrecorded_weights():
    base, mask = np.array([100., 200.]), np.array([True, False])
    with pytest.raises(ValueError):
        combine_routes(base, [80., 199.], [100., 160.], mask, 1.)
    with pytest.raises(ValueError):
        combine_routes(base, [80., 200.], [100., 160.], mask, .75)
