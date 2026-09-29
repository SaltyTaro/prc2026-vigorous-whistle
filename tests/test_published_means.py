import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_published_means import fit_mapping, shift_for  # noqa: E402


def test_gate_and_shrinkage():
    shift, record = shift_for(level=38.0, bias=8.0, sigma=4.0, u=0.0, u_max=0.01)
    assert record["eligible"] and record["effective_level_s"] == 30.0
    assert shift == pytest.approx(30.0 * (1 - 16.0 / 900.0))
    assert shift_for(10.0, 0.0, 4.0, 0.0, 0.01)[0] == 0.0          # below three sigma
    assert shift_for(60.0, 0.0, 7.0, 0.0, 0.01, max_sigma=5.0)[0] == 0.0 and shift_for(60.0, 0.0, 7.0, 0.0, 0.01)[0] > 0   # the noise limit
    assert shift_for(60.0, 0.0, 2.0, 0.20, 0.10)[0] == 0.0         # left-out share beyond the fitting months: no extrapolation
    assert shift_for(-30.0, 0.0, 2.0, 0.0, 0.01)[0] == pytest.approx(-30.0 * (1 - 4.0 / 900.0))


def test_mapping_is_a_constant_or_a_line_and_excludes_the_month_read():
    months = [f"2025-{m:02}" for m in range(1, 13)]
    flat = pd.DataFrame({"airport": "AAAA", "month": months, "u": 0.001, "P": 900.0, "label_mean": 905.0 + np.arange(12) % 2})
    m = fit_mapping(flat, "AAAA", None)
    assert not m["slope"] and m["function"](0.5) == pytest.approx(5.5) and m["months"] == 12
    u = np.linspace(0.0, 0.2, 12)
    line = pd.DataFrame({"airport": "BBBB", "month": months, "u": u, "P": 800.0, "label_mean": 800.0 + 10.0 + 700.0 * u})
    m = fit_mapping(line, "BBBB", "2025-12")
    assert m["slope"] and m["months"] == 11 and m["function"](0.1) == pytest.approx(80.0) and m["sd_map"] < 1e-6
    assert m["u_max"] == pytest.approx(u[10])
