import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_tree_mask import FOCAL_COLUMNS, MATCHED, augment  # noqa: E402


def frame():
    rng = np.random.default_rng(0)
    n = 6
    data = {c: rng.normal(size=n).astype("float32") for c in FOCAL_COLUMNS}
    data[MATCHED] = np.array([1, 0, 1, 1, 0, 0], dtype="float32")
    for c in FOCAL_COLUMNS:
        data[c][data[MATCHED] == 0] = np.nan
    data["nb_runway_10m_count"] = np.arange(n, dtype="float32")
    data["airport"] = pd.Categorical(["EDDF", "EGLL", "EDDF", "LTFM", "LTFM", "EGLL"], categories=["EDDF", "EGLL", "LTFM", "__MISSING__"])
    return pd.DataFrame(data), np.arange(n, dtype=float) * 100


def test_zero_is_identity():
    x, y = frame()
    xa, ya, w = augment(x, y, 0.0)
    assert xa is x and w is None and np.array_equal(ya, y)


def test_copies_weights_and_columns():
    x, y = frame()
    xa, ya, w = augment(x, y, 0.25)
    matched = x[MATCHED].to_numpy() == 1
    assert len(xa) == len(x) + matched.sum()
    pd.testing.assert_frame_equal(xa.iloc[: len(x)].reset_index(drop=True), x)
    copies = xa.iloc[len(x):].reset_index(drop=True)
    assert (copies[MATCHED] == 0).all()
    assert copies[FOCAL_COLUMNS].isna().all().all()
    np.testing.assert_array_equal(copies["nb_runway_10m_count"].to_numpy(), x.loc[matched, "nb_runway_10m_count"].to_numpy())
    assert list(copies["airport"]) == list(x.loc[matched, "airport"])
    assert (xa.dtypes == x.dtypes).all()
    np.testing.assert_array_equal(ya, np.concatenate([y, y[matched]]))
    np.testing.assert_allclose(w, np.concatenate([np.where(matched, 0.75, 1.0), np.full(matched.sum(), 0.25)]))
    # each matched row's total weight stays 1
    assert np.isclose(w.sum(), len(x))


def test_rejects_bad_inputs():
    x, y = frame()
    with pytest.raises(ValueError):
        augment(x, y, 1.0)
    with pytest.raises(ValueError):
        augment(x.drop(columns=[FOCAL_COLUMNS[0]]), y, 0.25)
    bad = x.copy()
    bad.loc[0, MATCHED] = 2.0
    with pytest.raises(ValueError):
        augment(bad, y, 0.25)
