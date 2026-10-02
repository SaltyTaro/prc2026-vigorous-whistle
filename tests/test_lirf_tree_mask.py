"""Masked LIRF tree design (tools/run_lirf_tree_mask.py): row-restricted masking with the focal and dark semantics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import run_lirf_tree_mask as R  # noqa: E402
from prc2026_tree_mask import augment  # noqa: E402


def frame(n: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    cols = {c: rng.normal(size=n).astype("float32") for c in [*R.FOCAL, *R.NB_COUNTS, *R.NB_OTHER, "other"]}
    cols[R.MATCHED] = np.array([1, 1, 0, 1, 0, 1], dtype="float32")[:n]
    return pd.DataFrame(cols), np.arange(n, dtype=float)


def test_p_zero_is_the_unweighted_call():
    x, y = frame()
    xa, ya, w = R.augment_rows(x, y, 0.0, x[R.MATCHED].eq(1).to_numpy(), "dark")
    assert xa is x and w is None and np.array_equal(ya, y)


def test_focal_on_selected_rows_only():
    x, y = frame()
    rows = np.array([True, False, False, True, False, False])
    xa, ya, w = R.augment_rows(x, y, 0.5, rows, "focal")
    assert len(xa) == 8 and np.array_equal(ya, np.r_[y, y[rows]])
    assert np.allclose(w, [0.5, 1, 1, 0.5, 1, 1, 0.5, 0.5])
    copies = xa.iloc[6:]
    assert copies[R.FOCAL].isna().all().all() and (copies[R.MATCHED] == 0).all()
    pd.testing.assert_frame_equal(copies[[*R.NB_COUNTS, *R.NB_OTHER, "other"]].reset_index(drop=True), x.loc[rows, [*R.NB_COUNTS, *R.NB_OTHER, "other"]].reset_index(drop=True))
    pd.testing.assert_frame_equal(xa.iloc[:6], x)
    assert (xa.dtypes == x.dtypes).all()


def test_dark_sets_neighbour_encoding():
    x, y = frame()
    rows = x[R.MATCHED].eq(1).to_numpy()
    xa, _, _ = R.augment_rows(x, y, 0.25, rows, "dark")
    copies = xa.iloc[6:]
    assert (copies[R.NB_COUNTS] == 0).all().all() and copies[R.NB_OTHER].isna().all().all()
    assert np.allclose(copies["other"].to_numpy(), x.loc[rows, "other"].to_numpy())


def test_focal_equals_tree_mask_augment_on_its_columns():
    x, y = frame()
    rows = x[R.MATCHED].eq(1).to_numpy()
    a1 = R.augment_rows(x, y, 0.5, rows, "focal")
    a2 = augment(x, y, 0.5, columns=R.FOCAL)
    pd.testing.assert_frame_equal(a1[0], a2[0])
    assert np.array_equal(a1[1], a2[1]) and np.allclose(a1[2], a2[2])


def test_unmatched_rows_are_refused():
    x, y = frame()
    with pytest.raises(ValueError):
        R.augment_rows(x, y, 0.5, np.ones(len(x), dtype=bool), "focal")
