"""Exercise fitted native output, provenance and reload as one complete operation."""
import numpy as np
import pandas as pd
from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash
from prc2026_timeline.encoding import contract
from prc2026_timeline.native import fit_node


def test_timeline_native_fit_saves_correct_population_and_reproduces(tmp_path):
    spec = {"kind": "lightgbm", "features": "flown", "target": "nm_residual", "route": "clock_available"}
    names, cats = contract(spec)
    rng = np.random.default_rng(41)
    frame = pd.DataFrame({name: (pd.Series(["known"]*80, dtype="string") if name in cats else rng.normal(size=80)) for name in names})
    frame.insert(0, ID, np.arange(80, dtype=float))
    frame["nmclock_available"], frame["nmclock_proxy_s"] = 1., 900.
    truth = pd.DataFrame({ID: frame[ID], TARGET: 920.+30*frame[names[-1]]})
    settings = {"threads": 1, "seed": 41, "lightgbm": {"rounds": 12, "early_stopping_rounds": 3,
                "parameters": {"objective": "regression", "metric": "rmse", "verbosity": -1,
                               "num_leaves": 3, "min_data_in_leaf": 2, "force_col_wise": True}}}
    node = fit_node(tmp_path, "test", spec, settings, frame.iloc[:60], truth.iloc[:60], frame.iloc[60:], truth.iloc[60:])
    assert node["schema"] == "timeline"
    assert node["fit_ids_sha256"] == identity_hash(frame.iloc[:60][ID])
    assert node["early_stop_ids_sha256"] == identity_hash(frame.iloc[60:][ID])
    assert node["fit_rows"] == 60 and node["early_stop_rows"] == 20
    assert node["max_model_reload_difference_seconds"] == 0.
    assert (tmp_path/node["model_path"]).is_file()
