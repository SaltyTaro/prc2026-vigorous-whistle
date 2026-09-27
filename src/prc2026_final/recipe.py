"""S10: v30's arithmetic on the recomputed components, the detail table (v30 detail's columns) and the nearest-integer export.

v30 = the combined round's selection (configs/adsb_ctx_v5.json + runs/adsb_ctx/dev5/selection.json: committee mix 0.25 ens_ctx_e24 +
0.25 ens_ctx_e24b + 0.5 ens_ctx_e32 at share 0.75) with round B's LIRF blends (configs/adsb_ctx_lirf_v1.json lctx_b100_up), the v26
residual and v9, through the frozen arithmetic: run_adsb_nn_dev.mixed, run_adsb_nn2_dev.weighted / lirf_blends / spec_for and
run_adsb_v11.candidate_values (floor -12), exactly as tools/export_adsb_ctx4.py computes v30 for the ranking. The scope column follows
tools/export_adsb_nn2.py. The export goes through prc2026_submission.contract.export_submission with the run's template, into the run
root (never runs/submissions); nothing is uploaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import object_hash, read_json, sha256_file
from prc2026.features import ID
from prc2026_submission.contract import export_submission

from .common import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import run_adsb_v24  # noqa: E402,F401  (registers the lirf_nonpop_unmatched scope)
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_v11 import SCOPES, candidate_values  # noqa: E402

SCOPE_ORDER = ["matched_no_lirf", "all_no_lirf", "lirf_nonpop_matched", "lirf_nonpop_unmatched"]
DECLARATION = "configs/adsb_ctx_v5.json"
SELECTION = "runs/adsb_ctx/dev5/selection.json"
ROUND_B = "configs/adsb_ctx_lirf_v1.json"
ROUND_B_CANDIDATE = "lctx_b100_up"


def recipe_spec() -> dict:
    decl = read_json(ROOT / DECLARATION)
    selection = read_json(ROOT / SELECTION)
    name = selection["selected_candidate"]
    if name != "c_e24x32_e32h":
        raise ValueError("The combined round's selection is not v30's candidate.")
    decl_b = read_json(ROOT / ROUND_B)
    return {"name": name, "candidate": decl["candidates"][name], "lirf": decl_b["candidates"][ROUND_B_CANDIDATE],
            "declaration_sha256": object_hash(decl), "selection_sha256": sha256_file(ROOT / SELECTION), "round_b_sha256": object_hash(decl_b)}


def scope_column(table: pd.DataFrame) -> np.ndarray:
    scope = np.full(len(table), "fixed", dtype=object)
    for s in SCOPE_ORDER:
        scope[SCOPES[s](table).to_numpy() & (scope == "fixed")] = s
    return scope


def compute(table: pd.DataFrame, floor: float) -> tuple[pd.DataFrame, dict]:
    """`table`: the residual table (scopes' inputs) joined with v9, residual, committee, ens_v22_lirf_q and the network ensembles."""
    spec = recipe_spec()
    config = merged_config(read_json(ROOT / DECLARATION))
    committee = table[config["committee_table"]].to_numpy(dtype=float)
    candidate = spec["candidate"]
    column = f"committee_mix:{spec['name']}"
    table = table.copy()
    table[column] = mixed(committee, weighted(table, candidate["network"]), float(candidate["network_share"]))
    blends = lirf_blends(config, table, {**config["v25_recipe"], "lirf": spec["lirf"]["lirf"], "lirf_network": spec["lirf"]["lirf_network"]}, ROUND_B_CANDIDATE)
    v9 = table["v9"].to_numpy(dtype=float)
    residual = table["residual"].to_numpy(dtype=float)
    values, touched = candidate_values(table, v9, residual, spec_for(config, column, blends), floor)
    detail = pd.DataFrame({ID: table[ID].to_numpy(), "airport": table["airport"].to_numpy(), "month": table["month"].to_numpy(), "scope": scope_column(table),
                           "v9": v9, "residual": residual, "committee": committee, config["lirf_table"]: table[config["lirf_table"]].to_numpy(dtype=float)})
    for name in [c for c in table.columns if c.startswith("ens_") and c not in (config["committee_table"], config["lirf_table"])]:
        detail[name] = table[name].to_numpy(dtype=float)
    detail["committee_mix"] = table[column].to_numpy(dtype=float)
    for blend in blends:
        detail[blend["name"]] = table[blend["name"]].to_numpy(dtype=float)
    detail["v30"] = values
    record = {"candidate": spec["name"], "network": candidate["network"], "network_share": candidate["network_share"], "lirf": spec["lirf"],
              "declaration_sha256": spec["declaration_sha256"], "selection_sha256": spec["selection_sha256"], "round_b_sha256": spec["round_b_sha256"],
              "floor_seconds": floor, "outer_blend": config["recipe_blend"], "rows_touched": int(touched.sum()),
              "rows_by_scope": {s: int((detail["scope"] == s).sum()) for s in [*SCOPE_ORDER, "fixed"]}}
    return detail, record


def export(detail: pd.DataFrame, template: Path, output: Path, provenance: dict) -> dict:
    return export_submission(pd.DataFrame({ID: detail[ID].to_numpy(), "prediction": detail["v30"].to_numpy(dtype=float)}), template, output, provenance)
