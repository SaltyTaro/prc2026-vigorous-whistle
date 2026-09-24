"""adsb_rnn2 design readings: October 2025 only, from the sixteen adsb_rnn leave-one-month-out October predictions (seeds 501-516).

The question is what doubling the residual network's seeds (16 -> 32) can buy. Averaging theory: the network's error variance part
halves with each doubling, so the gain of 16 -> 32 is about half the gain of 8 -> 16, which is measurable here. Readings, on October's
in-scope rows (matched_no_lirf) and on v25's non-LIRF recipe rebuilt on October (tools/rnn_design_report.py's construction: 0.3 x
(v9 + r) + 0.7 x (0.5 committee + 0.5 32-seed pool), floored; LIRF rows at v9):
  * the seed curve of the network alone (first k seeds, k = 1, 2, 4, 8, 16) and the fit RMSE^2 = B + V/k, extrapolated to k = 32;
  * the v26 analogue (residual = 0.05 adj2 + 0.05 cat + 0.9 net16) against the same with net8, and against w = 1.0 (residual = net16).
No recipe or ranking month is read. Output: a JSON report and a printed summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from nn_design_report import day_interval  # noqa: E402
from run_adsb_v11 import candidate_values, scope_mask  # noqa: E402
from run_adsb_v24 import adsb_features, load_frame, month_table  # noqa: E402

OCTOBER = "2025-10"
SEEDS = list(range(501, 517))
POOL_SEEDS = list(range(101, 133))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_rnn2/design_report.json"))
    args = parser.parse_args()
    config = read_json(ROOT / "configs/adsb_rnn_v1.json")
    stage = read_json(ROOT / "configs/adsb_rnn_network_v1.json")["residual_stage"]
    table = month_table(load_frame(stage, OCTOBER, "cross_fit"), adsb_features(OCTOBER, ROOT / stage["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), OCTOBER)
    ids = table[ID]
    label, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
    floor = float(table["floor"].iloc[0])
    inside = scope_mask(table, "matched_no_lirf")
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    columns = pd.read_parquet(ROOT / config["residual_columns"][OCTOBER]).set_index(ID).reindex(ids)
    trees = (columns["residual:adj2"].to_numpy(dtype=float) + columns["residual:cat"].to_numpy(dtype=float)) / 2
    nets = []
    for s in SEEDS:
        run = ROOT / f"runs/adsb_rnn/rnb_s{s}/{OCTOBER}"
        record = read_json(run / "run.json")
        if record["stage"] != "development" or record["held_out"] != OCTOBER or record["seed"] != s:
            raise ValueError(f"{run} is not the declared October leave-one-month-out fit.")
        nets.append(pd.read_parquet(run / "predictions.parquet").set_index(ID)["residual"].reindex(ids).to_numpy(dtype=float))
    committee = pd.read_parquet(ROOT / "runs/adsb_nn/design/committee_october.parquet").set_index(ID)["committee"].reindex(ids).to_numpy(dtype=float)
    pool = np.mean([pd.read_parquet(ROOT / f"runs/adsb_nn/nnb_s{s}/seasonal/stop_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
                    for s in POOL_SEEDS], axis=0)
    table["_mix"] = np.where(np.isfinite(committee) & np.isfinite(pool), 0.5 * committee + 0.5 * pool, committee)

    def recipe(residual: np.ndarray) -> np.ndarray:
        spec = {"weight": 1.0, "scope": "matched_no_lirf", "blend_with": {"name": "_mix", "scope": "all_no_lirf", "weight": 0.7}, "extra_blends": []}
        return candidate_values(table, v9, residual, spec, floor)[0]

    y, b = label[inside], v9[inside]
    curve = {k: rmse(y, b + np.mean(nets[:k], axis=0)[inside]) for k in (1, 2, 4, 8, 16)}
    ks = np.array(list(curve), dtype=float)
    fit_b, fit_v = np.linalg.lstsq(np.column_stack([np.ones(len(ks)), 1 / ks]), np.array(list(curve.values())) ** 2, rcond=None)[0]
    net8, net16 = np.mean(nets[:8], axis=0), np.mean(nets, axis=0)
    v26 = recipe(0.1 * trees + 0.9 * net16)
    base = rmse(label, v26)
    readings = {"v26_analogue_net8": recipe(0.1 * trees + 0.9 * net8), "w100_net16": recipe(net16), "w80_net16": recipe(0.2 * trees + 0.8 * net16)}
    report = {"created_at_utc": utc_now(), "rows": int(len(table)), "rows_in_scope": int(inside.sum()),
              "scope_rmse": {"v9": rmse(y, b), "trees": rmse(y, b + trees[inside]), **{f"net_first{k}": v for k, v in curve.items()},
                             "v26_residual": rmse(y, b + (0.1 * trees + 0.9 * net16)[inside]), "net16_alone": curve[16]},
              "seed_fit": {"B": float(fit_b), "V": float(fit_v), "extrapolated_net32_scope_rmse": float(np.sqrt(fit_b + fit_v / 32)),
                           "extrapolated_net_infinite_scope_rmse": float(np.sqrt(max(fit_b, 0.0)))},
              "recipe_v26_analogue_rmse": base,
              "recipe_ratio_to_v26_analogue": {k: rmse(label, v) / base for k, v in readings.items()},
              "doubling": {"8_to_16_recipe_pct": 100 * (base / rmse(label, readings["v26_analogue_net8"]) - 1)},
              "day_interval_w100_vs_v26_analogue_seconds": day_interval(days, label, readings["w100_net16"], v26),
              "inputs": {"residual_columns": sha256_file(ROOT / config["residual_columns"][OCTOBER]), "committee_october": sha256_file(ROOT / "runs/adsb_nn/design/committee_october.parquet")},
              "audit_used": False, "note": "October only; no recipe or ranking month read."}
    report["doubling"]["expected_16_to_32_recipe_pct"] = report["doubling"]["8_to_16_recipe_pct"] / 2
    write_json(ROOT / args.report, report)
    print({k: round(v, 3) for k, v in report["scope_rmse"].items()})
    print("seed fit", {k: round(v, 3) for k, v in report["seed_fit"].items()})
    print("recipe vs v26 analogue (%)", {k: round(100 * (v - 1), 4) for k, v in report["recipe_ratio_to_v26_analogue"].items()})
    print("doubling", {k: round(v, 4) for k, v in report["doubling"].items()}, "w100 interval", [round(x, 3) for x in report["day_interval_w100_vs_v26_analogue_seconds"]])


if __name__ == "__main__":
    main()
