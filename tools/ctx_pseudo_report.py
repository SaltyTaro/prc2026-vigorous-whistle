"""Readings of the pseudo-label adaptation test (configs/adsb_ctx_pseudo_v1.json) on December 2025: 3-seed mean of each variant (pg =
December gate rows pseudo-labelled and added to the fit, pgd = gate + dark) against the seed-matched control networks' December leaf
predictions, on non-LIRF rows split by ADS-B status (gate / dark / unmatched) and airport; the target rows are dark + unmatched (not
pseudo-labelled by pg; dark rows are pseudo-labelled by pgd). Paired-day interval on the target rows. Writes docs/adsb_ctx_pseudo/report.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, utc_now, write_json  # noqa: E402
from ctx_season_report import interval, load, rmse, truth  # noqa: E402


def main():
    config = read_json(ROOT / "configs/adsb_ctx_pseudo_v1.json")
    seeds = config["seeds"]
    t = truth("2025-12")
    gate_tab = pd.read_parquet(ROOT / "runs/adsb_v24/features/matches_2025-12.parquet", columns=["MVT_ID_mvt", "adsb_gate"])
    t = t.merge(gate_tab, on="MVT_ID_mvt", how="left")
    ids, y = t["MVT_ID_mvt"], t["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    matched = t["adsb_matched"].eq(1).to_numpy()
    gate = matched & t["adsb_gate"].eq(1).to_numpy()
    status = np.where(gate, "gate", np.where(matched, "dark", "unmatched"))
    control = np.mean([load(ROOT / f"runs/december/learners/runs__adsb_ctx_gpu__ctx_e32_s{s}.parquet", ids) for s in seeds], axis=0)
    rng = np.random.default_rng(20260927)
    report = {"created_at_utc": utc_now(), "seeds": seeds, "variants": {}}
    for name in ["pg", "pgd"]:
        runs = [ROOT / f"runs/adsb_ctx_pseudo/{name}_s{s}/refit" for s in seeds]
        values = np.mean([load(r / "december_predictions.parquet", ids) for r in runs], axis=0)
        info = [read_json(r / "run.json")["inputs"].get("pseudo_labels", {}) for r in runs]
        entry = {"pseudo_rows": info[0].get("rows"), "pseudo_rmse_vs_true": info[0].get("pseudo_rmse_vs_true"),
                 "all": 100 * (rmse(y, values) / rmse(y, control) - 1), "by_status": {}, "airports": {}}
        for st in ["gate", "dark", "unmatched"]:
            k = status == st
            entry["by_status"][st] = {"rows": int(k.sum()), "control": rmse(y[k], control[k]), "variant": rmse(y[k], values[k]),
                                      "change_pct": 100 * (rmse(y[k], values[k]) / rmse(y[k], control[k]) - 1)}
        target = status != "gate"
        entry["target_change_pct"] = 100 * (rmse(y[target], values[target]) / rmse(y[target], control[target]) - 1)
        entry["target_day_interval_seconds"] = interval(t["day"].to_numpy()[target], y[target], values[target], control[target], rng)
        airports = t["ADEP_mvt"].to_numpy()
        for a in sorted(set(airports)):
            k = (airports == a) & target
            if k.any():
                entry["airports"][a] = round(100 * (rmse(y[k], values[k]) / rmse(y[k], control[k]) - 1), 3)
        entry["go"] = bool(entry["target_change_pct"] <= -0.5 and entry["target_day_interval_seconds"][1] < 0)
        report["variants"][name] = entry
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items() if k not in ("by_status", "airports")},
              {st: round(e["change_pct"], 3) for st, e in entry["by_status"].items()}, flush=True)
    write_json(ROOT / "docs/adsb_ctx_pseudo/report.json", report)


if __name__ == "__main__":
    main()
