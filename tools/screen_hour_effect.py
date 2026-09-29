"""Hour-effect screen: can label-free aggregates of an airport-hour predict the stack's mean residual in that hour?

Motivation (scratch census of 29 September 2026): on the development months the airport-hour mean residual of the stack explains
about 5.5% of the ordinary squared error beyond a permutation floor, and the hour's mean residual of ADS-B-matched rows correlates
with that of unmatched rows (+0.5 at EGLL and LFPG): an hour state the row models miss. This screen asks how much of it label-free
hour aggregates can recover. Descriptive only: nothing is declared, selected, exported or uploaded.

Rows     runs/clean_line_screens/hour/dev_stack.parquet: runs/lomo_dev/rows.parquet with the v46 development stack
         (tools/run_share_gate_reread.full_stack with the n85_nd1200 mix; --build-stack writes it and checks the recorded RMSE).
Unit     airport x UTC hour of the take-off.
Target   the hour's mean residual (label - v46), each residual clipped to +-CLIP seconds.
Inputs   label-free aggregates of the hour and of the hours h-2 .. h+2: departures and arrivals, inter-departure gaps, the stack's and
         its components' predictions and disagreements, schedule intervals, NM off-block proxies, rows without an NM clock, ADS-B
         match share and the matched rows' ADS-B taxi against their prediction, arrival taxi-in (supplied for every arrival, in the
         ranking file too) against the airport-month median, hour of day and weekday. No departure label or block time enters an input.
Model    LightGBM, fixed settings, fitted on three months' hours (weighted by their rows) and read on the fourth (month-blocked).
Reading  RMSE of v46 + w x predicted hour effect against v46 on every row of the held-out month, w in W_VALUES, with the paired-day
         95% interval, by scope and by airport; the same with calendar-only inputs (what a plain hour-of-day calibration would give).
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
from prc2026.features import ID  # noqa: E402

MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL_FOOTING = ["2025-01", "2025-07", "2025-12"]
STACK = "runs/clean_line_screens/hour/dev_stack.parquet"
CLIP = 3600.0
W_VALUES = [0.5, 1.0]
LAGS = [-2, -1, 1, 2]
MIN_ROWS = 3
SEED = 20260929
PARAMS = {"objective": "regression", "learning_rate": 0.03, "n_estimators": 400, "num_leaves": 15, "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
          "colsample_bytree": 0.7, "reg_lambda": 10.0, "random_state": SEED, "n_jobs": 8, "verbose": -1}
CALENDAR = ["airport_code", "hour_of_day", "weekday"]


def raw_file(month: str) -> Path:
    start = pd.Timestamp(month + "-01")
    return ROOT / "data/raw" / f"training_{start:%Y-%m-%d}_{start + pd.offsets.MonthBegin(1):%Y-%m-%d}.parquet"


def build_stack() -> pd.DataFrame:
    from prc2026_final import share_gate
    from run_callsign_fix import offsets, scope_table
    from run_queue_trust import offsets_all, segments
    from run_share_gate_reread import STRETCH, full_stack
    share, gate = share_gate.candidate_spec(read_json(ROOT / "configs/thinned_regime_round_v1.json"), "n85_nd1200")
    rows = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet")
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off, seg = offsets(), offsets_all(), segments(MONTHS)
    k = read_json(ROOT / STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    recorded = read_json(ROOT / "docs/thinned_regime/reread_n85_nd1200.json")["months"]
    parts = []
    for month in MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        mix, _ = share_gate.mix(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), share, gate)
        f["v46"] = full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)
        value = float(np.sqrt(np.mean((f["label"] - f["v46"]) ** 2)))
        if abs(value - recorded[month]["candidate_rmse"]) > 1e-6:
            raise ValueError(f"The rebuilt v46 stack does not reproduce the recorded RMSE on {month}.")
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    (ROOT / STACK).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ROOT / STACK, index=False)
    return out


def seconds(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds()


def hour_table(stack: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows with their airport-hour key, airport-hour table of label-free aggregates and the target)."""
    columns = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "LOBT_flt"]
    raw = pd.concat([pd.read_parquet(raw_file(m), columns=columns) for m in MONTHS], ignore_index=True)
    dep = raw[raw["PHASE_mvt"].eq("DEP")].drop(columns=["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])          # departure labels never enter an input
    rows = stack.merge(dep, on=ID, how="left", validate="one_to_one")
    if rows["MVT_TIME_UTC_mvt"].isna().any():
        raise ValueError("A development row has no raw movement.")
    matches = pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/matches/matches2_{m}.parquet", columns=[ID, "takeoff", "motion_start", "taxi_stopped_s", "taxi_stops", "gate"])
                         for m in MONTHS], ignore_index=True).drop_duplicates(ID)
    rows = rows.merge(matches, on=ID, how="left")
    rows["hour"] = rows["MVT_TIME_UTC_mvt"].dt.floor("h")
    rows["sched_int"] = seconds(rows["MVT_TIME_UTC_mvt"], rows["SCHED_TIME_UTC_mvt"])
    rows["proxy"] = seconds(rows["MVT_TIME_UTC_mvt"], rows["AOBT_3_flt"])
    rows["noclock"] = (rows["AOBT_3_flt"].isna() & rows["LOBT_flt"].isna()).astype(float)
    rows["adsb_taxi"] = rows["takeoff"] - rows["motion_start"]
    rows["adsb_minus_pred"] = rows["adsb_taxi"] - rows["v46"]
    rows["nets_minus_committee"] = rows["nets"] - rows["committee"]
    rows["stack_minus_v9"] = rows["v46"] - rows["v9"]
    rows["residual_clipped"] = (rows["label"] - rows["v46"]).clip(-CLIP, CLIP)
    rows = rows.sort_values(["airport", "MVT_TIME_UTC_mvt"]).reset_index(drop=True)
    rows["gap"] = rows.groupby("airport")["MVT_TIME_UTC_mvt"].diff().dt.total_seconds().clip(upper=14400.0)

    g = rows.groupby(["airport", "hour"], observed=True)
    table = g.agg(n=("v46", "size"), target=("residual_clipped", "mean"), pred_mean=("v46", "mean"), pred_p90=("v46", lambda s: s.quantile(0.9)), pred_max=("v46", "max"),
                  v9_mean=("v9", "mean"), committee_mean=("committee", "mean"), nets_mean=("nets", "mean"), nets_minus_committee=("nets_minus_committee", "mean"),
                  nets_committee_abs=("nets_minus_committee", lambda s: s.abs().mean()), stack_minus_v9=("stack_minus_v9", "mean"), matched_share=("matched", "mean"),
                  sched_int_med=("sched_int", "median"), sched_int_p90=("sched_int", lambda s: s.quantile(0.9)), sched_late_share=("sched_int", lambda s: (s > 3600).mean()),
                  proxy_med=("proxy", "median"), proxy_p10=("proxy", lambda s: s.quantile(0.1)), proxy_short_share=("proxy", lambda s: (s < 300).mean()),
                  noclock_share=("noclock", "mean"), gap_max=("gap", "max"), adsb_taxi_mean=("adsb_taxi", "mean"), adsb_minus_pred=("adsb_minus_pred", "mean"),
                  stopped_mean=("taxi_stopped_s", "mean"), stops_mean=("taxi_stops", "mean")).reset_index()

    arr = raw[raw["PHASE_mvt"].eq("ARR") & raw["ADES_mvt"].isin(stack["airport"].unique())].copy()
    arr["airport"] = arr["ADES_mvt"].astype(str)
    arr["hour"] = arr["MVT_TIME_UTC_mvt"].dt.floor("h")
    arr["month"] = arr["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    arr["taxi_in"] = arr["TAXITIME_SEC_mvt"].astype(float).clip(-600.0, 7200.0)
    arr["taxi_in_rel"] = arr["taxi_in"] - arr.groupby(["airport", "month"])["taxi_in"].transform("median")
    arrivals = arr.groupby(["airport", "hour"], observed=True).agg(arr_n=("taxi_in", "size"), taxi_in_mean=("taxi_in", "mean"), taxi_in_rel=("taxi_in_rel", "mean"),
                                                                    taxi_in_p90=("taxi_in", lambda s: s.quantile(0.9))).reset_index()
    table["airport"] = table["airport"].astype(str)
    table = table.merge(arrivals, on=["airport", "hour"], how="left")
    table["arr_n"] = table["arr_n"].fillna(0.0)
    table["month"] = table["hour"].dt.strftime("%Y-%m")
    table["hour_of_day"] = table["hour"].dt.hour
    table["weekday"] = table["hour"].dt.weekday
    table["airport_code"] = table["airport"].astype("category").cat.codes
    for name in ("n", "pred_mean", "arr_n"):
        typical = table.groupby(["airport", "month", "hour_of_day"])[name].transform("median")
        table[f"{name}_vs_typical"] = table[name] - typical
    lagged = ["n", "arr_n", "pred_mean", "sched_int_med", "taxi_in_rel", "gap_max", "proxy_med", "adsb_minus_pred", "nets_minus_committee"]
    keyed = table.set_index(["airport", "hour"])
    for lag in LAGS:
        shifted = keyed[lagged].copy()
        shifted.index = pd.MultiIndex.from_arrays([shifted.index.get_level_values(0), shifted.index.get_level_values(1) - pd.Timedelta(hours=lag)], names=["airport", "hour"])
        shifted.columns = [f"{c}_h{lag:+d}" for c in lagged]
        table = table.merge(shifted.reset_index(), on=["airport", "hour"], how="left")
        table[f"n_h{lag:+d}"] = table[f"n_h{lag:+d}"].fillna(0.0)
        table[f"arr_n_h{lag:+d}"] = table[f"arr_n_h{lag:+d}"].fillna(0.0)
    return rows, table


def day_interval(days: np.ndarray, label: np.ndarray, candidate: np.ndarray, reference: np.ndarray) -> list[float]:
    frame = pd.DataFrame({"day": days, "c": (candidate - label) ** 2, "r": (reference - label) ** 2})
    grouped = frame.groupby("day").agg(c=("c", "sum"), r=("r", "sum"), n=("c", "size"))
    idx = np.random.default_rng(SEED).integers(0, len(grouped), size=(2000, len(grouped)))
    n = grouped["n"].to_numpy()[idx].sum(axis=1)
    delta = np.sqrt(grouped["c"].to_numpy()[idx].sum(axis=1) / n) - np.sqrt(grouped["r"].to_numpy()[idx].sum(axis=1) / n)
    return list(map(float, np.quantile(delta, [0.025, 0.975])))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-stack", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("docs/clean_line_screens/hour_effect.json"))
    args = parser.parse_args()
    import lightgbm as lgb
    stack = build_stack() if args.build_stack or not (ROOT / STACK).exists() else pd.read_parquet(ROOT / STACK)
    rows, table = hour_table(stack)
    excluded = {"airport", "hour", "month", "target", "n_vs_typical_dummy"}
    features = [c for c in table.columns if c not in excluded]
    table.to_parquet(ROOT / "runs/clean_line_screens/hour/hour_table.parquet", index=False)
    key = pd.MultiIndex.from_frame(rows[["airport", "hour"]].astype({"airport": str}))
    label, base, floor = rows["label"].to_numpy(dtype=float), rows["v46"].to_numpy(dtype=float), rows["floor"].to_numpy(dtype=float)
    report = {"created_at_utc": utc_now(), "rows": int(len(rows)), "airport_hours": int(len(table)), "features": features, "params": PARAMS, "clip": CLIP, "min_rows": MIN_ROWS,
              "stack_sha256": sha256_file(ROOT / STACK), "tool_sha256": sha256_file(ROOT / "tools/screen_hour_effect.py"), "months": {}, "audit_used": False,
              "note": "descriptive screen on the development months; November's fold side is on nine months; no ranking row and no official score is read"}
    importance = np.zeros(len(features))
    for month in MONTHS:
        fit = table[(table["month"] != month) & (table["n"] >= MIN_ROWS)]
        held = table[table["month"] == month]
        entry = {"fit_hours": int(len(fit)), "held_hours": int(len(held))}
        m = (rows["month"] == month).to_numpy()
        y, b, days, fl = label[m], base[m], rows.loc[m, "day"].to_numpy(), floor[m]
        scope = {"all": np.ones(m.sum(), dtype=bool), "matched": rows.loc[m, "matched"].to_numpy(dtype=bool), "unmatched": ~rows.loc[m, "matched"].to_numpy(dtype=bool)}
        airports = rows.loc[m, "airport"].astype(str).to_numpy()
        entry["control_rmse"] = rmse(y, b)
        for name, columns in (("aggregates", features), ("calendar_only", CALENDAR)):
            model = lgb.LGBMRegressor(**PARAMS).fit(fit[columns], fit["target"], sample_weight=fit["n"].to_numpy(dtype=float), categorical_feature=["airport_code"])
            effect = pd.Series(model.predict(held[columns]), index=pd.MultiIndex.from_frame(held[["airport", "hour"]]))
            e = effect.reindex(key[m]).to_numpy()
            if not np.isfinite(e).all():
                raise ValueError("A held-out row has no hour effect.")
            target = held.set_index(["airport", "hour"])["target"].reindex(effect.index).to_numpy()
            weights = held["n"].to_numpy(dtype=float)
            out = {"effect_sd_s": float(np.sqrt(np.average((effect.to_numpy() - np.average(effect.to_numpy(), weights=weights)) ** 2, weights=weights))),
                   "hour_correlation": float(np.corrcoef(effect.to_numpy()[weights >= MIN_ROWS], target[weights >= MIN_ROWS])[0, 1]), "by_w": {}}
            for w in W_VALUES:
                values = np.maximum(b + w * e, fl)
                item = {"change_pct": 100 * (rmse(y, values) / rmse(y, b) - 1), "interval_s": day_interval(days, y, values, b),
                        "by_scope_pct": {s: 100 * (rmse(y[k], values[k]) / rmse(y[k], b[k]) - 1) for s, k in scope.items() if k.any()},
                        "by_airport_pct": {a: 100 * (rmse(y[airports == a], values[airports == a]) / rmse(y[airports == a], b[airports == a]) - 1) for a in sorted(set(airports))}}
                ordinary = y < 10800
                item["ordinary_change_pct"] = 100 * (rmse(y[ordinary], values[ordinary]) / rmse(y[ordinary], b[ordinary]) - 1)
                out["by_w"][str(w)] = item
            entry[name] = out
            if name == "aggregates":
                importance += model.booster_.feature_importance(importance_type="gain")
        report["months"][month] = entry
        print(month, {"control": round(entry["control_rmse"], 3),
                      **{f"{n}_w{w}": (round(entry[n]["by_w"][str(w)]["change_pct"], 4), [round(v, 3) for v in entry[n]["by_w"][str(w)]["interval_s"]]) for n in ("aggregates", "calendar_only") for w in W_VALUES},
                      "hour_corr": round(entry["aggregates"]["hour_correlation"], 4)}, flush=True)
    report["mean_change_pct_equal_footing"] = {n: {str(w): float(np.mean([report["months"][m][n]["by_w"][str(w)]["change_pct"] for m in EQUAL_FOOTING])) for w in W_VALUES}
                                               for n in ("aggregates", "calendar_only")}
    order = np.argsort(-importance)
    report["gain_importance_top"] = {features[i]: float(importance[i] / importance.sum()) for i in order[:20]}
    write_json(ROOT / args.report, report)
    print("mean over equal-footing months:", {n: {w: round(v, 4) for w, v in d.items()} for n, d in report["mean_change_pct_equal_footing"].items()})
    print("top inputs by gain:", {k: round(v, 3) for k, v in list(report["gain_importance_top"].items())[:12]})
    for month in MONTHS:
        item = report["months"][month]["aggregates"]["by_w"]["1.0"]
        print(month, "w 1.0 ordinary", round(item["ordinary_change_pct"], 4), "scope", {k: round(v, 3) for k, v in item["by_scope_pct"].items()}, "airports", {k: round(v, 3) for k, v in item["by_airport_pct"].items()})


if __name__ == "__main__":
    main()
