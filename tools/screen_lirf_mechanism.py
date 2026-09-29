"""LIRF label-mechanism screen: does the stack already tell the label mechanisms of LIRF clock rows apart?

LIRF clock-present, non-population rows hold about a fifth of the ordinary development squared error. By the airport block time
(known on labelled months only) they fall into four classes:
  agree   block within 300 s of the NM off-block clock
  early   block more than 300 s before the NM clock
  late    block more than 300 s after the NM clock
  fill    block within 5 s of the schedule time
Fill separation is on record as closed (docs/error_map/README.md). This screen reads the split between `early` and `agree`:
  1. the stack's own separation: AUC of (v46 - NM proxy) per month;
  2. a month-blocked LightGBM classifier on label-free inputs, with and without the stack's outputs;
  3. the stack's calibration inside bands of the classifier's probability.
A lever would need the classifier to separate clearly better than the stack does already. Development months only, descriptive;
nothing is declared, exported or uploaded. Rows: runs/clean_line_screens/hour/dev_stack.parquet (tools/screen_hour_effect.py).
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
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from screen_hour_effect import MONTHS, STACK, raw_file  # noqa: E402

CATEGORICAL = ["AIRCRAFT_OPERATOR_flt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "RUNWAY_mvt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"]
NUMERIC = ["proxy", "sched_int", "aobt_sched", "aobt_eobt", "hour", "weekday", "aobt_second", "mvt_second", "sched_second", "matched_f", "gate_f", "motion_minus_aobt",
           "segment_minus_aobt", "adsb_taxi", "points", "first_dist_stand_m", "stop_seconds", "slow_start_s", "taxi_stopped_s", "taxi_path_m"]
STACK_OUTPUTS = ["s_v46", "s_v9", "s_tree", "s_net", "s_fill"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "n_estimators": 400, "num_leaves": 31, "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
          "colsample_bytree": 0.7, "reg_lambda": 10.0, "random_state": 1, "n_jobs": 8, "verbose": -1, "cat_smooth": 20, "min_data_per_group": 50}
BANDS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def rows() -> pd.DataFrame:
    stack = pd.read_parquet(ROOT / STACK)
    stack = stack[stack["airport"].astype(str).eq("LIRF")]
    columns = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "ADEP_flt", *CATEGORICAL]
    raw = pd.concat([pd.read_parquet(raw_file(m), columns=columns) for m in MONTHS], ignore_index=True)
    raw = raw[raw["PHASE_mvt"].eq("DEP") & raw["ADEP_mvt"].eq("LIRF")]
    t = stack.merge(raw, on=ID, how="left", validate="one_to_one")
    matches = pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/matches/matches2_{m}.parquet") for m in MONTHS], ignore_index=True).drop_duplicates(ID).drop(columns=["airport"])
    t = t.merge(matches, on=ID, how="left")

    def seconds(a: str, b: str) -> pd.Series:
        return (t[a] - t[b]).dt.total_seconds()

    aobt = (t["AOBT_3_flt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    t["proxy"], t["sched_int"] = seconds("MVT_TIME_UTC_mvt", "AOBT_3_flt"), seconds("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
    t["aobt_sched"], t["aobt_eobt"] = seconds("AOBT_3_flt", "SCHED_TIME_UTC_mvt"), seconds("AOBT_3_flt", "EOBT_1_flt")
    block_sched, block_aobt = seconds("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), seconds("BLOCK_TIME_UTC_mvt", "AOBT_3_flt")      # class definition only, never an input
    t["cls"] = np.where(block_sched.abs() <= 5, "fill", np.where(block_aobt.abs() <= 300, "agree", np.where(block_aobt < -300, "early", "late")))
    t["motion_minus_aobt"], t["segment_minus_aobt"], t["adsb_taxi"] = t["motion_start"] - aobt, t["segment_start"] - aobt, t["takeoff"] - t["motion_start"]
    t["hour"] = t["MVT_TIME_UTC_mvt"].dt.hour + t["MVT_TIME_UTC_mvt"].dt.minute / 60
    t["weekday"] = t["MVT_TIME_UTC_mvt"].dt.weekday
    t["aobt_second"], t["mvt_second"], t["sched_second"] = t["AOBT_3_flt"].dt.second, t["MVT_TIME_UTC_mvt"].dt.second, t["SCHED_TIME_UTC_mvt"].dt.second
    t["gate_f"], t["matched_f"] = t["gate"].fillna(False).astype(float), t["matched"].astype(float)
    for name in CATEGORICAL:
        t[name] = t[name].astype("string").fillna("NA").astype("category")
    for name, column in (("s_v46", "v46"), ("s_v9", "v9"), ("s_tree", "lirf_tree"), ("s_net", "lirf_net")):
        t[name] = t[column] - t["proxy"]
    t["s_fill"] = t["v46"] - t["sched_int"]
    t["err"] = t["label"] - t["v46"]
    population = t["ADEP_flt"].isna() & t["AIRCRAFT_TYPE_mvt"].astype(str).eq("NA")
    keep = ~population & t["AOBT_3_flt"].notna() & (t["label"] < 10800)
    return t[keep].reset_index(drop=True)


def blocked(frame: pd.DataFrame, target: np.ndarray, features: list[str]) -> np.ndarray:
    import lightgbm as lgb
    out = np.full(len(frame), np.nan)
    for month in MONTHS:
        fit, held = (frame["month"] != month).to_numpy(), (frame["month"] == month).to_numpy()
        model = lgb.LGBMClassifier(**PARAMS).fit(frame.loc[fit, features], target[fit])
        out[held] = model.predict_proba(frame.loc[held, features])[:, 1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path("docs/clean_line_screens/lirf_mechanism.json"))
    args = parser.parse_args()
    from sklearn.metrics import roc_auc_score
    k = rows()
    total = float((k["err"] ** 2).sum())
    classes = {c: {"rows": int(len(g)), "share": float(len(g) / len(k)), "rmse": float(np.sqrt((g["err"] ** 2).mean())), "bias": float(g["err"].mean()),
                   "share_of_squared_error": float((g["err"] ** 2).sum() / total), "label_median": float(g["label"].median()), "proxy_median": float(g["proxy"].median()),
                   "matched_share": float(g["matched"].mean())} for c, g in k.groupby("cls")}
    two = k[k["cls"].isin(["agree", "early"])].reset_index(drop=True)
    y = two["cls"].eq("early").to_numpy()
    month = two["month"].to_numpy()

    def by_month(score: np.ndarray) -> dict:
        return {**{m: float(roc_auc_score(y[month == m], score[month == m])) for m in MONTHS}, "all": float(roc_auc_score(y, score))}

    readings = {"stack_minus_proxy": by_month(two["s_v46"].to_numpy())}
    scores = {}
    for name, features in (("classifier_label_free", CATEGORICAL + NUMERIC), ("classifier_with_stack_outputs", CATEGORICAL + NUMERIC + STACK_OUTPUTS),
                           ("classifier_stack_outputs_only", STACK_OUTPUTS)):
        scores[name] = blocked(two, y, features)
        readings[name] = by_month(scores[name])
    two["p_early"] = scores["classifier_with_stack_outputs"]
    two["band"] = pd.cut(two["p_early"], BANDS, include_lowest=True).astype(str)
    calibration = {b: {"rows": int(len(g)), "early_share": float(g["cls"].eq("early").mean()), "stack_bias": float(g["err"].mean()), "stack_rmse": float(np.sqrt((g["err"] ** 2).mean())),
                       "stack_minus_proxy": float(g["s_v46"].mean()), "label_minus_proxy": float((g["label"] - g["proxy"]).mean())} for b, g in two.groupby("band")}
    gain = readings["classifier_with_stack_outputs"]["all"] - readings["stack_minus_proxy"]["all"]
    report = {"created_at_utc": utc_now(), "rows": int(len(k)), "two_class_rows": int(len(two)), "early_base_rate": float(y.mean()), "classes": classes, "auc_early_vs_agree": readings,
              "auc_gain_over_stack": gain, "calibration_by_probability_band": calibration, "params": PARAMS, "inputs": {"categorical": CATEGORICAL, "numeric": NUMERIC, "stack_outputs": STACK_OUTPUTS},
              "stack_sha256": sha256_file(ROOT / STACK), "tool_sha256": sha256_file(ROOT / "tools/screen_lirf_mechanism.py"), "audit_used": False,
              "note": "development months only; the block time defines the classes and never enters an input; no ranking row and no official score is read"}
    write_json(ROOT / args.report, report)
    print("classes:", {c: {key: round(v, 3) for key, v in e.items()} for c, e in classes.items()})
    for name, entry in readings.items():
        print(f"{name:32s}", {key: round(v, 4) for key, v in entry.items()})
    print("AUC gain of the classifier with stack outputs over the stack's own score:", round(gain, 4))
    print(pd.DataFrame(calibration).T.round(2))


if __name__ == "__main__":
    main()
