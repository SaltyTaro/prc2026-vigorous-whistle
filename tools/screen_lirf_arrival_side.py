"""LIRF label-mechanism screen, follow-up: do arrival-side reporting signals separate the label classes of LIRF clock rows outside the population?

Rows and classes: tools/screen_lirf_mechanism.rows() (v46 development stack, Jan/Jul/Nov/Dec 2025). Added label-free inputs, all
from ARRIVAL rows (their block times are supplied in every month, ranking months included):
  inbound leg = the last arrival at the same stand before the departure's NM off-block clock (within 12 h)
  its class (block = schedule fill / block vs landing), taxi-in, block second, gap to the departure's clock and schedule
  arrival fill share and arrival block-second-zero share: same operator that day, airport in the hour, airport that day
Readings: AUC early-vs-agree and fill-vs-rest, month-blocked, with and without the arrival-side inputs (stack outputs included);
and a month-blocked residual model on label - v46 with and without them (RMSE on these rows). Development months only, descriptive;
nothing is declared, exported or uploaded. Output: printed (quoted in docs/clean_line_screens/lirf_mechanism.md).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import screen_lirf_mechanism as L
from screen_hour_effect import MONTHS, raw_file
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ID = "MVT_ID_mvt"
k = L.rows()
print("rows", len(k), k["cls"].value_counts(normalize=True).round(3).to_dict(), flush=True)
cols = [ID, "PHASE_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AIRCRAFT_OPERATOR_flt", "ARVT_3_flt", "ADEP_flt"]
arr = pd.concat([pd.read_parquet(raw_file(m), columns=cols) for m in MONTHS], ignore_index=True)
arr = arr[arr["PHASE_mvt"].eq("ARR") & arr["ADES_mvt"].eq("LIRF") & arr["BLOCK_TIME_UTC_mvt"].notna()].reset_index(drop=True)
E = pd.Timestamp("1970-01-01", tz="UTC")
sec = lambda s: (s - E).dt.total_seconds()
arr["block"], arr["land"], arr["sched"] = sec(arr["BLOCK_TIME_UTC_mvt"]), sec(arr["MVT_TIME_UTC_mvt"]), sec(arr["SCHED_TIME_UTC_mvt"])
arr["fill"] = (arr["block"] - arr["sched"]).abs().le(5).astype(float)
arr["sec0"] = arr["BLOCK_TIME_UTC_mvt"].dt.second.eq(0).astype(float)
arr["block_minus_sched"] = arr["block"] - arr["sched"]
arr["taxi_in"] = arr["TAXITIME_SEC_mvt"].astype(float)
arr["nm_missing"] = arr["ADEP_flt"].isna().astype(float)
arr["arvt_gap"] = arr["land"] - sec(arr["ARVT_3_flt"])
arr["day"] = arr["BLOCK_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
arr["hour"] = arr["BLOCK_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d %H")
arr["op"] = arr["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("NA")

k = k.copy()
k["aobt"] = sec(k["AOBT_3_flt"]); k["sched_s"] = sec(k["SCHED_TIME_UTC_mvt"])
k["stand_s"] = k["STAND_mvt"].astype("string").fillna("NA").astype(str)
arr["stand_s"] = arr["STAND_mvt"].astype("string").fillna("NA").astype(str)
# inbound leg by stand: merge_asof on the departure's NM off-block clock
left = k[[ID, "stand_s", "aobt"]].sort_values("aobt")
right = arr[["stand_s", "block", "fill", "sec0", "block_minus_sched", "taxi_in", "nm_missing", "arvt_gap", "land"]].sort_values("block").rename(columns={"block": "in_block"})
right["key"] = right["in_block"]
j = pd.merge_asof(left, right, left_on="aobt", right_on="key", by="stand_s", direction="backward", tolerance=12 * 3600.0)
j = j.set_index(ID).reindex(k[ID])
k["in_gap_clock"] = (k["aobt"].to_numpy() - j["in_block"].to_numpy())
k["in_gap_sched"] = (k["sched_s"].to_numpy() - j["in_block"].to_numpy())
for c in ["fill", "sec0", "block_minus_sched", "taxi_in", "nm_missing", "arvt_gap"]:
    k["in_" + c] = j[c].to_numpy()
k["in_found"] = j["in_block"].notna().to_numpy().astype(float)
# shares: operator-day, airport-hour, airport-day (arrivals only)
k["day"] = k["AOBT_3_flt"].dt.strftime("%Y-%m-%d"); k["hourk"] = k["AOBT_3_flt"].dt.strftime("%Y-%m-%d %H"); k["op"] = k["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("NA").astype(str)
for name, keys_a, keys_k in [("opday", ["op", "day"], ["op", "day"]), ("hour", ["hour"], ["hourk"]), ("day", ["day"], ["day"])]:
    g = arr.groupby(keys_a).agg(n=("fill", "size"), fill=("fill", "mean"), sec0=("sec0", "mean"), taxi=("taxi_in", "median"), bms=("block_minus_sched", "median")).reset_index()
    g.columns = [*keys_k, f"a_{name}_n", f"a_{name}_fill", f"a_{name}_sec0", f"a_{name}_taxi", f"a_{name}_bms"]
    k = k.merge(g, on=keys_k, how="left")
ARR = [c for c in k.columns if c.startswith(("in_", "a_"))]
print("arrival-side inputs", len(ARR), "inbound found", round(k["in_found"].mean(), 3), "inbound fill share", round(k["in_fill"].mean(), 3), flush=True)
print(k.groupby("cls")[["in_fill", "in_sec0", "a_opday_fill", "a_hour_fill", "a_day_fill", "in_gap_clock", "in_taxi_in"]].mean().round(3).to_string(), flush=True)

BASE = L.CATEGORICAL + L.NUMERIC + L.STACK_OUTPUTS
def blocked(frame, y, features, objective="binary"):
    out = np.full(len(frame), np.nan)
    for month in MONTHS:
        fit, held = (frame["month"] != month).to_numpy(), (frame["month"] == month).to_numpy()
        if objective == "binary":
            model = lgb.LGBMClassifier(**L.PARAMS).fit(frame.loc[fit, features], y[fit]); out[held] = model.predict_proba(frame.loc[held, features])[:, 1]
        else:
            p = {**L.PARAMS, "objective": "regression"}; model = lgb.LGBMRegressor(**p).fit(frame.loc[fit, features], y[fit]); out[held] = model.predict(frame.loc[held, features])
    return out
def auc_by_month(frame, y, s):
    m = frame["month"].to_numpy()
    return {**{mm: round(float(roc_auc_score(y[m == mm], s[m == mm])), 4) for mm in MONTHS}, "all": round(float(roc_auc_score(y, s)), 4)}
two = k[k["cls"].isin(["agree", "early"])].reset_index(drop=True); y2 = two["cls"].eq("early").to_numpy()
print("early vs agree  without", auc_by_month(two, y2, blocked(two, y2, BASE)), flush=True)
print("early vs agree  with   ", auc_by_month(two, y2, blocked(two, y2, BASE + ARR)), flush=True)
print("early vs agree  arrival-side only", auc_by_month(two, y2, blocked(two, y2, ARR)), flush=True)
kk = k.reset_index(drop=True); yf = kk["cls"].eq("fill").to_numpy()
print("fill vs rest    without", auc_by_month(kk, yf, blocked(kk, yf, BASE)), flush=True)
print("fill vs rest    with   ", auc_by_month(kk, yf, blocked(kk, yf, BASE + ARR)), flush=True)
print("fill vs rest    arrival-side only", auc_by_month(kk, yf, blocked(kk, yf, ARR)), flush=True)
# residual model
err = kk["err"].to_numpy(dtype=float)
def rm(v, m=None):
    m = np.ones(len(v), bool) if m is None else m
    return float(np.sqrt(np.mean(v[m] ** 2)))
for name, feats in [("without", BASE), ("with", BASE + ARR)]:
    c = blocked(kk, np.clip(err, -1800, 1800), feats, objective="regression")
    for w in (0.5, 1.0):
        e2 = err - w * c
        print(f"residual model {name:8s} w {w}: rows RMSE {rm(err):.2f} -> {rm(e2):.2f} ({100 * (rm(e2) / rm(err) - 1):+.3f}%)", {m: round(100 * (rm(e2, (kk.month == m).to_numpy()) / rm(err, (kk.month == m).to_numpy()) - 1), 3) for m in MONTHS}, flush=True)
