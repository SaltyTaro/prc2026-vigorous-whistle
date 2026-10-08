"""aud17 (exploration, nothing declared): a second W25 key-memory layer on v76's development values (v75's + key_memory_v1 km_9's
out-of-month corrections). Same form as km_9 (c = beta n/(n+lam) mean of clip(label - base, +-1800) over the key's fitting-month rows,
cap 120, floor -12, non-fixed rows); Nov -> Dec, Dec -> Nov, cross-season Dec -> Jan 2025 (descriptive)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import run_key_memory as KM  # noqa: E402
import run_intruder_bound as R  # noqa: E402

ID = "MVT_ID_mvt"
T, K, B, info = KM.base_chain()
print("base", info)
FILES = {"2025-01": "training_2025-01-01_2025-02-01", "2025-07": "training_2025-07-01_2025-08-01",
         "2025-11": "training_2025-11-01_2025-12-01", "2025-12": "training_2025-12-01_2026-01-01"}
raw = pd.concat([pd.read_parquet(ROOT / f"data/raw/{f}.parquet", columns=[ID, "PHASE_mvt", "FLIGHT_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt",
                 "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]) for f in FILES.values()])
raw = raw[raw["PHASE_mvt"].eq("DEP")].drop(columns="PHASE_mvt").set_index(ID).reindex(T[ID])
T["fn"] = raw["FLIGHT_mvt"].astype("string").str.strip().str.upper().to_numpy()
T.loc[T["fn"].str.len() == 0, "fn"] = pd.NA
T["op"] = raw["AIRCRAFT_OPERATOR_flt"].astype("string").to_numpy()
T["ades"] = raw["ADES_mvt"].astype("string").to_numpy()
T["stand"] = raw["STAND_mvt"].astype("string").to_numpy()
T["rwy"] = raw["RUNWAY_mvt"].astype("string").to_numpy()
T["atype"] = raw["AIRCRAFT_TYPE_mvt"].astype("string").to_numpy()
T["hr"] = raw["SCHED_TIME_UTC_mvt"].dt.hour.to_numpy()
y = T["label"].to_numpy(dtype=float)
fixed = (T["kind"] == 3).to_numpy()
mon, air = T["month"].to_numpy(), T["airport"].to_numpy()
ordinary = y < 20000

# v76's development values: km_9 on v75's
base = T["base"].to_numpy()
T["r"] = np.clip(y - base, -1800, 1800)
fit_ok = ~fixed & ordinary & T["fn"].notna().to_numpy()
c76 = np.zeros(len(T))
for src, tgt in [("2025-11", "2025-12"), ("2025-12", "2025-11")]:
    tg = mon == tgt
    c = KM.corr(T[tg], KM.key_means(T, fit_ok & (mon == src)))
    c[fixed[tg]] = 0.0
    c76[tg] = c
km9 = ["EDDF", "EDDM", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
v76 = np.where(np.isin(air, km9), np.maximum(base + c76, -12.0), base)
rng = np.random.default_rng(20261004)
for m in ["2025-11", "2025-12"]:
    s = mon == m
    print("check km_9 on v75 dev", m, [round(v, 4) for v in R.interval(T[s].assign(value=base[s]), "value", v76[s], rng)])
T["r2"] = np.clip(y - v76, -1800, 1800)

KEYS = {"op_hr": ["airport", "op", "hr"], "op_ades": ["airport", "op", "ades"], "op_stand": ["airport", "op", "stand"],
        "stand_rwy": ["airport", "stand", "rwy"], "type_hr": ["airport", "atype", "hr"], "ades": ["airport", "ades"], "stand": ["airport", "stand"],
        "fn_rwy": ["airport", "fn", "rwy"], "op": ["airport", "op"]}


def corr2(src, tgt, key, beta, lam):
    s = (mon == src) & ~fixed & ordinary
    g = T[s].groupby(key, observed=True, dropna=True)["r2"].agg(["sum", "count"]).reset_index()
    tg = mon == tgt
    x = T.loc[tg, key].merge(g, on=key, how="left")
    n = x["count"].fillna(0).to_numpy()
    mean = np.where(n > 0, x["sum"].fillna(0).to_numpy() / np.maximum(n, 1), 0.0)
    c = np.clip(beta * n / (n + lam) * mean, -120, 120)
    c[fixed[tg]] = 0.0
    return tg, c


rows = []
for name, key in KEYS.items():
    for beta in [0.25, 0.5]:
        for lam in [10.0, 30.0, 100.0]:
            line = {"key": name, "beta": beta, "lam": lam}
            for src, tgt in [("2025-11", "2025-12"), ("2025-12", "2025-11"), ("2025-12", "2025-01"), ("2025-01", "2025-12")]:
                tg, c = corr2(src, tgt, key, beta, lam)
                b0 = v76[tg]
                new = np.maximum(b0 + c, -12.0)
                yy = y[tg]
                line[f"{src[5:]}>{tgt[5:]}"] = round(100 * (((yy - new) ** 2).sum() / ((yy - b0) ** 2).sum() - 1), 4)
            rows.append(line)
    print(pd.DataFrame([r for r in rows if r["key"] == name]).to_string(index=False), flush=True)
D = pd.DataFrame(rows)
D["w25"] = (D["11>12"] + D["12>11"]) / 2
print("\nby W25 mean (SSE %)\n", D.sort_values("w25").head(10).to_string(index=False))
for _, top in D.sort_values("w25").head(3).iterrows():
    for src, tgt in [("2025-11", "2025-12"), ("2025-12", "2025-11")]:
        tg, c = corr2(src, tgt, KEYS[top["key"]], top["beta"], top["lam"])
        sub = T[tg].assign(value=v76[tg])
        new = np.maximum(sub["value"].to_numpy() + c, -12.0)
        d = (sub["label"] - new) ** 2 - (sub["label"] - sub["value"]) ** 2
        print(top["key"], top["beta"], top["lam"], src, "->", tgt, "rmse%", [round(v, 4) for v in R.interval(sub, "value", new, np.random.default_rng(20261004))],
              "dSSE by airport e6", (d.groupby(sub["airport"]).sum() / 1e6).round(2).to_dict(), "S", round(float((c ** 2).sum()) / 1e6, 2), flush=True)
