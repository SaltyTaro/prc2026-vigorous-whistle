"""aud14 (exploration, nothing declared): same-season key memory on v74's development values.
Does the residual (label - v74 dev value) of a schedule key (flight number, callsign, operator + destination, operator + stand, ...)
carry from one W25 month to the adjacent one (Nov -> Dec, Dec -> Nov), while cross-season pairs (Jan 2025 <-> Dec, Jul -> Nov) do not?
The 29 September screen (docs/clean_line_screens/README.md section 1) pooled the other three months and read Nov / Dec corr +0.040,
Jan / Jul -0.01: the season pattern. Correction c = beta * n / (n + lam) * mean(clipped residual of the key in the fitting month), on
non-fixed rows; value = max(v74 + c, -12). Scored on every row of the target month."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import run_intruder_bound as R  # noqa: E402

ID = "MVT_ID_mvt"
FILES = {"2025-01": "training_2025-01-01_2025-02-01", "2025-07": "training_2025-07-01_2025-08-01",
         "2025-11": "training_2025-11-01_2025-12-01", "2025-12": "training_2025-12-01_2026-01-01"}
T = pd.read_parquet(ROOT / "runs/season_pass2/dev_sp2.parquet", columns=[ID, "month", "airport", "day", "label", "ship", "kind"])
raw = pd.concat([pd.read_parquet(ROOT / f"data/raw/{f}.parquet", columns=[ID, "PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADES_mvt",
                 "AIRCRAFT_OPERATOR_flt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]) for f in FILES.values()])
raw = raw[raw["PHASE_mvt"].eq("DEP")].drop(columns="PHASE_mvt")
t = T.merge(raw, on=ID, how="left", validate="one_to_one")
t["fn"] = t["FLIGHT_mvt"].astype("string").str.strip().str.upper()
t["cs"] = t["CALLSIGN_flt"].astype("string").str.strip().str.upper()
t["op"] = t["AIRCRAFT_OPERATOR_flt"].astype("string")
t["hr"] = t["SCHED_TIME_UTC_mvt"].dt.hour
t["dow"] = t["SCHED_TIME_UTC_mvt"].dt.dayofweek
t["res"] = t["label"] - t["ship"]
t["resc"] = t["res"].clip(-1800, 1800)
ordinary = (t["label"] < 20000).to_numpy()
fixed = (t["kind"] == 3).to_numpy()
KEYS = {"fn": ["airport", "fn"], "cs": ["airport", "cs"], "fn_dow": ["airport", "fn", "dow"], "op_ades": ["airport", "op", "ADES_mvt"],
        "op_stand": ["airport", "op", "STAND_mvt"], "op_hr": ["airport", "op", "hr"], "stand_rwy": ["airport", "STAND_mvt", "RUNWAY_mvt"],
        "type_hr": ["airport", "AIRCRAFT_TYPE_mvt", "hr"]}
PAIRS = [("2025-11", "2025-12"), ("2025-12", "2025-11"), ("2025-12", "2025-01"), ("2025-01", "2025-12"), ("2025-07", "2025-11"), ("2025-11", "2025-07")]


def correction(src_m: str, tgt_m: str, key: list[str], beta: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    s = (t["month"] == src_m).to_numpy() & ordinary & ~fixed
    g = t[s].groupby(key, observed=True, dropna=True)["resc"].agg(["sum", "count"]).reset_index()
    tg = (t["month"] == tgt_m).to_numpy()
    x = t[tg].merge(g, on=key, how="left")
    n = x["count"].fillna(0).to_numpy()
    mean = np.where(n > 0, x["sum"].fillna(0).to_numpy() / np.maximum(n, 1), 0.0)
    c = beta * n / (n + lam) * mean
    c[fixed[tg]] = 0.0
    return tg, c


rows = []
for name, key in KEYS.items():
    for beta in [0.25, 0.5]:
        for lam in [10.0, 30.0, 100.0]:
            line = {"key": name, "beta": beta, "lam": lam}
            for src_m, tgt_m in PAIRS:
                tg, c = correction(src_m, tgt_m, key, beta, lam)
                base = t["ship"].to_numpy()[tg]
                new = np.maximum(base + c, -12.0)
                y = t["label"].to_numpy()[tg]
                line[f"{src_m[5:]}>{tgt_m[5:]}"] = round(100 * (((y - new) ** 2).sum() / ((y - base) ** 2).sum() - 1), 4)
            rows.append(line)
    print(pd.DataFrame([r for r in rows if r["key"] == name]).to_string(index=False), flush=True)

best = pd.DataFrame(rows)
best["w25"] = (best["11>12"] + best["12>11"]) / 2
print("\nby W25 mean\n", best.sort_values("w25").head(12).to_string(index=False))
top = best.sort_values("w25").iloc[0]
rng = np.random.default_rng(20261004)
for src_m, tgt_m in [("2025-11", "2025-12"), ("2025-12", "2025-11")]:
    tg, c = correction(src_m, tgt_m, KEYS[top["key"]], top["beta"], top["lam"])
    sub = t[tg].assign(value=t["ship"].to_numpy()[tg])
    new = np.maximum(sub["value"].to_numpy() + c, -12.0)
    print(top["key"], src_m, "->", tgt_m, "interval", [round(v, 4) for v in R.interval(sub, "value", new, rng)],
          "by airport dSSE e6", (((sub["label"] - new) ** 2 - (sub["label"] - sub["value"]) ** 2).groupby(sub["airport"]).sum() / 1e6).round(2).to_dict())
