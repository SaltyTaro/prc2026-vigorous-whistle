"""aud18 scoring (exploration): new = max(v76dev + clip(0.5 x pred, +-120), -12) per airport and variant; per-airport dSSE (1e6 s^2) for
December (from November), November (from December) and January 2025 (from December, cross-season); month RMSE % with paired-day intervals
for each variant over the airports where both W25 directions gain."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import run_intruder_bound as R  # noqa: E402

ID = "MVT_ID_mvt"
T = pd.read_parquet(ROOT / "runs/season_pass2/dev_sp2.parquet", columns=[ID, "month", "airport", "day", "label", "kind"])
T = T.merge(pd.read_parquet(ROOT / "runs/audit_v72/aud18_dev.parquet", columns=[ID, "v76dev", "tr_points"]), on=ID, validate="one_to_one")
P = pd.concat([pd.read_parquet(ROOT / f"runs/audit_v72/aud18_preds_{v}.parquet").set_index(ID) for v in ["ctl", "trace", "tronly"]], axis=1).reindex(T[ID])
y, b = T["label"].to_numpy(dtype=float), T["v76dev"].to_numpy()
mon, air = T["month"].to_numpy(), T["airport"].to_numpy()
APS = sorted(set(air))
rows = []
for v in ["ctl", "trace", "tronly"]:
    for a in APS:
        p = P[f"{v}_{a}"].to_numpy()
        m = ~np.isnan(p)
        new = b.copy()
        new[m] = np.maximum(b[m] + np.clip(0.5 * p[m], -120, 120), -12)
        d = (y - new) ** 2 - (y - b) ** 2
        rows.append({"variant": v, "airport": a, **{mo: round(d[(mon == mo) & (air == a)].sum() / 1e6, 2) for mo in ["2025-11", "2025-12", "2025-01"]},
                     "S_nov_dec": round(((new - b) ** 2)[np.isin(mon, ["2025-11", "2025-12"]) & (air == a)].sum() / 1e6, 2)})
D = pd.DataFrame(rows)
print(D.pivot(index="airport", columns="variant", values=["2025-11", "2025-12", "2025-01"]).to_string())
print(D.pivot(index="airport", columns="variant", values="S_nov_dec").to_string())
rng = np.random.default_rng(20261004)
for v in ["ctl", "trace", "tronly"]:
    g = D[(D["variant"] == v) & (D["2025-11"] < 0) & (D["2025-12"] < 0)]
    aps = g["airport"].tolist()
    new = b.copy()
    for a in aps:
        p = P[f"{v}_{a}"].to_numpy()
        m = ~np.isnan(p)
        new[m] = np.maximum(b[m] + np.clip(0.5 * p[m], -120, 120), -12)
    line = {mo: [round(x, 4) for x in R.interval(T[mon == mo].assign(value=b[mon == mo]), "value", new[mon == mo], rng)] for mo in ["2025-11", "2025-12", "2025-01"]}
    print(v, "airports gaining both ways:", aps, line, flush=True)
