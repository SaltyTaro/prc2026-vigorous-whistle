"""aud18 (exploration, nothing declared): table for a W25 raw-trace pass on v76. Writes runs/audit_v72/aud18_dev.parquet with MVT_ID_mvt,
v76's development value (v75's + key_memory_v1 km_9's out-of-month corrections; January / July 2025 = v74's) and the 38 raw-trace features of
the 29 September screen (runs/clean_line_screens/trace/features_<month>.parquet, matched departures only). Light; workstation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import run_key_memory as KM  # noqa: E402

ID = "MVT_ID_mvt"
T, K, B, info = KM.base_chain()
assert info["base_version"] == 75
FILES = {"2025-01": "training_2025-01-01_2025-02-01", "2025-07": "training_2025-07-01_2025-08-01",
         "2025-11": "training_2025-11-01_2025-12-01", "2025-12": "training_2025-12-01_2026-01-01"}
T["fn"] = KM.flight_numbers([ROOT / f"data/raw/{f}.parquet" for f in FILES.values()], T[ID]).to_numpy()
y, base = T["label"].to_numpy(dtype=float), T["base"].to_numpy()
T["r"] = np.clip(y - base, -1800, 1800)
fixed, mon, air = (T["kind"] == 3).to_numpy(), T["month"].to_numpy(), T["airport"].to_numpy()
fit_ok = ~fixed & (y < 20000) & T["fn"].notna().to_numpy()
c = np.zeros(len(T))
for src, tgt in [("2025-11", "2025-12"), ("2025-12", "2025-11")]:
    tg = mon == tgt
    cc = KM.corr(T[tg], KM.key_means(T, fit_ok & (mon == src)))
    cc[fixed[tg]] = 0.0
    c[tg] = cc
km9 = ["EDDF", "EDDM", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
T["v76dev"] = np.where(np.isin(air, km9), np.maximum(base + c, -12.0), base)
tr = pd.concat([pd.read_parquet(ROOT / f"runs/clean_line_screens/trace/features_{m}.parquet") for m in FILES])
out = T[[ID, "v76dev"]].merge(tr, on=ID, how="left", validate="one_to_one")
out.to_parquet(ROOT / "runs/audit_v72/aud18_dev.parquet", index=False)
print(out.shape, "trace coverage by month", out.assign(m=mon)["tr_points"].notna().groupby(mon).mean().round(3).to_dict())
