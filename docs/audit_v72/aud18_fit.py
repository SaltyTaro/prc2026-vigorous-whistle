"""aud18 (exploration, nothing declared; Azure): W25 pass on v76 with and without the 38 raw-trace features. Per airport, single seed 7,
the season passes' LightGBM settings; target clip(label - v76dev, +-1800) on non-fixed rows; December predicted from November and November
from December (and January 2025 from December, descriptive). Variants: ctl = season_pass_v2's 151 inputs + v76dev; trace = ctl + 38 tr_*;
tronly = 38 tr_* + v76dev + ship + kind. Writes runs/audit_v72/aud18_preds.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from run_season_pass2 import PARAMS, ROUNDS, features  # noqa: E402

ID = "MVT_ID_mvt"
T = pd.read_parquet(ROOT / "runs/season_pass2/dev_sp2.parquet")
A = pd.read_parquet(ROOT / "runs/audit_v72/aud18_dev.parquet")
T = T.merge(A, on=ID, how="left", validate="one_to_one")
TR = [c for c in A.columns if c.startswith("tr_")]
base = features(T)
VARIANTS = {"ctl": base + ["v76dev"], "trace": base + ["v76dev"] + TR, "tronly": TR + ["v76dev", "ship", "kind"]}
y = np.clip(T["label"].to_numpy(dtype=float) - T["v76dev"].to_numpy(), -1800, 1800)
fixed = (T["kind"] == 3).to_numpy()
mon, air = T["month"].to_numpy(), T["airport"].to_numpy()
out = pd.DataFrame({ID: T[ID]})
RUN = sys.argv[1:] or list(VARIANTS)
for v, cols in [(k, VARIANTS[k]) for k in RUN]:
    cats = [c for c in ["kind", "ap"] if c in cols]
    X = T[cols].astype(np.float32)
    for a in sorted(set(air)):
        col = np.full(len(T), np.nan)
        for src, tgts in [("2025-11", ["2025-12"]), ("2025-12", ["2025-11", "2025-01"])]:
            fit = (air == a) & (mon == src) & ~fixed
            m = lgb.train({**PARAMS, "seed": 7}, lgb.Dataset(X[fit], y[fit], categorical_feature=cats), ROUNDS)
            for tgt in tgts:
                s = (air == a) & (mon == tgt) & ~fixed
                if s.sum():
                    col[s] = m.predict(X[s])
        out[f"{v}_{a}"] = col
        print(v, a, "done", flush=True)
out.to_parquet(ROOT / f"runs/audit_v72/aud18_preds_{'_'.join(RUN)}.parquet", index=False)
print("written", flush=True)
