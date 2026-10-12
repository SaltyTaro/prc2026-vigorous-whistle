"""configs/lirf_cap_v1.json: cap LIRF fill-population predictions at max(schedule interval, the largest 2025 one-day label).

On population rows the 2025 labels with s > 43,200 s take two values only (the fill, label = s; the one-day record, 86,400 s + a taxi),
so a prediction above both is improved by the cap whichever is true. Light pandas, workstation.

    run_lirf_cap.py derive                      # U from the twelve 2025 training files (must equal the declared 88,392 s)
    run_lirf_cap.py dev                         # 2025 out-of-sample reading on runs/daily_totals/rows12.parquet
    run_lirf_cap.py apply --base B --movements M --out O   # the layer, applied last; writes O and O's receipt
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_lirf_mixture.features import DEPARTURE_COLUMNS, population_rows  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DECL = ROOT / "configs/lirf_cap_v1.json"
TRAINING = sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))
ROWS12 = ROOT / "runs/daily_totals/rows12.parquet"
DOCS = ROOT / "docs/lirf_cap"
U_DECLARED, ONE_DAY, FILL_TOL = 88392.0, 86400.0, 60.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def population(path: Path, label: bool = False) -> pd.DataFrame:
    cols = DEPARTURE_COLUMNS + ([TARGET] if label else [])
    pop = population_rows(pd.read_parquet(path, columns=cols))
    pop["s"] = (pop["MVT_TIME_UTC_mvt"] - pop["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    return pop[[ID, "s"] + ([TARGET] if label else [])]


def ceiling(pop: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    y, s = pop[TARGET].astype(float), pop["s"]
    one_day = pop[(y >= ONE_DAY) & ((y - s).abs() > FILL_TOL)]
    return float(one_day[TARGET].max()), one_day


def cap(pred: np.ndarray, s: np.ndarray, u: float) -> np.ndarray:
    limit = np.maximum(s, u)
    return np.where(np.isfinite(limit) & (pred > limit), limit, pred)


def derive() -> dict:
    pop = pd.concat([population(p, label=True) for p in TRAINING], ignore_index=True)
    u, one_day = ceiling(pop)
    if u != U_DECLARED:
        raise ValueError(f"U = {u} differs from the declared {U_DECLARED}")
    long_rows = pop[pop["s"] > 43200]
    fills = int(((long_rows[TARGET] - long_rows["s"]).abs() <= FILL_TOL).sum())
    return {"U": u, "population_rows_2025": len(pop), "one_day_rows": len(one_day),
            "one_day_label_range": [float(one_day[TARGET].min()), u],
            "long_rows_s_over_43200": len(long_rows), "long_fills": fills, "long_one_day": len(long_rows) - fills,
            "training_sha256": {p.name: sha(p) for p in TRAINING}}


def dev() -> dict:
    r = pd.read_parquet(ROWS12, columns=[ID, "month", "label", "stack", "extra", "complete"])
    pop = pd.concat([population(p) for p in TRAINING], ignore_index=True)
    d = r.merge(pop, on=ID)
    d = d[~d["extra"].astype(bool)]
    d["new"] = cap(d["stack"].to_numpy(float), d["s"].to_numpy(float), U_DECLARED)
    fired = d[d["new"] != d["stack"]]
    months = {}
    for m in sorted(fired["month"].unique()):
        full = r[(r["month"] == m) & r["complete"].astype(bool)]
        sse = float(((full["stack"] - full["label"]) ** 2).sum())
        f = fired[fired["month"] == m]
        gain = float(((f["stack"] - f["label"]) ** 2).sum() - ((f["new"] - f["label"]) ** 2).sum())
        months[m] = {"rows": len(full), "rmse": (sse / len(full)) ** 0.5,
                     "rmse_capped": ((sse - gain) / len(full)) ** 0.5, "gain_s2": gain,
                     "change_pct": 100 * (((sse - gain) / sse) ** 0.5 - 1)}
    return {"population_rows_with_prediction": len(d), "by_month": d["month"].value_counts().sort_index().to_dict(),
            "fired": fired[[ID, "month", "s", "label", "stack", "new"]].to_dict("records"), "months": months,
            "rows12_sha256": sha(ROWS12)}


def apply(base: Path, movements: Path, out: Path) -> dict:
    b = pd.read_parquet(base)
    pop = population(movements)
    d = b[[ID]].merge(pop, on=ID, how="left")
    pred = b[TARGET].to_numpy()
    new = cap(pred.astype(float), d["s"].to_numpy(float), U_DECLARED)
    moved = new != pred
    o = b.copy()
    o.loc[moved, TARGET] = np.rint(new[moved]).astype(b[TARGET].dtype)
    if not (o[ID].equals(b[ID]) and o.dtypes.equals(b.dtypes)):
        raise ValueError("order or dtypes changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out, index=False)
    rows = pd.DataFrame({ID: b.loc[moved, ID], "s": d.loc[moved, "s"], "base": pred[moved], "new": o.loc[moved, TARGET]})
    receipt = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "declaration_sha256": sha(DECL), "base": str(base), "base_sha256": sha(base),
               "movements": str(movements), "movements_sha256": sha(movements), "out": str(out), "out_sha256": sha(out),
               "rows": len(o), "population_rows": int(pop[ID].isin(b[ID]).sum()), "moved": rows.to_dict("records"),
               "squared_change": float(((rows["new"] - rows["base"]).astype(float) ** 2).sum())}
    out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1, default=str), encoding="utf-8")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["derive", "dev", "apply"])
    ap.add_argument("--base", type=Path)
    ap.add_argument("--movements", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.mode == "apply":
        print(json.dumps(apply(a.base, a.movements, a.out), indent=1, default=str))
        return
    res = derive() if a.mode == "derive" else dev()
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "results_v1.json"
    allres = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    allres[a.mode] = res
    path.write_text(json.dumps(allres, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
