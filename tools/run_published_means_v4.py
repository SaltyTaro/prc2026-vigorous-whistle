"""configs/published_means_v4.json: published_means_v3's pm_np procedure applied last on a final-file candidate, all four months.

The mapping (published mean -> label mean, per airport, twelve 2025 months, LIRF without its fill population), the method bias and noise
(v72's development values, the four development months) and the gate / shrinkage are tools/run_published_means_v3.py's, unchanged; only the
target differs: every airport-month of the final file (January, February, June, July 2026) instead of the ranking file's January / July.
LIRF fill-population rows and the guard rows never move and are left out of LIRF's means. Light pandas, workstation.

    run_published_means_v4.py cells --base B                 # the cell table only (no file written)
    run_published_means_v4.py apply --base B --out O         # O and O's receipt (never under runs/submissions)
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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import run_published_means_v3 as V3  # noqa: E402
from prc2026_lirf_mixture.features import DEPARTURE_COLUMNS, population_rows  # noqa: E402
from run_published_means import fit_mapping, shift_for  # noqa: E402
from screen_published_means import AIRPORTS, CAP, WORKBOOK, published  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DECL = ROOT / "configs/published_means_v4.json"
MOVEMENTS = ROOT / "data/raw/final_ranking.parquet"
MONTHS = ["2026-01", "2026-02", "2026-06", "2026-07"]
GUARD = [213142471, 213142469]
CAND = "pm_np"
FLOOR = V3.FLOOR


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def method_terms() -> dict:
    """V3.run()'s development levels for pm_np and its method(a, None): bias and noise per airport over the four development months."""
    T = V3.dev_values()
    table = V3.label_table(True)
    pub = published()
    y, air, fx = T["label"].to_numpy(dtype=float), T["airport"].to_numpy(), T["fixed"].to_numpy()
    ctl = T["v72"].to_numpy(dtype=float)
    terms = {}
    for a in AIRPORTS:
        d = []
        for m in V3.MONTHS:
            sel = (T["month"] == m).to_numpy() & V3.selected(air, fx, a, CAND) & (ctl > 0) & (ctl <= CAP)
            q = pub[(pub["airport"] == a) & (pub["month"] == m)].iloc[0]
            mp = fit_mapping(table, a, m)
            ordy = sel & (y > 0) & (y <= CAP)
            d.append(float(ctl[sel].mean() - (q["P"] + mp["function"](q["u"]))) - float(ctl[ordy].mean() - y[ordy].mean()))
        d = np.array(d)
        terms[a] = (float(d.mean()), float(np.sqrt(np.mean((d - d.mean()) ** 2))))
    return {"terms": terms, "table": table, "pub": pub}


def cells(base: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    b = pd.read_parquet(base)
    mv = pd.read_parquet(MOVEMENTS, columns=DEPARTURE_COLUMNS)
    dep = mv[mv["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(b[ID])
    if dep["ADEP_mvt"].isna().any():
        raise ValueError("base rows outside the final file's departures")
    fixed = b[ID].isin(set(population_rows(mv)[ID]) | set(GUARD)).to_numpy()
    air, mon = dep["ADEP_mvt"].to_numpy(), dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()
    val = b[TARGET].to_numpy(dtype=float)
    M = method_terms()
    new, rec = val.copy(), {}
    for a in AIRPORTS:
        mp = fit_mapping(M["table"], a, None)
        bias, noise = M["terms"][a]
        for mo in MONTHS:
            q = M["pub"][(M["pub"]["airport"] == a) & (M["pub"]["month"] == mo)]
            if q.empty:
                raise ValueError(f"no published figure for {a} {mo}")
            q = q.iloc[0]
            inside = (air == a) & (mon == mo) & (val > 0) & (val <= CAP)
            sel = inside & ~fixed if a == "LIRF" else inside
            lev = float(val[sel].mean() - (q["P"] + mp["function"](q["u"])))
            shift, r = shift_for(lev, bias, max(mp["sd_map"], noise), float(q["u"]), mp["u_max"], V3.MAX_SIGMA)
            mvk = inside & ~fixed
            if r["eligible"]:
                new[mvk] = np.maximum(val[mvk] - shift, FLOOR)
            rec[f"{a} {mo}"] = {**r, "rows_moved": int(mvk.sum()) if r["eligible"] else 0,
                                "expected_gain_s2_if_estimate_holds": float(mvk.sum() * (2 * shift * r["effective_level_s"] - shift ** 2)) if r["eligible"] else 0.0}
    out = b.copy()
    moved = np.rint(new) != val
    out.loc[moved, TARGET] = np.rint(new[moved]).astype(b[TARGET].dtype)
    if not (out[ID].equals(b[ID]) and out.dtypes.equals(b.dtypes)):
        raise ValueError("order or dtypes changed")
    if (out[TARGET].to_numpy()[fixed] != b[TARGET].to_numpy()[fixed]).any():
        raise ValueError("a fixed row moved")
    return out, moved, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cells", "apply"])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if sha(ROOT / WORKBOOK) != json.loads(DECL.read_text(encoding="utf-8"))["external_data"]["sha256"]:
        raise ValueError("workbook differs from the declared file")
    out, moved, rec = cells(a.base)
    elig = {k: round(v["shift_s"], 2) for k, v in rec.items() if v["eligible"]}
    gain = sum(v["expected_gain_s2_if_estimate_holds"] for v in rec.values())
    print("eligible cells (shift s):", elig, "| rows moved", int(moved.sum()), "| expected gain if estimates hold", round(gain / 1e6, 2), "e6 s^2")
    if a.mode == "cells":
        return
    if a.out.exists():
        raise ValueError(f"{a.out} exists; never overwrite")
    if "submissions" in a.out.parts:
        raise ValueError("never under runs/submissions")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    base_v = pd.read_parquet(a.base)[TARGET].to_numpy(dtype=float)
    ch = out[TARGET].to_numpy(dtype=float) - base_v
    receipt = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "declaration_sha256": sha(DECL),
               "workbook_sha256": sha(ROOT / WORKBOOK), "movements_sha256": sha(MOVEMENTS), "base": a.base.as_posix(), "base_sha256": sha(a.base),
               "out": a.out.as_posix(), "out_sha256": sha(a.out), "rows": len(out), "rows_changed": int((ch != 0).sum()),
               "squared_change": float((ch ** 2).sum()), "max_abs_change": float(np.abs(ch).max()), "eligible": elig,
               "expected_gain_s2_if_estimates_hold": gain, "cells": rec,
               "code": {p: sha(ROOT / p) for p in ["tools/run_published_means_v4.py", "tools/run_published_means_v3.py", "tools/run_published_means.py",
                                                   "tools/screen_published_means.py"]}}
    a.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["out_sha256", "rows_changed", "squared_change", "max_abs_change", "eligible"]}, indent=1))


if __name__ == "__main__":
    main()
