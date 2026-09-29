"""Stack-weight screen: would weights per state, per airport or per airport and state beat the shipped weights?

Rows: the v46 development stack (runs/clean_line_screens/hour/dev_stack.parquet, tools/screen_hour_effect.py), non-LIRF rows with a
committee and a network value. Prediction before the gates: base + a1 (nets - base) + a2 (committee - base), base = v9 + residual.
The shipped weights in this form: outer weight 1.0 (LTFM) / 0.6 (matched) / 0.7 (unmatched), network share 0.85.
For each held month the weights (a1, a2) are fitted by least squares on the other three months, per cell of a scheme, shrunk toward
the global solution by `lam`, and read on the held month against the shipped weights. Development months only, descriptive; nothing
is declared, exported or uploaded.
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
from screen_hour_effect import EQUAL_FOOTING, MONTHS, STACK  # noqa: E402

LAMBDAS = [0.0, 0.3, 1.0]
MINIMUM = 2000


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path("docs/clean_line_screens/segment_weights.json"))
    args = parser.parse_args()
    r = pd.read_parquet(ROOT / STACK)
    r = r[r["airport"].astype(str).ne("LIRF") & r["nets"].notna() & r["committee"].notna()].reset_index(drop=True)
    airport, state = r["airport"].astype(str).to_numpy(), r["matched"].to_numpy().astype(bool)
    y, b, c, n = (r[name].to_numpy(dtype=float) for name in ("label", "base", "committee", "nets"))
    month = r["month"].to_numpy()
    outer = np.where(airport == "LTFM", 1.0, np.where(state, 0.6, 0.7))
    shipped = (1 - outer) * b + outer * (0.15 * c + 0.85 * n)
    x, t = np.c_[n - b, c - b], y - b
    schemes = {"global": np.full(len(r), "all"), "state": np.where(airport == "LTFM", "LTFM", np.where(state, "matched", "unmatched")), "airport": airport,
               "airport_state": np.char.add(np.char.add(airport, "|"), state.astype(str))}
    optimum = {}
    for m in MONTHS:
        for key in np.unique(schemes["state"]):
            k = (month == m) & (schemes["state"] == key)
            a = np.linalg.lstsq(x[k], t[k], rcond=None)[0]
            optimum[f"{m}|{key}"] = {"outer": float(a.sum()), "share": float(a[0] / a.sum()), "rows": int(k.sum())}
    readings = {}
    for name, keys in schemes.items():
        for lam in ([0.0] if name == "global" else LAMBDAS):
            out = {}
            for held in MONTHS:
                train = month != held
                g = np.linalg.lstsq(x[train], t[train], rcond=None)[0]
                p = b.copy()
                for key in np.unique(keys):
                    cell = keys == key
                    k = train & cell
                    a = g
                    if k.sum() >= MINIMUM:
                        gram, v = x[k].T @ x[k], x[k].T @ t[k]
                        ridge = lam * np.trace(gram) / 2
                        a = np.linalg.solve(gram + ridge * np.eye(2), v + ridge * g)
                    p[cell] = b[cell] + x[cell] @ a
                test = ~train
                out[held] = 100 * (rmse((y - p)[test]) / rmse((y - shipped)[test]) - 1)
            readings[f"{name}|lam{lam}"] = {**out, "mean": float(np.mean(list(out.values()))), "mean_equal_footing": float(np.mean([out[m] for m in EQUAL_FOOTING]))}
    report = {"created_at_utc": utc_now(), "rows": int(len(r)), "shipped_rmse_by_month": {m: rmse((y - shipped)[month == m]) for m in MONTHS},
              "in_sample_optimum_by_month_and_state": optimum, "held_month_change_pct_against_shipped_weights": readings, "lambdas": LAMBDAS, "minimum_rows_per_cell": MINIMUM,
              "stack_sha256": sha256_file(ROOT / STACK), "tool_sha256": sha256_file(ROOT / "tools/screen_segment_weights.py"), "audit_used": False,
              "note": "development months only; the gates, the stretch and the LIRF layers are left out on both sides; no ranking row and no official score is read"}
    write_json(ROOT / args.report, report)
    for name, e in readings.items():
        print(f"{name:22s}", {k: round(v, 3) for k, v in e.items()})
    print({k: (round(v["outer"], 2), round(v["share"], 2)) for k, v in optimum.items()})


if __name__ == "__main__":
    main()
