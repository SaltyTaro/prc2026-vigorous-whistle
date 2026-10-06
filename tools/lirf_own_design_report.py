"""Design reading of configs/lirf_own_design_v1.json: LIRF non-population rows of october, 2025-01_natural, 2025-07_natural,
2025-07_thinned and 2025-12_natural, from tools/lirf_own_predict.py's transfer files (runs/lirf_own/transfer).

Reproduction first: the refitted controls' October stop predictions (runs/lirf_own/design/ctx_lirf_s1, lm25_32_s21) against the saved
runs' (max absolute difference must be 0.0 s), and their transfer predictions against the saved models' on every reading.
Composition: N = (1 - w) C + w M, w = 0.40 matched / 0.55 unmatched, 0.70 on every row of the dark readings; C = mean ctx_lirf s1-4,
M = mean lm25_32 s21-24 (saved); Cown = mean ctx_lirf_own s1-4, Mown = mean lm25_32own s21-24. Own route: p_first_mv not null in the
reading's own table (none on 2025-07_thinned). Items M_routed, C_routed, CM_routed (rule) and M_all, C_all, CM_all (reported).
L2 = (1 - b) v9 + b N, b in 0.35 / 0.5, change in % of RMSE against the control N; paired-day 95% interval at b = 0.35 (seconds).
GO (declared): october change at b 0.35 <= -0.20% with the interval below zero, 2025-07_natural and 2025-12_natural <= +0.01%, the
dark readings <= +0.01%. Carried: lowest october change among GO items; CM_routed only if GO and below both singles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402

DECL = "configs/lirf_own_design_v1.json"
TRANSFER = ROOT / "runs/lirf_own/transfer"
STACK = ROOT / "runs/clean_line_screens/hour/dev_stack.parquet"
COMMITTEE = ROOT / "runs/adsb_nn/design/committee_october.parquet"
THINNED = ROOT / "runs/coverage_regime/a13_jul/features/thinned/committee/adsb_training_2025-07-01_2025-08-01.parquet"
OWN = ROOT / "data/external/adsblol/features_v19own"
READINGS = ["october", "2025-01_natural", "2025-07_natural", "2025-07_thinned", "2025-12_natural"]
DARK = {"2025-01_natural", "2025-07_thinned"}
B_VALUES, BAR, SAFE = [0.35, 0.5], -0.20, 0.01
RULE_ITEMS, REPORTED = ["M_routed", "C_routed", "CM_routed"], ["M_all", "C_all", "CM_all"]


def frame_for(reading: str, proxy: RecipeProxy, stack: pd.DataFrame) -> pd.DataFrame:
    if reading == "october":
        t = proxy.table
        keep = t["airport"].eq("LIRF").to_numpy() & ~t["population"].astype(bool).to_numpy()
        out = pd.DataFrame({ID: t[ID].to_numpy()[keep], "label": t[TARGET].to_numpy(dtype=float)[keep], "v9": t["v9"].to_numpy(dtype=float)[keep],
                            "day": t[TIME].dt.strftime("%Y-%m-%d").to_numpy()[keep], "matched": t["adsb_matched"].eq(1).to_numpy()[keep]})
        month = "2025-10"
    else:
        month, regime = reading.split("_")
        s = stack[stack["month"].eq(month) & (stack["lirf_m"] | stack["lirf_u"])]
        out = pd.DataFrame({ID: s[ID].to_numpy(), "label": s["label"].to_numpy(dtype=float), "v9": s["v9"].to_numpy(dtype=float),
                            "day": s["day"].astype(str).to_numpy(), "matched": s["lirf_m"].to_numpy(dtype=bool)})
        if regime == "thinned":
            flag = pd.read_parquet(THINNED, columns=[ID, "adsb_matched"]).set_index(ID)["adsb_matched"].reindex(out[ID])
            if flag.isna().any():
                raise ValueError(f"{reading}: thinned table misses rows.")
            out["matched"] = flag.eq(1).to_numpy()
    if reading == "2025-07_thinned":
        out["route"] = False
    else:
        own = pd.read_parquet(OWN / f"features_v19_{month}.parquet", columns=[ID, "p_first_mv"]).set_index(ID)["p_first_mv"]
        out["route"] = own.reindex(out[ID]).notna().to_numpy()
    return out


def main() -> None:
    decl = read_json(ROOT / DECL)
    files: dict[str, str] = {}

    def values(name: str, reading: str, ids: pd.Series) -> np.ndarray:
        path = TRANSFER / name / f"{reading}.parquet"
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        v = pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        if not np.isfinite(v).all():
            raise ValueError(f"{path} does not cover the reading's rows.")
        return v

    # reproduction of the saved design runs by the refitted controls
    repro = {}
    for new, old in [("runs/lirf_own/design/ctx_lirf_s1/design", "runs/adsb_ctx/lirf_design/ctx_lirf_s1/design"),
                     ("runs/lirf_own/design/lm25_32_s21/design", "runs/adsb_ctx3_lirf/design/lm25_32_s21/design")]:
        a = pd.read_parquet(ROOT / new / "stop_predictions.parquet").set_index(ID)["adsb_full"]
        b = pd.read_parquet(ROOT / old / "stop_predictions.parquet").set_index(ID)["adsb_full"].reindex(a.index)
        d = np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
        repro[new] = {"rows": int(len(d)), "max_abs_s": float(np.max(d)), "exact": bool(np.max(d) == 0.0),
                      "model_sha256_equal": sha256_file(ROOT / new / "model.pt") == sha256_file(ROOT / old / "model.pt")}
    for name_new, name_old in [("new_ctx_lirf_s1", "ctx_lirf_s1"), ("new_lm25_32_s21", "lm25_32_s21")]:
        for reading in READINGS:
            a = pd.read_parquet(TRANSFER / name_new / f"{reading}.parquet")["adsb_full"].to_numpy()
            b = pd.read_parquet(TRANSFER / name_old / f"{reading}.parquet")["adsb_full"].to_numpy()
            repro[f"{name_new}:{reading}"] = float(np.max(np.abs(a - b)))
    print("reproduction:", repro)

    proxy = RecipeProxy(pd.read_parquet(COMMITTEE))
    stack = pd.read_parquet(STACK, columns=[ID, "month", "day", "airport", "label", "v9", "lirf_m", "lirf_u"])
    seeds = {k: v["seeds"] for k, v in decl["fits"].items()}
    out, go_parts = {}, {k: {} for k in RULE_ITEMS}
    for reading in READINGS:
        f = frame_for(reading, proxy, stack)
        ids, y, v9, days, m, route = f[ID], f["label"].to_numpy(), f["v9"].to_numpy(), f["day"].to_numpy(), f["matched"].to_numpy(), f["route"].to_numpy()
        C = np.mean([values(f"ctx_lirf_s{s}", reading, ids) for s in (1, 2, 3, 4)], axis=0)
        M = np.mean([values(f"lm25_32_s{s}", reading, ids) for s in (21, 22, 23, 24)], axis=0)
        Co = np.mean([values(f"new_ctx_lirf_own_s{s}", reading, ids) for s in seeds["ctx_lirf_own"]], axis=0)
        Mo = np.mean([values(f"new_lm25_32own_s{s}", reading, ids) for s in seeds["lm25_32own"]], axis=0)
        w = np.full(len(y), 0.70) if reading in DARK else np.where(m, 0.40, 0.55)
        comp = lambda c, mm: (1 - w) * c + w * mm  # noqa: E731
        nets = {"control": comp(C, M),
                "M_routed": comp(C, np.where(route, Mo, M)), "C_routed": comp(np.where(route, Co, C), M),
                "CM_routed": comp(np.where(route, Co, C), np.where(route, Mo, M)),
                "M_all": comp(C, Mo), "C_all": comp(Co, M), "CM_all": comp(Co, Mo)}
        parts = {"all": np.ones(len(y), dtype=bool), "route": route, "matched": m, "unmatched": ~m}
        e = {"rows": {k: int(v.sum()) for k, v in parts.items()}, "v9_rmse": rmse(y, v9),
             "L1": {name: {k: (rmse(y[v], x[v]) if v.any() else None) for k, v in parts.items()}
                    for name, x in {"C": C, "M": M, "Cown": Co, "Mown": Mo, "N_control": nets["control"]}.items()},
             "L2": {}}
        for name in [*RULE_ITEMS, *REPORTED]:
            x = nets[name]
            q = {}
            for b in B_VALUES:
                cand, ctrl = (1 - b) * v9 + b * x, (1 - b) * v9 + b * nets["control"]
                q[str(b)] = {"rmse": rmse(y, cand), "control_rmse": rmse(y, ctrl), "change_pct": 100 * (rmse(y, cand) / rmse(y, ctrl) - 1),
                             "route_change_pct": (100 * (rmse(y[route], cand[route]) / rmse(y[route], ctrl[route]) - 1)) if route.any() else None,
                             "matched_change_pct": (100 * (rmse(y[m], cand[m]) / rmse(y[m], ctrl[m]) - 1)) if m.any() else None,
                             "unmatched_change_pct": (100 * (rmse(y[~m], cand[~m]) / rmse(y[~m], ctrl[~m]) - 1)) if (~m).any() else None}
            cand, ctrl = 0.65 * v9 + 0.35 * x, 0.65 * v9 + 0.35 * nets["control"]
            q["interval_b0.35_s"] = day_interval(days, y, cand, ctrl)
            q["days_better_b0.35"] = int(pd.DataFrame({"d": days, "c": (cand - y) ** 2 - (ctrl - y) ** 2}).groupby("d")["c"].sum().lt(0).sum())
            q["days"] = int(len(set(days)))
            e["L2"][name] = q
        out[reading] = e
        for name in RULE_ITEMS:
            go_parts[name][reading] = (e["L2"][name]["0.35"]["change_pct"], e["L2"][name]["interval_b0.35_s"])
    go = {}
    for name, g in go_parts.items():
        a = g["october"][0] <= BAR and g["october"][1][1] < 0
        b = all(g[r][0] <= SAFE for r in ("2025-07_natural", "2025-12_natural"))
        c = all(g[r][0] <= SAFE for r in DARK)
        go[name] = {"october": a, "natural_not_worse": b, "dark_not_worse": c, "go": bool(a and b and c), "october_change_pct": g["october"][0]}
    eligible = [n for n in RULE_ITEMS if go[n]["go"]]
    carried = None
    if eligible:
        singles = [n for n in eligible if n != "CM_routed"]
        if "CM_routed" in eligible and all(go["CM_routed"]["october_change_pct"] < go[s]["october_change_pct"] for s in ("M_routed", "C_routed")):
            carried = "CM_routed"
        elif singles:
            carried = min(singles, key=lambda n: go[n]["october_change_pct"])
        else:
            carried = None
    controls_exact = all(v["exact"] for k, v in repro.items() if isinstance(v, dict))
    rep = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "reproduction": repro,
           "controls_exact": controls_exact, "readings": out, "go": go, "carried": carried if controls_exact else None,
           "carried_if_controls_exact": carried, "inputs_sha256": files, "audit_used": False,
           "note": "Design-fold models (fit 2025-02..06, 08, 09; October monitored); January / July / December are transfer readings. No official score, leaderboard, ranking label or published aggregate read."}
    path = ROOT / "docs/lirf_own/design_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, rep)
    for reading, e in out.items():
        l1 = e["L1"]
        print(f"{reading:16s} rows {e['rows']['all']:6d} route {e['rows']['route']:6d} unm {e['rows']['unmatched']:6d} | L1 route C {l1['C']['route'] or float('nan'):.2f} Cown {l1['Cown']['route'] or float('nan'):.2f} M {l1['M']['route'] or float('nan'):.2f} Mown {l1['Mown']['route'] or float('nan'):.2f}")
        for name in [*RULE_ITEMS, *REPORTED]:
            q = e["L2"][name]
            rc = q["0.35"]["route_change_pct"]
            print(f"    {name:10s} b.35 {q['0.35']['change_pct']:+.3f}% [{q['interval_b0.35_s'][0]:+.2f}, {q['interval_b0.35_s'][1]:+.2f}] s  route {rc if rc is None else round(rc, 3)}%  b.5 {q['0.5']['change_pct']:+.3f}%  days {q['days_better_b0.35']}/{q['days']}")
    print("GO:", go)
    print("carried:", rep["carried"], "controls exact:", controls_exact)


if __name__ == "__main__":
    main()
