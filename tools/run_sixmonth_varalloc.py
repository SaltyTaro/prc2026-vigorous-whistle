"""configs/sixmonth_varalloc_v1.json: six-month refit of the variance re-allocation layer (the early chain's step 5).

The shipped layer (tools/scout_var_alloc.py export, configs/scout_var_alloc_v1.json) fits its two LightGBMs (log squared residual and
P(late)) on the four development months' eligible rows (runs/daily_totals/rows.parquet, the extreme-day layer month-blocked), and k as the
least-squares slope over the same rows with out-of-month weights. The procedure P(pool) below is that fit for any pool of months:
  models = fit on the pool's eligible rows; weights of each pool month from models fitted on the pool without that month;
  k = slope over the pool's eligible rows on those out-of-month weights.
P(four development months) is the shipped layer (checked: its k and its values on the Tier B file equal the Tier B chain's step 5).
Tier B rows (February / June 2025, runs/febjun_tierb/chain/work, lane M's rows read) enter a pool with their own extreme-day input
(the Tier B chain's steps R / X) and labels. Settings, features, caps and eligibility are var_alloc's, unchanged.

    run_sixmonth_varalloc.py read     # the declared readings -> docs/sixmonth/varalloc_results_v1.json (heavy part: ~30 pool fits; Azure)

No official score enters; nothing is uploaded; nothing is written under runs/submissions.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_scout import extreme_shift, var_alloc  # noqa: E402

DECLARATION = "configs/sixmonth_varalloc_v1.json"
RESULTS = ROOT / "docs/sixmonth/varalloc_results_v1.json"
TB = ROOT / "runs/febjun_tierb/chain"
F_RAW = "runs/febjun_tierb/chain/input/febjun2025.parquet"
F_DETAIL = "runs/febjun_tierb/chain/root/recipe/detail.parquet"
F_BASE = "runs/febjun_tierb/chain/work/steps/rematch/base/vigorous-whistle_v55m_rematch.parquet"
F_R, F_X, F_Y = TB / "work/steps/R", TB / "work/steps/X/vigorous-whistle_v66.parquet", TB / "work/steps/Y"
DEV = ["2025-01", "2025-07", "2025-11", "2025-12"]
TIERB = ["2025-02", "2025-06"]
HELD = ["2025-01", "2025-07", "2025-02", "2025-06"]
SEED = 20261004
SHIPPED_K = 0.24186632360173135
CODE = ["tools/run_sixmonth_varalloc.py", "tools/scout_var_alloc.py", "src/prc2026_scout/var_alloc.py", "tools/scout_extreme_shift.py",
        "src/prc2026_scout/extreme_shift.py", "tools/export_daily_totals.py"]


def labels(ids: np.ndarray) -> np.ndarray:
    lab = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", TARGET]) for p in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))])
    lab = lab[lab["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID)[TARGET].astype(float)
    out = lab.reindex(ids).to_numpy()
    if np.isnan(out).any():
        raise ValueError("a row without a label")
    return out


def dev_side() -> dict:
    """scout_var_alloc.layered's table, without its fits: the four development months, the extreme-day layer month-blocked."""
    import scout_extreme_shift as SE
    import scout_var_alloc as SV
    KEYS = SE.KEYS
    t = SE.dev_rows("stack")
    days = extreme_shift.day_frame(t, KEYS)
    months = sorted(t.loc[~t["extra"], "month"].unique())
    if months != DEV:
        raise ValueError(f"development months {months}")
    ext, I = t["v48"].to_numpy(dtype=float).copy(), np.zeros(len(t))
    for m in months:
        params = extreme_shift.fit(days[days.index.get_level_values("month") != m].droplevel("month"))
        use = days[days.index.get_level_values("month") == m]
        ch = extreme_shift.shifts(use.droplevel("month"), params)
        ch.index = pd.MultiIndex.from_tuples([(m, a, b) for a, b in ch.index], names=KEYS)
        idx = np.flatnonzero(t["month"].eq(m).to_numpy())
        ext[idx] = extreme_shift.apply(t.iloc[idx], ch, KEYS)[0]
        I[idx] = SV.implied(t.iloc[idx], use.droplevel("month"), params["c0"], ["airport", "bday"])
    t["ext"], t["I"] = ext, I
    t["r"] = t["label"] - t["ext"]
    f = var_alloc.features(t, SV.departures(sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))), KEYS)
    params_all = extreme_shift.fit(days.droplevel("month"))
    return {"t": t.reset_index(drop=True), "f": f.reset_index(drop=True), "elig": var_alloc.eligible(t), "pool_month": t["month"].to_numpy(),
            "use": np.ones(len(t), dtype=bool), "params_all": params_all}


def f_side(params: dict) -> dict:
    """The Tier B file's rows at step 5's input, exactly as tools/scout_var_alloc.py export builds the ranking side."""
    import export_daily_totals
    import export_v33
    import scout_extreme_shift as SE
    import scout_var_alloc as SV
    import run_sixmonth as M
    export_daily_totals.RANKING, SE.RANKING, SV.RANKING = F_RAW, F_RAW, F_RAW
    export_v33.DRY = F_DETAIL
    t = SE.ranking_rows(F_BASE, F_R)
    days = extreme_shift.day_frame(t, ["airport", "bday"])
    ext, mask = extreme_shift.apply(t, extreme_shift.shifts(days, params), ["airport", "bday"])
    v55 = np.where(mask, np.rint(ext), t["v48_exported"].to_numpy(dtype=float)).astype(np.int64)
    x = pd.read_parquet(F_X).set_index(ID)[TARGET].reindex(t[ID]).to_numpy().astype(np.int64)
    if not np.array_equal(v55, x):
        raise ValueError("Tier B: the extreme-day layer does not reproduce steps/X")
    t["month"] = t["bday"].str[:7]
    I = SV.implied(t, days, params["c0"], ["airport", "bday"])
    t["ext"], t["I"], t["v55"] = ext, I, v55
    t["label"] = labels(t[ID].to_numpy())
    t["r"] = t["label"] - t["ext"]
    rows = M.tierb_rows(TB / "work").set_index(ID).reindex(t[ID])
    f = var_alloc.features(t, SV.departures([str(ROOT / F_RAW)]), SE.KEYS)
    return {"t": t.reset_index(drop=True), "f": f.reset_index(drop=True), "elig": var_alloc.eligible(t, SV.GUARD),
            "pool_month": rows["month"].to_numpy(), "use": rows["lane_m"].to_numpy(dtype=bool), "takeoff_day": rows["day"].to_numpy()}


class Pools:
    """P(pool) on the development table D and the Tier B table F, with every pool fit cached."""

    def __init__(self, D: dict, F: dict):
        self.D, self.F = D, F
        self.f = pd.concat([D["f"], F["f"]], ignore_index=True)
        cat = lambda k: np.concatenate([D["t"][k].to_numpy(dtype=float), F["t"][k].to_numpy(dtype=float)])
        self.a, self.label, self.r = cat("a"), cat("label"), cat("r")
        self.month = np.concatenate([D["pool_month"], F["pool_month"]])
        self.ok = np.concatenate([D["elig"] & D["use"], F["elig"] & F["use"]])
        self.nd = len(D["t"])
        self.cache, self.log, self.k_log = {}, [], {}

    def models(self, pool: frozenset):
        if pool not in self.cache:
            t0 = time.time()
            rows = self.ok & np.isin(self.month, sorted(pool))
            if set(pool) <= set(DEV):   # a development-only pool: fitted on the development frame alone (the shipped call)
                d = self.D
                rows_d = rows[: self.nd]
                self.cache[pool] = var_alloc.fit_models(d["f"], d["t"]["a"].to_numpy(dtype=float), d["t"]["label"].to_numpy(dtype=float),
                                                        d["t"]["r"].to_numpy(dtype=float), rows_d)
            else:
                self.cache[pool] = var_alloc.fit_models(self.f, self.a, self.label, self.r, rows)
            self.log.append({"pool": sorted(pool), "rows": int(rows.sum()), "seconds": round(time.time() - t0, 1)})
            print("fit", sorted(pool), int(rows.sum()), round(time.time() - t0, 1), "s", flush=True)
        return self.cache[pool]

    def weights_on(self, side: str, months_rows: np.ndarray, models) -> np.ndarray:
        S = self.D if side == "D" else self.F
        w = np.full(len(S["t"]), np.nan)
        sel = months_rows & S["elig"]
        if sel.any():
            w[sel] = var_alloc.weights(models, S["f"].loc[sel], np.ones(int(sel.sum()), dtype=bool))
        return w

    def procedure(self, pool: set) -> tuple:
        """(models on the pool, k from the pool's out-of-month weights)."""
        pool = frozenset(pool)
        if ("k", pool) in self.cache:
            return self.models(pool), self.cache[("k", pool)]
        full = self.models(pool)
        import scout_extreme_shift as SE
        bases, rs, rows_all = [], [], []
        for side, S in (("D", self.D), ("F", self.F)):
            w = np.full(len(S["t"]), np.nan)
            for j in sorted(pool):
                mrows = S["pool_month"] == j
                if mrows.any():
                    wj = self.weights_on(side, mrows, self.models(pool - {j}))
                    w[mrows] = wj[mrows]
            bases.append(var_alloc.basis(S["t"], w, S["t"]["I"].to_numpy(dtype=float), SE.KEYS))
            rs.append(S["t"]["r"].to_numpy(dtype=float))
            rows_all.append(S["elig"] & S["use"] & np.isin(S["pool_month"], sorted(pool)))
        if set(pool) <= set(DEV):   # the shipped call: the development arrays alone (same summation as scout_var_alloc.export)
            k = var_alloc.slope(rs[0], bases[0], rows_all[0])
        else:
            k = var_alloc.slope(np.concatenate(rs), np.concatenate(bases), np.concatenate(rows_all))
        self.cache[("k", pool)] = k
        self.k_log[",".join(sorted(pool))] = k
        return full, k

    def values(self, side: str, month: str, pool: set) -> np.ndarray:
        """Step 5's value on `side`'s rows of `month` with P(pool) (development: float, as the governing run; Tier B: the exported integer)."""
        import scout_extreme_shift as SE
        S = self.D if side == "D" else self.F
        models, k = self.procedure(pool)
        mrows = S["pool_month"] == month
        w = self.weights_on(side, mrows, models)
        base = var_alloc.basis(S["t"], w, S["t"]["I"].to_numpy(dtype=float), SE.KEYS)
        el = S["elig"] & mrows
        new = var_alloc.update(S["t"]["ext"].to_numpy(dtype=float), k, base, el)
        if side == "F":
            return np.where(el, np.rint(new), S["t"]["v55"].to_numpy(dtype=float))[mrows]
        return new[mrows]


def paired(days, y, cand, ref, seed=SEED) -> dict:
    f = pd.DataFrame({"d": days, "c": (cand - y) ** 2, "r": (ref - y) ** 2}).groupby("d").agg(c=("c", "sum"), r=("r", "sum"), n=("c", "size"))
    idx = np.random.default_rng(seed).integers(0, len(f), size=(2000, len(f)))
    n = f["n"].to_numpy()[idx].sum(1)
    rc, rr = np.sqrt(f["c"].to_numpy()[idx].sum(1) / n), np.sqrt(f["r"].to_numpy()[idx].sum(1) / n)
    return {"interval_pct": [float(v) for v in np.quantile(100 * (rc / rr - 1), [0.025, 0.975])],
            "interval_s": [float(v) for v in np.quantile(rc - rr, [0.025, 0.975])], "days": int(len(f)), "days_better": int((f["c"] < f["r"]).sum())}


def reading(side: str, S: dict, month: str, a: np.ndarray, b: np.ndarray) -> dict:
    mrows = S["pool_month"] == month
    gov = S["use"][mrows]
    t = S["t"][mrows]
    y, air = t["label"].to_numpy(dtype=float)[gov], t["airport"].to_numpy()[gov]
    days = (S["takeoff_day"][mrows] if side == "F" else t["bday"].to_numpy())[gov]
    a, b = a[gov], b[gov]
    sa, sb = (y - a) ** 2, (y - b) ** 2
    out = {"rows": int(gov.sum()), "rmse_a": float(np.sqrt(sa.mean())), "rmse_b": float(np.sqrt(sb.mean())),
           "change_pct": float(100 * (np.sqrt(sb.mean()) / np.sqrt(sa.mean()) - 1)), "dsse": float(sb.sum() - sa.sum()),
           "rows_moved": int((np.abs(b - a) > 1e-9).sum()), "squared_change": float(((b - a) ** 2).sum()), **paired(days, y, b, a)}
    g = pd.DataFrame({"airport": air, "a": sa, "b": sb, "ch": b - a}).groupby("airport")
    out["by_airport"] = {k: {"dsse": float(v["b"].sum() - v["a"].sum()), "change_pct": float(100 * (np.sqrt(v["b"].mean()) / np.sqrt(v["a"].mean()) - 1)),
                             "mean_change_s": float(v["ch"].mean())} for k, v in g}
    return out


def read() -> None:
    decl = read_json(ROOT / DECLARATION)
    if RESULTS.exists():
        raise ValueError("Preserve the reading.")
    t0 = time.time()
    D = dev_side()
    F = f_side(D["params_all"])
    P = Pools(D, F)
    res = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "official_scores_used": "none",
           "code": {c: sha256_file(ROOT / c) for c in CODE}, "readings": {}}
    # control: P(four development months) is the shipped layer
    _, k4 = P.procedure(set(DEV))
    yv = pd.read_parquet(F_Y / "rows.parquet").set_index(ID)["value"].reindex(F["t"][ID]).to_numpy(dtype=float)
    ctl = {m: P.values("F", m, set(DEV)) for m in TIERB}
    match = {m: bool(np.array_equal(ctl[m], yv[F["pool_month"] == m])) for m in TIERB}
    res["control"] = {"k_four_months": k4, "k_shipped": SHIPPED_K, "k_equal": bool(k4 == SHIPPED_K), "tierb_values_equal_steps_Y": match}
    print("control", res["control"], flush=True)
    if not (res["control"]["k_equal"] and all(match.values())):
        write_json(RESULTS.with_name("varalloc_control_failed.json"), res)
        raise ValueError("control failed: P(four development months) does not reproduce the shipped layer; nothing is read.")
    for m in HELD:
        if m in TIERB:
            other = [x for x in TIERB if x != m]
            a, b = ctl[m], P.values("F", m, set(DEV) | set(other))
            res["readings"][m] = {"a": "four development months (shipped)", "b": f"four development months + {other[0]}", **reading("F", F, m, a, b)}
        else:
            rest = set(DEV) - {m}
            a, b = P.values("D", m, rest), P.values("D", m, rest | set(TIERB))
            res["readings"][m] = {"a": f"{sorted(rest)}", "b": f"{sorted(rest)} + Tier B February / June", **reading("D", D, m, a, b)}
        r = res["readings"][m]
        print(m, f"{r['change_pct']:+.4f}%", [round(v, 4) for v in r["interval_pct"]], "dSSE", f"{r['dsse']:.3e}", flush=True)
    R = res["readings"]
    total = float(sum(R[m]["dsse"] for m in HELD))
    rule = {"sum_dsse_negative": total < 0, "february_june_point_le_zero": all(R[m]["change_pct"] <= 0 for m in TIERB),
            "no_month_worse_than_0.005pct": all(R[m]["change_pct"] <= 0.005 for m in HELD)}
    res.update({"sum_dsse": total, "rule": rule, "accepted": bool(all(rule.values())), "pool_fits": P.log, "seconds": round(time.time() - t0, 1),
                "k_by_pool": P.k_log})
    write_json(RESULTS, res)
    print("accepted", res["accepted"], rule, "sum dSSE", total, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["read"])
    a = ap.parse_args()
    if a.mode == "read":
        read()


if __name__ == "__main__":
    main()
