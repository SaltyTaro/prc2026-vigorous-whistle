"""Stack residual learner v3 (configs/stack_resid_v3.json): stack_resid_v2's guarded learner refitted with 19 own-aircraft turnaround /
pre-gap ADS-B features (dig tunnel 18, docs/dig/pregap.md); LIRF keeps v2's value. Imports tools/run_stack_resid.py unchanged.

  --score       governing run: v2 and v3 month-blocked on the four development months, the full chain (rematch -> stack -> intruder bound
                -> identity extension -> own-motion bound) against the v62 development proxy, v2's rule in-chain, the transfer guard on the
                ranking -> docs/stack_resid/results_v3.json + README section
  --dry-run     ranking: rematch chain file (c67f20d6) -> stack v3 -> intruder bound + identity extension -> own-motion bound
                -> runs/stack_resid/dry_v<N>/vigorous-whistle_v<N>.parquet (steps under steps/); also re-runs the same chain from v2's
                stack file (cff778e0) and requires the uploaded v62 byte for byte
  --build       rerun the dry run into runs/stack_resid/build_v<N>, require byte identity with the dry run, write
                runs/submissions/vigorous-whistle_v<N>.parquet (never overwrite) + docs/stack_resid/export_v<N>.json
Nothing is uploaded. No official score is used.
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
import run_stack_resid as S  # noqa: E402
import run_intruder_bound as R  # noqa: E402
import run_intruder_identity as I  # noqa: E402
import run_own_motion as OM  # noqa: E402

DECL = "configs/stack_resid_v3.json"
V2_DECL = "configs/stack_resid_v2.json"
V60_STACK = ("runs/stack_resid/dry_v60/vigorous-whistle_v60.parquet", "cff778e06fa7e402ee4170c64b6404e9c6bf394e11570ca79e128d18a3f45b1f")
V62 = ("runs/submissions/vigorous-whistle_v62.parquet", "cba143f84debdca0d9aff345ff4669ff6a3f0f7f5c4e02d08bcd3e0d02c32d9d")
DEV61 = ("runs/dig/census61/dev_v61.parquet", "9739d9c5bfa6258affaf8f9a6db3c001a46c1e1e92308bebcdcd6920c51448de")
FEAT = ROOT / "runs/stack_resid/v3_features"
SURF = ROOT / "data/external/adsblol/surface"
KEYMAP = ROOT / "runs/rematch_nonm/key_map.parquet"
TOL = 600
CORE = ["p_inb_rel", "p_tow", "p_n", "p_first_any", "p_first_mv", "p_last_park", "p_park10", "p_park20", "p_park40",
        "p_pregap_n", "p_pregap_last", "p_pregap_maxdist", "p_gap_len", "p_seg_start"]
REL = [("q_mv_vs_pred", "p_first_mv"), ("q_any_vs_pred", "p_first_any"), ("q_park_vs_pred", "p_last_park"),
       ("q_pregap_vs_pred", "p_pregap_last"), ("q_seg_vs_pred", "p_seg_start")]
NEW_COLS = CORE + [q for q, _ in REL]
CODE = ["tools/run_stack_resid_v3.py", "tools/run_stack_resid.py", "tools/run_intruder_bound.py", "tools/run_intruder_identity.py", "tools/run_own_motion.py"]
MOVE_COLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]


# ---------------------------------------------------------------- own-point features (declaration: own_point_features)

def _load(paths: list[Path]) -> pd.DataFrame:
    km = pd.read_parquet(KEYMAP)[["ADEP_mvt", "FLIGHT_mvt", "key"]].rename(columns={"ADEP_mvt": "airport"})
    mv = pd.concat([pd.read_parquet(p, columns=MOVE_COLS) for p in paths], ignore_index=True)
    mv["airport"] = np.where(mv["PHASE_mvt"].eq("ARR"), mv["ADES_mvt"], mv["ADEP_mvt"])
    mv = mv.merge(km, on=["airport", "FLIGHT_mvt"], how="left")
    mv["cs"] = mv["CALLSIGN_flt"].fillna(mv["key"]).str.upper().str.strip()
    mv["t"] = (mv["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    mv["day"] = mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m%d")
    return mv


def _surf(day: str, cols: list[str]) -> pd.DataFrame | None:
    p = SURF / f"surface_{day}.parquet"
    return pd.read_parquet(p, columns=cols) if p.exists() else None


def _identify(mv: pd.DataFrame) -> pd.DataFrame:
    out = []
    for day, g in mv[mv["cs"].notna()].groupby("day"):
        x = _surf(day, ["icao", "airport", "time", "callsign"])
        if x is None:
            continue
        x = x[x["callsign"].notna()].copy()
        x["cs"] = x["callsign"].str.upper().str.strip()
        m = pd.merge_asof(g.sort_values("t")[[ID, "PHASE_mvt", "airport", "cs", "t"]], x.sort_values("time")[["airport", "cs", "time", "icao"]],
                          left_on="t", right_on="time", by=["airport", "cs"], direction="nearest", tolerance=TOL)
        out.append(m[[ID, "PHASE_mvt", "icao"]])
    return mv.merge(pd.concat(out), on=[ID, "PHASE_mvt"], how="left")


def _dist_m(lat1, lon1, lat2, lon2):
    k = np.pi / 180
    x = (lon2 - lon1) * k * np.cos(0.5 * (lat1 + lat2) * k)
    y = (lat2 - lat1) * k
    return 6371000.0 * np.sqrt(x * x + y * y)


def _points(mv: pd.DataFrame, dep_ids: set) -> pd.DataFrame:
    arr = mv[mv["PHASE_mvt"].eq("ARR") & mv["icao"].notna() & mv["BLOCK_TIME_UTC_mvt"].notna()].copy()
    arr["inb"] = (arr["BLOCK_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    arr = arr.rename(columns={"STAND_mvt": "inb_stand"})[["airport", "icao", "inb", "inb_stand"]].sort_values("inb")
    dep = mv[mv["PHASE_mvt"].eq("DEP") & mv[ID].isin(dep_ids) & mv["icao"].notna()].copy()
    dep["ref"] = dep["t"] - 600.0
    dep = pd.merge_asof(dep.sort_values("ref"), arr, left_on="ref", right_on="inb", by=["airport", "icao"], direction="backward")
    dep["p_inb_rel"] = dep["t"] - dep["inb"]
    dep = dep[dep["p_inb_rel"].between(600, 18 * 3600)].copy()
    dep["p_tow"] = (dep["inb_stand"].astype(str) != dep["STAND_mvt"].astype(str)).astype(float)
    dep["lo"] = dep["inb"] + 120.0
    res = []
    for day, g in dep.groupby("day"):
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        cols = ["icao", "airport", "time", "ground", "gs", "lat", "lon"]
        pts = [p for p in (_surf(prev, cols), _surf(day, cols)) if p is not None]
        if not pts:
            continue
        p = pd.concat(pts)
        p = p[p["icao"].isin(set(g["icao"]))]
        j = g[[ID, "airport", "icao", "t", "lo"]].merge(p, on=["airport", "icao"])
        j = j[(j["time"] > j["lo"]) & (j["time"] <= j["t"] + 120)].sort_values([ID, "time"])
        if j.empty:
            continue
        j["rel"] = j["t"] - j["time"]
        j["park"] = j["ground"] & (j["gs"].fillna(0) < 1)
        j["mv"] = j["ground"] & (j["gs"].fillna(0) >= 3)
        pk = j[j["park"] & (j["time"] <= j["lo"] + 1800)].groupby(ID)[["lat", "lon"]].median()
        fp = j.groupby(ID)[["lat", "lon"]].first()
        P = fp.copy()
        P.loc[pk.index] = pk
        j = j.merge(P.rename(columns={"lat": "plat", "lon": "plon"}), left_on=ID, right_index=True)
        j["d"] = _dist_m(j["plat"], j["plon"], j["lat"], j["lon"])
        pre = j["rel"] >= 60
        j["dt"] = j.groupby(ID)["time"].diff()
        j["gap"] = (j["dt"] > 600).astype(int)
        j["gk"] = j.groupby(ID)["gap"].cumsum()
        j["gmax"] = j.groupby(ID)["gk"].transform("max")
        j["pregap"] = (j["gk"] < j["gmax"]) & (j["gmax"] > 0)
        a = pd.DataFrame(index=j[ID].unique())
        jp = j[pre]
        a["p_n"] = jp.groupby(ID).size()
        a["p_first_any"] = jp.groupby(ID)["rel"].max()
        a["p_first_mv"] = jp[jp["mv"]].groupby(ID)["rel"].max()
        a["p_last_park"] = jp[jp["park"]].groupby(ID)["rel"].min()
        for T in (600, 1200, 2400):
            q = jp[jp["park"] & (jp["d"] < 150) & (jp["rel"] <= T)]
            a[f"p_park{T // 60}"] = 0.0
            a.loc[q[ID].unique(), f"p_park{T // 60}"] = 1.0
        jg = j[j["pregap"]]
        a["p_pregap_n"] = jg.groupby(ID).size()
        a["p_pregap_last"] = jg.groupby(ID)["rel"].min()
        a["p_pregap_maxdist"] = jg.groupby(ID)["d"].max()
        a["p_gap_len"] = j[(j["gap"] == 1) & (j["gk"] == j["gmax"])].groupby(ID)["dt"].first()
        a["p_seg_start"] = j[~j["pregap"]].groupby(ID)["rel"].max()
        a["p_pregap_n"] = a["p_pregap_n"].fillna(0.0)
        res.append(a)
    f = pd.concat(res)
    f.index.name = ID
    out = dep.set_index(ID)[["p_inb_rel", "p_tow"]].join(f, how="left")
    out["p_n"] = out["p_n"].fillna(0.0)
    return out.reset_index()[[ID] + CORE]


def own_points(which: str, ids: np.ndarray, raw: Path | None = None) -> pd.DataFrame:
    """Columns CORE for `ids` (cached under runs/stack_resid/v3_features/<which>.parquet; the cache is keyed on the row set)."""
    cache = FEAT / f"{which}.parquet"
    if cache.exists():
        f = pd.read_parquet(cache)
    else:
        if which == "dev":
            files = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
            mv = _load([Path(files[i]) for i in (0, 5, 6, 9, 10, 11)])
            mv = mv[mv["day"].str[:6].isin(["202501", "202507", "202511", "202512"]) | mv["day"].isin(["20250629", "20250630", "20251030", "20251031"])]
        else:
            mv = _load([raw if raw is not None else ROOT / S.RANKING])
        mv = _identify(mv)
        f = _points(mv, set(ids))
        FEAT.mkdir(parents=True, exist_ok=True)
        f.to_parquet(cache, index=False)
        write_json(cache.with_suffix(".receipt.json"), {"written_at_utc": utc_now(), "sha256": sha256_file(cache), "rows": int(len(f)),
                                                        "code_sha256": sha256_file(Path(__file__)), "keymap_sha256": sha256_file(KEYMAP)})
    f = f.set_index(ID).reindex(ids)
    return f[CORE].astype(np.float32).reset_index(drop=True)


def new_columns(core: pd.DataFrame, new: np.ndarray) -> pd.DataFrame:
    P = core.copy()
    for q, c in REL:
        P[q] = (P[c].to_numpy(dtype=np.float64) - new).astype(np.float32)
    return P[NEW_COLS]


def guard_new(Xf: pd.DataFrame, af: np.ndarray, Xe: pd.DataFrame, ae: np.ndarray) -> np.ndarray:
    held = np.zeros(len(Xe), bool)
    for a in np.unique(ae):
        fa, ea = af == a, ae == a
        for c in Xe.columns:
            v = Xf.loc[fa, c].dropna()
            if len(v) < 50:
                continue
            lo, hi = np.percentile(v, [0.1, 99.9])
            x = Xe.loc[ea, c].to_numpy()
            held[np.flatnonzero(ea)] |= (x < lo) | (x > hi)
    return held


# ---------------------------------------------------------------- development

def dev_inputs():
    dev = S.dev_table()
    X46, res = S.dev_matrix(dev)
    core = own_points("dev", dev[ID].to_numpy())
    Pn = new_columns(core, dev["new"].to_numpy(dtype=float))
    X65 = pd.concat([X46.reset_index(drop=True), Pn], axis=1)
    return dev, X46.reset_index(drop=True), X65, Pn, res


def dev_predictions(dev, X46, X65, Pn, res) -> dict:
    """Month-blocked guarded bag predictions of v2 (46) and v3 (65) on the development rows."""
    fx = S.fixed_rows(dev)
    month, air, dom, new = dev["month"].to_numpy(), dev["airport"].to_numpy(), dev["dom"].to_numpy(), dev["new"].to_numpy(dtype=float)
    out = {k: np.zeros(len(dev)) for k in ["p2", "p3"]}
    held = {k: np.zeros(len(dev), bool) for k in ["h2", "h3"]}
    trees = {}
    for m in S.MONTHS:
        fit, ev = (month != m) & ~fx, month == m
        h2 = S.outside_support(X46[fit], air[fit], X46[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
        hn = guard_new(Pn[fit].reset_index(drop=True), air[fit], Pn[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
        for key, X, h in [("2", X46, h2), ("3", X65, h2 | hn)]:
            models = S.fit_bag(X.to_numpy(np.float32), res, fit, dom)
            p = S.predict_bag(models, X.to_numpy(np.float32)[ev])
            p[fx[ev]] = 0.0
            p[h] = 0.0
            out["p" + key][ev] = p
            held["h" + key][ev] = h
            trees[f"v{key} {m}"] = [int(b.best_iteration) for b in models]
            print(m, "v" + key, "held", int(h.sum()), "trees", trees[f"v{key} {m}"], flush=True)
    base2 = S.apply(new, out["p2"], fx)
    base3 = S.apply(new, out["p3"], fx)
    return {"fx": fx, "p2": out["p2"], "p3": out["p3"], "h2": held["h2"], "h3": held["h3"], "base2": base2, "base3": base3, "trees": trees}


class Chain:
    """intruder bound -> identity extension -> own-motion bound, month-blocked on the development months (as runs/dig/census61/build.py
    and tools/run_own_motion.score)."""

    def __init__(self):
        files = I.training_files()
        self.rdep, self.rarr = R.movements(files)
        self.dep, self.arr = I.movements(files)
        twelve, p12 = R.read_table("twelve", self.rdep)
        self.acc, self.icfg = read_json(R.CONFIG)["candidate"], read_json(I.CONFIG)
        self.c12acc = R.candidates(twelve, p12, self.rdep, self.rarr, self.acc["G_s"])
        months12 = sorted(twelve["month"].unique())
        after12 = pd.concat([I.accepted_after(twelve, p12, self.rdep, self.rarr, self.c12acc, m, self.acc)[0] for m in months12])
        self.cid12 = I.candidates(after12, "after", self.dep, self.arr, self.icfg["candidate"]["G_s"])
        self.ocfg = read_json(OM.CONFIG)
        self.trig = OM.triggers(I.training_files())
        self.c12om = OM.candidates(OM.twelve_after(), "after2", self.trig, self.ocfg["candidate"]["B_max_s"])

    def run(self, d: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
        """d: ID, month, airport, label, cls, col (the chain after the stacking layer). Returns (after identity, after own-motion)."""
        tab = d[[ID, "month", "airport", "label", "cls", col]].copy()
        s3 = pd.Series(d[col].to_numpy(dtype=float), index=d[ID].to_numpy())
        s4 = s3.copy()
        for m in R.DEV_MONTHS:
            model = R.fit(self.c12acc[self.c12acc["month"] != m])
            s = tab[tab["month"] == m]
            c = R.values(R.candidates(s, col, self.rdep, self.rarr, self.acc["G_s"]), col, model, self.acc["w"])
            s3.loc[c[ID].to_numpy()] = c["value"].to_numpy()
            t3 = s.assign(after=s3.loc[s[ID].to_numpy()].to_numpy())
            imodel = I.fit(self.cid12[self.cid12["month"] != m], model["airports"])
            ci = I.values(I.candidates(t3, "after", self.dep, self.arr, self.icfg["candidate"]["G_s"]), "after", imodel)
            s4.loc[t3[ID].to_numpy()] = t3["after"].to_numpy()
            s4.loc[ci[ID].to_numpy()] = ci["value"].to_numpy()
        v61 = s4.loc[d[ID].to_numpy()].to_numpy()
        spec = self.ocfg["candidate"]
        t = d[[ID, "month", "airport", "label", "cls"]].assign(v61=v61)
        cdv = OM.candidates(t, "v61", self.trig, spec["B_max_s"])
        s5 = pd.Series(v61.copy(), index=d[ID].to_numpy())
        for m in R.DEV_MONTHS:
            mod = OM.fit(self.c12om[self.c12om["month"] != m])
            c = OM.values(cdv[cdv["month"] == m], "v61", mod, spec["w"])
            s5.loc[c[ID].to_numpy()] = c["value"].to_numpy()
        return v61, s5.loc[d[ID].to_numpy()].to_numpy()


def score() -> None:
    decl = read_json(ROOT / DECL)
    if sha256_file(ROOT / DEV61[0]) != DEV61[1]:
        raise ValueError("The v61 development proxy differs from the declared file.")
    dev, X46, X65, Pn, res = dev_inputs()
    pr = dev_predictions(dev, X46, X65, Pn, res)
    fx = pr["fx"]
    y, new, month, air = dev["label"].to_numpy(dtype=float), dev["new"].to_numpy(dtype=float), dev["month"].to_numpy(), dev["airport"].to_numpy()
    checks = {}
    # v2 reproduction
    r2 = read_json(ROOT / "docs/stack_resid/results_v2.json")["months"]
    got2 = {m: float(np.sqrt(np.mean((y[month == m] - pr["base2"][month == m]) ** 2))) for m in S.MONTHS}
    checks["v2_month_rmse_vs_results_v2_max_abs_s"] = max(abs(got2[m] - r2[m]["rmse"]) for m in S.MONTHS)
    if checks["v2_month_rmse_vs_results_v2_max_abs_s"] > 1e-6:
        raise ValueError(f"v2 does not reproduce results_v2.json: {got2}")
    d61 = pd.read_parquet(ROOT / DEV61[0]).set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(d61["label"].to_numpy(), y):
        raise ValueError("Row alignment with the v61 proxy failed.")
    s1 = d61["s1"].to_numpy(dtype=float)
    d = d61[[ID, "month", "airport", "label", "cls"]].copy()
    d["s2_v2"] = s1 + (pr["base2"] - new)
    checks["s2_v2_vs_census61_max_abs_s"] = float(np.abs(d["s2_v2"].to_numpy() - d61["s2"].to_numpy()).max())
    lirf = air == "LIRF"
    base3_nolirf = np.where(lirf, pr["base2"], pr["base3"])
    d["s2_v3"] = s1 + (base3_nolirf - new)
    d["s2_v3_lirf"] = s1 + (pr["base3"] - new)
    ch = Chain()
    v61_ctl, v62_ctl = ch.run(d, "s2_v2")
    checks["v61_proxy_rebuilt_max_abs_s"] = float(np.abs(v61_ctl - d61["v61"].to_numpy()).max())
    if checks["v61_proxy_rebuilt_max_abs_s"] > 1e-6 or checks["s2_v2_vs_census61_max_abs_s"] > 1e-6:
        raise ValueError(f"The control chain does not reproduce the v61 proxy: {checks}")
    om = read_json(ROOT / "docs/own_motion/results_v1.json")
    want = {m: om["control_month_rmse_s"][m] * (1 + om["candidates"]["om_b2700_w100"]["development"][m]["pct"] / 100) for m in S.MONTHS}
    got = {m: float(np.sqrt(np.mean((y[month == m] - v62_ctl[month == m]) ** 2))) for m in S.MONTHS}
    checks["v62_proxy_month_rmse_s"] = got
    checks["v62_proxy_vs_own_motion_results_max_rel"] = max(abs(got[m] / want[m] - 1) for m in S.MONTHS)
    if checks["v62_proxy_vs_own_motion_results_max_rel"] > 1e-9:
        raise ValueError(f"The v62 proxy does not reproduce docs/own_motion/results_v1.json: {got} vs {want}")
    print("checks", checks, flush=True)
    days = dev["takeoff_day"].to_numpy()
    out = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL),
           "code": {c: sha256_file(ROOT / c) for c in CODE}, "features": {k: read_json(FEAT / f"{k}.receipt.json") for k in ["dev"]},
           "control": "v62 development proxy (v61 proxy rebuilt + own-motion bound, month-blocked)", "checks": checks, "trees": pr["trees"], "candidates": {}}
    for name, col in [(decl["candidate"]["name"], "s2_v3"), ("pregap_replace_w50_withlirf", "s2_v3_lirf")]:
        v61c, v62c = ch.run(d, col)
        res_m = {}
        for m in S.MONTHS:
            ev = month == m
            c0, c1 = float(np.sqrt(np.mean((y[ev] - v62_ctl[ev]) ** 2))), float(np.sqrt(np.mean((y[ev] - v62c[ev]) ** 2)))
            mv = v62c[ev] - v62_ctl[ev]
            res_m[m] = {"control_rmse": c0, "rmse": c1, "change_pct": 100 * (c1 / c0 - 1), **S.paired(days[ev], y[ev], v62c[ev], v62_ctl[ev]),
                        "rows_moved": int((np.abs(mv) > 1e-9).sum()), "rms_move_s": float(np.sqrt(np.mean(mv ** 2))), "max_abs_move_s": float(np.abs(mv).max()),
                        "rows_held_by_guard": int(pr["h3"][ev].sum()), "lirf_rows_moved": int((np.abs(mv[lirf[ev]]) > 1e-9).sum())}
        eq = float(np.mean([res_m[m]["change_pct"] for m in S.EQUAL]))
        by = pd.DataFrame({"airport": air, "month": month, "c": (v62_ctl - y) ** 2, "n": (v62c - y) ** 2}).groupby(["airport", "month"]).sum()
        out["candidates"][name] = {"months": res_m, "equal_footing_mean_pct": eq,
                                   "airport_month_change_pct": {f"{a} {m}": float(100 * (np.sqrt(r["n"]) / np.sqrt(r["c"]) - 1)) for (a, m), r in by.iterrows()},
                                   "fixed_rows_moved": int((np.abs(v62c - v62_ctl)[fx] > 1e-9).sum())}
        print(name, " ".join(f"{m[5:]} {res_m[m]['change_pct']:+.4f}% [{res_m[m]['interval_s'][0]:+.3f},{res_m[m]['interval_s'][1]:+.3f}]s" for m in S.MONTHS),
              f"| eq {eq:+.4f}%", flush=True)
    # rule (4): ranking side, bag on all four months
    applies = ~lirf & ~fx
    dev_rms = S.by_airport_rms(dev, np.where(applies, pr["p3"], 0.0), ~applies)
    rk, pr_rk, held_rk, X65r, Pr = ranking_prediction(dev, X46, X65, Pn, res, fx)
    fxr = S.fixed_rows(rk)
    applies_r = ~rk["airport"].eq("LIRF").to_numpy() & ~fxr
    rk_rms = S.by_airport_rms(rk, np.where(applies_r, pr_rk, 0.0), ~applies_r)
    ratios = {a: rk_rms[a]["rms_s"] / dev_rms[a]["rms_s"] for a in rk_rms}
    out["dev_oof_prediction_by_airport"] = dev_rms
    out["ranking_prediction_by_airport"] = rk_rms
    out["ranking_over_dev_rms"] = ratios
    out["ranking_rows_held_by_guard"] = int((held_rk & applies_r).sum())
    out["coverage_shift"] = coverage_shift(dev, Pn, pr["h3"], rk, Pr, held_rk, fx, fxr)
    c = out["candidates"][decl["candidate"]["name"]]
    rule = {"1_no_month_worse": all(c["months"][m]["change_pct"] <= 0 for m in S.MONTHS),
            "2_equal_footing_mean_at_most_-0.05pct": c["equal_footing_mean_pct"] <= -0.05,
            "3_equal_footing_intervals_below_zero": all(c["months"][m]["interval_s"][1] < 0 for m in S.EQUAL),
            "4_transfer_guard_0.5x_2x": all(0.5 <= v <= 2.0 for v in ratios.values())}
    out["rule"] = rule
    out["accepted"] = all(rule.values())
    out["selected"] = decl["candidate"]["name"] if out["accepted"] else None
    out["twelve_month_check"] = "none exists (as stack_resid_v2): the component tables cover only the four development months"
    out["official_scores_used"] = "none"
    write_json(ROOT / "docs/stack_resid/results_v3.json", out)
    append_readme(out, decl)
    print("rule", rule, "accepted", out["accepted"], "ratios", {a: round(v, 2) for a, v in ratios.items()})


def coverage_shift(dev, Pn, h3d, rk, Pr, h3r, fx, fxr) -> dict:
    res = {}
    for a in sorted(set(dev["airport"])):
        ad, ar = (dev["airport"].eq(a).to_numpy() & ~fx), (rk["airport"].eq(a).to_numpy() & ~fxr)
        e = {"guard_share_dev_oof": float(h3d[ad].mean()), "guard_share_ranking": float(h3r[ar].mean())}
        for mo in sorted(set(rk["month"][ar])):
            mm = ar & rk["month"].eq(mo).to_numpy()
            e[f"guard_share_ranking_{mo}"] = float(h3r[mm].mean())
        for c in ["p_inb_rel", "p_first_mv", "q_mv_vs_pred", "q_park_vs_pred", "p_pregap_n"]:
            dv, rv = Pn.loc[ad, c].dropna(), Pr.loc[ar, c].dropna()
            e[c] = {"dev_share": float(Pn.loc[ad, c].notna().mean()), "rank_share": float(Pr.loc[ar, c].notna().mean()),
                    "dev_q10_50_90": [float(v) for v in dv.quantile([0.1, 0.5, 0.9])] if len(dv) else None,
                    "rank_q10_50_90": [float(v) for v in rv.quantile([0.1, 0.5, 0.9])] if len(rv) else None}
        res[a] = e
    return res


def append_readme(out: dict, decl: dict) -> None:
    path = S.OUT_DOCS / "README.md"
    text = path.read_text(encoding="utf-8")
    marker = "## v3: own-aircraft turnaround / pre-gap features (configs/stack_resid_v3.json)"
    if marker in text:
        raise ValueError("The README already has this section; preserve it.")
    c = out["candidates"][decl["candidate"]["name"]]
    lines = ["", marker, "", f"Declaration [`configs/stack_resid_v3.json`](../../configs/stack_resid_v3.json); results [`results_v3.json`](results_v3.json); "
             f"exploration [`docs/dig/pregap.md`](../dig/pregap.md). **Accepted by the declared rule: {out['accepted']}.** In-chain against the v62 "
             "development proxy (the stacking layer replaced, then the intruder bound, identity extension and own-motion bound re-evaluated month-blocked); "
             "LIRF keeps v2's value; the control chain reproduces the v61 proxy and the own-motion results exactly.", "",
             "| Month | Control (v62 proxy) | Candidate | Change | Paired-day 95% (s) | Days better | Rows moved | Held by guard | RMS move (s) | Max move (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in S.MONTHS:
        r = c["months"][m]
        lines.append(f"| {m} | {r['control_rmse']:.4f} | {r['rmse']:.4f} | {r['change_pct']:+.4f}% | [{r['interval_s'][0]:+.3f}, {r['interval_s'][1]:+.3f}] | "
                     f"{r['days_better']}/{r['days']} | {r['rows_moved']:,} | {r['rows_held_by_guard']:,} | {r['rms_move_s']:.1f} | {r['max_abs_move_s']:.0f} |")
    s = out["candidates"]["pregap_replace_w50_withlirf"]
    lines += ["", f"Equal-footing mean (January, July, December): **{c['equal_footing_mean_pct']:+.4f}%**.", "",
              "Rule: " + "; ".join(f"{k} = {v}" for k, v in out["rule"].items()) + ".", "",
              "Transfer guard after both support guards (ranking prediction RMS over development out-of-month RMS; the nine airports where the learner "
              "applies): " + ", ".join(f"{a} {v:.2f}" for a, v in sorted(out["ranking_over_dev_rms"].items())) + f"; {out['ranking_rows_held_by_guard']:,} ranking rows held.", "",
              "Sensitivity, reported and not selectable (the same learner applied at LIRF too): " +
              " / ".join(f"{s['months'][m]['change_pct']:+.4f}%" for m in S.MONTHS) + f", equal-footing {s['equal_footing_mean_pct']:+.4f}%.", "",
              "No twelve-month check exists (as for v2). No official score is used."]
    path.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- ranking

def ranking_prediction(dev, X46, X65, Pn, res, fx):
    """The v3 bag on all four development months applied to the ranking rows of v2's ranking table (base c67f20d6), guarded."""
    spec2 = S.spec_of(V2_DECL)
    checks: dict = {}
    rk = S.spec_ranking(spec2, spec2["base"], checks)
    X46r = S.features(rk).reset_index(drop=True)
    core = own_points("rank", rk[ID].to_numpy())
    Pr = new_columns(core, rk["new"].to_numpy(dtype=float))
    X65r = pd.concat([X46r, Pr], axis=1)
    air = dev["airport"].to_numpy()
    models = S.fit_bag(X65.to_numpy(np.float32), res, ~fx, dev["dom"].to_numpy())
    p = S.predict_bag(models, X65r.to_numpy(np.float32))
    fxr = S.fixed_rows(rk)
    p[fxr] = 0.0
    held = (S.outside_support(X46[~fx], air[~fx], X46r, rk["airport"].to_numpy())
            | guard_new(Pn[~fx].reset_index(drop=True), air[~fx], Pr, rk["airport"].to_numpy())) & ~fxr
    p[held] = 0.0
    rk.attrs["checks"], rk.attrs["trees"] = checks, [int(m.best_iteration) for m in models]
    return rk, p, held, X65r, Pr


def chain_from_stack(stack: Path, version: int, out: Path) -> Path:
    """stack file -> intruder bound + identity extension (tools/run_intruder_identity.dry_run, no v60 check) -> own-motion bound."""
    I.V60 = None
    I.dry_run(stack, sha256_file(stack), version, out / "identity")
    ident = out / "identity" / f"vigorous-whistle_v{version}.parquet"
    OM.dry_run(ident, sha256_file(ident), version, out / "own")
    return out / "own" / f"vigorous-whistle_v{version}.parquet"


def dry_run(version: int, out_dir: Path) -> None:
    decl = read_json(ROOT / DECL)
    res_v3 = read_json(ROOT / "docs/stack_resid/results_v3.json")
    if not res_v3.get("accepted"):
        raise ValueError("stack_resid_v3 is not accepted by its declared rule; no ranking file.")
    final = out_dir / f"vigorous-whistle_v{version}.parquet"
    if final.exists():
        raise ValueError(f"{final} exists; never overwrite.")
    for f, h in [V60_STACK, V62]:
        if sha256_file(ROOT / f) != h:
            raise ValueError(f"{f} does not have the expected SHA-256.")
    steps = out_dir / "steps"
    # control: the same chain from v2's stack file must give the uploaded v62
    ctl = chain_from_stack(ROOT / V60_STACK[0], version, steps / "control_from_v60_stack")
    ctl_ok = sha256_file(ctl) == V62[1]
    if not ctl_ok:
        raise ValueError("The chain from v2's stack file does not reproduce v62.")
    dev, X46, X65, Pn, res = dev_inputs()
    fx = S.fixed_rows(dev)
    rk, p, held, X65r, Pr = ranking_prediction(dev, X46, X65, Pn, res, fx)
    fxr = S.fixed_rows(rk)
    lirf = rk["airport"].eq("LIRF").to_numpy()
    b = rk["new"].to_numpy(dtype=float)
    v60s = pd.read_parquet(ROOT / V60_STACK[0]).set_index(ID)[TARGET].reindex(rk[ID]).to_numpy(dtype=float)
    values = np.where(lirf, v60s, S.apply(b, np.where(lirf, 0.0, p), fxr))
    (steps / "stack").mkdir(parents=True, exist_ok=True)
    stack = steps / "stack" / f"vigorous-whistle_v{version}.parquet"
    provenance = {"experiment_id": decl["experiment_id"], "dry_run": True, "base": {"path": S.spec_of(V2_DECL)["base"][0], "sha256": S.spec_of(V2_DECL)["base"][1]},
                  "declaration": {DECL: sha256_file(ROOT / DECL)}, "code": {c: sha256_file(ROOT / c) for c in CODE}, "official_scores_used": "none",
                  "audit_used": False, "recipe": "stack_resid_v2's learner refitted with 19 own-aircraft turnaround / pre-gap features, guarded; LIRF keeps v2 (v60 stack file)"}
    export_submission(pd.DataFrame({ID: rk[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", stack, provenance)
    sv = pd.read_parquet(stack).set_index(ID)[TARGET].reindex(rk[ID]).to_numpy()
    if not np.array_equal(sv[lirf], v60s[lirf]) or not np.array_equal(sv[fxr | (held & ~lirf)], np.rint(b[fxr | (held & ~lirf)])):
        raise ValueError("A LIRF, fixed or guarded row does not keep its value in the stack step.")
    last = chain_from_stack(stack, version, steps)
    shutil.copyfile(last, final)
    report(version, out_dir, final, rk, p, held, fxr, Pr, ctl_ok)


def report(version, out_dir, final, rk, p, held, fxr, Pr, ctl_ok) -> None:
    v = pd.read_parquet(final)
    v62 = pd.read_parquet(ROOT / V62[0])
    if not v["MVT_ID_mvt"].equals(v62["MVT_ID_mvt"]):
        raise ValueError("Row order differs from v62.")
    raw = pd.read_parquet(ROOT / S.RANKING, columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v[ID])
    pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    lirf = raw["ADEP_mvt"].eq("LIRF").to_numpy()
    guard = v[ID].isin(R.GUARD).to_numpy()
    diff = v[TARGET].to_numpy(dtype=float) - v62[TARGET].to_numpy(dtype=float)
    ch = diff != 0
    by = pd.DataFrame({"airport": raw["ADEP_mvt"].to_numpy(), "month": raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy(), "d": diff})
    rep = {"created_at_utc": utc_now(), "version": version, "file": final.relative_to(ROOT).as_posix(), "sha256": sha256_file(final), "bytes": final.stat().st_size,
           "control_chain_from_v60_stack_reproduces_v62": ctl_ok, "vs_v62": {"sha256": V62[1], "rows_changed": int(ch.sum()), "squared_change": float((diff ** 2).sum()),
           "rms_change_s": float(np.sqrt((diff ** 2).mean())), "max_up_s": float(diff.max()), "max_down_s": float(diff.min()), "mean_change_s": float(diff.mean())},
           "lirf_rows_changed": int((ch & lirf).sum()), "lirf_population_rows_changed": int((ch & pop).sum()), "guard_rows_changed": int((ch & guard).sum()),
           "by_airport_month": {f"{a} {m}": {"rows_changed": int((g["d"] != 0).sum()), "squared_change": float((g["d"] ** 2).sum()), "max_abs_s": float(g["d"].abs().max())}
                                for (a, m), g in by.groupby(["airport", "month"])},
           "ranking_rows_held_by_guard": int((held & ~fxr).sum()), "trees": rk.attrs.get("trees"), "ranking_table_checks": rk.attrs.get("checks"),
           "steps": {k: sha256_file(out_dir / "steps" / k / f"vigorous-whistle_v{version}.parquet") for k in ["stack", "identity", "own"]}}
    if rep["lirf_rows_changed"] or rep["lirf_population_rows_changed"] or rep["guard_rows_changed"]:
        raise ValueError("A LIRF, population or guard row changed against v62.")
    write_json(out_dir / f"vigorous-whistle_v{version}.report.json", rep)
    print({k: rep[k] for k in ["sha256", "bytes", "vs_v62", "lirf_rows_changed", "lirf_population_rows_changed", "guard_rows_changed", "ranking_rows_held_by_guard"]})


def build(version: int, decision: str) -> None:
    dry = ROOT / f"runs/stack_resid/dry_v{version}/vigorous-whistle_v{version}.parquet"
    target = ROOT / f"runs/submissions/vigorous-whistle_v{version}.parquet"
    if target.exists():
        raise ValueError(f"{target.name} exists; never overwrite a submission file.")
    if not dry.exists() or not decision:
        raise ValueError("Run the dry run first and record the user's decision.")
    tmp = ROOT / f"runs/stack_resid/build_v{version}"
    if tmp.exists() and any(tmp.iterdir()):
        raise ValueError(f"{tmp} is not empty; never overwrite a build directory.")
    dry_run(version, tmp)
    built = tmp / f"vigorous-whistle_v{version}.parquet"
    if sha256_file(built) != sha256_file(dry):
        raise ValueError("The rebuilt file differs from the dry run.")
    target.write_bytes(built.read_bytes())
    from prc2026_submission.contract import validate_submission
    sub = validate_submission(target, ROOT / "data/raw/submitting.parquet")
    decl = read_json(ROOT / DECL)
    sub.update({"version": version, "rounding": "numpy.rint, nearest integer with ties to even; no clipping",
                "provenance": {"audit_used": False, "dry_run": False, "experiment_id": decl["experiment_id"], "selected_candidate": decl["candidate"]["name"],
                               "selected_by": "the declared rule (docs/stack_resid/results_v3.json)", "official_scores_used": "none", "user_decision": decision,
                               "declaration": {DECL: sha256_file(ROOT / DECL)}, "code": {c: sha256_file(ROOT / c) for c in CODE},
                               "base": {"path": S.spec_of(V2_DECL)["base"][0], "sha256": S.spec_of(V2_DECL)["base"][1]},
                               "recipe": "v62's chain with stack_resid_v2 replaced by stack_resid_v3 (LIRF keeps v2): rematch chain file -> stack v3 -> intruder bound -> identity extension -> own-motion bound"}})
    write_json(target.with_suffix(".receipt.json"), sub)
    rep = read_json(tmp / f"vigorous-whistle_v{version}.report.json")
    rep.update({"kind": "submission build at the user's word", "file": str(target.relative_to(ROOT).as_posix()), "sha256": sha256_file(target), "dry_run_sha256": sha256_file(dry)})
    write_json(ROOT / f"docs/stack_resid/export_v{version}.json", rep)
    print("built", target.name, rep["sha256"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build", action="store_true", help="at the user's word: write runs/submissions/vigorous-whistle_v<version>.parquet")
    ap.add_argument("--version", type=int, default=63)
    ap.add_argument("--user-decision", default=None)
    a = ap.parse_args()
    if a.score:
        score()
    if a.dry_run:
        dry_run(a.version, ROOT / f"runs/stack_resid/dry_v{a.version}")
    if a.build:
        build(a.version, a.user_decision)


if __name__ == "__main__":
    main()
