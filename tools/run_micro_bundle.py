"""Micro bundle (configs/micro_bundle_v1.json): three small physical pieces inside v60's chain.

  C  day-boundary pair repair: a single row whose predicted off-block falls on the wrong side of midnight, detected label-free from two
     consecutive airport-days with opposite +-1 count differences and opposite published discrepancies its implied delay explains;
     its day moves in the extreme-day layer's and the variance re-allocation's day aggregations, and after the re-allocation its value
     is bounded to put its off-block just across that midnight
  B  count-aware extreme days: a day the extreme-day layer skips only for its count-sign rule qualifies with the published-figure noise
     model's count term (mu per missing / extra flight), outside month edges and +-1 neighbour patterns
  A  MARS stand families in the intruder stand bound (a sibling stand's different-type arrival), incremental over the accepted rule

  --score     development (four control tables of runs/scout/refit, rebuilt and checked against ctl_v55m) and the twelve-month table;
              the declared rule -> docs/micro_bundle/results_v1.json
  --chain     the ranking chain (model part -> v48 -> C -> extreme-day + B -> var-alloc -> C bound) into runs/micro_bundle/<out>/
              (--no-pieces: the same chain without C and B, which must reproduce runs/rematch_nonm/rank/Y c67f20d6...)
  --finish    A on top of the intruder step's dry run -> runs/micro_bundle/dry_v<version>/
Nothing is written under runs/submissions and nothing is uploaded. No official score is used.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_scout import extreme_shift, published_figure, var_alloc  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
import scout_extreme_shift as se  # noqa: E402
import scout_var_alloc as sva  # noqa: E402
import run_intruder_bound as R  # noqa: E402

DECL = "configs/micro_bundle_v1.json"
KEYS = ["month", "airport", "bday"]
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
REF = ROOT / "runs/scout/refit"
OUT = ROOT / "runs/micro_bundle"
GUARD = (213142471.0, 213142469.0)
C_MIN_S, C_TOL, C_WINDOW_S, C_STEP_S = 7200.0, 0.25, 3 * 3600.0, 60.0
SEED = 20261003
THREADS = 4            # the session's limit; the shipped chain used 6 (difference to ctl_v55m: float noise, below REPRO_TOL_S)
REPRO_TOL_S = 1e-6
var_alloc.PARAMS["num_threads"] = THREADS
TRAIN_FILES = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))


# ---------------------------------------------------------------- shared inputs

_DEP_CACHE: dict = {}


def departures_cached(paths: list[str]) -> pd.DataFrame:
    key = tuple(paths)
    if key not in _DEP_CACHE:
        _DEP_CACHE[key] = sva.departures(list(paths))
    return _DEP_CACHE[key]


def takeoffs(paths: list[str]) -> pd.Series:
    raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"]) for p in paths], ignore_index=True)
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
    return pd.Series(pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).to_numpy(), index=raw[ID].to_numpy())


def noise_mu(t: pd.DataFrame, fit_mask: np.ndarray) -> dict:
    f = t[fit_mask & t["in_figure"].to_numpy(dtype=bool) & ~t["extra"].to_numpy(dtype=bool)]
    fd = published_figure.fitting_days(f.assign(pos_true=np.maximum(f["a"] - f["label"], 0.0)))
    model = published_figure.fit(fd)
    return {a: {s: m["sign"][s]["mu"] for s in (+1, -1)} for a, m in model.items()}


# ---------------------------------------------------------------- C: day-boundary pairs

def detect_pairs(t: pd.DataFrame, days: pd.DataFrame, c0: dict, tk: pd.Series) -> pd.DataFrame:
    """days: one day frame indexed (airport, bday) for the rows t (same table, same keys after dropping the month)."""
    d = days[days["pub"].notna()].copy()
    d["D"] = d["pub"] - d.index.get_level_values("airport").map(c0).to_numpy(dtype=float) - d["implied"]
    found = []
    for (airport, bday), r in d.iterrows():
        nxt = (pd.Timestamp(bday) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if (airport, nxt) not in d.index:
            continue
        s = d.loc[(airport, nxt)]
        if not (abs(r["dn"]) == 1 and s["dn"] == -r["dn"]):
            continue
        if not (r["D"] * s["D"] < 0 and min(abs(r["D"]), abs(s["D"])) >= C_MIN_S):
            continue
        neg_day, pos_day, d_neg, d_pos, dn_neg = (bday, nxt, r["D"], s["D"], r["dn"]) if r["D"] < 0 else (nxt, bday, s["D"], r["D"], s["dn"])
        if dn_neg != -1:
            continue
        midnight = pd.Timestamp(nxt, tz="UTC")
        g = t[t["airport"].eq(airport) & t["bday"].eq(neg_day) & t["in_figure"].astype(bool) & ~t["cls"].eq("LIRF_pop") & ~t[ID].isin(list(GUARD))]
        if not len(g):
            continue
        take = pd.to_datetime(tk.reindex(g[ID]).to_numpy(), utc=True)
        off = take - pd.to_timedelta(g["stack"].to_numpy(dtype=float), unit="s")
        p = np.maximum(g["a"].to_numpy(dtype=float) - g["stack"].to_numpy(dtype=float), 0.0)
        near = np.abs((off - midnight).total_seconds().to_numpy()) <= C_WINDOW_S
        ok = near & (p >= C_MIN_S)
        if not ok.any():
            continue
        i = np.flatnonzero(ok)[np.argmin(np.abs(d_neg + p[ok]))]
        if abs(d_neg + p[i]) <= C_TOL * p[i] and abs(d_pos - p[i]) <= C_TOL * p[i]:
            found.append({ID: float(g[ID].iloc[i]), "airport": airport, "from_day": neg_day, "to_day": pos_day, "midnight": midnight,
                          "takeoff": take[i], "p": float(p[i]), "D_neg": float(d_neg), "D_pos": float(d_pos)})
    return pd.DataFrame(found, columns=[ID, "airport", "from_day", "to_day", "midnight", "takeoff", "p", "D_neg", "D_pos"])


def pair_bound(values: np.ndarray, ids: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    out = values.copy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    for _, r in pairs.iterrows():
        i = int(pos.loc[r[ID]])
        edge = (r["takeoff"] - r["midnight"]).total_seconds()
        out[i] = max(out[i], edge + C_STEP_S) if r["to_day"] < r["from_day"] else min(out[i], edge - C_STEP_S)
    return out


# ---------------------------------------------------------------- B: count-aware extreme days

def shifts_b(days: pd.DataFrame, params: dict, mu: dict) -> pd.DataFrame:
    """The extreme-day layer's days plus the count-aware ones (days indexed airport, bday)."""
    base = extreme_shift.shifts(days, params)
    d = days.copy()
    a = d.index.get_level_values("airport")
    b = pd.to_datetime(d.index.get_level_values("bday"))
    dn = pd.Series(d["dn"].to_numpy(), index=pd.MultiIndex.from_arrays([a, d.index.get_level_values("bday")]))
    dn = dn[~dn.index.duplicated()]
    prev = dn.reindex(pd.MultiIndex.from_arrays([a, (b - pd.Timedelta(days=1)).strftime("%Y-%m-%d")])).to_numpy()
    nxt = dn.reindex(pd.MultiIndex.from_arrays([a, (b + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])).to_numpy()
    own = d["dn"].to_numpy()
    pattern = (np.abs(own) == 1) & ((prev == -own) | (nxt == -own))
    edge = (b.day == 1) | ((b + pd.Timedelta(days=1)).day == 1)
    sel = (~d["eligible"].to_numpy(dtype=bool)) & d["pub"].notna().to_numpy() & (d["n_eff"].to_numpy() >= extreme_shift.MIN_NEFF) \
        & np.isin(a, list(params["beta"])) & ~np.asarray(edge) & ~pattern & (own != 0)
    x = d[sel].copy()
    if not len(x):
        return base
    ax = x.index.get_level_values("airport")
    ct = np.array([mu.get(ap, {1: 0.0, -1: 0.0})[1 if n > 0 else -1] * abs(n) for ap, n in zip(ax, x["dn"])])
    x["I"] = -(x["pub"] - ax.map(params["c0"]).to_numpy(dtype=float) - ct - x["implied"]) / x["n_eff"]
    x = x[x["I"].abs() >= extreme_shift.I_MIN_S]
    x["shift_new"] = x.index.get_level_values("airport").map(params["beta"]).to_numpy(dtype=float) * x["I"]
    return pd.concat([base, x[["I", "shift_new", "dn", "rows", "n_eff"]].assign(count_aware=True)])


# ---------------------------------------------------------------- the published chain on a development table

def chain(kind: str, pieces: bool, tk: pd.Series, log: dict | None = None) -> pd.DataFrame:
    """tools/scout_var_alloc.py layered(kind) with C and B when `pieces`; returns the table with ext, new (and the pair rows)."""
    t = se.dev_rows(kind)
    days = extreme_shift.day_frame(t, KEYS)
    months = sorted(t.loc[~t["extra"], "month"].unique())
    month_arr = t["month"].to_numpy()
    params_by, pairs_all = {}, []
    if pieces:
        for m in months:
            p0 = extreme_shift.fit(days[days.index.get_level_values("month") != m].droplevel("month"))
            use = days[days.index.get_level_values("month") == m].droplevel("month")
            pr = detect_pairs(t[month_arr == m], use, p0["c0"], tk).assign(month=m)
            if len(pr):
                pairs_all.append(pr)
        pairs = pd.concat(pairs_all, ignore_index=True) if pairs_all else pd.DataFrame(columns=[ID, "month", "from_day", "to_day"])
        if len(pairs):
            moved = t[ID].map(dict(zip(pairs[ID], pairs["to_day"])))
            t["bday"] = np.where(moved.notna(), moved, t["bday"])
            days = extreme_shift.day_frame(t, KEYS)
    else:
        pairs = pd.DataFrame(columns=[ID, "month", "from_day", "to_day"])
    ext, I = t["v48"].to_numpy(dtype=float).copy(), np.zeros(len(t))
    count_days = []
    for m in months:
        params = extreme_shift.fit(days[days.index.get_level_values("month") != m].droplevel("month"))
        use = days[days.index.get_level_values("month") == m]
        if pieces:
            mu = noise_mu(t, month_arr != m)
            ch = shifts_b(use.droplevel("month"), params, mu)
            cd = ch[ch.get("count_aware", pd.Series(False, index=ch.index)).fillna(False).astype(bool)] if "count_aware" in ch else ch.iloc[0:0]
            count_days += [{"month": m, "airport": a, "bday": b, **{k: float(v) for k, v in r[["I", "shift_new", "dn", "rows", "n_eff"]].items()}} for (a, b), r in cd.iterrows()]
            ch = ch[["I", "shift_new", "dn", "rows", "n_eff"]]
        else:
            ch = extreme_shift.shifts(use.droplevel("month"), params)
        ch.index = pd.MultiIndex.from_tuples([(m, a, b) for a, b in ch.index], names=KEYS)
        idx = np.flatnonzero(month_arr == m)
        ext[idx] = extreme_shift.apply(t.iloc[idx], ch, KEYS)[0]
        I[idx] = sva.implied(t.iloc[idx], use.droplevel("month"), params["c0"], ["airport", "bday"])
    t["ext"], t["I"] = ext, I
    t["r"] = t["label"] - t["ext"]
    elig = var_alloc.eligible(t)
    f = var_alloc.features(t, departures_cached(TRAIN_FILES), KEYS)
    a, label, r = t["a"].to_numpy(dtype=float), t["label"].to_numpy(dtype=float), t["r"].to_numpy(dtype=float)
    extra = t["extra"].to_numpy(dtype=bool)
    w = np.full(len(t), np.nan)
    for m in months:
        models = var_alloc.fit_models(f, a, label, r, elig & (month_arr != m))
        ev = month_arr == m
        w[ev] = var_alloc.weights(models, f.loc[ev], elig[ev])
    base = var_alloc.basis(t, w, I, KEYS)
    new = ext.copy()
    for m in months:
        k = var_alloc.slope(r, base, elig & (month_arr != m) & ~extra)
        ev = month_arr == m
        new[ev] = var_alloc.update(ext[ev], k, base[ev], elig[ev])
    if pieces and len(pairs):
        new = pair_bound(new, t[ID].to_numpy(), pairs)
    t["new"], t["w"], t["base"] = new, w, base
    if log is not None:
        log["pairs"] = [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in r.items()} for r in pairs.to_dict(orient="records")]
        log["count_aware_days"] = count_days
    return t


# ---------------------------------------------------------------- A: MARS siblings (tunnel 5) and the intruder step

def mars_inputs(paths: list[str]):
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
    raw = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    raw["airport"] = np.where(raw["PHASE_mvt"].eq("DEP"), raw["ADEP_mvt"], raw["ADES_mvt"])
    dep = raw[raw["PHASE_mvt"].eq("DEP")][[ID, "MVT_TIME_UTC_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]].dropna(subset=["STAND_mvt"]).drop_duplicates(ID)
    dep["MVT_TIME_UTC_mvt"] = dep["MVT_TIME_UTC_mvt"].astype("datetime64[us, UTC]")
    arr = raw[raw["PHASE_mvt"].eq("ARR")][["airport", "STAND_mvt", "BLOCK_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt"]].dropna()
    arr = arr.rename(columns={"BLOCK_TIME_UTC_mvt": "in_blk", "AIRCRAFT_TYPE_mvt": "a_type", "STAND_mvt": "sib"})
    arr["in_blk"] = arr["in_blk"].astype("datetime64[us, UTC]")
    arr["sib"] = arr["sib"].astype(str)
    codes = set(zip(raw["airport"], raw["STAND_mvt"].astype(str)))
    return dep, arr.sort_values("in_blk"), codes


def siblings(codes: set, ap: str, s) -> list[str]:
    s = str(s)
    if s[-1:] in ("L", "R") and s[:-1] and s[-2:-1].isdigit():
        c = s[:-1]
        return [c] if (ap, c) in codes else []
    return [s + x for x in ("L", "R") if (ap, s + x) in codes]


def mars_cands(t: pd.DataFrame, pred: str, intruded: np.ndarray, dep: pd.DataFrame, arr: pd.DataFrame, codes: set) -> pd.DataFrame:
    """Candidate rows (not intruded, not fixed) with B, gap and band against their sibling stands' arrivals."""
    t = t[~intruded & ~t["cls"].eq("LIRF_pop").to_numpy() & ~t[ID].isin(list(GUARD)).to_numpy()].merge(dep, on=ID)
    pairs = [(i, ap, sib) for i, ap, s in zip(t.index, t["airport"], t["STAND_mvt"]) for sib in siblings(codes, ap, s)]
    if not pairs:
        return t.iloc[0:0]
    pm = pd.DataFrame(pairs, columns=["i", "airport", "sib"])
    x = t.loc[pm["i"]].reset_index(drop=True).assign(sib=pm["sib"].values)
    x["pred_off"] = (x["MVT_TIME_UTC_mvt"] - pd.to_timedelta(x[pred].round(), unit="s")).astype("datetime64[us, UTC]")
    m = pd.merge_asof(x.sort_values("pred_off"), arr, left_on="pred_off", right_on="in_blk", by=["airport", "sib"], direction="backward")
    m["gap"] = (m["pred_off"] - m["in_blk"]).dt.total_seconds()
    m["B"] = (m["MVT_TIME_UTC_mvt"] - m["in_blk"]).dt.total_seconds()
    m = m[(m["gap"] > 0) & (m["gap"] <= 3600) & m["a_type"].ne(m["AIRCRAFT_TYPE_mvt"])]
    m = m.sort_values("B", ascending=False).drop_duplicates(ID)
    m["band"] = pd.cut(m["gap"], R.BANDS, labels=False)
    return m


def intruder_step(tab: pd.DataFrame, pred: str, c12: pd.DataFrame, rdep, rarr, months: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(values after the accepted intruder layer, intruded mask), month-blocked on c12 (twelve-month candidates on `stack`)."""
    cand = R.candidates(tab, pred, rdep, rarr, 3600)
    out, hit = tab[pred].to_numpy(dtype=float).copy(), np.zeros(len(tab), dtype=bool)
    pos = pd.Series(np.arange(len(tab)), index=tab[ID].to_numpy())
    for m in months:
        model = R.fit(c12[c12["month"] != m])
        c = R.values(cand[cand["month"] == m], pred, model, 1.0)
        i = pos.loc[c[ID]].to_numpy()
        out[i], hit[i] = c["value"].to_numpy(), True
    return out, hit


def mars_step(tab: pd.DataFrame, values: np.ndarray, hit: np.ndarray, m12: pd.DataFrame, c12: pd.DataFrame, dep, arr, codes, months, rows_out: list | None = None) -> tuple[np.ndarray, int]:
    t = tab.assign(p=values)
    mc = mars_cands(t, "p", hit, dep, arr, codes)
    out = values.copy()
    pos = pd.Series(np.arange(len(tab)), index=tab[ID].to_numpy())
    n = 0
    for m in months:
        f = m12[m12["month"] != m].assign(ok=lambda x: x["label"] >= x["B"] - 60)
        val = f.groupby("airport")["ok"].agg(["size", "mean"])
        aps = val[(val["size"] >= 10) & (val["mean"] >= 0.9)].index.tolist()
        model = R.fit(c12[c12["month"] != m])
        c = mc[mc["month"].eq(m) & mc["airport"].isin(aps) & mc["airport"].isin(model["airports"])].copy()
        if not len(c):
            continue
        off = np.array([model["offsets"].get(f"{a}|{int(b)}", model["airport_mean"][a]) for a, b in zip(c["airport"], c["band"])])
        i = pos.loc[c[ID]].to_numpy()
        out[i] = np.maximum(c["B"].to_numpy() + off, c["p"].to_numpy())
        n += len(c)
        if rows_out is not None:
            rows_out.extend(c[ID].tolist())
    return out, n


def twelve_fitting(rdep, rarr, dep, arr, codes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """c12 (intruder candidates on rows12 `stack`) and m12 (MARS candidates on rows12 `stack` after the intruder layer), with labels."""
    tw, p12 = R.read_table("twelve", rdep)
    c12 = R.candidates(tw, p12, rdep, rarr, 3600)
    vals, hit = intruder_step(tw, p12, c12, rdep, rarr, sorted(tw["month"].unique()))
    m12 = mars_cands(tw.assign(p=vals), "p", hit, dep, arr, codes)
    return c12, m12


# ---------------------------------------------------------------- scoring

def py(v):
    """JSON-safe copy (numpy scalars, timestamps, nested containers)."""
    if isinstance(v, dict):
        return {str(k): py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v


def month_reading(s: pd.DataFrame, ctl: np.ndarray, cand: np.ndarray, rng) -> dict:
    s = s.assign(c=ctl)
    pct, lo, hi = (float(x) for x in R.interval(s, "c", cand, rng, reps=2000))
    return {"control_rmse": float(np.sqrt(np.mean((s["label"].to_numpy() - ctl) ** 2))), "rmse": float(np.sqrt(np.mean((s["label"].to_numpy() - cand) ** 2))),
            "change_pct": pct, "interval_pct": [lo, hi], "rows_changed": int((np.abs(cand - ctl) > 1e-9).sum()),
            "squared_change": float(((cand - ctl) ** 2).sum())}


def score() -> None:
    decl = read_json(ROOT / DECL)
    target = ROOT / decl["report"]
    if target.exists():
        raise ValueError("Preserve the original result.")
    rng = np.random.default_rng(SEED)
    tk = takeoffs(TRAIN_FILES)
    rdep, rarr = R.movements([Path(f) for f in TRAIN_FILES])
    dep, arr, codes = mars_inputs(TRAIN_FILES)
    c12, m12 = twelve_fitting(rdep, rarr, dep, arr, codes)
    out = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": object_hash(decl), "official_scores_used": "none", "audit_used": False,
           "code": {c: sha256_file(ROOT / c) for c in ["tools/run_micro_bundle.py", "tools/run_intruder_bound.py", "tools/scout_var_alloc.py", "tools/scout_extreme_shift.py",
                                                         "src/prc2026_scout/extreme_shift.py", "src/prc2026_scout/var_alloc.py", "src/prc2026_scout/published_figure.py"]},
           "development": {}, "twelve": {}, "pieces_log": {}}
    ctl_ref = pd.read_parquet(REF / "ctl_v55m.parquet").set_index(ID)
    for m in MONTHS:
        kind = f"mb_ctl_v55m_{m}"
        se.TABLES[kind] = (REF / f"rows_v55m_ctl_{m}.parquet", REF / f"v48_ctl_v55m_{m}")
        c = chain(kind, False, tk)
        c = c[c["month"].eq(m) & ~c["extra"]].reset_index(drop=True)
        diff = float(np.abs(c["new"].to_numpy() - ctl_ref["new"].reindex(c[ID]).to_numpy()).max())
        print(m, "control vs ctl_v55m max |diff|", diff, flush=True)
        if diff > REPRO_TOL_S:
            raise ValueError("The rebuilt control does not reproduce ctl_v55m.")
        out.setdefault("control_reproduction_max_abs_s", {})[m] = diff
        log = {}
        n = chain(kind, True, tk, log)
        n = n[n["month"].eq(m) & ~n["extra"]].set_index(ID).reindex(c[ID]).reset_index()
        out["pieces_log"][m] = log
        tab = c[[ID, "month", "airport", "label", "cls"]].copy()
        tab["day"] = pd.to_datetime(tk.reindex(tab[ID]).to_numpy(), utc=True).strftime("%Y-%m-%d")
        tab["ctl"], tab["cand"] = c["new"].to_numpy(dtype=float), n["new"].to_numpy(dtype=float)
        vc, _ = intruder_step(tab, "ctl", c12, rdep, rarr, [m])
        vn, hn = intruder_step(tab, "cand", c12, rdep, rarr, [m])
        mars_ids: list = []
        va, na = mars_step(tab, vn, hn, m12, c12, dep, arr, codes, [m], mars_ids)
        r = month_reading(tab, vc, va, rng)
        # attribution (diagnostic, after the governing numbers): squared-error change on the pieces' own rows
        y = tab["label"].to_numpy(dtype=float)
        dse = (y - va) ** 2 - (y - vc) ** 2
        pair_ids = [p[ID] for p in log["pairs"]]
        is_pair, is_mars = tab[ID].isin(pair_ids).to_numpy(), tab[ID].isin(mars_ids).to_numpy()
        cd = pd.DataFrame(log["count_aware_days"], columns=["month", "airport", "bday"])
        n_days = n.set_index(ID).reindex(tab[ID])
        is_count = pd.MultiIndex.from_arrays([n_days["airport"].to_numpy(), n_days["bday"].to_numpy()]).isin(
            pd.MultiIndex.from_arrays([cd["airport"], cd["bday"]])) if len(cd) else np.zeros(len(tab), dtype=bool)
        r["attribution_dse"] = {"pair_rows": float(dse[is_pair].sum()), "mars_rows": float(dse[is_mars].sum()),
                                "count_aware_day_rows": float(dse[is_count & ~is_pair & ~is_mars].sum()),
                                "other_rows": float(dse[~is_pair & ~is_mars & ~is_count].sum())}
        r["pair_rows_in_month"] = [float(x) for x in tab.loc[is_pair, ID]]
        r.update({"pairs": len(log["pairs"]), "count_aware_days": len(log["count_aware_days"]), "mars_rows": na,
                  "rows_changed_by_chain": int((np.abs(tab["cand"] - tab["ctl"]) > 1e-9).sum())})
        out["development"][m] = r
        print("dev", m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "interval_pct"}, [round(v, 4) for v in r["interval_pct"]], flush=True)
    # twelve months
    c = chain("twelve", False, tk)
    log = {}
    n = chain("twelve", True, tk, log)
    out["pieces_log"]["twelve"] = log
    keep = ~c["extra"].to_numpy(dtype=bool)
    tab = c.loc[keep, [ID, "month", "airport", "label", "cls"]].reset_index(drop=True)
    tab["day"] = pd.to_datetime(tk.reindex(tab[ID]).to_numpy(), utc=True).strftime("%Y-%m-%d")
    tab["ctl"] = c.loc[keep, "new"].to_numpy(dtype=float)
    tab["cand"] = n.set_index(ID).loc[tab[ID], "new"].to_numpy(dtype=float)
    months12 = sorted(tab["month"].unique())
    vc, _ = intruder_step(tab, "ctl", c12, rdep, rarr, months12)
    vn, hn = intruder_step(tab, "cand", c12, rdep, rarr, months12)
    va, _ = mars_step(tab, vn, hn, m12, c12, dep, arr, codes, months12)
    for m in months12:
        k = tab["month"].eq(m).to_numpy()
        r = month_reading(tab[k].reset_index(drop=True), vc[k], va[k], rng)
        out["twelve"][m] = r
        print("twelve", m, round(r["change_pct"], 4), r["rows_changed"], flush=True)
    rule = decl["rule"]
    dv, tw = out["development"], out["twelve"]
    eq_mean = float(np.mean([dv[m]["change_pct"] for m in EQUAL]))
    checks = {"no_development_month_worse": max(dv[m]["change_pct"] for m in MONTHS) <= rule["max_month_worsening_pct"],
              "equal_footing_mean": eq_mean <= -rule["min_mean_improvement_pct"],
              "equal_footing_intervals_below_zero": all(dv[m]["interval_pct"][1] < 0 for m in EQUAL),
              "twelve_no_month_worse": max(v["change_pct"] for v in tw.values()) <= rule["max_month_worsening_pct"],
              "twelve_better_share": float(np.mean([v["change_pct"] < 0 for v in tw.values()])) >= rule["twelve_min_better_share"]}
    checks = {k: bool(v) for k, v in checks.items()}
    out.update({"equal_footing_mean_pct": eq_mean, "twelve_mean_pct": float(np.mean([v["change_pct"] for v in tw.values()])),
                "twelve_better": int(sum(v["change_pct"] < 0 for v in tw.values())), "checks": checks, "accepted": bool(all(checks.values()))})
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, py(out))
    print({"accepted": out["accepted"], "equal_footing_mean_pct": eq_mean, **checks})


def score_a(path: str) -> None:
    """micro_bundle_v2: piece A alone, incremental over the accepted intruder layer (control = ctl_v55m `new` + that layer)."""
    decl = read_json(ROOT / path)
    target = ROOT / decl["report"]
    if target.exists():
        raise ValueError("Preserve the original result.")
    rng = np.random.default_rng(SEED)
    tk = takeoffs(TRAIN_FILES)
    rdep, rarr = R.movements([Path(f) for f in TRAIN_FILES])
    dep, arr, codes = mars_inputs(TRAIN_FILES)
    c12, m12 = twelve_fitting(rdep, rarr, dep, arr, codes)
    out = {"created_at_utc": utc_now(), "declaration": path, "declaration_sha256": object_hash(decl), "official_scores_used": "none", "audit_used": False,
           "code": {c: sha256_file(ROOT / c) for c in ["tools/run_micro_bundle.py", "tools/run_intruder_bound.py"]}, "development": {}, "twelve": {}}
    ctl = pd.read_parquet(REF / "ctl_v55m.parquet")
    for name, tab_all, months in [("development", ctl, MONTHS), ("twelve", None, None)]:
        if name == "twelve":
            c = chain("twelve", False, tk)
            c = c[~c["extra"].to_numpy(dtype=bool)]
            tab_all = c[[ID, "month", "airport", "label", "cls", "new"]].reset_index(drop=True)
            months = sorted(tab_all["month"].unique())
        else:
            cls = pd.read_parquet(ROOT / "runs/daily_totals/rows.parquet", columns=[ID, "cls"])
            tab_all = tab_all[[ID, "month", "airport", "label", "new"]].merge(cls, on=ID)
        tab_all = tab_all.reset_index(drop=True)
        tab_all["day"] = pd.to_datetime(tk.reindex(tab_all[ID]).to_numpy(), utc=True).strftime("%Y-%m-%d")
        vc, hit = intruder_step(tab_all, "new", c12, rdep, rarr, months)
        ids: list = []
        va, na = mars_step(tab_all, vc, hit, m12, c12, dep, arr, codes, months, ids)
        moved = tab_all[ID].isin(ids).to_numpy()
        for m in months:
            k = tab_all["month"].eq(m).to_numpy()
            r = month_reading(tab_all[k].reset_index(drop=True), vc[k], va[k], rng)
            r["mars_rows"] = int((moved & k).sum())
            r["mars_rows_by_airport"] = tab_all.loc[moved & k, "airport"].value_counts().to_dict()
            out[name][m] = r
            print(name, m, round(r["change_pct"], 4), [round(x, 4) for x in r["interval_pct"]], "rows", r["mars_rows"], flush=True)
    rule = decl["rule"]
    dv, tw = out["development"], out["twelve"]
    eq_mean = float(np.mean([dv[m]["change_pct"] for m in EQUAL]))
    checks = {"no_development_month_worse": max(dv[m]["change_pct"] for m in MONTHS) <= rule["max_month_worsening_pct"],
              "equal_footing_mean": eq_mean <= -rule["min_mean_improvement_pct"],
              "equal_footing_intervals_below_zero": all(dv[m]["interval_pct"][1] < 0 for m in EQUAL),
              "twelve_no_month_worse": max(v["change_pct"] for v in tw.values()) <= rule["max_month_worsening_pct"],
              "twelve_better_share": float(np.mean([v["change_pct"] < 0 for v in tw.values()])) >= rule["twelve_min_better_share"]}
    checks = {k: bool(v) for k, v in checks.items()}
    out.update({"equal_footing_mean_pct": eq_mean, "twelve_mean_pct": float(np.mean([v["change_pct"] for v in tw.values()])),
                "twelve_better": int(sum(v["change_pct"] < 0 for v in tw.values())), "checks": checks, "accepted": bool(all(checks.values()))})
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, py(out))
    print({"accepted": out["accepted"], "equal_footing_mean_pct": eq_mean, **checks})


# ---------------------------------------------------------------- ranking

def ranking_chain(pieces: bool, out_name: str, version: int) -> None:
    base, recon = "runs/rematch_nonm/base/vigorous-whistle_v55m_rematch.parquet", ROOT / "runs/rematch_nonm/rank/R"
    out_dir = OUT / out_name
    output = out_dir / f"vigorous-whistle_v{version}.parquet"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    tk = takeoffs(TRAIN_FILES + [str(ROOT / "data/raw/ranking.parquet")])
    log = {}
    dev = chain("stack", pieces, tk, log if pieces else None)
    # the development side exactly as tools/scout_var_alloc.py export: k from the out-of-fold basis, models on all four months
    elig_d = var_alloc.eligible(dev)
    k = var_alloc.slope(dev["r"].to_numpy(dtype=float), dev["base"].to_numpy(dtype=float), elig_d)
    f_d = var_alloc.features(dev, departures_cached(TRAIN_FILES), KEYS)
    models = var_alloc.fit_models(f_d, dev["a"].to_numpy(dtype=float), dev["label"].to_numpy(dtype=float), dev["r"].to_numpy(dtype=float), elig_d)
    days_dev = extreme_shift.day_frame(dev, KEYS)
    params = extreme_shift.fit(days_dev.droplevel("month"))
    t = se.ranking_rows(base, recon)
    days = extreme_shift.day_frame(t, ["airport", "bday"])
    pairs = pd.DataFrame(columns=[ID])
    if pieces:
        pairs = detect_pairs(t, days, params["c0"], tk)
        if len(pairs):
            moved = t[ID].map(dict(zip(pairs[ID], pairs["to_day"])))
            t["bday"] = np.where(moved.notna(), moved, t["bday"])
            days = extreme_shift.day_frame(t, ["airport", "bday"])
        mu = noise_mu(dev, np.ones(len(dev), dtype=bool))
        ch = shifts_b(days, params, mu)
        count_days = ch[ch["count_aware"].fillna(False).astype(bool)] if "count_aware" in ch else ch.iloc[0:0]
        ch = ch[["I", "shift_new", "dn", "rows", "n_eff"]]
    else:
        ch = extreme_shift.shifts(days, params)
        count_days = ch.iloc[0:0]
    ext, mask = extreme_shift.apply(t, ch, ["airport", "bday"])
    v55 = np.where(mask, np.rint(ext), t["v48_exported"].to_numpy(dtype=float)).astype(np.int64)
    t["month"] = t["bday"].str[:7]
    I = sva.implied(t, days, params["c0"], ["airport", "bday"])
    elig = var_alloc.eligible(t, GUARD)
    f = var_alloc.features(t, sva.departures([str(ROOT / "data/raw/ranking.parquet")]), KEYS)
    w = var_alloc.weights(models, f, elig)
    basis = var_alloc.basis(t, w, I, KEYS)
    new = var_alloc.update(ext, k, basis, elig)
    values = np.where(elig, np.rint(new), v55).astype(float)
    if pieces and len(pairs):
        bounded = pair_bound(values, t[ID].to_numpy(), pairs)
        values = np.where(bounded != values, np.rint(bounded), values)
    provenance = {"experiment_id": "micro_bundle_v1", "dry_run": True, "pieces": pieces, "base": {"path": base, "sha256": sha256_file(ROOT / base)},
                  "reconciliation": {"dir": "runs/rematch_nonm/rank/R", "values_sha256": sha256_file(recon / "values.parquet"), "days_sha256": sha256_file(recon / "days.parquet")},
                  "declaration": {DECL: sha256_file(ROOT / DECL)}, "official_scores_used": "none", "audit_used": False,
                  "recipe": "model part (rematch) -> v48 -> [C] -> extreme-day [+ B] -> variance re-allocation -> [C bound]"}
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = export_submission(pd.DataFrame({ID: t[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    rep = {"created_at_utc": utc_now(), "sha256": sha256_file(output), "k": k, "receipt": receipt,
           "pairs": [{kk: (str(v) if isinstance(v, pd.Timestamp) else v) for kk, v in r.items()} for r in pairs.to_dict(orient="records")],
           "count_aware_days": [{"airport": a, "bday": b, **{kk: float(v) for kk, v in r.items()}} for (a, b), r in count_days[["I", "shift_new", "dn", "rows", "n_eff"]].iterrows()]}
    write_json(out_dir / f"chain_v{version}.json", rep)
    print({"sha256": rep["sha256"], "k": k, "pairs": len(pairs), "count_aware_days": len(count_days)})


def finish(intruder_file: str, intruder_sha: str, version: int) -> None:
    """A on top of the intruder step's dry run (whose base is the stack file) -> runs/micro_bundle/dry_v<version>/."""
    if sha256_file(ROOT / intruder_file) != intruder_sha:
        raise ValueError("The intruder step's file does not have the expected SHA-256.")
    out_dir = OUT / f"dry_v{version}"
    output = out_dir / f"vigorous-whistle_v{version}.parquet"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    rdep_t, rarr_t = R.movements([Path(f) for f in TRAIN_FILES])
    dep_t, arr_t, codes_t = mars_inputs(TRAIN_FILES)
    c12, m12 = twelve_fitting(rdep_t, rarr_t, dep_t, arr_t, codes_t)
    rk_path = str(ROOT / "data/raw/ranking.parquet")
    dep, arr, codes = mars_inputs([rk_path])
    codes = codes | codes_t
    b = pd.read_parquet(ROOT / intruder_file)
    receipt_in = read_json((ROOT / intruder_file).with_suffix(".receipt.json"))
    intruded = set(r[ID] for r in receipt_in["rows"])
    rk = pd.read_parquet(rk_path, columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    tab = b.merge(rk[[ID, "ADEP_mvt"]].assign(cls=np.where(pop, "LIRF_pop", "x")).rename(columns={"ADEP_mvt": "airport"}), on=ID)
    tab["month"] = "ranking"
    vals = tab[TARGET].to_numpy(dtype=float)
    hit = tab[ID].isin(intruded).to_numpy()
    mc = mars_cands(tab.assign(p=vals), "p", hit, dep, arr, codes)
    f = m12.assign(ok=lambda x: x["label"] >= x["B"] - 60)
    val = f.groupby("airport")["ok"].agg(["size", "mean"])
    aps = val[(val["size"] >= 10) & (val["mean"] >= 0.9)].index.tolist()
    model = R.fit(c12)
    c = mc[mc["airport"].isin(aps) & mc["airport"].isin(model["airports"])].copy()
    off = np.array([model["offsets"].get(f"{a}|{int(bb)}", model["airport_mean"][a]) for a, bb in zip(c["airport"], c["band"])]) if len(c) else np.array([])
    new = vals.copy()
    pos = pd.Series(np.arange(len(tab)), index=tab[ID].to_numpy())
    if len(c):
        new[pos.loc[c[ID]].to_numpy()] = np.maximum(np.rint(c["B"].to_numpy() + off), c["p"].to_numpy())
    if (new[tab[ID].isin(list(GUARD)).to_numpy() | tab["cls"].eq("LIRF_pop").to_numpy()] != vals[tab[ID].isin(list(GUARD)).to_numpy() | tab["cls"].eq("LIRF_pop").to_numpy()]).any():
        raise ValueError("A guard or LIRF population row moved.")
    provenance = {"experiment_id": "micro_bundle_v1", "dry_run": True, "intruder_step": {"path": intruder_file, "sha256": intruder_sha},
                  "declaration": {DECL: sha256_file(ROOT / DECL)}, "official_scores_used": "none", "audit_used": False,
                  "recipe": "v60's chain with C and B in the published chain -> stack_resid_v2 -> intruder stand bound -> A (MARS siblings, EGLL/LTFM by fitting validity)"}
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = export_submission(pd.DataFrame({ID: tab[ID].to_numpy(), "prediction": new}), ROOT / "data/raw/submitting.parquet", output, provenance)
    rep = {"created_at_utc": utc_now(), "sha256": sha256_file(output), "receipt": receipt, "mars_airports": aps, "mars_rows": int(len(c)),
           "mars": c[[ID, "airport", "p", "B", "gap"]].round(1).to_dict(orient="records")}
    write_json(out_dir / f"export_v{version}.json", rep)
    print({"sha256": rep["sha256"], "mars_rows": rep["mars_rows"], "airports": aps})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--declaration", default=DECL)
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--no-pieces", action="store_true")
    ap.add_argument("--out", default="chain_v61")
    ap.add_argument("--version", type=int, default=61)
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--intruder-file")
    ap.add_argument("--intruder-sha256")
    a = ap.parse_args()
    if a.score:
        if read_json(ROOT / a.declaration)["experiment_id"] == "micro_bundle_v2":
            score_a(a.declaration)
        else:
            score()
    if a.chain:
        ranking_chain(not a.no_pieces, a.out, a.version)
    if a.finish:
        finish(a.intruder_file, a.intruder_sha256, a.version)


if __name__ == "__main__":
    main()
