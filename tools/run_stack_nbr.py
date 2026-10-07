"""Governing run of configs/stack_nbr_v1.json (docs/stack_nbr/README.md): three label-free neighbour-state columns (tools/nbr_columns.py:
nx, nxk, nn) added to each learner of the guarded stacking residual (step 6 of docs/final_file/v69_chain.md). Candidates: sn65 (the stacker
with the columns at weight 0.65, ten seeds, then steps 7-11) and sn65_layers (sn65, then nbr_state_v1 w75 and nbr_dark_v1 dark_pool_w75
re-applied with sn65's values as the base). Control: v71's development values (v70's development proxy runs/oooi_lhr/dev_v70_proxy.parquet
with both layers applied; must reproduce docs/nbr_state/dark_results_v1.json's development RMSEs).

A copy of tools/run_stack_queue.py with the queue switch replaced by a neighbour switch (NBR off / on): with NBR on each learner gets the
three columns (v2 46 + 3 = 49, v3 65 + 3 = 68, OPDI 79 + 3 = 82). Guards on the unchanged matrices, LightGBM settings, month-blocking and the
ten seeds 20261003, 11-19 are tools/run_stack_weight.py's, unchanged. The neighbour layers reuse tools/run_nbr_state.py (state, slope, apply,
constants, dev_table) and tools/run_nbr_dark.py (with_gate, W_STATE) unchanged and replicate their main()'s arithmetic with the base values
replaced.

Neighbour columns (declaration 'columns'): tools/nbr_columns.columns on a table with value = the stacker table's 'new' (the pre-stacker chain
value), v9 = the stacker table's 'v9', matched = the stacker table's 'matched', airport, ts = take-off in seconds since 2025-01-01 UTC and
date = UTC take-off date, from the row's own month's training file (development) or data/raw/ranking.parquet (ranking).

  --warm fit|chain               VM preparation (open every input once; 'fit' also writes the input consistency report)
  --dev-month M --nbr off|on     month-blocked out-of-month predictions (ten seeds x v2 / v3 / OPDI) on development month M
                                 -> runs/stack_nbr/dev/<nbr>/preds_<M>.parquet
  --dev-identity                 columns off, w 0.65, ten seeds, steps 7-11 must equal the control's base (v70 development proxy value column:
                                 max |diff| 0 or the 4.5e-13 s Linux bound); with both layers the off path must reproduce dark_results' development
                                 RMSEs; else stop, nothing read -> runs/stack_nbr/dev/identity.json
  --rank-fit --nbr off|on        the three learners on all four development months, ten seeds, applied to v66's shipped step-5 ranking table
                                 -> runs/stack_nbr/rank/<nbr>/preds.parquet
  --rank-identity                columns off: steps 6-11 must reproduce the uploaded v70 (f1d17747...) byte for byte, and both layers on it
                                 (slopes from the off path's development values) the uploaded v71 (2dfb3b9f...) byte for byte
  --dev-score                    only after both identities: sn65 and sn65_layers against the control; clauses 1-3 -> docs/stack_nbr/results_v1.json
  --rank-check                   clause 4 (stacker transfer guard) for the columns-on ten-seed bags; sn65's ranking chain (steps 6-11, label-free,
                                 runs/stack_nbr/rank/on_chain/, not a candidate file) and the layers' transfer guards on it; selection -> results
  --rank-dry                     only if a candidate is accepted: its ranking file -> runs/stack_nbr/rank/dry_v72/vigorous-whistle_v72.parquet
  --verify-dry                   (workstation) independent check of the dry run against the uploaded v71 / v70

No official score enters. Nothing is uploaded; nothing is written under runs/submissions.
"""
from __future__ import annotations

import argparse
import json
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
import nbr_columns as NC  # noqa: E402
import run_nbr_dark as ND  # noqa: E402
import run_nbr_state as N  # noqa: E402
import run_stack_weight as SW  # noqa: E402

DECL = "configs/stack_nbr_v1.json"
OUT = ROOT / "runs/stack_nbr"
DOCS = ROOT / "docs/stack_nbr"
RESULTS = DOCS / "results_v1.json"
IDENT = OUT / "dev" / "identity.json"
RANK_IDENT = OUT / "rank" / "identity_v70" / "identity.json"
CONTROL = ("runs/oooi_lhr/dev_v70_proxy.parquet", "51aaa6a62f64897290199f6c6c9e7d3d43245a3862f36b2bdbc35be8012eabbe")
V69_PROXY = ("runs/stack_weight/dev/w65_bag10/proxy.parquet", "c088e409de153d81b006bcd26b5b2a28e0e306043a5172c2158d0ecc6e6f5cb8")
V69 = "192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665"
V70 = "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f"
V71_STATE = "7a0b1431eef1146d7afd40c728bea7af6b90a411e257ca277564866e410b84ed"   # nbr_state_v1 w75 on v70 (runs/nbr_state/rank/dry_v71)
V71 = "2dfb3b9f53284c7ae3a02882d19cfd61b19116b1b00504176551bcae8b82e6ba"
STACK_WEIGHT_RANK_PREDS = ("runs/stack_weight/rank/preds.parquet", "8083ffcfdf861e278ac8693fb2320872cd95378f3f9ac861decf23c9a6163772")
STACK_QUEUE_DEV_OFF = "runs/stack_queue/dev/off/preds_{m}.parquet"   # stack_queue_v1's columns-off predictions (= stack_weight's)
STATE_RESULTS = ROOT / "docs/nbr_state/results_v1.json"
DARK_RESULTS = ROOT / "docs/nbr_state/dark_results_v1.json"
DARK_DECL = ROOT / "configs/nbr_dark_v1.json"
LINUX_BOUND = 4.5e-13
RMSE_TOL = 1e-9   # tools/run_nbr_dark.py's own reproduction tolerance for the development RMSEs
WEIGHT, K = 0.65, 10
CANDIDATES = ["sn65", "sn65_layers"]
VERSION = 72
DEV_FILES = {m: f"data/raw/training_{m}-01_*.parquet" for m in SW.MONTHS}
RANKING = "data/raw/ranking.parquet"
RANK_MONTHS = ["2026-01", "2026-07"]
MATCHED_SCOPES = ["matched_no_lirf", "lirf_nonpop_matched"]
CODE = ["tools/run_stack_nbr.py", "tools/nbr_columns.py", "tools/run_nbr_state.py", "tools/run_nbr_dark.py"] + SW.CODE + [
    "tools/run_oooi_pin.py", "tools/run_oooi_lhr.py"]


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def w_dark() -> float:
    """nbr_dark_v1's declared weight (tools/run_nbr_dark.main reads it the same way)."""
    return float(read_json(DARK_DECL)["candidates"][0]["weight"])


# ---------------------------------------------------------------- neighbour columns (declaration: columns)

def takeoff_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"] == "DEP"][[ID, "MVT_TIME_UTC_mvt"]]
    if raw[ID].duplicated().any():
        raise ValueError(f"{path}: duplicate departure IDs.")
    return raw


def nbr_table(tab: pd.DataFrame, tk: pd.Series) -> pd.DataFrame:
    """The input table of tools/nbr_columns.columns: value = the stacker table's 'new', v9 and matched = the stacker table's, take-off from raw."""
    if tab["matched"].isna().any():
        raise ValueError("The stacker table's matched flag has missing values.")
    tk = pd.to_datetime(tk, utc=True).reset_index(drop=True)
    return pd.DataFrame({"airport": tab["airport"].to_numpy(),
                         "ts": (tk - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds().to_numpy(),
                         "date": tk.dt.strftime("%Y-%m-%d").to_numpy(),
                         "matched": tab["matched"].to_numpy(dtype=bool),
                         "value": tab["new"].to_numpy(dtype=float),
                         "v9": tab["v9"].to_numpy(dtype=float)})


def nbr_dev(dev: pd.DataFrame) -> pd.DataFrame:
    """Neighbour columns of the development rows; take-off times from each row's own month's training file."""
    parts = []
    for m, pat in DEV_FILES.items():
        files = sorted(ROOT.glob(pat))
        if len(files) != 1:
            raise ValueError(f"{pat}: {len(files)} files.")
        parts.append(takeoff_raw(files[0]).assign(qmonth=m))
    q = pd.concat(parts, ignore_index=True)
    if q[ID].duplicated().any():
        raise ValueError("A departure appears in two month files.")
    q = q.set_index(ID).reindex(dev[ID].to_numpy())
    if q["qmonth"].isna().any() or not np.array_equal(q["qmonth"].to_numpy(), dev["month"].to_numpy()):
        raise ValueError("A development row is not a departure of its own month's file.")
    return NC.columns(nbr_table(dev, q["MVT_TIME_UTC_mvt"]))


def nbr_rank(rk: pd.DataFrame) -> pd.DataFrame:
    q = takeoff_raw(ROOT / RANKING).set_index(ID).reindex(rk[ID].to_numpy())
    if q["MVT_TIME_UTC_mvt"].isna().any():
        raise ValueError("A ranking row is not a departure of data/raw/ranking.parquet.")
    return NC.columns(nbr_table(rk, q["MVT_TIME_UTC_mvt"]))


def with_nbr(X: pd.DataFrame, Q: pd.DataFrame) -> pd.DataFrame:
    if len(X) != len(Q):
        raise ValueError("Row counts differ.")
    return pd.concat([X.reset_index(drop=True), Q.reset_index(drop=True)], axis=1)


def coverage(Q: pd.DataFrame) -> dict:
    return {"rows": int(len(Q)), "nx_defined": int(np.isfinite(Q["nx"]).sum()), "nxk_defined": int(np.isfinite(Q["nxk"]).sum()),
            "nn_mean": float(Q["nn"].mean()), "nx_rms": float(np.sqrt(np.nanmean(Q["nx"].to_numpy(dtype=float) ** 2))),
            "nxk_rms": float(np.sqrt(np.nanmean(Q["nxk"].to_numpy(dtype=float) ** 2)))}


def consistency(dev1: pd.DataFrame, dev2: pd.DataFrame, rk: pd.DataFrame) -> dict:
    """The declaration's input checks: the stacker tables' matched flag against the layers' flag (development: the v70 development proxy's
    cls ending '_m'; ranking: the recipe scope in matched_no_lirf / lirf_nonpop_matched), and the development 'v9' against
    runs/lomo_dev/rows.parquet. A disagreement outside the fixed rows (LIRF population, guard rows) stops the round."""
    rep = {}
    proxy = pd.read_parquet(ROOT / CONTROL[0], columns=[ID, "cls"]).set_index(ID)
    lomo = pd.read_parquet(N.LOMO, columns=[ID, "v9"]).set_index(ID)["v9"]
    ok = True
    for name, dev in [("dev_inputs", dev1), ("rank_inputs_dev", dev2)]:
        cls = proxy["cls"].reindex(dev[ID].to_numpy())
        if cls.isna().any():
            raise ValueError(f"{name}: development rows outside the v70 development proxy.")
        pm = cls.str.endswith("_m").to_numpy()
        sm = dev["matched"].to_numpy(dtype=bool)
        dis = pm != sm
        fixed = cls.eq("LIRF_pop").to_numpy() | dev[ID].isin(N.GUARD).to_numpy()
        v9 = lomo.reindex(dev[ID].to_numpy()).to_numpy(dtype=float)
        sv9 = dev["v9"].to_numpy(dtype=float)
        v9_eq = (v9 == sv9) | (np.isnan(v9) & np.isnan(sv9))
        rep[name] = {"rows": int(len(dev)), "matched_equal": int((~dis).sum()), "matched_equal_share": float((~dis).mean()),
                     "matched_differ": int(dis.sum()), "matched_differ_outside_fixed_rows": int((dis & ~fixed).sum()),
                     "matched_differ_by_cls": {str(k): int(v) for k, v in pd.Series(cls.to_numpy()[dis]).value_counts().items()},
                     "matched_differ_stacker_true": int((dis & sm).sum()),
                     "v9_equal_to_lomo_rows": int(v9_eq.sum()), "v9_equal_share": float(v9_eq.mean())}
        ok &= bool((dis & ~fixed).sum() == 0 and v9_eq.all())
    rec = pd.read_parquet(N.RECIPE, columns=[ID, "scope", "v9"]).set_index(ID).reindex(rk[ID].to_numpy())
    if rec["scope"].isna().any():
        raise ValueError("Ranking rows outside the recipe.")
    rm = rec["scope"].isin(MATCHED_SCOPES).to_numpy()
    sm = rk["matched"].to_numpy(dtype=bool)
    dis = rm != sm
    fixed = rec["scope"].eq("fixed").to_numpy() | rk[ID].isin(N.GUARD).to_numpy()
    rv9, sv9 = rec["v9"].to_numpy(dtype=float), rk["v9"].to_numpy(dtype=float)
    rep["ranking"] = {"rows": int(len(rk)), "matched_equal": int((~dis).sum()), "matched_equal_share": float((~dis).mean()),
                      "matched_differ": int(dis.sum()), "matched_differ_outside_fixed_rows": int((dis & ~fixed).sum()),
                      "v9_equal_to_recipe_rows_for_information": int((rv9 == sv9).sum()),
                      "v9_max_abs_diff_to_recipe_for_information": float(np.nanmax(np.abs(rv9 - sv9)))}
    ok &= bool((dis & ~fixed).sum() == 0)
    rep["material_difference"] = not ok
    return rep


# ---------------------------------------------------------------- development

def dev_month(m: str, nbr: str) -> None:
    """run_stack_weight.dev_month with the neighbour switch (guards on the unchanged X46 / Pn / Pg, as declared)."""
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    path = OUT / "dev" / nbr / f"preds_{m}.parquet"
    if path.exists():
        raise ValueError(f"{path} exists.")
    dev, X46, X65, X79, Pn, Pg, Po, res = SW.dev_inputs()
    mats = {"p2": X46, "p3": X65, "po": X79}
    info = {}
    if nbr == "on":
        Q = nbr_dev(dev)
        info = {"nbr_coverage_dev": coverage(Q)}
        mats = {k: with_nbr(X, Q) for k, X in mats.items()}
    fx = S.fixed_rows(dev)
    month, air, dom = dev["month"].to_numpy(), dev["airport"].to_numpy(), dev["dom"].to_numpy()
    fit, ev = (month != m) & ~fx, month == m
    h2 = S.outside_support(X46[fit], air[fit], X46[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hn3 = V3.guard_new(Pn[fit].reset_index(drop=True), air[fit], Pn[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hno = V3.guard_new(Pg[fit].reset_index(drop=True), air[fit], Pg[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    out = pd.DataFrame({ID: dev[ID].to_numpy()[ev], "fx": fx[ev], "h2": h2, "h3": h2 | hn3, "ho": h2 | hno,
                        "o_has": Po["o_has"].to_numpy()[ev]})
    trees, ncol = {}, {}
    for key in ["p2", "p3", "po"]:
        Xa = mats[key].to_numpy(np.float32)
        ncol[key] = int(Xa.shape[1])
        for i, seed in enumerate(SW.SEEDS10):
            S.SEEDS = [seed]
            models = S.fit_bag(Xa, res, fit, dom)
            out[f"{key}_{i}"] = models[0].predict(Xa[ev], num_iteration=models[0].best_iteration)
            trees[f"{key} {seed}"] = int(models[0].best_iteration)
            print(m, nbr, key, ncol[key], seed, "trees", trees[f"{key} {seed}"], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "month": m, "nbr": nbr, "columns": ncol, "rows": int(ev.sum()),
                                           "trees": trees, **info, "sha256": sha256_file(path), "code": {p: sha256_file(ROOT / p) for p in CODE}})


def load_preds(nbr: str, dev: pd.DataFrame) -> pd.DataFrame:
    import run_stack_resid as S
    P = pd.concat([pd.read_parquet(OUT / "dev" / nbr / f"preds_{m}.parquet") for m in SW.MONTHS], ignore_index=True)
    P = P.set_index(ID).reindex(dev[ID]).reset_index()
    if P["p2_0"].isna().any() or not np.array_equal(P["fx"].to_numpy(dtype=bool), S.fixed_rows(dev)):
        raise ValueError("The stored predictions do not cover the development rows.")
    return P


def step11_dev_pins() -> pd.DataFrame:
    """Step 11's development pins exactly as tools/run_oooi_lhr.v70_dev_proxy (run_oooi_pin's development part): on the v69 proxy's rows,
    month-blocked offsets over July / November / December 2025 (labels and take-off - OUT only; independent of the chain's value)."""
    import run_oooi_pin as P
    import run_stack_resid as S
    if sha256_file(ROOT / V69_PROXY[0]) != V69_PROXY[1] or P.PROXY.resolve() != (ROOT / V69_PROXY[0]).resolve():
        raise ValueError("The v69 development proxy differs.")
    pin_res = read_json(ROOT / "docs/oooi_pin/results_v1.json")
    if sha256_file(P.RUN / "joined_dev.parquet") != pin_res["joined_sha256"]["dev"]:
        raise ValueError("runs/oooi_pin/joined_dev.parquet changed since oooi_pin_v1's governing run.")
    pin_dev = P.scope(pd.read_parquet(P.RUN / "joined_dev.parquet"))
    dev = pd.read_parquet(P.PROXY, columns=[ID, "month", "airport", "label", "cls", "day", "value"])
    dev = dev.merge(P.takeoff_days(P.DEV_MONTHS), on=ID, how="left", validate="one_to_one")
    dev["day"] = dev["tday"]
    dev = dev.merge(pin_dev[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    fixed = S.fixed_rows(dev)
    dev["in_scope"] = dev["raw"].notna().to_numpy() & ~fixed & dev["airport"].eq("LFPG").to_numpy()
    dev["pin"] = np.nan
    for m in P.PIN_DEV_MONTHS:
        off = P.offsets(dev[dev["in_scope"] & dev["month"].ne(m) & dev["month"].isin(P.PIN_DEV_MONTHS)])
        sel = dev["in_scope"] & dev["month"].eq(m)
        dev.loc[sel, "pin"] = dev.loc[sel, "raw"] + dev.loc[sel, "source"].map(off)
    return dev[[ID, "day", "in_scope", "pin"]]


def step11_dev(ids: np.ndarray, value: np.ndarray, pins: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """oooi_pin_v1 w50 on the in-scope LFPG rows: value = max(0.5 x pin + 0.5 x chain, -12)."""
    import run_oooi_pin as P
    p = pins.set_index(ID).reindex(ids)
    if p["day"].isna().any():
        raise ValueError("The pin table does not cover the rows.")
    sc = p["in_scope"].to_numpy(dtype=bool)
    v = value.astype(float).copy()
    v[sc] = P.cand_value(0.5, p["pin"].to_numpy()[sc], v[sc])
    return v, p["day"].to_numpy()


def dev_context() -> dict:
    """tools/run_stack_queue.dev_score's inputs (pinned v66 chain files, v61 keys, control, v69 proxy, step-11 pins, steps 7-10)."""
    import run_ctx_ownpoints_chain as CH
    import run_opdi_stack as O
    import run_own_motion_v2 as O2
    import run_stack_resid_v3 as V3
    if sha256_file(ROOT / CONTROL[0]) != CONTROL[1]:
        raise ValueError("The control differs from the declared file.")
    dev = CH.dev_table_from(O.SHIP / "ctl.parquet", O.SHIP / "rows_all.parquet", O.SHIP / "dmp.parquet")
    s1 = pd.read_parquet(O.SHIP / "cand.parquet").set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    keys, _ = O2.O.dev_table()
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    ctl = pd.read_parquet(ROOT / CONTROL[0]).set_index(ID).reindex(dev[ID]).reset_index()
    if ctl["value"].isna().any() or not np.allclose(ctl["label"].to_numpy(dtype=float), dev["label"].to_numpy(dtype=float)):
        raise ValueError("The control does not cover the development rows.")
    v69p = pd.read_parquet(ROOT / V69_PROXY[0]).set_index(ID).reindex(dev[ID]).reset_index()
    _, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    return {"dev": dev, "s1": s1, "keys": keys, "ctl": ctl, "v69p": v69p, "pins": step11_dev_pins(), "chain": V3.Chain(), "t12": t12, "trig": trig}


def through_chain(c: dict, nbr: str, name: str):
    """The stacking value at w 0.65 with ten seeds (run_stack_weight.dev_value), steps 7-10, step 11 -> runs/stack_nbr/dev/<name>/proxy.parquet."""
    import run_ctx_ownpoints_chain as CH
    dev, keys = c["dev"], c["keys"]
    P = load_preds(nbr, dev)
    value, use, p = SW.dev_value(dev, P, WEIGHT, K)
    t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
    t["s2"] = c["s1"] + (value - dev["new"].to_numpy(dtype=float))
    _, cand63 = c["chain"].run(t, "s2")
    t["c63"] = cand63
    c66 = CH.own_motion_v2(t, "c63", c["t12"], c["trig"])
    c70, tday = step11_dev(t[ID].to_numpy(), c66, c["pins"])
    d = OUT / "dev" / name
    d.mkdir(parents=True, exist_ok=True)
    out = t[[ID, "month", "airport", "label", "cls"]].copy()
    out["day"] = tday
    out["value"] = c70
    out["v69like"] = c66
    out["opdi_used"] = use
    out.to_parquet(d / "proxy.parquet", index=False)
    return c66, c70, use, p, d / "proxy.parquet"


# ---------------------------------------------------------------- neighbour layers (nbr_state_v1 w75, then nbr_dark_v1)

def layer_dev_table() -> pd.DataFrame:
    """tools/run_nbr_state.dev_table (v70 development proxy rows, matched = cls ending '_m', fixed, v9 from runs/lomo_dev/rows.parquet, take-off
    ts / date from the raw month files) with tools/run_nbr_dark.with_gate; each caller replaces 'value'."""
    return ND.with_gate(N.dev_table(), N.DEV_MONTHS)


def aligned(values: np.ndarray, ids: np.ndarray, L: pd.DataFrame) -> np.ndarray:
    v = pd.Series(np.asarray(values, dtype=float), index=ids).reindex(L[ID].to_numpy())
    if v.isna().any():
        raise ValueError("The values do not cover the layer table's rows.")
    return v.to_numpy(dtype=float)


def layers_dev(L: pd.DataFrame, value: np.ndarray, wd: float) -> dict:
    """tools/run_nbr_dark.main's development part with the base values replaced: x = run_nbr_state.state of the base; nbr_state_v1 w75 with
    per-airport month-blocked slopes on unmatched rows; nbr_dark_v1 with one pooled month-blocked slope on dark rows of the nbr_state values.
    Returns the values after each layer, the all-month slopes for the ranking and the development side of both transfer guards."""
    d = L.copy()
    d["value"] = np.asarray(value, dtype=float)
    d["x"] = N.state(d)
    x = d["x"].to_numpy()
    matched, fixed, gate = d["matched"].to_numpy(dtype=bool), d["fixed"].to_numpy(dtype=bool), d["gate"].to_numpy(dtype=bool)
    unm = ~matched & ~fixed & ~np.isnan(x)
    dk = matched & ~gate & ~fixed & ~np.isnan(x)
    air, month = d["airport"].to_numpy(), d["month"].to_numpy()
    airports = sorted(d["airport"].unique())
    v = d["value"].to_numpy(dtype=float)
    y0 = (d["label"] - d["value"]).to_numpy()
    b_dev, final, blocked = np.zeros(len(d)), {}, {}
    for ap in airports:
        a = unm & (air == ap)
        if a.sum() >= N.MIN_FIT:
            final[ap] = N.slope(x[a], y0[a])
        for m in N.DEV_MONTHS:
            fit, tgt = a & (month != m), a & (month == m)
            if fit.sum() >= N.MIN_FIT and tgt.any():
                blocked[f"{ap} {m}"] = N.slope(x[fit], y0[fit])
                b_dev[tgt] = blocked[f"{ap} {m}"]
    st = N.apply(v, x, b_dev, ND.W_STATE, unm)
    rd_state = {ap: float(np.sqrt(((st - v)[unm & (air == ap)] ** 2).mean())) for ap in airports if ap in final}
    y = np.clip((d["label"].to_numpy(dtype=float) - st), -N.CLIP_Y, N.CLIP_Y)
    B = float((x[dk] * y[dk]).sum() / (x[dk] ** 2).sum())
    bd, blocked_dark = np.zeros(len(d)), {}
    for m in N.DEV_MONTHS:
        fit, tgt = dk & (month != m), dk & (month == m)
        blocked_dark[m] = float((x[fit] * y[fit]).sum() / (x[fit] ** 2).sum())
        bd[tgt] = blocked_dark[m]
    dark = st.copy()
    dark[dk] = np.maximum(dark[dk] + np.clip(wd * bd[dk] * x[dk], -N.CAP, N.CAP), N.FLOOR)
    cd = dark - st
    rd_dark = {ap: float(np.sqrt((cd[dk & (air == ap)] ** 2).mean())) for ap in airports if (dk & (air == ap)).any()}
    return {"state": st, "dark": dark, "airports": airports, "final": final, "blocked_state": blocked, "B": B, "blocked_dark": blocked_dark,
            "rd_state": rd_state, "rd_dark": rd_dark, "rows_unmatched_scope": int(unm.sum()), "rows_dark": int(dk.sum()),
            "rows_changed_state": int((st != v).sum()), "rows_changed_dark": int((dark != st).sum())}


def fit_summary(f: dict) -> dict:
    return {k: f[k] for k in ["final", "blocked_state", "B", "blocked_dark", "rd_state", "rd_dark", "rows_unmatched_scope", "rows_dark",
                              "rows_changed_state", "rows_changed_dark"]}


def month_rmse(L: pd.DataFrame, values: np.ndarray) -> dict:
    lab, mon = L["label"].to_numpy(dtype=float), L["month"].to_numpy()
    return {m: float(np.sqrt(((lab[mon == m] - values[mon == m]) ** 2).mean())) for m in N.DEV_MONTHS}


def layer_rank_table(base: Path) -> pd.DataFrame:
    """tools/run_nbr_state.ranking_table with the values read from `base` (a full file in the template order) instead of the uploaded v70,
    with tools/run_nbr_dark.with_gate (January / July 2026)."""
    sub = pd.read_parquet(base)
    rec = pd.read_parquet(N.RECIPE, columns=[ID, "airport", "month", "scope", "v9"])
    raw = pd.read_parquet(ROOT / RANKING, columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"] == "DEP"][[ID, "MVT_TIME_UTC_mvt"]]
    k = sub.rename(columns={TARGET: "value"}).merge(rec, on=ID, how="left").merge(raw, on=ID, how="left")
    if len(k) != len(sub) or k[["v9", "MVT_TIME_UTC_mvt", "scope"]].isna().any().any():
        raise ValueError("ranking join incomplete")
    k["value"] = k["value"].astype(float)
    k["ts"] = (k["MVT_TIME_UTC_mvt"] - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds()
    k["date"] = k["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    k["matched"] = k["scope"].isin(MATCHED_SCOPES).to_numpy()
    k["fixed"] = k["scope"].eq("fixed").to_numpy() | k[ID].isin(N.GUARD).to_numpy()
    return ND.with_gate(k, RANK_MONTHS)


def layers_rank(base: Path, f: dict, out: Path, wd: float, version: int) -> dict:
    """tools/run_nbr_state.main's ranking part (all-month per-airport slopes, transfer guard, np.rint into the base's dtype) on `base`, written
    to out/state/; then tools/run_nbr_dark.main's ranking part (pooled all-month slope on dark rows, transfer guard, np.rint) on that file,
    written to out/. x is computed from the base's values in both layers, as in the two tools."""
    sub = pd.read_parquet(base)
    k = layer_rank_table(base)
    k["x"] = N.state(k)
    x = k["x"].to_numpy()
    matched, fixed, gate = k["matched"].to_numpy(dtype=bool), k["fixed"].to_numpy(dtype=bool), k["gate"].to_numpy(dtype=bool)
    unm = ~matched & ~fixed & ~np.isnan(x)
    kdk = matched & ~gate & ~fixed & ~np.isnan(x)
    rair, val = k["airport"].to_numpy(), k["value"].to_numpy(dtype=float)
    # nbr_state_v1 w75
    b_k = k["airport"].map(f["final"]).fillna(0.0).to_numpy(dtype=float)
    nk = N.apply(val, x, b_k, ND.W_STATE, unm)
    g1 = {}
    for ap in f["airports"]:
        sk = unm & (rair == ap)
        if ap not in f["final"] or not sk.any():
            continue
        rd = f["rd_state"][ap]
        rk = float(np.sqrt(((nk - val)[sk] ** 2).mean()))
        g1[ap] = {"rms_dev": rd, "rms_rank": rk, "ratio": rk / rd if rd > 0 else None, "rows_rank": int(sk.sum()),
                  "ok": bool(rd > 0 and 0.5 <= rk / rd <= 2.0)}
    ex1 = sorted(a for a, g in g1.items() if not g["ok"])
    nk2 = np.where(k["airport"].isin(ex1).to_numpy(), val, nk)
    res = sub.copy()
    newv = pd.Series(np.rint(nk2), index=k[ID].to_numpy()).reindex(sub[ID].to_numpy()).to_numpy()
    res[TARGET] = newv.astype(sub[TARGET].dtype)
    fixed_ids = set(k.loc[k["fixed"], ID])
    fx_sub = sub[ID].isin(fixed_ids).to_numpy()
    if (res[TARGET].to_numpy()[fx_sub] != sub[TARGET].to_numpy()[fx_sub]).any():
        raise ValueError("a fixed row moved (nbr_state layer)")
    p1 = out / "state" / f"vigorous-whistle_v{version}.parquet"
    p1.parent.mkdir(parents=True, exist_ok=True)
    if p1.exists():
        raise ValueError(f"{p1} exists.")
    res.to_parquet(p1, index=False)
    # nbr_dark_v1 dark_pool_w75 on the rinted nbr_state file
    s71 = pd.read_parquet(p1)
    ctl = s71.set_index(ID)[TARGET].reindex(k[ID]).to_numpy(dtype=float)
    new_k = ctl.copy()
    new_k[kdk] = np.maximum(new_k[kdk] + np.clip(wd * f["B"] * x[kdk], -N.CAP, N.CAP), N.FLOOR)
    g2 = {}
    for ap in f["airports"]:
        sk = kdk & (rair == ap)
        if ap not in f["rd_dark"] or not sk.any():
            continue
        rd = f["rd_dark"][ap]
        rk = float(np.sqrt(((new_k - ctl)[sk] ** 2).mean()))
        g2[ap] = {"rms_dev": rd, "rms_rank": rk, "ratio": rk / rd if rd > 0 else None, "rows_rank": int(sk.sum()),
                  "ok": bool(rd > 0 and 0.5 <= rk / rd <= 2.0)}
    ex2 = sorted(a for a, g in g2.items() if not g["ok"])
    final_k = np.where(k["airport"].isin(ex2).to_numpy(), ctl, new_k)
    res2 = s71.copy()
    res2[TARGET] = pd.Series(np.rint(final_k), index=k[ID].to_numpy()).reindex(s71[ID].to_numpy()).to_numpy().astype(s71[TARGET].dtype)
    if (res2[TARGET].to_numpy()[fx_sub] != sub[TARGET].to_numpy()[fx_sub]).any():
        raise ValueError("a fixed row moved (nbr_dark layer)")
    p2 = out / p1.name
    if p2.exists():
        raise ValueError(f"{p2} exists.")
    res2.to_parquet(p2, index=False)
    back = pd.read_parquet(p2)
    if not back.equals(res2):
        raise ValueError("The written file does not read back identically.")
    b, s, fv = sub[TARGET].to_numpy(dtype=float), s71[TARGET].to_numpy(dtype=float), back[TARGET].to_numpy(dtype=float)
    return {"base_sha256": sha256_file(base), "state_file": rel(p1), "state_sha256": sha256_file(p1),
            "file": rel(p2), "sha256": sha256_file(p2),
            "state_guard": g1, "state_excluded_airports": ex1, "dark_guard": g2, "dark_excluded_airports": ex2,
            "rows_unmatched_scope": int(unm.sum()), "rows_dark": int(kdk.sum()),
            "state_vs_base": {"rows_changed": int((s != b).sum()), "squared_change": float(((s - b) ** 2).sum())},
            "dark_vs_state": {"rows_changed": int((fv != s).sum()), "squared_change": float(((fv - s) ** 2).sum())},
            "final_vs_base": {"rows_changed": int((fv != b).sum()), "squared_change": float(((fv - b) ** 2).sum())},
            "fixed_rows_unchanged": True, "template_order_same": bool(back[ID].equals(sub[ID]))}


def dev_layer_fit(proxy: Path, wd: float) -> tuple[pd.DataFrame, dict]:
    """The layers' development fit on a stored development proxy's value column (the off path's or sn65's)."""
    L = layer_dev_table()
    p = pd.read_parquet(proxy, columns=[ID, "value"])
    return L, layers_dev(L, aligned(p["value"].to_numpy(), p[ID].to_numpy(), L), wd)


# ---------------------------------------------------------------- identity (development)

def dev_identity() -> None:
    if IDENT.exists():
        raise ValueError("Preserve the identity record.")
    c = dev_context()
    dev, ctl = c["dev"], c["ctl"]
    rec = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "control_base": CONTROL[0],
           "control_base_sha256": sha256_file(ROOT / CONTROL[0]), "official_scores_used": "none", "audit_used": False,
           "code": {p: sha256_file(ROOT / p) for p in CODE}, "platform": sys.platform, "bound_s": LINUX_BOUND, "rmse_tolerance": RMSE_TOL}
    # the columns-off per-seed predictions against stack_queue_v1's (identical to stack_weight_v1's)
    seeds = {}
    for m in SW.MONTHS:
        ref = ROOT / STACK_QUEUE_DEV_OFF.format(m=m)
        if ref.exists():
            a, b = pd.read_parquet(OUT / "dev" / "off" / f"preds_{m}.parquet"), pd.read_parquet(ref)
            cols = [f"{kk}_{i}" for kk in SW.LEARNERS for i in range(10)] + ["fx", "h2", "h3", "ho"]
            seeds[m] = (float(np.abs(a[cols].to_numpy(dtype=float) - b[cols].to_numpy(dtype=float)).max()) if a[ID].equals(b[ID])
                        else "row order differs")
    rec["off_preds_vs_stack_queue_off_max_abs"] = seeds
    print("columns-off per-seed predictions vs stack_queue_v1's:", seeds, flush=True)
    c66_off, c70_off, _, _, path_off = through_chain(c, "off", "off_w65_bag10")
    ident = float(np.abs(c70_off - ctl["value"].to_numpy(dtype=float)).max())
    ident69 = float(np.abs(c66_off - c["v69p"]["value"].to_numpy(dtype=float)).max())
    rec["columns_off_vs_control_base_max_abs_s"] = ident
    rec["columns_off_v69like_vs_v69_proxy_max_abs_s"] = ident69
    rec["off_proxy_sha256"] = sha256_file(path_off)
    hold_base = bool(ident == 0.0 or ident <= LINUX_BOUND)
    print("identity: columns off vs v70 dev proxy max |diff|", ident, "| v69-like vs v69 proxy", ident69, flush=True)
    # the control (v71's development values) and the off path with both layers
    wd = w_dark()
    L = layer_dev_table()
    lc = layers_dev(L, L["value"].to_numpy(dtype=float), wd)
    lo = layers_dev(L, aligned(c70_off, dev[ID].to_numpy(), L), wd)
    state_res, dark_res = read_json(STATE_RESULTS), read_json(DARK_RESULTS)
    want = {m: float(dark_res["dev"][m]["rmse"]) for m in N.DEV_MONTHS}
    want_state = {m: float(state_res["candidates"]["w75"]["dev"][m]["rmse"]) for m in N.DEV_MONTHS}
    rc, ro = month_rmse(L, lc["dark"]), month_rmse(L, lo["dark"])
    rcs, ros = month_rmse(L, lc["state"]), month_rmse(L, lo["state"])
    rec["control"] = {"rmse": rc, "dark_results_rmse": want, "max_abs_rmse_diff": max(abs(rc[m] - want[m]) for m in want),
                      "state_rmse": rcs, "state_results_rmse": want_state, "state_max_abs_rmse_diff": max(abs(rcs[m] - want_state[m]) for m in want),
                      "slopes_final_max_abs_diff": max(abs(lc["final"][a] - state_res["slopes_final"][a]) for a in state_res["slopes_final"]),
                      "dark_slope_final_diff": abs(lc["B"] - float(dark_res["slope_final"])),
                      "dark_slopes_blocked_max_abs_diff": max(abs(lc["blocked_dark"][m] - float(dark_res["slopes_blocked"][m])) for m in N.DEV_MONTHS),
                      "fit": fit_summary(lc)}
    rec["off_with_layers"] = {"rmse": ro, "max_abs_rmse_diff": max(abs(ro[m] - want[m]) for m in want),
                              "state_rmse": ros, "state_max_abs_rmse_diff": max(abs(ros[m] - want_state[m]) for m in want),
                              "values_vs_control_max_abs_s": float(np.abs(lo["dark"] - lc["dark"]).max()),
                              "slopes_final_max_abs_diff": max(abs(lo["final"][a] - state_res["slopes_final"][a]) for a in state_res["slopes_final"]),
                              "dark_slope_final_diff": abs(lo["B"] - float(dark_res["slope_final"]))}
    hold_ctl = bool(rec["control"]["max_abs_rmse_diff"] <= RMSE_TOL and rec["control"]["state_max_abs_rmse_diff"] <= RMSE_TOL
                    and rec["control"]["slopes_final_max_abs_diff"] <= 1e-12)
    hold_off = bool(rec["off_with_layers"]["max_abs_rmse_diff"] <= RMSE_TOL and rec["off_with_layers"]["state_max_abs_rmse_diff"] <= RMSE_TOL)
    rec["holds"] = {"columns_off_vs_control_base": hold_base, "control_reproduces_dark_results": hold_ctl,
                    "off_with_layers_reproduces_dark_results": hold_off, "all": bool(hold_base and hold_ctl and hold_off)}
    print("control RMSE vs dark_results", rec["control"]["max_abs_rmse_diff"], "| off + layers", rec["off_with_layers"]["max_abs_rmse_diff"],
          "| off + layers values vs control", rec["off_with_layers"]["values_vs_control_max_abs_s"], "| holds", rec["holds"], flush=True)
    write_json(IDENT, rec)
    if not rec["holds"]["all"]:
        raise ValueError("identity: the columns-off path does not reproduce the control; nothing is read.")


# ---------------------------------------------------------------- development reading

def dev_score() -> None:
    import run_intruder_bound as R
    import run_stack_resid as S
    if RESULTS.exists():
        raise ValueError("Preserve the reading.")
    ident, rident = read_json(IDENT), read_json(RANK_IDENT)
    if not ident["holds"]["all"] or not rident["holds"]:
        raise ValueError("An identity does not hold; nothing is read.")
    c = dev_context()
    dev = c["dev"]
    res = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "control": "v71's development values: "
           + CONTROL[0] + " with nbr_state_v1 w75 and nbr_dark_v1 dark_pool_w75 (tools/run_nbr_state.py / run_nbr_dark.py arithmetic)",
           "control_base_sha256": sha256_file(ROOT / CONTROL[0]), "official_scores_used": "none", "audit_used": False,
           "code": {p: sha256_file(ROOT / p) for p in CODE}, "nbr_columns": NC.COLS, "weight": WEIGHT, "seeds": SW.SEEDS10,
           "identity": ident, "consistency": read_json(OUT / "consistency.json")}
    _, c70, use, p, path_on = through_chain(c, "on", "sn65")
    wd = w_dark()
    L = layer_dev_table()
    lc = layers_dev(L, L["value"].to_numpy(dtype=float), wd)
    ctl71 = lc["dark"]
    rc = month_rmse(L, ctl71)
    if max(abs(rc[m] - ident["control"]["dark_results_rmse"][m]) for m in N.DEV_MONTHS) > RMSE_TOL:
        raise ValueError("The control does not reproduce v71's development RMSEs.")
    sn65 = aligned(c70, dev[ID].to_numpy(), L)
    ls = layers_dev(L, sn65, wd)
    cands = {"sn65": sn65, "sn65_layers": ls["dark"]}
    d = OUT / "dev" / "sn65_layers"
    d.mkdir(parents=True, exist_ok=True)
    out = L[[ID, "month", "airport", "label", "cls", "day"]].copy()
    out["value"] = ls["dark"]
    out["state"] = ls["state"]
    out["sn65"] = sn65
    out.to_parquet(d / "proxy.parquet", index=False)
    ctlp = L[[ID, "month", "airport", "label", "cls", "day"]].assign(value=ctl71)
    (OUT / "dev" / "control_v71").mkdir(parents=True, exist_ok=True)
    ctlp.to_parquet(OUT / "dev" / "control_v71" / "proxy.parquet", index=False)
    res["control_rmse"] = rc
    res["control_proxy_sha256"] = sha256_file(OUT / "dev" / "control_v71" / "proxy.parquet")
    res["layers_dev"] = {"control": fit_summary(lc), "sn65": fit_summary(ls)}
    air, month = L["airport"].to_numpy(), L["month"].to_numpy()
    fxm = L["fixed"].to_numpy(dtype=bool)
    tt = L[[ID, "month", "label", "day"]].assign(ctl=ctl71)
    res["candidates"] = {}
    for name, cand in cands.items():
        rng = np.random.default_rng(20261004)
        months = {}
        for m in SW.MONTHS:
            sel = (tt["month"] == m).to_numpy()
            s = tt[sel]
            pct, lo, hi = R.interval(s, "ctl", cand[sel], rng)
            ch = cand[sel] - s["ctl"].to_numpy()
            months[m] = {"control_rmse": float(np.sqrt(np.mean((s["label"] - s["ctl"]) ** 2))),
                         "rmse": float(np.sqrt(np.mean((s["label"].to_numpy() - cand[sel]) ** 2))),
                         "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum()),
                         "rms_change_s": float(np.sqrt(np.mean(ch ** 2))), "max_abs_change_s": float(np.abs(ch).max())}
        eq = float(np.mean([months[m]["pct"] for m in SW.EQUAL]))
        checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= 0.005 for m in SW.MONTHS), "2_equal_footing_mean_pct": eq,
                  "2_mean_ok": eq <= -0.02, "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in SW.EQUAL)}
        diff = cand - ctl71
        by = pd.DataFrame({"airport": air, "month": month, "c": (ctl71 - L["label"].to_numpy()) ** 2,
                           "n": (cand - L["label"].to_numpy()) ** 2}).groupby(["airport", "month"]).sum()
        res["candidates"][name] = {"weight": WEIGHT, "seeds": SW.SEEDS10, "layers_after": name == "sn65_layers", "months": months,
                                   "checks_1_to_3": checks,
                                   "clauses_1_to_3": bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"]
                                                          and checks["3_every_equal_interval_below_zero"]),
                                   "airport_month_change_pct": {f"{a} {m}": float(100 * (np.sqrt(r["n"]) / np.sqrt(r["c"]) - 1)) for (a, m), r in by.iterrows()},
                                   "fixed_rows_moved": int((np.abs(diff)[fxm] > 1e-9).sum()),
                                   "proxy_sha256": sha256_file(path_on if name == "sn65" else d / "proxy.parquet")}
        print(name, " ".join(f"{m[5:]} {months[m]['pct']:+.4f}% [{months[m]['lo']:+.4f},{months[m]['hi']:+.4f}]" for m in SW.MONTHS),
              f"| eq {eq:+.4f}%", "clauses 1-3", res["candidates"][name]["clauses_1_to_3"], flush=True)
    # development side of clause 4 (out-of-month guarded predictions of the columns-on bags, rows where each bag applies)
    fxd = S.fixed_rows(dev)
    dair = dev["airport"].to_numpy()
    lirf = dair == "LIRF"
    res["dev_oof_prediction_by_airport"] = {
        "v3": S.by_airport_rms(dev, np.where(~lirf & ~fxd, p["p3"], 0.0), ~(~lirf & ~fxd)),
        "v2_LIRF": S.by_airport_rms(dev, np.where(lirf & ~fxd, p["p2"], 0.0), ~(lirf & ~fxd)),
        "opdi_LSZH_used": S.by_airport_rms(dev, np.where(use, p["po"], 0.0), ~use)}
    write_json(RESULTS, res)


# ---------------------------------------------------------------- ranking

def rank_fit(nbr: str) -> None:
    import run_stack_resid as S
    path = OUT / "rank" / nbr / "preds.parquet"
    if path.exists():
        raise ValueError(f"{path} exists.")
    dev, res, fx, mats, rk, fxr, h2, h3, ho, Por = SW.rank_inputs()
    info = {}
    if nbr == "on":
        Qd, Qr = nbr_dev(dev), nbr_rank(rk)
        mats = [(k, with_nbr(X, Qd), with_nbr(Xr, Qr)) for k, X, Xr in mats]
        info = {"nbr_coverage_dev": coverage(Qd), "nbr_coverage_rank": coverage(Qr)}
        print("neighbour columns:", info, flush=True)
    dom = dev["dom"].to_numpy()
    out = pd.DataFrame({ID: rk[ID].to_numpy(), "airport": rk["airport"].to_numpy(), "new": rk["new"].to_numpy(dtype=float), "fx": fxr,
                        "h2": h2, "h3": h3, "ho": ho, "o_has": Por["o_has"].to_numpy()})
    trees, ncol = {}, {}
    for key, X, Xr in mats:
        Xa, Xra = X.to_numpy(np.float32), Xr.to_numpy(np.float32)
        ncol[key] = int(Xa.shape[1])
        if list(X.columns) != list(Xr.columns):
            raise ValueError("Development and ranking columns differ.")
        for i, seed in enumerate(SW.SEEDS10):
            S.SEEDS = [seed]
            models = S.fit_bag(Xa, res, ~fx, dom)
            out[f"{key}_{i}"] = models[0].predict(Xra, num_iteration=models[0].best_iteration)
            trees[f"{key} {seed}"] = int(models[0].best_iteration)
            print("rank", nbr, key, ncol[key], seed, "trees", trees[f"{key} {seed}"], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "nbr": nbr, "columns": ncol, "rows": int(len(out)), "trees": trees,
                                           **info, "sha256": sha256_file(path), "code": {p: sha256_file(ROOT / p) for p in CODE}})


def step11_rank(base: Path, out_path: Path) -> dict:
    """Step 11 exactly as tools/run_oooi_pin.py's ranking part (selected w50): offsets from July / November / December 2025 on the v69
    development proxy, LFPG in-scope rows of the base file, np.rint into the file's dtype."""
    import run_intruder_bound as R
    import run_oooi_pin as P
    import run_stack_resid as S
    if sha256_file(P.PROXY) != P.PROXY_SHA:
        raise ValueError("The v69 development proxy differs.")
    pin_res = read_json(ROOT / "docs/oooi_pin/results_v1.json")
    for k, f in [("dev", "joined_dev.parquet"), ("rank", "joined_rank.parquet")]:
        if sha256_file(P.RUN / f) != pin_res["joined_sha256"][k]:
            raise ValueError(f"runs/oooi_pin/{f} changed since oooi_pin_v1's governing run.")
    pin_dev = P.scope(pd.read_parquet(P.RUN / "joined_dev.parquet"))
    pin_rk = P.scope(pd.read_parquet(P.RUN / "joined_rank.parquet"))
    dev = pd.read_parquet(P.PROXY, columns=[ID, "month", "airport", "label", "cls", "day", "value"])
    dev = dev.merge(P.takeoff_days(P.DEV_MONTHS), on=ID, how="left", validate="one_to_one")
    dev["day"] = dev["tday"]
    dev = dev.merge(pin_dev[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    fixed = S.fixed_rows(dev)
    dev["in_scope"] = dev["raw"].notna().to_numpy() & ~fixed & dev["airport"].eq("LFPG").to_numpy()
    off_all = P.offsets(dev[dev["in_scope"] & dev["month"].isin(P.PIN_DEV_MONTHS)])
    v69 = pd.read_parquet(base)
    rk = pd.read_parquet(P.RANKING, columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    if list(v69.columns) != [ID, "TAXITIME_SEC_mvt"] or len(v69) != len(rk):
        raise ValueError("The base is not a full submission.")
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    rfixed_ids = set(rk.loc[pop, ID]) | set(R.GUARD) | set(rk.loc[rk["ADEP_mvt"].eq("LIRF"), ID])
    rkt = v69.merge(rk[[ID, "ADEP_mvt"]], on=ID, how="left", validate="one_to_one")
    rkt = rkt.merge(pin_rk[[ID, "source", "raw", "month_dep"]], on=ID, how="left", validate="one_to_one")
    rkt["in_scope"] = rkt["raw"].notna() & rkt["ADEP_mvt"].eq("LFPG") & ~rkt[ID].isin(rfixed_ids)
    rkt["pin"] = np.where(rkt["in_scope"], rkt["raw"] + rkt["source"].map(off_all), np.nan)
    rkt["chain"] = rkt["TAXITIME_SEC_mvt"].astype(float)
    rs = rkt["in_scope"].to_numpy()
    rv = rkt["chain"].to_numpy().copy()
    rv[rs] = P.cand_value(0.5, rkt.loc[rs, "pin"].to_numpy(), rkt.loc[rs, "chain"].to_numpy())
    new_int = np.rint(rv).astype(v69["TAXITIME_SEC_mvt"].dtype)
    out_df = v69.copy()
    out_df["TAXITIME_SEC_mvt"] = pd.Series(new_int, index=v69.index).astype(v69["TAXITIME_SEC_mvt"].dtype)
    if not out_df[ID].equals(v69[ID]):
        raise ValueError("Row order changed.")
    changed = out_df["TAXITIME_SEC_mvt"].ne(v69["TAXITIME_SEC_mvt"])
    fixed_r = v69[ID].isin(rfixed_ids)
    if (changed & fixed_r).any() or (changed & ~rkt["in_scope"]).any():
        raise ValueError("A row outside the scope, a LIRF, population or guard row moved.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise ValueError(f"{out_path} exists.")
    out_df.to_parquet(out_path, index=False)
    if not pd.read_parquet(out_path).equals(out_df):
        raise ValueError("The written file does not read back identically.")
    rec = {"base_sha256": sha256_file(base), "sha256": sha256_file(out_path), "rows_in_scope": int(rs.sum()), "rows_changed": int(changed.sum()),
           "offsets_jul_nov_dec": {k: float(v) for k, v in off_all.items()}}
    write_json(out_path.with_suffix(".receipt.json"), rec)
    return rec


def steps_6_to_11(P: pd.DataFrame, out: Path, version: int) -> tuple[Path, Path, np.ndarray]:
    """Step 6 at weight 0.65 with ten seeds (run_stack_weight.rank_value), steps 7-10 (chain_from_stack, own_motion_v2), step 11."""
    import run_own_motion_v2 as O2
    import run_stack_resid_v3 as V3
    from prc2026_submission.contract import export_submission
    values, use, _ = SW.rank_value(P, WEIGHT, K)
    st = out / f"steps/stack/vigorous-whistle_v{version}.parquet"
    export_submission(pd.DataFrame({ID: P[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", st,
                      {"dry_run": True, "step": f"stack_nbr_v1: stacking value at weight {WEIGHT} with {K} seeds", "official_scores_used": "none",
                       "audit_used": False})
    v63like = V3.chain_from_stack(st, version, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), version, out / "steps/om2")
    v69like = out / f"steps/om2/vigorous-whistle_v{version}.parquet"
    step11_rank(v69like, out / f"steps/pin/vigorous-whistle_v{version}.parquet")
    return v69like, out / f"steps/pin/vigorous-whistle_v{version}.parquet", use


def rank_identity() -> None:
    out = OUT / "rank" / "identity_v70"
    if out.exists():
        raise ValueError(f"{out} exists.")
    if not read_json(IDENT)["holds"]["all"]:
        raise ValueError("The development identity does not hold.")
    P = pd.read_parquet(OUT / "rank" / "off" / "preds.parquet")
    rec = {"created_at_utc": utc_now(), "code": {p: sha256_file(ROOT / p) for p in CODE}}
    ref = ROOT / STACK_WEIGHT_RANK_PREDS[0]
    if ref.exists() and sha256_file(ref) == STACK_WEIGHT_RANK_PREDS[1]:
        R0 = pd.read_parquet(ref)
        cols = [f"{k}_{i}" for k in SW.LEARNERS for i in range(10)]
        rec["columns_off_preds_vs_stack_weight_preds_max_abs"] = (float(np.abs(P[cols].to_numpy() - R0[cols].to_numpy()).max())
                                                                  if P[ID].equals(R0[ID]) else "row order differs")
        print("columns-off ranking predictions vs stack_weight's (8083ffcf):", rec["columns_off_preds_vs_stack_weight_preds_max_abs"], flush=True)
    v69like, final, use = steps_6_to_11(P, out, 70)
    rec.update({"v69like_sha256": sha256_file(v69like), "reproduces_v69": sha256_file(v69like) == V69, "final_sha256": sha256_file(final),
                "reproduces_v70": sha256_file(final) == V70, "opdi_rows_used": int(use.sum())})
    print({k: rec[k] for k in ["reproduces_v69", "reproduces_v70", "final_sha256"]}, flush=True)
    rec["holds"] = False
    if rec["reproduces_v70"]:
        # both layers on the off path's file, slopes from the off path's development values
        _, f = dev_layer_fit(OUT / "dev" / "off_w65_bag10" / "proxy.parquet", w_dark())
        lr = layers_rank(final, f, out / "layers", w_dark(), 71)
        rec["layers"] = lr
        rec["reproduces_v71_state"] = lr["state_sha256"] == V71_STATE
        rec["reproduces_v71"] = lr["sha256"] == V71
        rec["holds"] = bool(rec["reproduces_v70"] and rec["reproduces_v71"])
        print("layers on the off path:", {k: lr[k] for k in ["state_sha256", "sha256", "state_excluded_airports", "dark_excluded_airports"]},
              "reproduces v71", rec["reproduces_v71"], flush=True)
    write_json(out / "identity.json", rec)
    if not rec["holds"]:
        raise ValueError("identity: the ranking path with the columns off does not reproduce v70 / v71; no candidate ranking file.")


def rank_check() -> None:
    import run_stack_resid as S
    res = read_json(RESULTS)
    ident = read_json(RANK_IDENT)
    res["rank_identity"] = ident
    if not ident["holds"]:
        write_json(RESULTS, res)
        raise ValueError("The ranking identity failed.")
    P = pd.read_parquet(OUT / "rank" / "on" / "preds.parquet")
    fxr, lirf = P["fx"].to_numpy(dtype=bool), P["airport"].eq("LIRF").to_numpy()
    _, use, p = SW.rank_value(P, WEIGHT, K)
    rk = {"v3": S.by_airport_rms(P, np.where(~lirf & ~fxr, p["p3"], 0.0), ~(~lirf & ~fxr)),
          "v2_LIRF": S.by_airport_rms(P, np.where(lirf & ~fxr, p["p2"], 0.0), ~(lirf & ~fxr)),
          "opdi_LSZH_used": S.by_airport_rms(P, np.where(use, p["po"], 0.0), ~use)}
    dv = res["dev_oof_prediction_by_airport"]
    ratios = {f"{part} {a}": rk[part][a]["rms_s"] / dv[part][a]["rms_s"] for part in rk for a in rk[part] if a in dv[part]}
    ok = all(0.5 <= r <= 2.0 for r in ratios.values())
    res["clause_4_transfer_guard"] = {"ranking_prediction_by_airport": rk, "ratios": ratios, "ok": bool(ok)}
    print("clause 4", {k: round(r, 3) for k, r in ratios.items()}, "ok", ok, flush=True)
    # sn65's ranking chain (label-free; not a candidate file) and the layers' transfer guards on it, for the record and for sn65_layers
    oc = OUT / "rank" / "on_chain"
    if oc.exists():
        raise ValueError(f"{oc} exists.")
    _, pinned, use_r = steps_6_to_11(P, oc, VERSION)
    wd = w_dark()
    _, f = dev_layer_fit(OUT / "dev" / "sn65" / "proxy.parquet", wd)
    if any(abs(f["final"][a] - res["layers_dev"]["sn65"]["final"][a]) > 0 for a in f["final"]) or f["B"] != res["layers_dev"]["sn65"]["B"]:
        raise ValueError("The layers' development fit on sn65 does not reproduce dev_score's.")
    lr = layers_rank(pinned, f, oc / "layers", wd, VERSION)
    res["ranking_chain_sn65"] = {"note": "label-free ranking chain of sn65 (steps 6-11) in runs/stack_nbr/rank/on_chain/, used for the layers' "
                                         "transfer guards; not a candidate file", "file": pinned.relative_to(ROOT).as_posix(),
                                 "sha256": sha256_file(pinned), "opdi_rows_used": int(use_r.sum())}
    res["ranking_layers_sn65_layers"] = lr
    print("layers on sn65's ranking chain: state excluded", lr["state_excluded_airports"], "dark excluded", lr["dark_excluded_airports"],
          {a: round(g["ratio"], 3) for a, g in lr["state_guard"].items()}, {a: round(g["ratio"], 3) for a, g in lr["dark_guard"].items()}, flush=True)
    sel, best = None, None
    for name in CANDIDATES:
        cnd = res["candidates"][name]
        acc = bool(cnd["clauses_1_to_3"] and ok)
        cnd["clause_4"] = bool(ok)
        cnd["accepted"] = acc
        e = cnd["checks_1_to_3"]["2_equal_footing_mean_pct"]
        if acc and (best is None or e < best):
            sel, best = name, e
    res["selected"] = sel
    write_json(RESULTS, res)
    print("selected", sel, flush=True)


def rank_dry() -> None:
    import run_intruder_bound as R
    res = read_json(RESULTS)
    sel = res.get("selected")
    if sel not in CANDIDATES:
        raise ValueError("No candidate is accepted; no ranking file.")
    out = OUT / "rank" / f"dry_v{VERSION}"
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite a dry run.")
    P = pd.read_parquet(OUT / "rank" / "on" / "preds.parquet")
    _, pinned, use = steps_6_to_11(P, out, VERSION)
    if sha256_file(pinned) != res["ranking_chain_sn65"]["sha256"]:
        raise ValueError("sn65's ranking chain does not reproduce rank_check's.")
    src, lr = pinned, None
    if sel == "sn65_layers":
        wd = w_dark()
        _, f = dev_layer_fit(OUT / "dev" / "sn65" / "proxy.parquet", wd)
        lr = layers_rank(pinned, f, out / "layers", wd, VERSION)
        if lr["sha256"] != res["ranking_layers_sn65_layers"]["sha256"]:
            raise ValueError("The layered file does not reproduce rank_check's.")
        src = ROOT / lr["file"]
    final = out / f"vigorous-whistle_v{VERSION}.parquet"
    shutil.copyfile(src, final)
    ref71 = OUT / "rank" / "identity_v70" / "layers" / "vigorous-whistle_v71.parquet"
    ref70 = OUT / "rank" / "identity_v70" / "steps/pin/vigorous-whistle_v70.parquet"
    if sha256_file(ref71) != V71 or sha256_file(ref70) != V70:
        raise ValueError("The v70 / v71 references differ.")
    v, v71, v70 = pd.read_parquet(final), pd.read_parquet(ref71), pd.read_parquet(ref70)
    if not v[ID].equals(v71[ID]):
        raise ValueError("Row order differs from v71.")
    rec = pd.read_parquet(N.RECIPE, columns=[ID, "airport", "month", "scope"]).set_index(ID).reindex(v[ID])
    fixed = rec["scope"].eq("fixed").to_numpy()
    guard = v[ID].isin(R.GUARD).to_numpy()
    d71 = v[TARGET].to_numpy(dtype=float) - v71[TARGET].to_numpy(dtype=float)
    d70 = v[TARGET].to_numpy(dtype=float) - v70[TARGET].to_numpy(dtype=float)
    by = pd.DataFrame({"airport": rec["airport"].to_numpy(), "month": rec["month"].astype(str).to_numpy(), "d": d71})
    rep = {"created_at_utc": utc_now(), "candidate": sel, "file": final.relative_to(ROOT).as_posix(), "sha256": sha256_file(final),
           "bytes": final.stat().st_size, "opdi_rows_used_ranking": int(use.sum()), "layers": lr,
           "vs_v71": {"rows_changed": int((d71 != 0).sum()), "squared_change": float((d71 ** 2).sum()), "rms_change_s": float(np.sqrt((d71 ** 2).mean())),
                      "max_up_s": float(d71.max()), "max_down_s": float(d71.min()), "mean_change_s": float(d71.mean())},
           "vs_v70": {"rows_changed": int((d70 != 0).sum()), "squared_change": float((d70 ** 2).sum())},
           "lirf_population_rows_changed_vs_v71": int(((d71 != 0) & fixed).sum()), "guard_rows_changed_vs_v71": int(((d71 != 0) & guard).sum()),
           "min_value": int(v[TARGET].min()), "dtype": str(v[TARGET].dtype),
           "by_airport": {a: {"rows_changed": int((g["d"] != 0).sum()), "squared_change": float((g["d"] ** 2).sum()),
                              "max_abs_s": float(g["d"].abs().max())} for a, g in by.groupby("airport")},
           "official_scores_used": "none"}
    if rep["lirf_population_rows_changed_vs_v71"] or rep["guard_rows_changed_vs_v71"] or rep["min_value"] < -12:
        raise ValueError("A LIRF population or guard row changed against v71, or a value is below -12.")
    write_json(out / f"vigorous-whistle_v{VERSION}.report.json", rep)
    res["ranking_dry_run"] = {k: rep[k] for k in ["candidate", "file", "sha256", "vs_v71", "vs_v70", "lirf_population_rows_changed_vs_v71",
                                                  "guard_rows_changed_vs_v71", "min_value", "dtype"]}
    write_json(RESULTS, res)
    print({k: rep[k] for k in ["sha256", "vs_v71", "vs_v70", "lirf_population_rows_changed_vs_v71", "guard_rows_changed_vs_v71", "min_value"]},
          flush=True)


def verify_dry() -> None:
    """Workstation: the dry run against the uploaded v71 / v70 (read only), the recipe's fixed scope and the guard rows."""
    import hashlib
    final = OUT / "rank" / f"dry_v{VERSION}" / f"vigorous-whistle_v{VERSION}.parquet"
    up71, up70 = ROOT / "runs/submissions/vigorous-whistle_v71.parquet", ROOT / "runs/submissions/vigorous-whistle_v70.parquet"
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    if h(up71) != V71 or h(up70) != V70:
        raise ValueError("The uploaded v71 / v70 files differ from their recorded hashes.")
    v, v71, v70 = pd.read_parquet(final), pd.read_parquet(up71), pd.read_parquet(up70)
    tmpl = pd.read_parquet(ROOT / "data/raw/submitting.parquet")
    rec = pd.read_parquet(N.RECIPE, columns=[ID, "scope"]).set_index(ID)["scope"].reindex(v[ID])
    a, b, c = v[TARGET].to_numpy(dtype=np.int64), v71[TARGET].to_numpy(dtype=np.int64), v70[TARGET].to_numpy(dtype=np.int64)
    fixed = rec.eq("fixed").to_numpy()
    guard = v[ID].isin([213142471, 213142469]).to_numpy()
    out = {"created_at_utc": utc_now(), "file": final.relative_to(ROOT).as_posix(), "sha256": h(final), "bytes": final.stat().st_size,
           "columns": list(v.columns), "dtype": str(v[TARGET].dtype), "rows": int(len(v)),
           "template_order_equals_v71": bool(v[ID].equals(v71[ID])), "template_order_equals_template": bool(v[ID].equals(tmpl[ID])),
           "rows_changed_vs_v71": int((a != b).sum()), "squared_change_vs_v71": float(((a - b) ** 2).sum()),
           "rows_changed_vs_v70": int((a != c).sum()), "squared_change_vs_v70": float(((a - c) ** 2).sum()),
           "fixed_rows": int(fixed.sum()), "fixed_rows_changed_vs_v71": int(((a != b) & fixed).sum()),
           "guard_rows_found": int(guard.sum()), "guard_rows_changed_vs_v71": int(((a != b) & guard).sum()),
           "guard_values": {str(int(i)): int(x) for i, x in zip(v.loc[guard, ID], v.loc[guard, TARGET])},
           "min_value": int(a.min()), "min_ok": bool(a.min() >= -12), "nan": int(v[TARGET].isna().sum())}
    out["ok"] = bool(out["template_order_equals_v71"] and out["fixed_rows_changed_vs_v71"] == 0 and out["guard_rows_changed_vs_v71"] == 0
                     and out["guard_rows_found"] == 2 and out["min_ok"] and out["dtype"] == "int32" and out["nan"] == 0)
    write_json(final.parent / "verification_workstation.json", out)
    print(json.dumps(out, indent=1))


def warm(part: str) -> None:
    """VM preparation only (no fit, no score): open every input once, in one process, so that the lazily fetched files arrive before
    parallel jobs read them. 'fit' also writes the declaration's input consistency report (runs/stack_nbr/consistency.json)."""
    if part == "fit":
        dev = SW.dev_inputs()[0]
        Qd = nbr_dev(dev)
        ri = SW.rank_inputs()
        dev2, rk = ri[0], ri[4]
        nbr_dev(dev2)
        Qr = nbr_rank(rk)
        rep = consistency(dev, dev2, rk)
        rep["nbr_coverage_dev"], rep["nbr_coverage_rank"] = coverage(Qd), coverage(Qr)
        rep["created_at_utc"] = utc_now()
        write_json(OUT / "consistency.json", rep)
        print("warm fit: development", len(dev), "ranking", len(rk), json.dumps(rep), flush=True)
        if rep["material_difference"]:
            raise ValueError("The stacker tables' matched flag or v9 differ materially from the layers' inputs; stop.")
    else:
        import run_ctx_ownpoints_chain as CH
        import run_opdi_stack as O
        import run_own_motion_v2 as O2
        import run_stack_resid_v3 as V3
        if sha256_file(ROOT / CONTROL[0]) != CONTROL[1]:
            raise ValueError("The control differs from the declared file.")
        CH.dev_table_from(O.SHIP / "ctl.parquet", O.SHIP / "rows_all.parquet", O.SHIP / "dmp.parquet")
        pd.read_parquet(O.SHIP / "cand.parquet")
        O2.O.dev_table()
        step11_dev_pins()
        V3.Chain()
        O2.v62_tables()
        O2.triggers_2025(O2.IDS_DIR, use_cache=True)
        layer_dev_table()
        pd.read_parquet(N.RECIPE)
        ND.gates(RANK_MONTHS)
        read_json(STATE_RESULTS), read_json(DARK_RESULTS), read_json(DARK_DECL)
        for m in SW.MONTHS:
            ref = ROOT / STACK_QUEUE_DEV_OFF.format(m=m)
            if ref.exists():
                pd.read_parquet(ref, columns=[ID])
        ref = ROOT / STACK_WEIGHT_RANK_PREDS[0]
        if ref.exists():
            sha256_file(ref)
        print("warm chain: done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm", choices=["fit", "chain"])
    ap.add_argument("--dev-month")
    ap.add_argument("--nbr", choices=["off", "on"])
    ap.add_argument("--dev-identity", action="store_true")
    ap.add_argument("--dev-score", action="store_true")
    ap.add_argument("--rank-fit", action="store_true")
    ap.add_argument("--rank-identity", action="store_true")
    ap.add_argument("--rank-check", action="store_true")
    ap.add_argument("--rank-dry", action="store_true")
    ap.add_argument("--verify-dry", action="store_true")
    a = ap.parse_args()
    if (a.dev_month or a.rank_fit) and not a.nbr:
        ap.error("--nbr off|on is required with --dev-month / --rank-fit")
    if a.warm:
        warm(a.warm)
    if a.dev_month:
        dev_month(a.dev_month, a.nbr)
    if a.dev_identity:
        dev_identity()
    if a.rank_fit:
        rank_fit(a.nbr)
    if a.rank_identity:
        rank_identity()
    if a.dev_score:
        dev_score()
    if a.rank_check:
        rank_check()
    if a.rank_dry:
        rank_dry()
    if a.verify_dry:
        verify_dry()


if __name__ == "__main__":
    main()
