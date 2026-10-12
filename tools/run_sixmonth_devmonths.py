"""configs/sixmonth_devmonths_v1.json: the development-month reading of final candidate 4's six-month C6 stacker and C14 second pass.

For each development month m, the learner fitted on the other three development months (a) and the same plus Tier B's February / June
2025 rows (b), predicting month m. Fitting code, guards and value arithmetic are tools/run_sixmonth.py's and the chain's, unchanged.

  fit --layer stack|ss --month M --variant a|b   one fit set -> runs/sixmonth/devmonths/<layer>_<M>_<variant>.parquet (+ .json); Azure VM
  read                                          the reading and the decision -> docs/sixmonth/devmonths_results_v1.json (refuses unless the
                                                declaration is committed and unchanged)
  splice                                        only if the decision splices a month: runs/final_blind/final_candidate5/chain_out_cap.parquet

No official score enters. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
import run_sixmonth as X  # noqa: E402

ID, TARGET = X.ID, X.TARGET
DECLARATION = "configs/sixmonth_devmonths_v1.json"
OUT = ROOT / "runs/sixmonth/devmonths"
RESULTS = ROOT / "docs/sixmonth/devmonths_results_v1.json"
WORK = ROOT / "runs/febjun_tierb/chain/work"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
ANALOGUE = {"2026-01": "2025-01", "2026-07": "2025-07"}
SS_CANDIDATE = "sp_hicov7"
READ_SEED = 20261004
C3 = ("runs/final_blind/final_candidate3/chain_out_cap.parquet", "b673791dfdaf8e909175001387183946236075280c9b3457822b52157c9cebc8")
C4 = ("runs/final_blind/final_candidate4/chain_out_cap.parquet", "8fc8828d1f5338bda08972f16af61c0f5f288b5d50992c1750d4d768d89b0a42")
C5 = ROOT / "runs/final_blind/final_candidate5/chain_out_cap.parquet"
CODE = X.CODE + ["tools/run_sixmonth_devmonths.py"]


def rel(p) -> str:
    return X.rel(p)


# ---------------------------------------------------------------- C6 stacker

def fit_stack(m: str, variant: str) -> pd.DataFrame:
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    dev, res, fx, mats, Pn, Pg = X.stack_dev()
    extra = None
    if variant == "b":
        (dev2, res2, fx2, _), extra = X.tierb_stack_extra(WORK, X.FEBJUN)   # the final file's selection of Tier B rows, unchanged
        if not (np.array_equal(dev2[ID].to_numpy(), dev[ID].to_numpy()) and np.array_equal(res2, res) and np.array_equal(fx2, fx)):
            raise ValueError("tierb_stack_extra's development half differs from stack_dev()")
    month, air, dom = dev["month"].to_numpy(), dev["airport"].to_numpy(), dev["dom"].to_numpy()
    fit, ev = (month != m) & ~fx, month == m
    X46 = mats["p2"]
    # run_stack_weight.dev_month's guards (support = the fitting months' non-fixed rows), the same for (a) and (b)
    h2 = S.outside_support(X46[fit], air[fit], X46[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hn3 = V3.guard_new(Pn[fit].reset_index(drop=True), air[fit], Pn[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hno = V3.guard_new(Pg[fit].reset_index(drop=True), air[fit], Pg[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    out = pd.DataFrame({ID: dev[ID].to_numpy()[ev], "airport": air[ev], "new": dev["new"].to_numpy(dtype=float)[ev], "fx": fx[ev],
                        "h2": h2, "h3": h2 | hn3, "ho": h2 | hno, "o_has": Pg["o_has"].fillna(0.0).to_numpy()[ev]})
    rm = {k: v[ev].reset_index(drop=True) for k, v in mats.items()}
    P, trees = X.stack_fit_predict(mats, res, fit, dom, rm, X.STACK["k"], extra)
    for c in P.columns:
        out[c] = P[c].to_numpy()
    info = {"dev_fit_rows": int(fit.sum()), "extra_fit_rows": int(len(extra["res"])) if extra is not None else 0, "held_rows": int(ev.sum()),
            "trees": trees}
    return out, info


# ---------------------------------------------------------------- C14 second pass

def ss_tables():
    d = WORK / "late" / "stack_second"
    T, KF = pd.read_parquet(d / "dev_table.parquet"), pd.read_parquet(d / "rank_table.parquet")
    return d, T, KF


def fit_ss(m: str, variant: str) -> pd.DataFrame:
    d, T, KF = ss_tables()
    c = next(c for c in X.pass_candidates("ss") if c["name"] == SS_CANDIDATE)
    extra = None
    if variant == "b":
        rows = X.tierb_rows(WORK).set_index(ID).reindex(KF[ID]).reset_index()
        sel = rows["month"].isin(X.FEBJUN).to_numpy() & rows["lane_m"].to_numpy() & ~KF["fixed"].to_numpy(dtype=bool)
        extra = KF.assign(label=rows["label"].to_numpy())[sel].reset_index(drop=True)
    tf = (T["month"] != m).to_numpy()
    tm = (T["month"] == m).to_numpy()
    p = X.pass_fit_predict("ss", c, T[tf].reset_index(drop=True), T[tm].reset_index(drop=True), extra)
    out = pd.DataFrame({ID: T[ID].to_numpy()[tm], SS_CANDIDATE: p})
    info = {"dev_fit_rows_offered": int(tf.sum()), "extra_rows_offered": int(len(extra)) if extra is not None else 0, "held_rows": int(tm.sum()),
            "tables": {rel(d / f): sha256_file(d / f) for f in ["dev_table.parquet", "rank_table.parquet"]}}
    return out, info


def fit(layer: str, m: str, variant: str) -> dict:
    path = OUT / f"{layer}_{m}_{variant}.parquet"
    if path.exists():
        raise ValueError(f"{path} exists; never overwrite.")
    out, info = fit_stack(m, variant) if layer == "stack" else fit_ss(m, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    rec = {"created_at_utc": utc_now(), "declaration": DECLARATION, "layer": layer, "month": m, "variant": variant, **info, "file": rel(path),
           "sha256": sha256_file(path), "code": {p: sha256_file(ROOT / p) for p in CODE}}
    write_json(path.with_suffix(".json"), rec)
    return rec


# ---------------------------------------------------------------- reading

def stack_values(m: str, variant: str) -> pd.DataFrame:
    import run_stack_weight as SW
    P = pd.read_parquet(OUT / f"stack_{m}_{variant}.parquet")
    v, _, _ = SW.rank_value(P, X.STACK["weight"], X.STACK["k"])
    return pd.DataFrame({ID: P[ID].to_numpy(), "airport": P["airport"].to_numpy(), "value": np.rint(v)})


def ss_values(m: str, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import run_stack_second as SS
    _, T, _ = ss_tables()
    c = next(c for c in X.pass_candidates("ss") if c["name"] == SS_CANDIDATE)
    P = pd.read_parquet(OUT / f"ss_{m}_{variant}.parquet")
    t = T[T["month"] == m].reset_index(drop=True)
    if not np.array_equal(t[ID].to_numpy(), P[ID].to_numpy()):
        raise ValueError("ss rows differ from the development table's month")
    base = t["v71"].to_numpy(dtype=float)
    p = P[SS_CANDIDATE].to_numpy()
    mv = ~np.isnan(p)
    new = base.copy()
    new[mv] = np.maximum(base[mv] + np.clip(c["weight"] * p[mv], -SS.CAP, SS.CAP), SS.FLOOR)
    return pd.DataFrame({ID: t[ID].to_numpy(), "airport": t["airport"].to_numpy(), "value": np.rint(new)}), t


def compare(lab: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, rng) -> dict:
    import run_intruder_bound as R
    s = lab.merge(a.rename(columns={"value": "base"}), on=ID, validate="one_to_one").merge(b[[ID, "value"]], on=ID, validate="one_to_one")
    y, va, vb = s["label"].to_numpy(dtype=float), s["base"].to_numpy(dtype=float), s["value"].to_numpy(dtype=float)
    pct, lo, hi = R.interval(s, "base", vb, rng)
    da, db = (y - va) ** 2, (y - vb) ** 2
    by = pd.DataFrame({"airport": s["airport"], "d": db - da}).groupby("airport")["d"].sum()
    return {"rows": int(len(s)), "rows_moved": int((va != vb).sum()), "rmse_a": float(np.sqrt(da.mean())), "rmse_b": float(np.sqrt(db.mean())),
            "change_pct": float(pct), "lo_pct": lo, "hi_pct": hi, "dSSE": float(db.sum() - da.sum()), "squared_change": float(((vb - va) ** 2).sum()),
            "dSSE_by_airport": {k: float(v) for k, v in by.items()}}


def read() -> None:
    if not X.committed_unchanged(DECLARATION):
        raise ValueError(f"{DECLARATION} must be committed and unchanged before the reading.")
    if RESULTS.exists():
        raise ValueError(f"{RESULTS} exists; never overwrite.")
    out ={"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
           "official_scores_used": "none", "code": {p: sha256_file(ROOT / p) for p in CODE}, "months": {}, "fits": {}}
    lab_stack = stack_labels()
    for m in MONTHS:
        rng = np.random.default_rng(READ_SEED)
        sa, sb = stack_values(m, "a"), stack_values(m, "b")
        c6 = compare(lab_stack[lab_stack["month"] == m][[ID, "label", "day"]], sa, sb, rng)
        (qa, t), (qb, _) = ss_values(m, "a"), ss_values(m, "b")
        rng = np.random.default_rng(READ_SEED)
        c14 = compare(t[[ID, "label", "day"]], qa, qb, rng)
        # control: (a) against the stored out-of-month development predictions of the chain's table
        stored = pd.read_parquet(WORK / "late" / "stack_second" / "preds_dev_stored.parquet").set_index(ID)[SS_CANDIDATE]
        pa = pd.read_parquet(OUT / f"ss_{m}_a.parquet").set_index(ID)[SS_CANDIDATE]
        st = stored.reindex(pa.index).to_numpy()
        same_nan = bool((np.isnan(st) == np.isnan(pa.to_numpy())).all())
        c14["control_a_vs_stored"] = {"nan_pattern_equal": same_nan,
                                      "max_abs_diff": float(np.nanmax(np.abs(st - pa.to_numpy()))) if same_nan else None}
        out["months"][m] = {"C6": c6, "C14": c14, "dSSE_sum": c6["dSSE"] + c14["dSSE"]}
        for layer in ["stack", "ss"]:
            for v in "ab":
                p = OUT / f"{layer}_{m}_{v}.parquet"
                out["fits"][f"{layer}_{m}_{v}"] = sha256_file(p)
        print(m, f"C6 {c6['change_pct']:+.4f}% [{c6['lo_pct']:+.4f}, {c6['hi_pct']:+.4f}] dSSE {c6['dSSE']:+.4g} |",
              f"C14 {c14['change_pct']:+.4f}% [{c14['lo_pct']:+.4f}, {c14['hi_pct']:+.4f}] dSSE {c14['dSSE']:+.4g} |",
              "control", c14["control_a_vs_stored"], flush=True)
    out["decision"] = {k: ("candidate 3 (splice)" if out["months"][m]["dSSE_sum"] > 0 else "candidate 4") for k, m in ANALOGUE.items()}
    out["decision"].update({"2026-02": "candidate 4", "2026-06": "candidate 4"})
    out["build_candidate5"] = any(v.startswith("candidate 3") for v in out["decision"].values())
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS, out)
    print(json.dumps(out["decision"], indent=1), "build candidate 5:", out["build_candidate5"])


def stack_labels() -> pd.DataFrame:
    """Labels / days of the C6 development table, read light (no matrices): run_ctx_ownpoints_chain.dev_table_from's month / label /
    take-off day on the same files as tools/run_sixmonth.stack_dev()."""
    import export_ctx_ownpoints as E
    import run_ctx_ownpoints_chain as CH
    import run_opdi_stack_decl as D
    import run_ownpoints_weight as W
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    d = E.CHAIN / D.CANDIDATE_CHAIN
    dev = CH.dev_table_from(d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet")
    return dev[[ID, "month", "label"]].assign(day=dev["takeoff_day"].to_numpy())


# ---------------------------------------------------------------- splice

def splice() -> dict:
    from prc2026_lirf_mixture.features import DEPARTURE_COLUMNS, population_rows
    res = read_json(RESULTS)
    if not res["build_candidate5"]:
        raise ValueError("The decision keeps candidate 4; no candidate 5.")
    for p, h in [C3, C4]:
        if sha256_file(ROOT / p) != h:
            raise ValueError(f"{p} differs from {h}")
    c3, c4 = pd.read_parquet(ROOT / C3[0]), pd.read_parquet(ROOT / C4[0])
    if not (c3[ID].equals(c4[ID]) and c3.dtypes.equals(c4.dtypes)):
        raise ValueError("candidates 3 and 4 differ in order or dtypes")
    mv = pd.read_parquet(ROOT / "data/raw/final_ranking.parquet", columns=DEPARTURE_COLUMNS)
    dep = mv[mv["PHASE_mvt"] == "DEP"].set_index(ID)["MVT_TIME_UTC_mvt"]
    month = dep.reindex(c4[ID]).dt.strftime("%Y-%m").to_numpy()
    if pd.isna(month).any():
        raise ValueError("a row without a departure")
    take3 = np.isin(month, [k for k, v in res["decision"].items() if v.startswith("candidate 3")])
    o = c4.copy()
    o.loc[take3, TARGET] = c3.loc[take3, TARGET].to_numpy()
    if C5.exists():
        raise ValueError(f"{C5} exists; never overwrite.")
    C5.parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(C5, index=False)
    pop = population_rows(mv)[ID]
    fixed = c4[ID].isin(pop).to_numpy()
    rec = {"created_at_utc": utc_now(), "decision": res["decision"], "rows": len(o), "rows_from_candidate3": int(take3.sum()),
           "rows_differ_from_c4": int((o[TARGET] != c4[TARGET]).sum()), "rows_differ_from_c3": int((o[TARGET] != c3[TARGET]).sum()),
           "population_rows": int(fixed.sum()), "population_rows_differ": int(((o[TARGET] != c4[TARGET]) | (o[TARGET] != c3[TARGET]))[fixed].sum()),
           "fixed": {str(int(i)): int(o.loc[o[ID] == i, TARGET].iloc[0]) for i in [213142471, 213142469, 213145579, 213152353]},
           "inputs": {C3[0]: C3[1], C4[0]: C4[1]}, "file": rel(C5), "sha256": sha256_file(C5)}
    write_json(C5.with_suffix(".receipt.json"), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fit", "read", "splice"])
    ap.add_argument("--layer", choices=["stack", "ss"])
    ap.add_argument("--month", choices=MONTHS)
    ap.add_argument("--variant", choices=["a", "b"])
    a = ap.parse_args()
    if a.action == "fit":
        if not X.committed_unchanged(DECLARATION):
            raise ValueError(f"{DECLARATION} must be committed and unchanged before any fit.")
        rec = fit(a.layer, a.month, a.variant)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("trees", "code")}, indent=1))
    elif a.action == "read":
        read()
    else:
        print(json.dumps(splice(), indent=1))


if __name__ == "__main__":
    main()
