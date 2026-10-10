"""Six-month refit of the chain's late learners (docs/sixmonth/README.md, configs/sixmonth_refit_v1.json).

The guarded stacker of step 6 (stack_weight_v1 w65_bag10: v2 / v3 / OPDI bags, ten seeds, weight 0.65), the second pass (stack_second_v1
sp_hicov7) and the third pass (stack_third_v1 t3_ro_3) were fitted on the four development months (January, July, November, December 2025).
Tier B gives February / June 2025 out-of-sample chain tables (runs/febjun_tierb/chain/work); this tool refits the same learners with those
months' rows added, everything else frozen (features, parameters, seeds, rounds, early-stopping rule, weights, caps, guards, airport sets).

  control --kind stack|ss|st            four-month mode on the ranking file: the shipped ranking-mode predictions reproduced exactly
                                         (stack: runs/final_early_chain/rehearsal_shipped3/steps/stack/preds.parquet; ss / st:
                                         runs/stack_second|stack_third/preds_rank.parquet) -> runs/sixmonth/control_<kind>.json
  held --kind K --held M [--work W]     learner fitted on the four development months + the OTHER Tier B month (lane M's rows read, not
                                         fixed), predicting month M's rows of F -> runs/sixmonth/held/<K>_<M>.parquet (+ .json)
  read [--work W]                       configs/sixmonth_refit_v1.json's readings (refuses unless it is committed and unchanged)
                                         -> docs/sixmonth/results_v1.json
  fit-final --kind K --table ...        (final file, later) the learner on all six months for a new file's tables

No official score enters. Nothing is uploaded; nothing is written under runs/submissions. Heavy commands run on an Azure VM.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

ID, TARGET, TIME = "MVT_ID_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"
DECLARATION = "configs/sixmonth_refit_v1.json"
OUT = ROOT / "runs/sixmonth"
DOCS = ROOT / "docs/sixmonth"
TIERB_WORK = ROOT / "runs/febjun_tierb/chain/work"
LANE_M = ROOT / "runs/febjun_tierb/model"
FEBJUN = ["2025-02", "2025-06"]
OTHER = {"2025-02": "2025-06", "2025-06": "2025-02"}
READ_SEED = 20261004
STACK = {"weight": 0.65, "k": 10, "candidate": "w65_bag10"}            # configs/stack_weight_v1.json, selected
CONTROL_STACK = ROOT / "runs/final_early_chain/rehearsal_shipped3/steps/stack"   # ranking-file rehearsal of v69's step 6 (byte for byte)
CODE = ["tools/run_sixmonth.py", "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py", "tools/run_stack_weight.py", "tools/run_opdi_stack.py",
        "tools/run_stack_second.py", "tools/run_stack_third.py"]


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# ---------------------------------------------------------------- the step-6 stacker

def stack_dev():
    """run_stack_weight.rank_inputs()'s development half, unchanged: the four development months' rows, matrices, targets and fixed rows."""
    import export_ctx_ownpoints as E
    import run_ctx_ownpoints_chain as CH
    import run_opdi_stack as O
    import run_opdi_stack_decl as D
    import run_ownpoints_weight as W
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    d = E.CHAIN / D.CANDIDATE_CHAIN
    dev = CH.dev_table_from(d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet")
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    new_d = dev["new"].to_numpy(dtype=float)
    Pn = V3.new_columns(V3.own_points("dev", dev[ID].to_numpy()), new_d)
    X65 = pd.concat([X46, Pn], axis=1)
    Pg = pd.concat([Pn, O.opdi_columns("dev", dev[ID].to_numpy(), new_d)], axis=1)
    X79 = pd.concat([X46, Pg], axis=1)
    return dev, res, S.fixed_rows(dev), {"p2": X46, "p3": X65, "po": X79}, Pn, Pg


def stack_rank_mats(stacker_rank: Path, dev_mats: dict) -> tuple[pd.Series, dict]:
    """A file's 79 stacker columns (final_early_chain step 6's stacker_rank.parquet) as the three learners' matrices (X46 / X65 / X79 are
    prefixes of the 'po' matrix: X65 = [X46, own points], X79 = [X46, own points, OPDI])."""
    sr = pd.read_parquet(stacker_rank)
    X = sr.drop(columns=[ID]).reset_index(drop=True)
    cols = list(dev_mats["po"].columns)
    if X.shape[1] != len(cols):
        raise ValueError("stacker_rank: column count differs from the development 'po' matrix")
    X.columns = cols
    mats = {k: X.iloc[:, :dev_mats[k].shape[1]] for k in ["p2", "p3", "po"]}
    for k in mats:
        if list(mats[k].columns) != list(dev_mats[k].columns):
            raise ValueError(f"{k}: the prefix columns differ from the development matrix")
    return sr[ID], mats


def stack_guards(dev, fx, mats, Pn, Pg, rmats, rair, fxr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """run_stack_weight.rank_inputs()'s guards, unchanged (support = the four development months' non-fixed rows)."""
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    air = dev["airport"].to_numpy()
    h2 = S.outside_support(mats["p2"][~fx], air[~fx], rmats["p2"], rair) & ~fxr
    Pr = rmats["p3"].iloc[:, mats["p2"].shape[1]:]
    Pgr = rmats["po"].iloc[:, mats["p2"].shape[1]:]
    h3 = (h2 | V3.guard_new(Pn[~fx].reset_index(drop=True), air[~fx], Pr.reset_index(drop=True), rair)) & ~fxr
    ho = (h2 | V3.guard_new(Pg[~fx].reset_index(drop=True), air[~fx], Pgr.reset_index(drop=True), rair)) & ~fxr
    return h2, h3, ho


def stack_fit_predict(dev_mats: dict, res: np.ndarray, fit: np.ndarray, dom: np.ndarray, rmats: dict, k: int,
                      extra: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """final_early_chain.step6's fitting loop, unchanged; extra = {'mats', 'res', 'fit', 'dom'} rows appended to the development rows."""
    import run_stack_resid as S
    import run_stack_weight as SW
    P, trees = pd.DataFrame(index=range(len(rmats["p2"]))), {}
    saved = list(S.SEEDS)
    try:
        for key in ["p2", "p3", "po"]:
            Xa = dev_mats[key].to_numpy(np.float32)
            r, f, d = res, fit, dom
            if extra is not None:
                Xa = np.vstack([Xa, extra["mats"][key].to_numpy(np.float32)])
                r, f, d = (np.concatenate([res, extra["res"]]), np.concatenate([fit, extra["fit"]]), np.concatenate([dom, extra["dom"]]))
            Xra = rmats[key].to_numpy(np.float32)
            for i, seed in enumerate(SW.SEEDS10[:k]):
                S.SEEDS = [seed]
                models = S.fit_bag(Xa, r, f, d)
                P[f"{key}_{i}"] = models[0].predict(Xra, num_iteration=models[0].best_iteration)
                trees[f"{key} {seed}"] = int(models[0].best_iteration)
                print("stack", key, seed, "trees", trees[f"{key} {seed}"], flush=True)
    finally:
        S.SEEDS = saved
    return P, trees


def control_stack() -> dict:
    """Four-month mode on the ranking file: v69's step-6 predictions (rehearsal_shipped3, which reproduces v69 byte for byte) reproduced."""
    import run_stack_weight as SW
    dev, res, fx, mats, Pn, Pg = stack_dev()
    ids, rmats = stack_rank_mats(CONTROL_STACK / "stacker_rank.parquet", mats)
    P0 = pd.read_parquet(CONTROL_STACK / "preds.parquet")
    if not np.array_equal(P0[ID].to_numpy(), ids.to_numpy()):
        raise ValueError("control: preds and stacker_rank rows differ")
    rair, fxr = P0["airport"].to_numpy(), P0["fx"].to_numpy(dtype=bool)
    h2, h3, ho = stack_guards(dev, fx, mats, Pn, Pg, rmats, rair, fxr)
    rec = {"guards_equal": {g: bool(np.array_equal(v, P0[g].to_numpy(dtype=bool))) for g, v in [("h2", h2), ("h3", h3), ("ho", ho)]}}
    P, trees = stack_fit_predict(mats, res, ~fx, dev["dom"].to_numpy(), rmats, SW.k_of(SW.SEEDS10))
    cols = [c for c in P.columns]
    rec["max_abs_pred_diff"] = {c: float(np.abs(P[c].to_numpy() - P0[c].to_numpy()).max()) for c in cols}
    rec["trees"] = trees
    rec["all_equal"] = all(rec["guards_equal"].values()) and max(rec["max_abs_pred_diff"].values()) == 0.0
    # the step's value at w65_bag10 recomputed from the reproduced predictions against the rehearsal's step-6 file
    Q = P0.copy()
    for c in cols:
        Q[c] = P[c].to_numpy()
    values, _, _ = SW.rank_value(Q, STACK["weight"], STACK["k"])
    step = pd.read_parquet(CONTROL_STACK / "vigorous-whistle_v69.parquet").set_index(ID)[TARGET].reindex(P0[ID]).to_numpy(dtype=float)
    rec["step_file_rows_differ_after_rint"] = int((np.rint(values) != step).sum())
    rec["inputs"] = {rel(CONTROL_STACK / f): sha256_file(CONTROL_STACK / f) for f in ["preds.parquet", "stacker_rank.parquet", "vigorous-whistle_v69.parquet"]}
    return rec


# ---------------------------------------------------------------- second / third pass

def pass_module(kind: str):
    import run_stack_second as SS
    import run_stack_third as ST
    return {"ss": SS, "st": ST}[kind]


def pass_candidates(kind: str) -> list[dict]:
    """Every declared candidate of the pass (the chain's score() re-selects among them on each file)."""
    return json.loads(pass_module(kind).DECL.read_text(encoding="utf-8"))["candidates"]


def pass_target(kind: str, T: pd.DataFrame, label: np.ndarray) -> np.ndarray:
    M = pass_module(kind)
    base = T["v71"] if kind == "ss" else T["ship"]
    return np.clip(label - base.to_numpy(dtype=float), -M.CLIP_Y, M.CLIP_Y)


def pass_fit_predict(kind: str, c: dict, T: pd.DataFrame, K: pd.DataFrame, extra: pd.DataFrame | None = None) -> np.ndarray:
    """run_stack_second.fit / run_stack_third.fit's ranking bag of candidate c, unchanged; extra = rows (with 'label') appended to the
    development table's fitting rows (they pass the same airport / kind filter)."""
    import lightgbm as lgb
    M = pass_module(kind)
    F = M.features(T)
    fit_air, app_air = (c["airports"], c["airports"]) if kind == "ss" else (c["fit_airports"], c["apply_airports"])
    tr = T["airport"].isin(fit_air).to_numpy() & (T["kind"] != 3).to_numpy()
    XT, y = T[F].astype(np.float32)[tr], pass_target(kind, T, T["label"].to_numpy(dtype=float))[tr]
    if extra is not None:
        te = extra["airport"].isin(fit_air).to_numpy() & (extra["kind"] != 3).to_numpy()
        XT = pd.concat([XT, extra[F].astype(np.float32)[te]], ignore_index=False)
        y = np.concatenate([y, pass_target(kind, extra, extra["label"].to_numpy(dtype=float))[te]])
    XK = K[F].astype(np.float32)
    kin = K["airport"].isin(app_air).to_numpy() & (K["kind"] != 3).to_numpy()
    acc = np.zeros(int(kin.sum()))
    for s in (M.SEEDS if kin.any() else []):
        mdl = lgb.train({**M.PARAMS, "seed": s}, lgb.Dataset(XT, y, categorical_feature=["kind", "ap"]), M.ROUNDS)
        acc += mdl.predict(XK[kin]) / len(M.SEEDS)
        print(kind, c["name"], "seed", s, "done", flush=True)
    pr = np.full(len(K), np.nan)
    pr[kin] = acc
    return pr


def control_pass(kind: str) -> dict:
    d = ROOT / ("runs/stack_second" if kind == "ss" else "runs/stack_third")
    tn, kn = ("dev_table", "rank_table") if kind == "ss" else ("dev_t3", "rank_t3")
    T, K = pd.read_parquet(d / f"{tn}.parquet"), pd.read_parquet(d / f"{kn}.parquet")
    shipped = pd.read_parquet(d / "preds_rank.parquet").set_index(ID).reindex(K[ID])
    rec = {"candidates": {}, "inputs": {rel(d / f): sha256_file(d / f) for f in [f"{tn}.parquet", f"{kn}.parquet", "preds_rank.parquet"]}}
    for c in pass_candidates(kind):
        pr = pass_fit_predict(kind, c, T, K)
        ship = shipped[c["name"]].to_numpy()
        same_nan = bool((np.isnan(pr) == np.isnan(ship)).all())
        diff = float(np.nanmax(np.abs(pr - ship))) if same_nan else None
        rec["candidates"][c["name"]] = {"nan_pattern_equal": same_nan, "max_abs_pred_diff": diff, "equal": bool(same_nan and diff == 0.0)}
    rec["all_equal"] = all(v["equal"] for v in rec["candidates"].values())
    return rec


# ---------------------------------------------------------------- Tier B side (February / June 2025)

def tierb_rows(work: Path) -> pd.DataFrame:
    """F's departures with month, label, take-off day / day of month and lane M's read flag (the readings' governing population)."""
    import febjun_tierb_chain as TB
    dep = TB.departures()
    lab = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", TARGET]) for p in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))])
    lab = lab[lab["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID)[TARGET].astype(float)
    lane_m = set(pd.concat([pd.read_parquet(LANE_M / f"chain_{m}.parquet", columns=[ID]) for m in FEBJUN])[ID])
    out = pd.DataFrame({ID: dep[ID].to_numpy(), "month": dep["month"].to_numpy(), "label": lab.reindex(dep[ID]).to_numpy(),
                        "day": dep[TIME].dt.strftime("%Y-%m-%d").to_numpy(), "dom": dep[TIME].dt.day.to_numpy(),
                        "lane_m": dep[ID].isin(lane_m).to_numpy()})
    if out["label"].isna().any():
        raise ValueError("A departure of F without a label.")
    return out


def held_stack(work: Path, held: str) -> dict:
    import run_stack_weight as SW
    dev, res, fx, mats, Pn, Pg = stack_dev()
    st = work / "steps" / "stack"
    ids, rmats = stack_rank_mats(st / "stacker_rank.parquet", mats)
    P0 = pd.read_parquet(st / "preds.parquet")
    if not np.array_equal(P0[ID].to_numpy(), ids.to_numpy()):
        raise ValueError("preds and stacker_rank rows differ")
    rows = tierb_rows(work).set_index(ID).reindex(P0[ID]).reset_index()
    if rows["month"].isna().any():
        raise ValueError("stacker rows outside F")
    fxr = P0["fx"].to_numpy(dtype=bool)
    other = (rows["month"] == OTHER[held]).to_numpy()
    fit_x = other & ~fxr & rows["lane_m"].to_numpy()
    sel = fit_x
    extra = {"mats": {k: v[sel].reset_index(drop=True) for k, v in rmats.items()},
             "res": np.clip(rows["label"].to_numpy()[sel] - P0["new"].to_numpy(dtype=float)[sel], -3600.0, 3600.0),
             "fit": np.ones(int(sel.sum()), dtype=bool), "dom": rows["dom"].to_numpy()[sel]}
    tgt = (rows["month"] == held).to_numpy()
    rm_t = {k: v[tgt].reset_index(drop=True) for k, v in rmats.items()}
    P, trees = stack_fit_predict(mats, res, ~fx, dev["dom"].to_numpy(), rm_t, STACK["k"], extra)
    out = P0[tgt].reset_index(drop=True)[[ID, "airport", "new", "fx", "h2", "h3", "ho", "o_has"]].copy()
    for c in P.columns:
        out[c] = P[c].to_numpy()
    path = OUT / "held" / f"stack_{held}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    rec = {"created_at_utc": utc_now(), "kind": "stack", "held": held, "extra_fit_rows": int(sel.sum()), "dev_fit_rows": int((~fx).sum()),
           "held_rows": int(tgt.sum()), "trees": trees, "file": rel(path), "sha256": sha256_file(path),
           "inputs": {rel(st / f): sha256_file(st / f) for f in ["preds.parquet", "stacker_rank.parquet"]},
           "code": {p: sha256_file(ROOT / p) for p in CODE}}
    write_json(path.with_suffix(".json"), rec)
    return rec


def held_pass(kind: str, work: Path, held: str) -> dict:
    d = work / "late" / ("stack_second" if kind == "ss" else "stack_third")
    tn, kn = ("dev_table", "rank_table") if kind == "ss" else ("dev_t3", "rank_t3")
    T, K = pd.read_parquet(d / f"{tn}.parquet"), pd.read_parquet(d / f"{kn}.parquet")
    rows = tierb_rows(work).set_index(ID).reindex(K[ID]).reset_index()
    if rows["month"].isna().any():
        raise ValueError("pass rows outside F")
    Kl = K.assign(label=rows["label"].to_numpy())
    sel = (rows["month"] == OTHER[held]).to_numpy() & rows["lane_m"].to_numpy() & ~K["fixed"].to_numpy(dtype=bool)
    tgt = (rows["month"] == held).to_numpy()
    out = pd.DataFrame({ID: K[ID].to_numpy()[tgt]})
    for c in pass_candidates(kind):
        out[c["name"]] = pass_fit_predict(kind, c, T, K[tgt].reset_index(drop=True), Kl[sel].reset_index(drop=True))
    path = OUT / "held" / f"{kind}_{held}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    rec = {"created_at_utc": utc_now(), "kind": kind, "held": held, "candidates": [c["name"] for c in pass_candidates(kind)], "extra_rows_offered": int(sel.sum()),
           "held_rows": int(tgt.sum()), "file": rel(path), "sha256": sha256_file(path),
           "inputs": {rel(d / f): sha256_file(d / f) for f in [f"{tn}.parquet", f"{kn}.parquet"]}, "code": {p: sha256_file(ROOT / p) for p in CODE}}
    write_json(path.with_suffix(".json"), rec)
    return rec


# ---------------------------------------------------------------- final file (after acceptance)

def tierb_stack_extra(work: Path, months: list[str]) -> tuple[dict, dict]:
    """Tier B rows (lane M's rows read, not fixed) of the given months as step-6 training rows."""
    dev, res, fx, mats, Pn, Pg = stack_dev()
    st = work / "steps" / "stack"
    ids, rmats = stack_rank_mats(st / "stacker_rank.parquet", mats)
    P0 = pd.read_parquet(st / "preds.parquet")
    if not np.array_equal(P0[ID].to_numpy(), ids.to_numpy()):
        raise ValueError("preds and stacker_rank rows differ")
    rows = tierb_rows(work).set_index(ID).reindex(P0[ID]).reset_index()
    sel = rows["month"].isin(months).to_numpy() & ~P0["fx"].to_numpy(dtype=bool) & rows["lane_m"].to_numpy()
    extra = {"mats": {k: v[sel].reset_index(drop=True) for k, v in rmats.items()},
             "res": np.clip(rows["label"].to_numpy()[sel] - P0["new"].to_numpy(dtype=float)[sel], -3600.0, 3600.0),
             "fit": np.ones(int(sel.sum()), dtype=bool), "dom": rows["dom"].to_numpy()[sel]}
    return (dev, res, fx, mats), extra


def fit_final_stack(target: Path, out: Path, work: Path) -> dict:
    """The step-6 bags fitted on the four development months + both Tier B months, predicting a file's rows from its early chain's
    steps/stack/{stacker_rank,preds}.parquet; writes a preds file for tools/final_early_chain.py --stack-preds (rows, base and guard columns
    unchanged, p2_ / p3_ / po_ columns replaced)."""
    (dev, res, fx, mats), extra = tierb_stack_extra(work, FEBJUN)
    ids, rmats = stack_rank_mats(target / "stacker_rank.parquet", mats)
    P0 = pd.read_parquet(target / "preds.parquet")
    if not np.array_equal(P0[ID].to_numpy(), ids.to_numpy()):
        raise ValueError("target preds and stacker_rank rows differ")
    P, trees = stack_fit_predict(mats, res, ~fx, dev["dom"].to_numpy(), rmats, STACK["k"], extra)
    Q = P0.copy()
    for c in P.columns:
        Q[c] = P[c].to_numpy()
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)
    Q.to_parquet(out, index=False)
    rec = {"created_at_utc": utc_now(), "kind": "stack", "target": {rel(target / f): sha256_file(target / f) for f in ["preds.parquet", "stacker_rank.parquet"]},
           "extra_fit_rows": int(len(extra["res"])), "trees": trees, "file": rel(out), "sha256": sha256_file(out), "code": {p: sha256_file(ROOT / p) for p in CODE}}
    write_json(out.with_suffix(".json"), rec)
    return rec


def fit_final_pass(kind: str, table_dir: Path, work: Path) -> dict:
    """In place of tools/final_late_chain.py fit --stage ss|st: the ranking bags of every candidate fitted on the development table +
    both Tier B months' rows, predicting the work table's rows; the development predictions are the stored ones (as the late chain copies
    them). Writes table_dir/preds_{dev,rank}.parquet in the shipped layout."""
    sub = "stack_second" if kind == "ss" else "stack_third"
    tn, kn = ("dev_table", "rank_table") if kind == "ss" else ("dev_t3", "rank_t3")
    T, K = pd.read_parquet(table_dir / f"{tn}.parquet"), pd.read_parquet(table_dir / f"{kn}.parquet")
    d = work / "late" / sub
    KF = pd.read_parquet(d / f"{kn}.parquet")
    rows = tierb_rows(work).set_index(ID).reindex(KF[ID]).reset_index()
    sel = rows["month"].isin(FEBJUN).to_numpy() & rows["lane_m"].to_numpy() & ~KF["fixed"].to_numpy(dtype=bool)
    extra = KF.assign(label=rows["label"].to_numpy())[sel].reset_index(drop=True)
    stored = pd.read_parquet(table_dir / "preds_dev_stored.parquet")
    pd_dev, pd_rank = pd.DataFrame({ID: T[ID]}), pd.DataFrame({ID: K[ID]})
    for c in pass_candidates(kind):
        pd_dev[c["name"]] = stored.set_index(ID)[c["name"]].reindex(T[ID]).to_numpy()
        pd_rank[c["name"]] = pass_fit_predict(kind, c, T, K, extra)
    for f in ["preds_dev.parquet", "preds_rank.parquet"]:
        if (table_dir / f).exists():
            raise ValueError(f"{table_dir / f} exists; never overwrite (run in place of the late chain's fit stage).")
    pd_dev.to_parquet(table_dir / "preds_dev.parquet", index=False)
    pd_rank.to_parquet(table_dir / "preds_rank.parquet", index=False)
    rec = {"created_at_utc": utc_now(), "kind": kind, "table_dir": rel(table_dir), "extra_rows_offered": int(sel.sum()),
           "preds_rank": sha256_file(table_dir / "preds_rank.parquet"), "preds_dev": sha256_file(table_dir / "preds_dev.parquet"),
           "tierb_table": {rel(d / f"{kn}.parquet"): sha256_file(d / f"{kn}.parquet")}, "code": {p: sha256_file(ROOT / p) for p in CODE}}
    write_json(table_dir / "sixmonth_fit.json", rec)
    return rec


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["control", "held", "read", "fit-final"])
    ap.add_argument("--kind", choices=["stack", "ss", "st"])
    ap.add_argument("--held", choices=FEBJUN)
    ap.add_argument("--work", default="runs/febjun_tierb/chain/work", help="the Tier B chain's work directory (training rows / readings)")
    ap.add_argument("--target", help="fit-final stack: a file's early-chain steps/stack directory; ss / st: its late work table directory")
    ap.add_argument("--out", help="fit-final stack: the preds file to write (for final_early_chain.py --stack-preds)")
    a = ap.parse_args()
    work = ROOT / a.work
    if a.action == "fit-final":
        if not committed_unchanged(DECLARATION):
            raise ValueError(f"{DECLARATION} must be committed and unchanged.")
        acc = read_json(DOCS / "results_v1.json")["final_file"]
        layer = {"stack": "C6 guarded stacker", "ss": "C14 second pass", "st": "C15 third pass"}[a.kind]
        if acc[layer] != "six-month learner":
            raise ValueError(f"{layer} was not accepted; the final file keeps the shipped learner.")
        rec = fit_final_stack(ROOT / a.target, ROOT / a.out, work) if a.kind == "stack" else fit_final_pass(a.kind, ROOT / a.target, work)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("trees", "code")}, indent=1))
        return
    if a.action == "control":
        path = OUT / f"control_{a.kind}.json"
        if path.exists():
            raise ValueError(f"{path} exists; never overwrite.")
        rec = control_stack() if a.kind == "stack" else control_pass(a.kind)
        rec.update({"created_at_utc": utc_now(), "kind": a.kind, "code": {p: sha256_file(ROOT / p) for p in CODE}})
        write_json(path, rec)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("trees", "code")}, indent=1))
    elif a.action == "held":
        rec = held_stack(work, a.held) if a.kind == "stack" else held_pass(a.kind, work, a.held)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("trees", "code")}, indent=1))
    else:
        read(work)


def committed_unchanged(path: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=ROOT).returncode == 0
    return tracked and clean


def file_values(path: Path, ids: np.ndarray) -> np.ndarray:
    v = pd.read_parquet(path).set_index(ID)[TARGET].reindex(ids)
    if v.isna().any():
        raise ValueError(f"{path} does not cover F")
    return v.to_numpy(dtype=float)


def stack_variant(work: Path, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """C6 with the four-month bags (the Tier B chain's step-6 file) and with each month's five-month bags; guards, weight, seeds, floor and
    fixed rows as shipped. Rows whose shipped value is not the bag arithmetic (none expected on 2025 rows: published_missing) keep the file."""
    import febjun_tierb_chain as TB
    import run_stack_weight as SW
    st = work / "steps" / "stack"
    b0 = file_values(st / f"vigorous-whistle_v{TB.VERSION}.parquet", ids)
    P0 = pd.read_parquet(st / "preds.parquet")
    pos = pd.Series(np.arange(len(P0)), index=P0[ID].to_numpy()).reindex(ids).to_numpy()
    if np.isnan(pos).any():
        raise ValueError("the step-6 predictions do not cover F")
    pos = pos.astype(int)
    v0, use0, p0 = SW.rank_value(P0, STACK["weight"], STACK["k"])
    same = np.rint(v0)[pos] == b0
    b1 = b0.copy()
    info = {"rows_not_bag_arithmetic": int((~same).sum()), "transfer": {}}
    ref = read_json(ROOT / "docs/stack_weight/results_v1.json")["dev_oof_prediction_by_airport"]["bag10"]
    for m in FEBJUN:
        H = pd.read_parquet(OUT / "held" / f"stack_{m}.parquet")
        P0m = P0.set_index(ID).loc[H[ID].to_numpy()].reset_index()
        for c in ["airport", "new", "fx", "h2", "h3", "ho", "o_has"]:
            if not np.array_equal(P0m[c].to_numpy(), H[c].to_numpy()):
                raise ValueError(f"held stack {m}: column {c} differs from the chain's preds")
        v1, use1, p1 = SW.rank_value(H, STACK["weight"], STACK["k"])
        hpos = pd.Series(np.arange(len(H)), index=H[ID].to_numpy())
        rows = np.flatnonzero(pd.Index(ids).isin(H[ID]))
        hp = hpos.reindex(ids[rows]).to_numpy().astype(int)
        b1[rows] = np.where(same[rows], np.rint(v1)[hp], b0[rows])
        # transfer: per airport, RMS of the guarded bag prediction (fixed / held rows at 0) against the stored out-of-month development RMS
        air, fx = H["airport"].to_numpy()[hp], H["fx"].to_numpy(dtype=bool)[hp]
        lirf = air == "LIRF"
        p0m = {k: v[pos[rows]] for k, v in p0.items()}
        p1m = {k: v[hp] for k, v in p1.items()}
        use0m, use1m = use0[pos[rows]], use1[hp]
        tr = {}
        for bag, key, mask0, mask1 in [("v3", "p3", ~lirf & ~fx, ~lirf & ~fx), ("v2_LIRF", "p2", lirf & ~fx, lirf & ~fx),
                                       ("opdi_LSZH_used", "po", use0m, use1m)]:
            for a, r in ref[bag].items():
                s0, s1 = mask0 & (air == a), mask1 & (air == a)
                if not s0.any() and not s1.any():
                    continue
                r0 = float(np.sqrt(np.mean(p0m[key][s0] ** 2))) if s0.any() else float("nan")
                r1 = float(np.sqrt(np.mean(p1m[key][s1] ** 2))) if s1.any() else float("nan")
                q0, q1 = r0 / r["rms_s"], r1 / r["rms_s"]
                ok0, ok1 = bool(0.5 <= q0 <= 2.0), bool(0.5 <= q1 <= 2.0)
                tr[f"{bag} {a}"] = {"dev_rms_s": r["rms_s"], "four_month_rms_s": r0, "six_month_rms_s": r1, "ratio_four": q0, "ratio_six": q1,
                                    "fails": bool(ok0 and not ok1)}
        info["transfer"][m] = tr
    info["transfer_fail"] = any(v["fails"] for t in info["transfer"].values() for v in t.values())
    return b0, b1, info


def pass_variant(kind: str, work: Path, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """C14 / C15 with the Tier B chain's four-month ranking bags (its output file) and with each month's five-month bags: the layer's rule
    as in its governing score() (clauses 1-3 from the stored development predictions, as the chain read them; clause 4 on F, both months
    pooled; selection by the lowest equal-footing mean; airports outside the transfer range keep the base)."""
    M = pass_module(kind)
    sub = "stack_second" if kind == "ss" else "stack_third"
    late = read_json(work / "late" / "late_chain.json")["outputs"]
    a = file_values(ROOT / late["nbr_dark" if kind == "ss" else "ss"]["file"], ids)
    b0 = file_values(ROOT / late[kind]["file"], ids)
    d = work / "late" / sub
    T = pd.read_parquet(d / ("dev_table.parquet" if kind == "ss" else "dev_t3.parquet"))
    K = pd.read_parquet(d / ("rank_table.parquet" if kind == "ss" else "rank_t3.parquet"))
    PD = pd.read_parquet(d / "preds_dev_stored.parquet").set_index(ID).reindex(T[ID]).reset_index()
    chain_res = read_json(work / "late" / f"{sub}_docs" / "results_v1.json")
    PR = pd.concat([pd.read_parquet(OUT / "held" / f"{kind}_{m}.parquet") for m in FEBJUN]).set_index(ID).reindex(K[ID]).reset_index()
    base_col = "v71" if kind == "ss" else "ship"
    base_d, base_k = T[base_col].to_numpy(dtype=float), K[base_col].to_numpy(dtype=float)
    kpos = pd.Series(np.arange(len(K)), index=K[ID].to_numpy()).reindex(ids).to_numpy()
    if np.isnan(kpos).any():
        raise ValueError(f"{kind}: the rank table does not cover F")
    kpos = kpos.astype(int)
    if not np.array_equal(np.rint(base_k)[kpos], a):
        raise ValueError(f"{kind}: the rank table's base differs from the layer's input file")
    info = {"candidates": {}}
    for c in pass_candidates(kind):
        name, aps = c["name"], (c["airports"] if kind == "ss" else c["apply_airports"])
        p = PD[name].to_numpy()
        mv = ~np.isnan(p)
        new = base_d.copy()
        new[mv] = np.maximum(base_d[mv] + np.clip(c["weight"] * p[mv], -M.CAP, M.CAP), M.FLOOR)
        pk = PR[name].to_numpy()
        mk = ~np.isnan(pk)
        newk = base_k.copy()
        newk[mk] = np.maximum(base_k[mk] + np.clip(c["weight"] * pk[mk], -M.CAP, M.CAP), M.FLOOR)
        cd, ck = new - base_d, newk - base_k
        guard = {}
        for ap in aps:
            sd, sk = mv & (T["airport"] == ap).to_numpy(), mk & (K["airport"] == ap).to_numpy()
            if kind == "ss" and (not sd.any() or not sk.any()):
                continue
            rd = float(np.sqrt((cd[sd] ** 2).mean())) if sd.any() else float("nan")
            rk_ = float(np.sqrt((ck[sk] ** 2).mean())) if sk.any() else float("nan")
            guard[ap] = {"rms_dev": rd, "rms_rank": rk_, "ratio": rk_ / rd, "ok": bool(0.5 <= rk_ / rd <= 2.0), "rows_rank": int(sk.sum())}
        excl = sorted(x for x, g in guard.items() if not g["ok"])
        cr = chain_res["candidates"][name]
        c4 = len(excl) <= (2 if kind == "ss" else 1)
        acc = bool(cr["clause_1"] and cr["clause_2"] and cr["clause_3"] and c4)
        info["candidates"][name] = {"eq_mean_pct": cr["eq_mean_pct"], "clauses_1_3": [cr["clause_1"], cr["clause_2"], cr["clause_3"]],
                                    "transfer": guard, "excluded_airports": excl, "clause_4": c4, "accepted": acc,
                                    "chain_four_month": {"excluded_airports": cr["excluded_airports"], "accepted": cr["accepted"]},
                                    "_v": np.where(K["airport"].isin(excl).to_numpy(), base_k, newk)}
    passing = [n for n, r in info["candidates"].items() if r["accepted"]]
    sel = min(passing, key=lambda n: info["candidates"][n]["eq_mean_pct"]) if passing else None
    info["selected"], info["chain_selected"] = sel, chain_res.get("selected")
    b1 = np.rint(info["candidates"][sel]["_v"])[kpos] if sel else a.copy()
    for r in info["candidates"].values():
        r.pop("_v")
    return b0, b1, info


def read(work: Path) -> None:
    import run_intruder_bound as R
    if not committed_unchanged(DECLARATION):
        raise ValueError(f"{DECLARATION} must be committed and unchanged before any reading.")
    path = DOCS / "results_v1.json"
    if path.exists():
        raise ValueError(f"{path} exists; never overwrite.")
    decl = read_json(ROOT / DECLARATION)
    rows = tierb_rows(work)
    ids = rows[ID].to_numpy()
    out = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "rule": decl["rule"],
           "work": rel(work), "code": {p: sha256_file(ROOT / p) for p in CODE}, "learners": {}}
    for kind, layer in [("stack", "C6 guarded stacker"), ("ss", "C14 second pass"), ("st", "C15 third pass")]:
        b0, b1, info = stack_variant(work, ids) if kind == "stack" else pass_variant(kind, work, ids)
        rng = np.random.default_rng(READ_SEED)
        months = {}
        for m in FEBJUN:
            sel = (rows["month"] == m).to_numpy() & rows["lane_m"].to_numpy()
            y = rows["label"].to_numpy()[sel]
            s = pd.DataFrame({"label": y, "base": b0[sel], "day": rows["day"].to_numpy()[sel]})
            pct, lo, hi = R.interval(s, "base", b1[sel], rng)
            months[m] = {"rows": int(sel.sum()), "change_pct": pct, "lo_pct": lo, "hi_pct": hi, "rows_moved": int((b0[sel] != b1[sel]).sum()),
                         "rmse_four_month": float(np.sqrt(np.mean((y - b0[sel]) ** 2))), "rmse_six_month": float(np.sqrt(np.mean((y - b1[sel]) ** 2))),
                         "squared_change": float(((b1[sel] - b0[sel]) ** 2).sum())}
        better = all(months[m]["change_pct"] < 0 and months[m]["hi_pct"] < 0 for m in FEBJUN)
        guard_ok = (not info["transfer_fail"]) if kind == "stack" else True
        out["learners"][layer] = {"kind": kind, "months": months, "info": info, "both_months_better_intervals_below_zero": better,
                                  "transfer_ok": guard_ok, "accepted": bool(better and guard_ok)}
        print(layer, " | ".join(f"{m} {months[m]['change_pct']:+.4f}% [{months[m]['lo_pct']:+.4f}, {months[m]['hi_pct']:+.4f}] n{months[m]['rows_moved']}"
                                 for m in FEBJUN), "ACCEPTED" if out["learners"][layer]["accepted"] else "not accepted", flush=True)
    out["final_file"] = {layer: ("six-month learner" if r["accepted"] else "shipped four-month learner") for layer, r in out["learners"].items()}
    DOCS.mkdir(parents=True, exist_ok=True)
    write_json(path, out)


if __name__ == "__main__":
    main()
