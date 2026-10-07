"""Governing run of configs/stack_weight_v1.json (docs/stack_weight/README.md): the guarded stacking residual's weight (0.5 shipped) and a
ten-seed bag, read in v68's pinned development chain against v68's development proxy.

  --dev-month M     month-blocked out-of-month predictions of the v2 (46), v3 (65) and OPDI (79) learners on development month M, one model
                    per seed for the ten seeds (seed s's model does not depend on the other seeds: tools/run_stack_resid.fit_bag builds its
                    datasets per seed), with the guards -> runs/stack_weight/dev/preds_<M>.parquet
  --dev-score       identity first (w50_bag3 through the chain must equal runs/opdi_stack_lszh/dev_v68_proxy.parquet exactly), then the six
                    declared candidates through the chain -> runs/stack_weight/dev/<name>/proxy.parquet; clauses 1-3 ->
                    docs/stack_weight/results_v1.json
  --rank-fit        ranking: the three learners on all four development months, one model per seed for the ten seeds, guarded, applied to
                    v66's shipped step-5 ranking table (tools/run_opdi_stack_decl.step_stack_opdi's inputs) -> runs/stack_weight/rank/preds.parquet;
                    check: w50_bag3 recombined equals step_stack_opdi(..., ['LSZH']) unchanged; clause 4 for the ten-seed bags -> results
  --rank-identity   tools/run_opdi_stack_decl.step_stack_opdi(..., ['LSZH']) unchanged -> chain_from_stack -> own_motion_v2 must reproduce the
                    uploaded v68 (d41d8705...) byte for byte
  --rank-dry        the selected candidate (if any) -> runs/stack_weight/rank/dry_v69/ + report against v68
Value of a candidate (weight w, seeds k): LIRF rows v2's bag, LSZH rows with an OPDI match not held by the OPDI guard the OPDI bag, every
other row v3's bag; max(chain + w x bag, -12); LIRF population and guard rows keep the chain; held rows keep the chain.
No official score enters. Nothing is uploaded; nothing is written under runs/submissions.
"""
from __future__ import annotations

import argparse
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

DECL = "configs/stack_weight_v1.json"
OUT = ROOT / "runs/stack_weight"
DOCS = ROOT / "docs/stack_weight"
RESULTS = DOCS / "results_v1.json"
CONTROL = ("runs/opdi_stack_lszh/dev_v68_proxy.parquet", "938a91cd84a9e6c0773afb72dec490fdf30c6762ee3f505469b3a26e3ab56aca")
V68 = "d41d8705e75a71e865fe21a8d6ad2f62044a5ed07f125fd208067914b2b10eab"
SEEDS10 = [20261003, 11, 12, 13, 14, 15, 16, 17, 18, 19]
LEARNERS = ["p2", "p3", "po"]
AIRPORTS = ["LSZH"]
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
CODE = ["tools/run_stack_weight.py", "tools/run_opdi_stack.py", "tools/run_opdi_stack_decl.py", "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py",
        "tools/run_ctx_ownpoints_chain.py", "tools/export_ctx_ownpoints.py", "tools/run_ownpoints_weight.py", "tools/run_intruder_bound.py",
        "tools/run_intruder_identity.py", "tools/run_own_motion.py", "tools/run_own_motion_v2.py"]


def candidates() -> list[dict]:
    decl = read_json(ROOT / DECL)
    return [{"name": "w50_bag3", "weight": 0.5, "seeds": SEEDS10[:3]}] + decl["candidates"]


def k_of(seeds: list[int]) -> int:
    if seeds != SEEDS10[:len(seeds)]:
        raise ValueError(f"Seeds {seeds} are not a prefix of the ten declared seeds.")
    return len(seeds)


def bag(t: pd.DataFrame, learner: str, k: int) -> np.ndarray:
    """tools/run_stack_resid.predict_bag's arithmetic on the first k seeds' stored predictions."""
    return np.mean([t[f"{learner}_{i}"].to_numpy(dtype=np.float64) for i in range(k)], axis=0)


# ---------------------------------------------------------------- development

def dev_inputs():
    """tools/run_opdi_stack.score's inputs on the pinned v66 chain files."""
    import run_ctx_ownpoints_chain as CH
    import run_opdi_stack as O
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    files = (O.SHIP / "ctl.parquet", O.SHIP / "rows_all.parquet", O.SHIP / "dmp.parquet")
    dev = CH.dev_table_from(*files)
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    new = dev["new"].to_numpy(dtype=float)
    Pn = V3.new_columns(V3.own_points("dev", dev[ID].to_numpy()), new)
    Po = O.opdi_columns("dev", dev[ID].to_numpy(), new)
    Pg = pd.concat([Pn, Po], axis=1)
    return dev, X46, pd.concat([X46, Pn], axis=1), pd.concat([X46, Pg], axis=1), Pn, Pg, Po, res


def dev_month(m: str) -> None:
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    path = OUT / "dev" / f"preds_{m}.parquet"
    if path.exists():
        raise ValueError(f"{path} exists.")
    dev, X46, X65, X79, Pn, Pg, Po, res = dev_inputs()
    fx = S.fixed_rows(dev)
    month, air, dom = dev["month"].to_numpy(), dev["airport"].to_numpy(), dev["dom"].to_numpy()
    fit, ev = (month != m) & ~fx, month == m
    # V3.dev_predictions' guards, unchanged
    h2 = S.outside_support(X46[fit], air[fit], X46[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hn3 = V3.guard_new(Pn[fit].reset_index(drop=True), air[fit], Pn[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hno = V3.guard_new(Pg[fit].reset_index(drop=True), air[fit], Pg[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    out = pd.DataFrame({ID: dev[ID].to_numpy()[ev], "fx": fx[ev], "h2": h2, "h3": h2 | hn3, "ho": h2 | hno,
                        "o_has": Po["o_has"].to_numpy()[ev]})
    trees = {}
    for key, X in [("p2", X46), ("p3", X65), ("po", X79)]:
        Xa = X.to_numpy(np.float32)
        for i, seed in enumerate(SEEDS10):
            S.SEEDS = [seed]
            models = S.fit_bag(Xa, res, fit, dom)
            out[f"{key}_{i}"] = models[0].predict(Xa[ev], num_iteration=models[0].best_iteration)
            trees[f"{key} {seed}"] = int(models[0].best_iteration)
            print(m, key, seed, "trees", trees[f"{key} {seed}"], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "month": m, "rows": int(ev.sum()), "trees": trees, "sha256": sha256_file(path),
                                           "code": {p: sha256_file(ROOT / p) for p in CODE}})


def dev_value(dev: pd.DataFrame, P: pd.DataFrame, w: float, k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """The stacking value at weight w with the first k seeds (tools/run_opdi_stack.score's arithmetic with S.WEIGHT = w)."""
    import run_stack_resid as S
    new, air = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy()
    fx = P["fx"].to_numpy(dtype=bool)
    p = {}
    for key, h in [("p2", "h2"), ("p3", "h3"), ("po", "ho")]:
        q = bag(P, key, k)
        q[fx] = 0.0
        q[P[h].to_numpy(dtype=bool)] = 0.0
        p[key] = q
    S.WEIGHT = w
    try:
        base2, base3, baseo = S.apply(new, p["p2"], fx), S.apply(new, p["p3"], fx), S.apply(new, p["po"], fx)
    finally:
        S.WEIGHT = 0.5
    use = (P["o_has"].fillna(0.0).to_numpy() > 0) & ~P["ho"].to_numpy(dtype=bool) & np.isin(air, AIRPORTS)
    base3 = np.where(use, baseo, base3)
    return np.where(air == "LIRF", base2, base3), use, p


def dev_score() -> None:
    import run_ctx_ownpoints_chain as CH
    import run_intruder_bound as R
    import run_opdi_stack as O
    import run_own_motion_v2 as O2
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    if RESULTS.exists():
        raise ValueError("Preserve the reading.")
    if sha256_file(ROOT / CONTROL[0]) != CONTROL[1]:
        raise ValueError("The control differs from the declared file.")
    dev = CH.dev_table_from(O.SHIP / "ctl.parquet", O.SHIP / "rows_all.parquet", O.SHIP / "dmp.parquet")
    P = pd.concat([pd.read_parquet(OUT / "dev" / f"preds_{m}.parquet") for m in MONTHS], ignore_index=True).set_index(ID).reindex(dev[ID]).reset_index()
    if P["p2_0"].isna().any() or not np.array_equal(P["fx"].to_numpy(dtype=bool), S.fixed_rows(dev)):
        raise ValueError("The stored predictions do not cover the development rows.")
    s1 = pd.read_parquet(O.SHIP / "cand.parquet").set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    keys, _ = O2.O.dev_table()
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    ctl = pd.read_parquet(ROOT / CONTROL[0]).set_index(ID).reindex(dev[ID]).reset_index()
    chain = V3.Chain()
    _, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    new, air, month = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy(), dev["month"].to_numpy()
    res = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "control": CONTROL[0],
           "control_sha256": sha256_file(ROOT / CONTROL[0]), "official_scores_used": "none", "audit_used": False,
           "code": {p: sha256_file(ROOT / p) for p in CODE}, "candidates": {}}
    oof = {}
    for c in candidates():
        name, w, k = c["name"], float(c["weight"]), k_of(c["seeds"])
        value, use, p = dev_value(dev, P, w, k)
        oof[name] = (p, use)
        t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
        t["s2"] = s1 + (value - new)
        _, cand63 = chain.run(t, "s2")
        t["c63"] = cand63
        c66 = CH.own_motion_v2(t, "c63", t12, trig)
        d = OUT / "dev" / name
        d.mkdir(parents=True, exist_ok=True)
        out = t[[ID, "month", "airport", "label", "cls", "day"]].copy()
        out["value"] = c66
        out["opdi_used"] = use
        out.to_parquet(d / "proxy.parquet", index=False)
        diff = c66 - ctl["value"].to_numpy(dtype=float)
        if name == "w50_bag3":
            ident = float(np.abs(diff).max())
            res["identity"] = {"w50_bag3_max_abs_vs_control": ident, "proxy_sha256": sha256_file(d / "proxy.parquet")}
            print("identity: w50_bag3 vs v68 dev proxy max |diff|", ident, flush=True)
            if ident != 0.0:
                write_json(OUT / "dev" / "identity_failed.json", res)
                raise ValueError("identity: w50_bag3 does not reproduce v68's development proxy; nothing is read.")
            continue
        tt = ctl[[ID, "month", "label", "day"]].rename(columns={}).assign(ctl=ctl["value"].to_numpy(dtype=float), cand=c66)
        rng = np.random.default_rng(20261004)
        months = {}
        for m in MONTHS:
            s = tt[tt["month"] == m]
            pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
            ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
            months[m] = {"control_rmse": float(np.sqrt(np.mean((s["label"] - s["ctl"]) ** 2))), "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))),
                         "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum()),
                         "rms_change_s": float(np.sqrt(np.mean(ch ** 2))), "max_abs_change_s": float(np.abs(ch).max())}
        eq = float(np.mean([months[m]["pct"] for m in EQUAL]))
        checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= 0.005 for m in MONTHS), "2_equal_footing_mean_pct": eq,
                  "2_mean_ok": eq <= -0.02, "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in EQUAL)}
        by = pd.DataFrame({"airport": air, "month": month, "c": (ctl["value"].to_numpy() - ctl["label"].to_numpy()) ** 2,
                           "n": (c66 - ctl["label"].to_numpy()) ** 2}).groupby(["airport", "month"]).sum()
        fxm = S.fixed_rows(dev)
        res["candidates"][name] = {"weight": w, "seeds": c["seeds"], "months": months, "checks_1_to_3": checks,
                                   "clauses_1_to_3": bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"]),
                                   "airport_month_change_pct": {f"{a} {m}": float(100 * (np.sqrt(r["n"]) / np.sqrt(r["c"]) - 1)) for (a, m), r in by.iterrows()},
                                   "fixed_rows_moved": int((np.abs(diff)[fxm] > 1e-9).sum()), "proxy_sha256": sha256_file(d / "proxy.parquet")}
        print(name, " ".join(f"{m[5:]} {months[m]['pct']:+.4f}% [{months[m]['lo']:+.4f},{months[m]['hi']:+.4f}]" for m in MONTHS), f"| eq {eq:+.4f}%",
              "clauses 1-3", res["candidates"][name]["clauses_1_to_3"], flush=True)
    # development side of clause 4 (out-of-month guarded predictions, rows where each bag applies)
    fx = S.fixed_rows(dev)
    lirf = air == "LIRF"
    res["dev_oof_prediction_by_airport"] = {}
    for name in ["w50_bag3", "w50_bag10"]:
        p, use = oof[name]
        res["dev_oof_prediction_by_airport"][name.split("_")[1]] = {
            "v3": S.by_airport_rms(dev, np.where(~lirf & ~fx, p["p3"], 0.0), ~(~lirf & ~fx)),
            "v2_LIRF": S.by_airport_rms(dev, np.where(lirf & ~fx, p["p2"], 0.0), ~(lirf & ~fx)),
            "opdi_LSZH_used": S.by_airport_rms(dev, np.where(use, p["po"], 0.0), ~use)}
    write_json(RESULTS, res)


# ---------------------------------------------------------------- ranking

def rank_inputs():
    """tools/run_opdi_stack_decl.step_stack_opdi's inputs, unchanged."""
    import export_ctx_ownpoints as E
    import run_ctx_ownpoints_chain as CH
    import run_opdi_stack as O
    import run_opdi_stack_decl as D
    import run_ownpoints_weight as W
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    b, c, c_rows = D.SH / "rematch/base/vigorous-whistle_v55m_rematch.parquet", D.SH / "Y/vigorous-whistle_v66.parquet", D.SH / "Y/rows.parquet"
    delta_path = D.SH / "model_part/member_delta.parquet"
    d = E.CHAIN / D.CANDIDATE_CHAIN
    dev = CH.dev_table_from(d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet")
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    new_d = dev["new"].to_numpy(dtype=float)
    Pn = V3.new_columns(V3.own_points("dev", dev[ID].to_numpy()), new_d)
    X65 = pd.concat([X46, Pn], axis=1)
    Pg = pd.concat([Pn, O.opdi_columns("dev", dev[ID].to_numpy(), new_d)], axis=1)
    X79 = pd.concat([X46, Pg], axis=1)
    fx = S.fixed_rows(dev)
    checks: dict = {}
    rk = S.ranking_table((E.rel(c), sha256_file(c)), checks, E.rel(b), E.rel(c_rows))
    delta = pd.read_parquet(delta_path).set_index(ID).reindex(rk[ID])
    lirf = rk["airport"].eq("LIRF").to_numpy()
    for col, dc in [("v51m", "d51"), ("v52m", "d52"), ("v53m", "d52"), ("v54m", "d52")]:
        rk[col] = rk[col].to_numpy(dtype=float) + np.where(lirf, 0.0, delta[dc].to_numpy(dtype=float))
    X46r = S.features(rk).reset_index(drop=True)
    new_r = rk["new"].to_numpy(dtype=float)
    Pr = V3.new_columns(V3.own_points("rank", rk[ID].to_numpy()), new_r)
    X65r = pd.concat([X46r, Pr], axis=1)
    Por = O.opdi_columns("rank", rk[ID].to_numpy(), new_r)
    Pgr = pd.concat([Pr, Por], axis=1)
    X79r = pd.concat([X46r, Pgr], axis=1)
    air, rair = dev["airport"].to_numpy(), rk["airport"].to_numpy()
    fxr = S.fixed_rows(rk)
    h2 = S.outside_support(X46[~fx], air[~fx], X46r, rair) & ~fxr
    h3 = (h2 | V3.guard_new(Pn[~fx].reset_index(drop=True), air[~fx], Pr, rair)) & ~fxr
    ho = (h2 | V3.guard_new(Pg[~fx].reset_index(drop=True), air[~fx], Pgr, rair)) & ~fxr
    return dev, res, fx, [("p2", X46, X46r), ("p3", X65, X65r), ("po", X79, X79r)], rk, fxr, h2, h3, ho, Por


def rank_fit() -> None:
    import run_stack_resid as S
    path = OUT / "rank" / "preds.parquet"
    if path.exists():
        raise ValueError(f"{path} exists.")
    dev, res, fx, mats, rk, fxr, h2, h3, ho, Por = rank_inputs()
    dom = dev["dom"].to_numpy()
    out = pd.DataFrame({ID: rk[ID].to_numpy(), "airport": rk["airport"].to_numpy(), "new": rk["new"].to_numpy(dtype=float), "fx": fxr,
                        "h2": h2, "h3": h3, "ho": ho, "o_has": Por["o_has"].to_numpy()})
    trees = {}
    for key, X, Xr in mats:
        Xa, Xra = X.to_numpy(np.float32), Xr.to_numpy(np.float32)
        for i, seed in enumerate(SEEDS10):
            S.SEEDS = [seed]
            models = S.fit_bag(Xa, res, ~fx, dom)
            out[f"{key}_{i}"] = models[0].predict(Xra, num_iteration=models[0].best_iteration)
            trees[f"{key} {seed}"] = int(models[0].best_iteration)
            print("rank", key, seed, "trees", trees[f"{key} {seed}"], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "rows": int(len(out)), "trees": trees, "sha256": sha256_file(path),
                                           "code": {p: sha256_file(ROOT / p) for p in CODE}})


def rank_value(P: pd.DataFrame, w: float, k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """tools/run_opdi_stack_decl.step_stack_opdi's arithmetic at weight w with the first k seeds."""
    import run_stack_resid as S
    new_r, rair = P["new"].to_numpy(dtype=float), P["airport"].to_numpy()
    fxr = P["fx"].to_numpy(dtype=bool)
    lirf = rair == "LIRF"
    p = {}
    for key, h in [("p2", "h2"), ("p3", "h3"), ("po", "ho")]:
        q = bag(P, key, k)
        q[fxr], q[P[h].to_numpy(dtype=bool)] = 0.0, 0.0
        p[key] = q
    use = (P["o_has"].to_numpy() > 0) & ~P["ho"].to_numpy(dtype=bool) & np.isin(rair, AIRPORTS) & ~lirf & ~fxr
    S.WEIGHT = w
    try:
        v3vals = np.where(lirf, S.apply(new_r, p["p2"], fxr), S.apply(new_r, p["p3"], fxr))
        values = np.where(use, S.apply(new_r, p["po"], fxr), v3vals)
    finally:
        S.WEIGHT = 0.5
    return values, use, p


def rank_identity() -> None:
    import run_opdi_stack_decl as D
    import run_own_motion_v2 as O2
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import export_ctx_ownpoints as E
    import run_ownpoints_weight as W
    out = OUT / "rank" / "identity_v68"
    if out.exists():
        raise ValueError(f"{out} exists.")
    if sha256_file(ROOT / D.V66[0]) != D.V66[1]:
        raise ValueError("The uploaded v66 file changed.")
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    b, c, c_rows = D.SH / "rematch/base/vigorous-whistle_v55m_rematch.parquet", D.SH / "Y/vigorous-whistle_v66.parquet", D.SH / "Y/rows.parquet"
    S.SEEDS, S.WEIGHT = SEEDS10[:3], 0.5
    st, rk, v3vals, values, use = D.step_stack_opdi(b, c, c_rows, D.SH / "model_part/member_delta.parquet", out / "steps/stack", 68, AIRPORTS)
    v63like = V3.chain_from_stack(st, 68, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), 68, out / "steps/om2")
    final = out / "steps/om2/vigorous-whistle_v68.parquet"
    rec = {"created_at_utc": utc_now(), "final_sha256": sha256_file(final), "reproduces_v68": sha256_file(final) == V68,
           "stack_step_sha256": sha256_file(st), "opdi_rows_used": int(use.sum())}
    pd.DataFrame({ID: rk[ID].to_numpy(), "values": values}).to_parquet(out / "stack_values.parquet", index=False)
    write_json(out / "identity.json", rec)
    print(rec, flush=True)
    if not rec["reproduces_v68"]:
        raise ValueError("identity: the ranking path does not reproduce v68.")


def rank_check() -> None:
    """The stored per-seed ranking predictions recombined at w50_bag3 equal step_stack_opdi's values; clause 4 for the ten-seed bags."""
    import run_stack_resid as S
    res = read_json(RESULTS)
    P = pd.read_parquet(OUT / "rank" / "preds.parquet")
    ident = pd.read_parquet(OUT / "rank" / "identity_v68" / "stack_values.parquet")
    if not ident[ID].equals(P[ID]):
        raise ValueError("Row order differs.")
    v, _, _ = rank_value(P, 0.5, 3)
    mx = float(np.abs(v - ident["values"].to_numpy()).max())
    res["rank_recombination_vs_step_stack_opdi_max_abs"] = mx
    print("recombined w50_bag3 vs step_stack_opdi values max |diff|", mx, flush=True)
    if mx != 0.0:
        write_json(RESULTS, res)
        raise ValueError("The recombined ranking values differ from step_stack_opdi's.")
    fxr, lirf = P["fx"].to_numpy(dtype=bool), P["airport"].eq("LIRF").to_numpy()
    tg = {}
    for tag, k in [("bag3", 3), ("bag10", 10)]:
        _, use, p = rank_value(P, 0.5, k)
        rk = {"v3": S.by_airport_rms(P, np.where(~lirf & ~fxr, p["p3"], 0.0), ~(~lirf & ~fxr)),
              "v2_LIRF": S.by_airport_rms(P, np.where(lirf & ~fxr, p["p2"], 0.0), ~(lirf & ~fxr)),
              "opdi_LSZH_used": S.by_airport_rms(P, np.where(use, p["po"], 0.0), ~use)}
        dv = res["dev_oof_prediction_by_airport"][tag]
        ratios = {f"{part} {a}": rk[part][a]["rms_s"] / dv[part][a]["rms_s"] for part in rk for a in rk[part] if a in dv[part]}
        tg[tag] = {"ranking_prediction_by_airport": rk, "ratios": ratios, "ok": all(0.5 <= r <= 2.0 for r in ratios.values())}
        print(tag, {k2: round(r, 3) for k2, r in ratios.items()}, "ok", tg[tag]["ok"], flush=True)
    res["clause_4_transfer_guard"] = {**tg, "applies_to": "ten-seed candidates (bag3 reported for reference; the shipped bags' ratios)"}
    sel, best = None, None
    for name, c in res["candidates"].items():
        acc = c["clauses_1_to_3"] and (tg["bag10"]["ok"] if len(c["seeds"]) == 10 else True)
        c["accepted"] = bool(acc)
        if acc and (best is None or c["checks_1_to_3"]["2_equal_footing_mean_pct"] < best):
            sel, best = name, c["checks_1_to_3"]["2_equal_footing_mean_pct"]
    res["selected"] = sel
    write_json(RESULTS, res)
    print("selected", sel, flush=True)


def rank_dry() -> None:
    import run_intruder_bound as R
    import run_own_motion_v2 as O2
    import run_stack_resid_v3 as V3
    import run_opdi_stack_decl as D
    from prc2026_submission.contract import export_submission
    res = read_json(RESULTS)
    sel = res.get("selected")
    if not sel:
        raise ValueError("No candidate is accepted; no ranking file.")
    c = res["candidates"][sel]
    out = OUT / "rank" / "dry_v69"
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite a dry run.")
    P = pd.read_parquet(OUT / "rank" / "preds.parquet")
    values, use, _ = rank_value(P, float(c["weight"]), k_of(c["seeds"]))
    st = out / "steps/stack/vigorous-whistle_v69.parquet"
    export_submission(pd.DataFrame({ID: P[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", st,
                      {"dry_run": True, "step": f"stack_weight_v1 {sel}: stacking value at weight {c['weight']} with {len(c['seeds'])} seeds",
                       "official_scores_used": "none", "audit_used": False})
    v63like = V3.chain_from_stack(st, 69, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), 69, out / "steps/om2")
    final = out / "vigorous-whistle_v69.parquet"
    shutil.copyfile(out / "steps/om2/vigorous-whistle_v69.parquet", final)
    v = pd.read_parquet(final)
    v68 = pd.read_parquet(ROOT / "runs/stack_weight/rank/identity_v68/steps/om2/vigorous-whistle_v68.parquet")
    if sha256_file(ROOT / "runs/stack_weight/rank/identity_v68/steps/om2/vigorous-whistle_v68.parquet") != V68 or not v[ID].equals(v68[ID]):
        raise ValueError("The v68 reference differs.")
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v[ID])
    pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    guard = v[ID].isin(R.GUARD).to_numpy()
    diff = v[TARGET].to_numpy(dtype=float) - v68[TARGET].to_numpy(dtype=float)
    ch = diff != 0
    by = pd.DataFrame({"airport": raw["ADEP_mvt"].to_numpy(), "month": raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy(), "d": diff})
    rep = {"created_at_utc": utc_now(), "candidate": sel, "file": final.relative_to(ROOT).as_posix(), "sha256": sha256_file(final), "bytes": final.stat().st_size,
           "opdi_rows_used_ranking": int(use.sum()),
           "vs_v68": {"rows_changed": int(ch.sum()), "squared_change": float((diff ** 2).sum()), "rms_change_s": float(np.sqrt((diff ** 2).mean())),
                      "max_up_s": float(diff.max()), "max_down_s": float(diff.min()), "mean_change_s": float(diff.mean())},
           "lirf_rows_changed": int((ch & raw["ADEP_mvt"].eq("LIRF").to_numpy()).sum()), "lirf_population_rows_changed": int((ch & pop).sum()),
           "guard_rows_changed": int((ch & guard).sum()), "min_value": int(v[TARGET].min()),
           "by_airport_month": {f"{a} {m}": {"rows_changed": int((g["d"] != 0).sum()), "squared_change": float((g["d"] ** 2).sum()),
                                             "rms_s": float(np.sqrt((g["d"] ** 2).mean())), "max_abs_s": float(g["d"].abs().max())}
                                for (a, m), g in by.groupby(["airport", "month"])},
           "steps": {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted((out / "steps").rglob("vigorous-whistle_v*.parquet"))},
           "official_scores_used": "none"}
    if rep["lirf_population_rows_changed"] or rep["guard_rows_changed"]:
        raise ValueError("A LIRF population or guard row changed against v68.")
    write_json(out / "vigorous-whistle_v69.report.json", rep)
    res["ranking_dry_run"] = {k: rep[k] for k in ["candidate", "file", "sha256", "vs_v68", "lirf_rows_changed", "lirf_population_rows_changed", "guard_rows_changed"]}
    write_json(RESULTS, res)
    print({k: rep[k] for k in ["sha256", "vs_v68", "lirf_rows_changed", "lirf_population_rows_changed", "guard_rows_changed"]}, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-month")
    ap.add_argument("--dev-score", action="store_true")
    ap.add_argument("--rank-fit", action="store_true")
    ap.add_argument("--rank-identity", action="store_true")
    ap.add_argument("--rank-check", action="store_true")
    ap.add_argument("--rank-dry", action="store_true")
    a = ap.parse_args()
    if a.dev_month:
        dev_month(a.dev_month)
    if a.dev_score:
        dev_score()
    if a.rank_identity:
        rank_identity()
    if a.rank_fit:
        rank_fit()
    if a.rank_check:
        rank_check()
    if a.rank_dry:
        rank_dry()


if __name__ == "__main__":
    main()
