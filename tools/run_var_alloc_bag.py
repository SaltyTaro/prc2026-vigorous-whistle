"""Governing run of configs/var_alloc_bag_v1.json (docs/var_alloc_bag/README.md): v57's variance re-allocation with its two LightGBMs
averaged over ten seeds, read in v69's development chain against the same-path single-seed control.

  --dev-layers KIND MONTH   tools/scout_var_alloc.layered on own_all_w100's mixed table rows_<KIND>_<MONTH> (pinned, as tools/run_pubrefit's
                            ship layers), once with the shipped seed alone (s1) and once with the ten-seed bag (b10; the same fitted models,
                            cached) -> runs/var_alloc_bag/dev/layers/<KIND>_<MONTH>.parquet
  --dev-chain               the four months' layers -> runs/var_alloc_bag/dev/chain/{s1,b10}/{ctl,cand}.parquet (+ ship's rows_all, dmp);
                            identity (a): s1 equals runs/pubrefit/layers/ship_{ctl,cand}.parquet 'new' to <= 1e-9 s
  --dev-stack MONTH         v69's stacker (v2 / v3 / OPDI, ten seeds): month-blocked models fitted on the SHIPPED development table
                            (runs/pubrefit/chain/ship), predicting MONTH from each variant's table (features and guards)
                            -> runs/var_alloc_bag/dev/stack/preds_<variant>_<MONTH>.parquet
  --dev-score               weight 0.65, chain (intruder bound, identity extension, own-motion v1, v2) for s1 and b10; identity (b): s1 equals
                            v69's development proxy (c088e409...) to <= 1e-9 s; readings of b10 against s1 (governing) and against the v69
                            proxy -> runs/var_alloc_bag/dev/score.json
  --rank-y                  step 5 (unpinned, as shipped) on the shipped step 1-4 outputs for s1 and b10 -> runs/var_alloc_bag/rank/<variant>/steps/Y;
                            identity: s1 equals the shipped Y (12a50ea0...) byte for byte, Y/rows to <= 1e-9
  --decide                  clauses 1-3 from score.json, clause 4 (transfer guard of b10's correction) -> docs/var_alloc_bag/results_v1.json
  --rank-final VARIANT VERSION   v69's steps 6-10 (tools/rehearse_v69.stage_c's path) on the variant's step 5 -> runs/var_alloc_bag/rank/<variant>/;
                            s1 with VERSION 69 must give v69 (192c4a30...) byte for byte; b10 only if accepted (VERSION 70: the dry run)
No official score enters. Nothing is uploaded; nothing is written under runs/submissions.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_scout import var_alloc as VA  # noqa: E402

DECL = "configs/var_alloc_bag_v1.json"
OUT = ROOT / "runs/var_alloc_bag"
DOCS = ROOT / "docs/var_alloc_bag"
RESULTS = DOCS / "results_v1.json"
SEEDS = [20261002, 11, 12, 13, 14, 15, 16, 17, 18, 19]
VARIANTS = {"s1": 1, "b10": 10}
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
KINDS = ["ctl", "cand"]
SRC = ROOT / "runs/ctx_ownpoints_weight/chain/own_all_w100"
SHIP = ROOT / "runs/pubrefit/chain/ship"
SHIP_LAYERS = ROOT / "runs/pubrefit/layers"
CONTROL = ("runs/stack_weight/dev/w65_bag10/proxy.parquet", "c088e409de153d81b006bcd26b5b2a28e0e306043a5172c2158d0ecc6e6f5cb8")
SH66 = ROOT / "runs/ctx_ownpoints_weight/rank/dry_v66/steps"
Y66 = ("Y/vigorous-whistle_v66.parquet", "12a50ea0011164f6ecdf2df1f227f930a9130b7f669eb523bb311d9e8771bad5")
V69 = ("runs/submissions/vigorous-whistle_v69.parquet", "192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665")
STACK_W, STACK_K = 0.65, 10
TOL = 1e-9
KEEP = [ID, "month", "airport", "bday", "label", "stack", "v48", "ext", "new", "I", "w", "base", "elig", "k"]
CODE = ["tools/run_var_alloc_bag.py", "src/prc2026_scout/var_alloc.py", "tools/scout_var_alloc.py", "tools/scout_extreme_shift.py",
        "tools/run_stack_weight.py", "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py", "tools/run_opdi_stack.py",
        "tools/run_opdi_stack_decl.py", "tools/run_ctx_ownpoints_chain.py", "tools/run_own_motion_v2.py", "tools/run_intruder_bound.py"]

# ---------------------------------------------------------------- the bag (patched into src/prc2026_scout/var_alloc)

_PARAMS0 = dict(VA.PARAMS)
_FIT = VA.fit_models
_CACHE: dict = {}
_SEL = {"k": len(SEEDS)}


def pin(on: bool) -> None:
    VA.PARAMS = {**_PARAMS0, "deterministic": True, "force_col_wise": True} if on else dict(_PARAMS0)


def _key(f: pd.DataFrame, r: np.ndarray, rows: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rows).tobytes())
    h.update(np.ascontiguousarray(np.asarray(r, dtype=float)).tobytes())
    h.update(pd.util.hash_pandas_object(f, index=False).to_numpy().tobytes())
    h.update(repr(sorted(VA.PARAMS.items())).encode())
    return h.hexdigest()


def fit_models_bag(f, a, label, r, rows):
    """var_alloc.fit_models once per seed (only 'seed' changes); cached so that s1 and b10 share the same fitted models."""
    key = _key(f, r, rows)
    if key not in _CACHE:
        base, models = VA.PARAMS, []
        try:
            for s in SEEDS:
                VA.PARAMS = {**base, "seed": s}
                models.append(_FIT(f, a, label, r, rows))
        finally:
            VA.PARAMS = base
        _CACHE[key] = models
    return _CACHE[key]


def weights_bag(models, f, rows):
    """w = exp(mean of the log-variance predictions) x mean of P(late), over the first k seeds (k = 1: var_alloc.weights exactly)."""
    use = models[: _SEL["k"]]
    w = np.full(len(f), np.nan)
    if rows.any():
        lv = np.mean([v.predict(f.loc[rows]) for v, _ in use], axis=0)
        pl = np.mean([l.predict(f.loc[rows]) for _, l in use], axis=0)
        w[rows] = np.exp(lv) * pl
    return w


VA.fit_models, VA.weights = fit_models_bag, weights_bag


def code_hashes() -> dict:
    return {p: sha256_file(ROOT / p) for p in CODE}


# ---------------------------------------------------------------- development: the layer

def dev_layers(kind: str, m: str) -> None:
    import scout_extreme_shift as se
    import scout_var_alloc as sva
    path = OUT / "dev" / "layers" / f"{kind}_{m}.parquet"
    if path.exists():
        raise ValueError(f"{path} exists.")
    pin(True)
    name = f"vab_{kind}_{m}"
    se.TABLES[name] = (SRC / f"rows_{kind}_{m}.parquet", SRC / f"v48_{kind}_{m}")
    outs = {}
    for v, k in VARIANTS.items():
        _SEL["k"] = k
        t, ks = sva.layered(name)
        t = t.assign(elig=VA.eligible(t), k=t["month"].map(ks))
        outs[v] = t.loc[t["month"].eq(m), KEEP].reset_index(drop=True)
        print(kind, m, v, "k", ks[m], flush=True)
    a, b = outs["s1"], outs["b10"]
    if not a[ID].equals(b[ID]):
        raise ValueError("Row order differs between s1 and b10.")
    for c in ["ext", "I", "elig", "v48", "stack", "label"]:
        if not np.array_equal(a[c].to_numpy(), b[c].to_numpy()):
            raise ValueError(f"{c} differs between s1 and b10 (only the weights may change).")
    df = a[[ID, "month", "airport", "bday", "label", "stack", "v48", "ext", "I", "elig"]].copy()
    for v, t in outs.items():
        for c in ["new", "w", "base", "k"]:
            df[f"{c}_{v}"] = t[c].to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "kind": kind, "month": m, "rows": int(len(df)), "sha256": sha256_file(path),
                                           "params_pinned": VA.PARAMS, "seeds": SEEDS, "code": code_hashes()})


def dev_chain() -> None:
    ident = {}
    for kind in KINDS:
        L = pd.concat([pd.read_parquet(OUT / "dev" / "layers" / f"{kind}_{m}.parquet") for m in MONTHS], ignore_index=True)
        ship = pd.read_parquet(SHIP_LAYERS / f"ship_{kind}.parquet")
        if not L[ID].equals(ship[ID]):
            L = L.set_index(ID).reindex(ship[ID]).reset_index()
        d_new = float(np.abs(L["new_s1"].to_numpy() - ship["new"].to_numpy()).max())
        d_k = float(np.abs(L["k_s1"].to_numpy() - ship["k"].to_numpy()).max())
        d_ext = float(np.abs(L["ext"].to_numpy() - ship["ext"].to_numpy()).max())
        ident[kind] = {"s1_new_max_abs_vs_ship": d_new, "s1_k_max_abs": d_k, "ext_max_abs": d_ext,
                       "b10_new_rms_vs_s1": float(np.sqrt(np.mean((L["new_b10"] - L["new_s1"]) ** 2))),
                       "b10_new_max_abs_vs_s1": float(np.abs(L["new_b10"] - L["new_s1"]).max())}
        print(kind, ident[kind], flush=True)
        if max(d_new, d_ext) > TOL:
            write_json(OUT / "dev" / "identity_a_failed.json", ident)
            raise ValueError(f"identity (a): s1 {kind} differs from the ship layers by {max(d_new, d_ext)} s; nothing is read.")
        for v in VARIANTS:
            d = OUT / "dev" / "chain" / v
            d.mkdir(parents=True, exist_ok=True)
            t = L[[ID, "month", "airport", "bday", "label", "stack", "v48", "ext"]].copy()
            t["new"], t["k"] = L[f"new_{v}"].to_numpy(), L[f"k_{v}"].to_numpy()
            t.to_parquet(d / f"{kind}.parquet", index=False)
    for v in VARIANTS:
        for f in ["rows_all.parquet", "dmp.parquet"]:
            shutil.copyfile(SHIP / f, OUT / "dev" / "chain" / v / f)
    write_json(OUT / "dev" / "identity_a.json", {"created_at_utc": utc_now(), "identity_a": ident, "tolerance_s": TOL,
                                                 "files": {p.relative_to(OUT).as_posix(): sha256_file(p) for p in sorted((OUT / "dev" / "chain").rglob("*.parquet"))}})


# ---------------------------------------------------------------- development: v69's stacker and chain

def matrices(d: Path):
    """tools/run_stack_weight.dev_inputs on the chain files in d."""
    import run_ctx_ownpoints_chain as CH
    import run_opdi_stack as O
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    dev = CH.dev_table_from(d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet")
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    new = dev["new"].to_numpy(dtype=float)
    Pn = V3.new_columns(V3.own_points("dev", dev[ID].to_numpy()), new)
    Po = O.opdi_columns("dev", dev[ID].to_numpy(), new)
    Pg = pd.concat([Pn, Po], axis=1)
    return {"dev": dev, "X46": X46, "X65": pd.concat([X46, Pn], axis=1), "X79": pd.concat([X46, Pg], axis=1), "Pn": Pn, "Pg": Pg, "Po": Po, "res": res}


def dev_stack(m: str) -> None:
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_stack_weight as SW
    tr = matrices(SHIP)
    ev_m = {v: matrices(OUT / "dev" / "chain" / v) for v in VARIANTS}
    dev = tr["dev"]
    for v, e in ev_m.items():
        if not e["dev"][ID].equals(dev[ID]):
            raise ValueError(f"Row order of {v}'s table differs from the shipped table.")
    fx = S.fixed_rows(dev)
    month, air, dom = dev["month"].to_numpy(), dev["airport"].to_numpy(), dev["dom"].to_numpy()
    fit, ev = (month != m) & ~fx, month == m
    outs = {}
    for v, e in ev_m.items():
        fxv = S.fixed_rows(e["dev"])
        if not np.array_equal(fxv, fx):
            raise ValueError("Fixed rows differ.")
        h2 = S.outside_support(tr["X46"][fit], air[fit], e["X46"][ev].reset_index(drop=True), air[ev]) & ~fx[ev]
        hn3 = V3.guard_new(tr["Pn"][fit].reset_index(drop=True), air[fit], e["Pn"][ev].reset_index(drop=True), air[ev]) & ~fx[ev]
        hno = V3.guard_new(tr["Pg"][fit].reset_index(drop=True), air[fit], e["Pg"][ev].reset_index(drop=True), air[ev]) & ~fx[ev]
        outs[v] = pd.DataFrame({ID: dev[ID].to_numpy()[ev], "fx": fx[ev], "h2": h2, "h3": h2 | hn3, "ho": h2 | hno,
                                "o_has": e["Po"]["o_has"].to_numpy()[ev]})
    trees = {}
    saved = list(S.SEEDS)
    try:
        for key, X in [("p2", "X46"), ("p3", "X65"), ("po", "X79")]:
            Xa = tr[X].to_numpy(np.float32)
            for i, seed in enumerate(SW.SEEDS10):
                S.SEEDS = [seed]
                models = S.fit_bag(Xa, tr["res"], fit, dom)
                for v, e in ev_m.items():
                    outs[v][f"{key}_{i}"] = models[0].predict(e[X].to_numpy(np.float32)[ev], num_iteration=models[0].best_iteration)
                trees[f"{key} {seed}"] = int(models[0].best_iteration)
                print(m, key, seed, "trees", trees[f"{key} {seed}"], flush=True)
    finally:
        S.SEEDS = saved
    d = OUT / "dev" / "stack"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"created_at_utc": utc_now(), "month": m, "trees": trees, "code": code_hashes(), "files": {}}
    for v, o in outs.items():
        p = d / f"preds_{v}_{m}.parquet"
        o.to_parquet(p, index=False)
        rec["files"][p.name] = sha256_file(p)
    write_json(d / f"preds_{m}.json", rec)


def readings(ctl: pd.DataFrame, cand: np.ndarray) -> dict:
    import run_intruder_bound as R
    tt = ctl[[ID, "month", "label", "day"]].assign(ctl=ctl["value"].to_numpy(dtype=float), cand=cand)
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
    checks["clauses_1_to_3"] = bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"])
    return {"months": months, "checks": checks}


def dev_score() -> None:
    import run_ctx_ownpoints_chain as CH
    import run_own_motion_v2 as O2
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_stack_weight as SW
    path = OUT / "dev" / "score.json"
    if path.exists():
        raise ValueError("Preserve the reading.")
    if sha256_file(ROOT / CONTROL[0]) != CONTROL[1]:
        raise ValueError("The v69 development proxy differs from the declared file.")
    v69 = pd.read_parquet(ROOT / CONTROL[0])
    keys, _ = O2.O.dev_table()
    chain = V3.Chain()
    _, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    proxies = {}
    for v in VARIANTS:
        d = OUT / "dev" / "chain" / v
        dev = CH.dev_table_from(d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet")
        P = pd.concat([pd.read_parquet(OUT / "dev" / "stack" / f"preds_{v}_{m}.parquet") for m in MONTHS], ignore_index=True).set_index(ID).reindex(dev[ID]).reset_index()
        if P["p2_0"].isna().any() or not np.array_equal(P["fx"].to_numpy(dtype=bool), S.fixed_rows(dev)):
            raise ValueError("The stacking predictions do not cover the development rows.")
        kk = keys.set_index(ID).reindex(dev[ID]).reset_index()
        if not np.allclose(kk["label"].to_numpy(), dev["label"].to_numpy()):
            raise ValueError("Row alignment with the v61 proxy failed.")
        value, use, _ = SW.dev_value(dev, P, STACK_W, STACK_K)
        s1 = pd.read_parquet(d / "cand.parquet").set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
        t = kk[[ID, "month", "airport", "label", "cls", "day"]].copy()
        t["s2"] = s1 + (value - dev["new"].to_numpy(dtype=float))
        _, cand63 = chain.run(t, "s2")
        t["c63"] = cand63
        c66 = CH.own_motion_v2(t, "c63", t12, trig)
        out = t[[ID, "month", "airport", "label", "cls", "day"]].copy()
        out["value"] = c66
        out["opdi_used"] = use
        p = OUT / "dev" / "chain" / v / "proxy.parquet"
        out.to_parquet(p, index=False)
        proxies[v] = out
        print(v, "proxy", sha256_file(p), {m: round(float(np.sqrt(np.mean((g["label"] - g["value"]) ** 2))), 4) for m, g in out.groupby("month")}, flush=True)
    a = proxies["s1"].set_index(ID).reindex(v69[ID])
    ident_b = float(np.abs(a["value"].to_numpy() - v69["value"].to_numpy()).max())
    res = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "official_scores_used": "none",
           "audit_used": False, "code": code_hashes(), "identity_a": read_json(OUT / "dev" / "identity_a.json")["identity_a"],
           "identity_b": {"s1_max_abs_vs_v69_dev_proxy": ident_b, "tolerance_s": TOL, "v69_dev_proxy": CONTROL[0], "v69_dev_proxy_sha256": CONTROL[1]},
           "proxies": {v: sha256_file(OUT / "dev" / "chain" / v / "proxy.parquet") for v in VARIANTS}}
    print("identity (b): s1 vs v69 development proxy max |diff|", ident_b, flush=True)
    if ident_b > TOL:
        write_json(OUT / "dev" / "identity_b_failed.json", res)
        raise ValueError("identity (b): s1 does not reproduce v69's development proxy; nothing is read.")
    ctl = proxies["s1"]
    b = proxies["b10"].set_index(ID).reindex(ctl[ID])
    if not np.allclose(b["label"].to_numpy(), ctl["label"].to_numpy()):
        raise ValueError("Row alignment between s1 and b10 failed.")
    res["governing_b10_vs_s1"] = readings(ctl, b["value"].to_numpy(dtype=float))
    v69a = v69.set_index(ID).reindex(ctl[ID]).reset_index()
    res["reported_b10_vs_v69_dev_proxy"] = readings(v69a, b["value"].to_numpy(dtype=float))
    fxm = S.fixed_rows(ctl)
    res["fixed_rows_moved"] = int((np.abs(b["value"].to_numpy() - ctl["value"].to_numpy())[fxm] > 1e-9).sum())
    write_json(path, res)
    g = res["governing_b10_vs_s1"]
    print("b10 vs s1", " ".join(f"{m[5:]} {g['months'][m]['pct']:+.4f}% [{g['months'][m]['lo']:+.4f},{g['months'][m]['hi']:+.4f}]" for m in MONTHS),
          "| eq", round(g["checks"]["2_equal_footing_mean_pct"], 4), "clauses 1-3", g["checks"]["clauses_1_to_3"], flush=True)


# ---------------------------------------------------------------- ranking

def rank_y() -> None:
    import scout_var_alloc as sva
    pin(False)   # step 5 runs as shipped in the ranking chain
    x = SH66 / "X/vigorous-whistle_v66.parquet"
    rec = {"created_at_utc": utc_now(), "params": VA.PARAMS, "seeds": SEEDS, "code": code_hashes(), "variants": {}}
    for v, k in VARIANTS.items():
        steps = OUT / "rank" / v / "steps"
        if (steps / "Y").exists():
            raise ValueError(f"{steps / 'Y'} exists.")
        steps.mkdir(parents=True, exist_ok=True)
        for sub in ["rematch/base", "model_part", "R", "X"]:
            shutil.copytree(SH66 / sub, steps / sub)
        (steps / "Y").mkdir()
        _SEL["k"] = k
        args = SimpleNamespace(declaration=None, dry_run=(steps / "Y").resolve(), version=66, user_override=None,
                               base=(SH66 / "rematch/base/vigorous-whistle_v55m_rematch.parquet").relative_to(ROOT).as_posix(),
                               recon=(SH66 / "R").relative_to(ROOT).as_posix(), previous=x.relative_to(ROOT).as_posix(), previous_sha256=sha256_file(x))
        sva.export(args)
        y = steps / Y66[0]
        rows = pd.read_parquet(steps / "Y/rows.parquet")
        ref = pd.read_parquet(SH66 / "Y/rows.parquet")
        num = [c for c in rows.columns if c in ref.columns and pd.api.types.is_numeric_dtype(rows[c])]
        side = {c: float(np.nanmax(np.abs(rows[c].to_numpy(dtype=float) - ref[c].to_numpy(dtype=float)))) for c in num} if rows[ID].equals(ref[ID]) else None
        rec["variants"][v] = {"Y_sha256": sha256_file(y), "Y_equals_shipped": sha256_file(y) == Y66[1], "rows_max_abs_vs_shipped": side,
                              "k": read_json(steps / "Y/export_v66.json")["k"]}
        print(v, {kk: vv for kk, vv in rec["variants"][v].items() if kk != "rows_max_abs_vs_shipped"}, flush=True)
    write_json(OUT / "rank" / "y.json", rec)
    s1 = rec["variants"]["s1"]
    if not s1["Y_equals_shipped"] or s1["rows_max_abs_vs_shipped"] is None or max(v for v in s1["rows_max_abs_vs_shipped"].values() if np.isfinite(v)) > TOL:
        raise ValueError("ranking identity: s1's step 5 does not reproduce the shipped step 5.")


def guard_table() -> dict:
    """Clause 4: RMS of b10's correction clip(k x base, +-100) on eligible rows, ranking (all four months' k) against development (out of fold, cand)."""
    dev = pd.concat([pd.read_parquet(OUT / "dev" / "layers" / f"cand_{m}.parquet") for m in MONTHS], ignore_index=True)
    e = dev["elig"].to_numpy(dtype=bool)
    cd = np.clip(dev["k_b10"].to_numpy() * dev["base_b10"].to_numpy(), -VA.CAP_S, VA.CAP_S)
    rows = pd.read_parquet(OUT / "rank" / "b10" / "steps/Y/rows.parquet")
    k = float(read_json(OUT / "rank" / "b10" / "steps/Y/export_v66.json")["k"])
    er = rows["w"].notna().to_numpy()
    cr = np.clip(k * rows["base"].to_numpy(dtype=float), -VA.CAP_S, VA.CAP_S)
    out = {}
    for a in sorted(set(dev["airport"]) | set(rows["airport"])):
        md, mr = e & dev["airport"].eq(a).to_numpy(), er & rows["airport"].eq(a).to_numpy()
        rd = float(np.sqrt(np.mean(cd[md] ** 2))) if md.any() else None
        rr = float(np.sqrt(np.mean(cr[mr] ** 2))) if mr.any() else None
        ratio = rr / rd if rd and rr is not None else None
        ruled = bool(md.sum() >= 200 and mr.sum() >= 200)
        out[a] = {"dev_rows": int(md.sum()), "rank_rows": int(mr.sum()), "dev_rms_s": rd, "rank_rms_s": rr, "ratio": ratio, "ruled": ruled,
                  "ok": (ratio is not None and 0.5 <= ratio <= 2.0) if ruled else None}
    return out


def decide() -> None:
    if RESULTS.exists():
        raise ValueError("Preserve the result.")
    sc = read_json(OUT / "dev" / "score.json")
    y = read_json(OUT / "rank" / "y.json")
    g = guard_table()
    c4 = all(v["ok"] for v in g.values() if v["ruled"])
    gov = sc["governing_b10_vs_s1"]
    accepted = bool(gov["checks"]["clauses_1_to_3"] and c4)
    res = {**sc, "rank_step5": y, "clause_4_transfer_guard": {"by_airport": g, "ok": c4}, "accepted": accepted,
           "selected": "va_bag10" if accepted else None, "created_at_utc_decision": utc_now()}
    DOCS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS, res)
    print("clauses 1-3", gov["checks"]["clauses_1_to_3"], "| clause 4", c4, {a: round(v["ratio"], 2) for a, v in g.items() if v["ratio"]}, "| accepted", accepted, flush=True)


def rank_final(v: str, version: int) -> None:
    """v69's steps 6-10 (tools/rehearse_v69.stage_c's arithmetic) on the variant's step 5."""
    import run_opdi_stack_decl as D
    import run_own_motion_v2 as O2
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_stack_weight as SW
    from prc2026_submission.contract import export_submission
    if v == "b10":
        res = read_json(RESULTS)
        if not res.get("accepted") or version == 69:
            raise ValueError("b10's ranking file only when accepted, and never as v69.")
    elif version != 69:
        raise ValueError("s1 is the identity run (version 69).")
    out = OUT / "rank" / v
    D.SH = out / "steps"
    dev, res_t, fx, mats, rk, fxr, h2, h3, ho, Por = SW.rank_inputs()
    dom = dev["dom"].to_numpy()
    P = pd.DataFrame({ID: rk[ID].to_numpy(), "airport": rk["airport"].to_numpy(), "new": rk["new"].to_numpy(dtype=float), "fx": fxr,
                      "h2": h2, "h3": h3, "ho": ho, "o_has": Por["o_has"].to_numpy()})
    saved = list(S.SEEDS)
    try:
        for key, X, Xr in mats:
            Xa, Xra = X.to_numpy(np.float32), Xr.to_numpy(np.float32)
            for i, seed in enumerate(SW.SEEDS10[:STACK_K]):
                S.SEEDS = [seed]
                models = S.fit_bag(Xa, res_t, ~fx, dom)
                P[f"{key}_{i}"] = models[0].predict(Xra, num_iteration=models[0].best_iteration)
                print("rank", v, key, seed, flush=True)
    finally:
        S.SEEDS = saved
    P.to_parquet(out / "preds.parquet", index=False)
    values, use, _ = SW.rank_value(P, STACK_W, STACK_K)
    st = out / f"steps/stack/vigorous-whistle_v{version}.parquet"
    export_submission(pd.DataFrame({ID: P[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", st,
                      {"dry_run": True, "step": f"var_alloc_bag_v1 {v}: step 5 with the variance re-allocation bag, then v69's stacker (w 0.65, ten seeds)",
                       "official_scores_used": "none", "audit_used": False})
    v63like = V3.chain_from_stack(st, version, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), version, out / "steps/om2")
    d = out / f"dry_v{version}"
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"vigorous-whistle_v{version}.parquet"
    shutil.copyfile(out / f"steps/om2/vigorous-whistle_v{version}.parquet", final)
    rep = {"created_at_utc": utc_now(), "variant": v, "version": version, "file": final.relative_to(ROOT).as_posix(), "sha256": sha256_file(final),
           "opdi_rows_used": int(use.sum()), "code": code_hashes()}
    if v == "s1":
        rep["equals_v69"] = rep["sha256"] == V69[1]
        print("ranking identity: s1 reproduces v69:", rep["equals_v69"], rep["sha256"], flush=True)
    else:
        a, b = pd.read_parquet(ROOT / V69[0]), pd.read_parquet(final)
        if not a[ID].equals(b[ID]):
            raise ValueError("Row order differs.")
        raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
        raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(b[ID])
        pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
        guard = b[ID].isin([213142471.0, 213142469.0]).to_numpy()
        dch = b[TARGET].to_numpy(dtype=float) - a[TARGET].to_numpy(dtype=float)
        ap = raw["ADEP_mvt"].to_numpy()
        mo = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m").to_numpy()
        rep.update({"vs_v69": {"rows_changed": int((dch != 0).sum()), "squared_change": float((dch ** 2).sum()), "rms_s": float(np.sqrt(np.mean(dch ** 2))),
                               "max_up_s": float(dch.max()), "max_down_s": float(dch.min()), "lirf_population_rows_changed": int(((dch != 0) & pop).sum()),
                               "guard_rows_changed": int(((dch != 0) & guard).sum()),
                               "by_airport_month": {f"{x} {m}": {"rows_changed": int((dch[(ap == x) & (mo == m)] != 0).sum()),
                                                                 "squared_change": float((dch[(ap == x) & (mo == m)] ** 2).sum())}
                                                    for x in sorted(set(ap)) for m in sorted(set(mo))}},
                    "min_value": int(b[TARGET].min()), "dtype": str(b[TARGET].dtype)})
        if rep["vs_v69"]["lirf_population_rows_changed"] or rep["vs_v69"]["guard_rows_changed"]:
            raise ValueError("A LIRF population or guard row changed.")
        print("b10 dry run", rep["sha256"], {k2: rep["vs_v69"][k2] for k2 in ["rows_changed", "squared_change", "max_up_s", "max_down_s"]}, flush=True)
    write_json(d / f"report_v{version}.json", rep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-layers", nargs=2, metavar=("KIND", "MONTH"))
    ap.add_argument("--dev-chain", action="store_true")
    ap.add_argument("--dev-stack", metavar="MONTH")
    ap.add_argument("--dev-score", action="store_true")
    ap.add_argument("--rank-y", action="store_true")
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--rank-final", nargs=2, metavar=("VARIANT", "VERSION"))
    a = ap.parse_args()
    if a.dev_layers:
        dev_layers(*a.dev_layers)
    if a.dev_chain:
        dev_chain()
    if a.dev_stack:
        dev_stack(a.dev_stack)
    if a.dev_score:
        dev_score()
    if a.rank_y:
        rank_y()
    if a.decide:
        decide()
    if a.rank_final:
        rank_final(a.rank_final[0], int(a.rank_final[1]))


if __name__ == "__main__":
    main()
