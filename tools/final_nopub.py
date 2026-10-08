"""Lane P (docs/final_file/lane_P.md): the chain on the 2025 development months when EUROCONTROL's published daily figures are absent.

Without published figures steps 3-5 (v48 reconciliation, extreme-day, variance re-allocation) change nothing, so the chain entering step 6
is the model part plus the no-NM rematch (`runs/pubrefit/chain/ship/cand.parquet` `stack`; the stacker's own base, as shipped, is the
rematch-free `ctl.parquet` `stack`), and the nine published-day stacker columns are missing. Variants of step 6 (v69's form: learners
v2 / v3 / OPDI, ten seeds, weight 0.65, guards as shipped), each then through steps 7-10 exactly as tools/run_stack_weight.dev_score:

  P1  the shipped learners (month-blocked fits on the published chain, as shipped) predicting from the no-published features (NaN, which
      LightGBM 4.7 reads as 0.0 for a feature never missing in training);
  P2  the same learners / seeds / weight refitted month-blocked on the no-published chain with the nine published columns dropped;
  P3  no step 6.
Reference (i): the shipped v69-level development proxy (runs/stack_weight/dev/w65_bag10/proxy.parquet, published figures present).

  build              (workstation, light) feature matrices -> runs/final_nopub/dev/{xpub,xnp,meta}.parquet
  fit VARIANT MONTH  (Azure) one held-out month's ten-seed predictions -> runs/final_nopub/dev/preds_<VARIANT>_<MONTH>.parquet
                     (P1 also predicts the published features of the held-out month: identity against runs/stack_weight/dev/preds_<M>)
  score              (workstation) guards, values, steps 7-10, readings -> docs/final_file/lane_P_results.json
No label of the ranking file, no official score; nothing is uploaded or written under runs/submissions.
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
from prc2026.features import ID  # noqa: E402

OUT = ROOT / "runs/final_nopub/dev"
SHIP = ROOT / "runs/pubrefit/chain/ship"
REF = ROOT / "runs/stack_weight/dev/w65_bag10/proxy.parquet"
RESULTS = ROOT / "docs/final_file/lane_P_results.json"
PUB_COLS = ["p_pub", "p_n_pub", "p_disc_new", "p_disc_m", "p_disc_new_pr", "p_disc_m_pr", "p_pub_pr", "p_n_ratio", "p_complete"]
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
WEIGHT, K = 0.65, 10
WIDTHS = {"p2": 46, "p3": 65, "po": 79}


def _unique(cols: list[str]) -> list[str]:
    seen, out = {}, []
    for c in cols:
        seen[c] = seen.get(c, -1) + 1
        out.append(c if seen[c] == 0 else f"{c}__{seen[c]}")
    return out


def no_published(dev: pd.DataFrame) -> pd.DataFrame:
    """The development table as step 6 would see it without published figures: base = the rematch-free model part, no published day."""
    ctl = pd.read_parquet(SHIP / "ctl.parquet", columns=[ID, "stack"]).set_index(ID)["stack"].reindex(dev[ID]).to_numpy(dtype=float)
    out = dev.copy()
    out["new"] = ctl
    out["pub"], out["n_pub"], out["complete"] = np.nan, np.nan, False
    return out


def matrices(dev: pd.DataFrame):
    import run_opdi_stack as O
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    new = dev["new"].to_numpy(dtype=float)
    Pn = V3.new_columns(V3.own_points("dev", dev[ID].to_numpy()), new)
    Po = O.opdi_columns("dev", dev[ID].to_numpy(), new)
    X79 = pd.concat([X46, Pn, Po], axis=1)
    return X79, res, Po["o_has"].to_numpy()


def build() -> None:
    import run_stack_resid as S
    import run_stack_weight as W
    dev, X46, X65, X79, Pn, Pg, Po, res = W.dev_inputs()
    devn = no_published(dev)
    Xn, resn, ohas_n = matrices(devn)
    if list(Xn.columns) != list(X79.columns):
        raise ValueError("Column order differs between the published and no-published matrices.")
    OUT.mkdir(parents=True, exist_ok=True)
    cols = _unique(list(X79.columns))
    X79.set_axis(cols, axis=1).to_parquet(OUT / "xpub.parquet", index=False)
    Xn.set_axis(cols, axis=1).to_parquet(OUT / "xnp.parquet", index=False)
    meta = pd.DataFrame({ID: dev[ID].to_numpy(), "month": dev["month"].to_numpy(), "airport": dev["airport"].to_numpy(), "dom": dev["dom"].to_numpy(),
                         "fx": S.fixed_rows(dev), "res_pub": res, "res_np": resn, "new_pub": dev["new"].to_numpy(dtype=float),
                         "new_np": devn["new"].to_numpy(dtype=float), "o_has_pub": Po["o_has"].to_numpy(), "o_has_np": ohas_n})
    meta.to_parquet(OUT / "meta.parquet", index=False)
    write_json(OUT / "build.json", {"created_at_utc": utc_now(), "files": {p.name: sha256_file(p) for p in sorted(OUT.glob("*.parquet"))},
                                    "columns": cols, "published_columns_dropped_in_P2": PUB_COLS,
                                    "nan_share_np": {c: float(Xn[c].isna().mean()) if Xn.columns.tolist().count(c) == 1 else None for c in PUB_COLS},
                                    "code": sha256_file(ROOT / "tools/final_nopub.py")})
    print("built", X79.shape, "nan in np published columns:", {c: round(float(Xn[c].isna().mean()), 3) for c in PUB_COLS if c != "p_complete"},
          "p_complete np mean", float(Xn["p_complete"].mean()), flush=True)


def keep_positions(cols: list[str], width: int, variant: str) -> list[int]:
    base = list(range(width))
    if variant == "P2":
        return [i for i in base if not (i < 46 and cols[i] in PUB_COLS)]
    return base


def fit(variant: str, month: str) -> None:
    import run_stack_resid as S
    import run_stack_weight as W
    path = OUT / f"preds_{variant}_{month}.parquet"
    if path.exists():
        raise ValueError(f"{path} exists.")
    meta = pd.read_parquet(OUT / "meta.parquet")
    xpub = pd.read_parquet(OUT / "xpub.parquet")
    xnp = pd.read_parquet(OUT / "xnp.parquet")
    cols = list(xpub.columns)
    fx, mon, dom = meta["fx"].to_numpy(dtype=bool), meta["month"].to_numpy(), meta["dom"].to_numpy()
    fitm, ev = (mon != month) & ~fx, mon == month
    Xfit_all = (xpub if variant == "P1" else xnp).to_numpy(np.float32)
    Xev_all = xnp.to_numpy(np.float32)
    res = meta["res_pub" if variant == "P1" else "res_np"].to_numpy(dtype=float)
    out = pd.DataFrame({ID: meta[ID].to_numpy()[ev]})
    trees = {}
    for key, width in WIDTHS.items():
        pos = keep_positions(cols, width, variant)
        Xf, Xe = Xfit_all[:, pos], Xev_all[ev][:, pos]
        Xp = xpub.to_numpy(np.float32)[ev][:, pos] if variant == "P1" else None
        for i, seed in enumerate(W.SEEDS10):
            S.SEEDS = [seed]
            m = S.fit_bag(Xf, res, fitm, dom)[0]
            out[f"{key}_{i}"] = m.predict(Xe, num_iteration=m.best_iteration)
            if Xp is not None:
                out[f"{key}_{i}_pubev"] = m.predict(Xp, num_iteration=m.best_iteration)
            trees[f"{key} {seed}"] = int(m.best_iteration)
            print(variant, month, key, seed, "trees", trees[f"{key} {seed}"], flush=True)
    out.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "variant": variant, "month": month, "rows": int(ev.sum()), "trees": trees,
                                           "sha256": sha256_file(path), "inputs": {p: sha256_file(OUT / p) for p in ["xpub.parquet", "xnp.parquet", "meta.parquet"]},
                                           "code": {p: sha256_file(ROOT / p) for p in ["tools/final_nopub.py", "tools/run_stack_resid.py", "tools/run_stack_weight.py"]}})


def guards(variant: str, meta: pd.DataFrame, xpub: pd.DataFrame, xnp: pd.DataFrame, month: str) -> dict:
    """tools/run_stack_weight.dev_month's guards for one held-out month: fitting rows from the variant's training matrix, target rows from
    the no-published matrix; P2 without the dropped published columns (its support guard on the remaining TOP features)."""
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    fx, mon, air = meta["fx"].to_numpy(dtype=bool), meta["month"].to_numpy(), meta["airport"].to_numpy()
    fitm, ev = (mon != month) & ~fx, mon == month
    cols = list(xpub.columns)
    Xf = (xpub if variant == "P1" else xnp)
    X46f, X46e = Xf.iloc[:, :46], xnp.iloc[:, :46]
    top_saved = list(S.TOP)
    try:
        if variant == "P2":
            S.TOP = [c for c in S.TOP if c not in PUB_COLS]
        h2 = S.outside_support(X46f[fitm], air[fitm], X46e[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    finally:
        S.TOP = top_saved
    Pn_f, Pn_e = Xf.iloc[:, 46:65], xnp.iloc[:, 46:65]
    Pg_f, Pg_e = Xf.iloc[:, 46:79], xnp.iloc[:, 46:79]
    hn3 = V3.guard_new(Pn_f[fitm].reset_index(drop=True), air[fitm], Pn_e[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hno = V3.guard_new(Pg_f[fitm].reset_index(drop=True), air[fitm], Pg_e[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    _ = cols
    return {"h2": h2, "h3": h2 | hn3, "ho": h2 | hno}


def score() -> None:
    import run_ctx_ownpoints_chain as CH
    import run_intruder_bound as R
    import run_opdi_stack as O
    import run_own_motion_v2 as O2
    import run_stack_resid_v3 as V3
    import run_stack_weight as W
    dev = CH.dev_table_from(O.SHIP / "ctl.parquet", O.SHIP / "rows_all.parquet", O.SHIP / "dmp.parquet")
    meta = pd.read_parquet(OUT / "meta.parquet")
    if not np.array_equal(meta[ID].to_numpy(), dev[ID].to_numpy()):
        raise ValueError("The stored matrices are not in the development table's order.")
    xpub, xnp = pd.read_parquet(OUT / "xpub.parquet"), pd.read_parquet(OUT / "xnp.parquet")
    keys, _ = O2.O.dev_table()
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    ref = pd.read_parquet(REF).set_index(ID).reindex(dev[ID]).reset_index()
    cand = pd.read_parquet(SHIP / "cand.parquet", columns=[ID, "stack", "new"]).set_index(ID).reindex(dev[ID])
    s1_np = cand["stack"].to_numpy(dtype=float)
    new_np = meta["new_np"].to_numpy(dtype=float)
    chain = V3.Chain()
    _, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    devn = dev.assign(new=new_np)

    def through(s2: np.ndarray, name: str) -> np.ndarray:
        t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
        t["s2"] = s2
        _, c63 = chain.run(t, "s2")
        t["c63"] = c63
        v = CH.own_motion_v2(t, "c63", t12, trig)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        t[[ID, "month", "airport", "label", "cls", "day"]].assign(value=v).to_parquet(d / "proxy.parquet", index=False)
        return v

    values = {"P3": through(s1_np, "P3")}
    identity = {}
    for variant in ["P1", "P2"]:
        files = [OUT / f"preds_{variant}_{m}.parquet" for m in MONTHS]
        if not all(f.exists() for f in files):
            print("missing predictions for", variant, flush=True)
            continue
        P = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).set_index(ID).reindex(dev[ID]).reset_index()
        for m in MONTHS:
            g = guards(variant, meta, xpub, xnp, m)
            ev = meta["month"].to_numpy() == m
            for k in ["h2", "h3", "ho"]:
                P.loc[ev, k] = g[k]
        P["fx"] = meta["fx"].to_numpy(dtype=bool)
        P["o_has"] = meta["o_has_np"].to_numpy()
        for k in ["h2", "h3", "ho"]:
            P[k] = P[k].astype(bool)
        if variant == "P1":
            shipped = pd.concat([pd.read_parquet(W.OUT / "dev" / f"preds_{m}.parquet") for m in MONTHS], ignore_index=True).set_index(ID).reindex(dev[ID])
            identity = {f"{key}_{i}": float(np.abs(P[f"{key}_{i}_pubev"].to_numpy() - shipped[f"{key}_{i}"].to_numpy()).max())
                        for key in WIDTHS for i in range(K)}
            print("P1 identity (refit on the published chain vs the shipped development predictions) max |diff|", max(identity.values()), flush=True)
        value, use, _ = W.dev_value(devn, P, WEIGHT, K)
        values[variant] = through(s1_np + (value - new_np), variant)
    rng = np.random.default_rng(20261004)
    out = {"created_at_utc": utc_now(), "reference": REF.relative_to(ROOT).as_posix(), "reference_sha256": sha256_file(REF),
           "identity_P1_max_abs": identity, "official_scores_used": "none", "readings": {},
           "code": sha256_file(ROOT / "tools/final_nopub.py")}
    tt = ref[[ID, "month", "label", "day"]].copy()
    tt["ref"] = ref["value"].to_numpy(dtype=float)
    comps = [("P3", "ref"), *[(v, "ref") for v in ["P1", "P2"] if v in values], *[(v, "P3") for v in ["P1", "P2"] if v in values]]
    for name, against in comps:
        tt["cand"], tt["base"] = values[name], (tt["ref"] if against == "ref" else values["P3"])
        months = {}
        for m in MONTHS:
            s = tt[tt["month"] == m]
            pct, lo, hi = R.interval(s.assign(value=s["base"]), "value", s["cand"].to_numpy(), rng)
            months[m] = {"rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))), "base_rmse": float(np.sqrt(np.mean((s["label"] - s["base"]) ** 2))),
                         "pct": pct, "lo": lo, "hi": hi}
        eq = float(np.mean([months[m]["pct"] for m in EQUAL]))
        out["readings"][f"{name}_vs_{against}"] = {"months": months, "equal_footing_mean_pct": eq}
        print(f"{name} vs {against}:", " ".join(f"{m[5:]} {months[m]['pct']:+.4f}% [{months[m]['lo']:+.3f},{months[m]['hi']:+.3f}]" for m in MONTHS),
              f"| eq {eq:+.4f}%", flush=True)
    write_json(RESULTS, out)


# ---------------------------------------------------------------- the second pass (v72) without published figures

PASS_AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LSZH"]   # configs/stack_second_v1.json sp_hicov7 (shipped)
S_PUB = [f"s_{c}" for c in PUB_COLS]


def pass2_build(base_variant: str) -> None:
    """The second pass's development table on the no-published chain: v70 / v71 = the variant's v69-level values plus the shipped
    development deltas of steps 11-13 (data-link pin, neighbour state and dark; none reads published figures; an approximation of
    re-running them on the new base), s_* = the no-published stacker inputs."""
    T = pd.read_parquet(ROOT / "runs/stack_second/dev_table.parquet")
    ids = T[ID].to_numpy()
    ref = pd.read_parquet(REF).set_index(ID)["value"].reindex(ids).to_numpy(dtype=float)
    base = pd.read_parquet(OUT / base_variant / "proxy.parquet").set_index(ID)["value"].reindex(ids).to_numpy(dtype=float)
    meta = pd.read_parquet(OUT / "meta.parquet")
    xpub = pd.read_parquet(OUT / "xpub.parquet").set_index(meta[ID].to_numpy()).reindex(ids)
    xnp = pd.read_parquet(OUT / "xnp.parquet").set_index(meta[ID].to_numpy()).reindex(ids)
    s_cols = [c for c in T.columns if c.startswith("s_")]
    if len(s_cols) != xpub.shape[1]:
        raise ValueError("s_* width differs")
    ident = float(np.nanmax(np.abs(T[s_cols].to_numpy(dtype=np.float64) - xpub.to_numpy(dtype=np.float64))))
    nan_same = bool((T[s_cols].isna().to_numpy() == xpub.isna().to_numpy()).all())
    out = T.copy()
    out["v70"] = base + (T["v70"].to_numpy(dtype=float) - ref)
    out["v71"] = base + (T["v71"].to_numpy(dtype=float) - ref)
    out[s_cols] = xnp.to_numpy(dtype=np.float32)
    out.to_parquet(OUT / f"pass2_{base_variant}.parquet", index=False)
    print("pass2 table on", base_variant, "| s_* (published) vs stored dev table: max |diff|", ident, "NaN pattern equal", nan_same, flush=True)


def pass2_fit(variant: str, base_variant: str, month: str) -> None:
    import lightgbm as lgb
    import run_stack_second as SS
    T = pd.read_parquet(ROOT / "runs/stack_second/dev_table.parquet")
    Tn = pd.read_parquet(OUT / f"pass2_{base_variant}.parquet")
    if not np.array_equal(T[ID].to_numpy(), Tn[ID].to_numpy()):
        raise ValueError("row order")
    F = SS.features(T)
    if variant == "L2":
        F = [c for c in F if c not in S_PUB]
    Tf = T if variant == "L1" else Tn
    y = np.clip((Tf["label"] - Tf["v71"]).to_numpy(), -SS.CLIP_Y, SS.CLIP_Y)
    inset = T["airport"].isin(PASS_AIRPORTS).to_numpy() & (T["kind"] != 3).to_numpy()
    f, t = inset & (T["month"] != month).to_numpy(), inset & (T["month"] == month).to_numpy()
    Xf, Xt = Tf[F].astype(np.float32)[f], Tn[F].astype(np.float32)[t]
    acc = np.zeros(int(t.sum()))
    for sd in SS.SEEDS:
        mdl = lgb.train({**SS.PARAMS, "seed": sd}, lgb.Dataset(Xf, y[f], categorical_feature=["kind", "ap"]), SS.ROUNDS)
        acc += mdl.predict(Xt) / len(SS.SEEDS)
        print(variant, base_variant, month, "seed", sd, flush=True)
    path = OUT / f"pass2_preds_{variant}_{base_variant}_{month}.parquet"
    pd.DataFrame({ID: T[ID].to_numpy()[t], "p": acc}).to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "rows": int(t.sum()), "features": len(F), "sha256": sha256_file(path),
                                           "code": {q: sha256_file(ROOT / q) for q in ["tools/final_nopub.py", "tools/run_stack_second.py"]}})


def pass2_score(base_variant: str) -> None:
    import run_intruder_bound as R
    T = pd.read_parquet(ROOT / "runs/stack_second/dev_table.parquet", columns=[ID, "month", "airport", "day", "label", "v71", "kind"])
    Tn = pd.read_parquet(OUT / f"pass2_{base_variant}.parquet", columns=[ID, "v71"])
    pd_dev = pd.read_parquet(ROOT / "runs/stack_second/preds_dev.parquet").set_index(ID)["sp_hicov7"].reindex(T[ID]).to_numpy()

    def apply(b, p):
        n = b.copy()
        m = ~np.isnan(p)
        n[m] = np.maximum(b[m] + np.clip(0.5 * p[m], -120, 120), -12)
        return n

    v72_ref = apply(T["v71"].to_numpy(dtype=float), pd_dev)
    base = Tn["v71"].to_numpy(dtype=float)
    vals = {"L3": base}
    for v in ["L1", "L2"]:
        files = [OUT / f"pass2_preds_{v}_{base_variant}_{m}.parquet" for m in MONTHS]
        if not all(f.exists() for f in files):
            continue
        p = pd.concat([pd.read_parquet(f) for f in files]).set_index(ID)["p"].reindex(T[ID]).to_numpy()
        vals[v] = apply(base, p)
    rng = np.random.default_rng(20261004)
    out = {}
    tt = T[[ID, "month", "label", "day"]].copy()
    for name, against in [("L3", "ref"), *[(v, "ref") for v in ["L1", "L2"] if v in vals], *[(v, "L3") for v in ["L1", "L2"] if v in vals]]:
        tt["cand"], tt["base"] = vals[name], (v72_ref if against == "ref" else vals["L3"])
        months = {}
        for m in MONTHS:
            s = tt[tt["month"] == m]
            pct, lo, hi = R.interval(s.assign(value=s["base"]), "value", s["cand"].to_numpy(), rng)
            months[m] = {"pct": pct, "lo": lo, "hi": hi, "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2)))}
        eq = float(np.mean([months[m]["pct"] for m in EQUAL]))
        out[f"{name}_vs_{against}"] = {"months": months, "equal_footing_mean_pct": eq}
        print(f"pass2 on {base_variant}: {name} vs {against}:", " ".join(f"{m[5:]} {months[m]['pct']:+.4f}% [{months[m]['lo']:+.3f},{months[m]['hi']:+.3f}]" for m in MONTHS),
              f"| eq {eq:+.4f}%", flush=True)
    res = __import__("json").loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    res.setdefault("pass2", {})[base_variant] = {"readings": out, "note": "steps 11-13 carried as the shipped development deltas (approximation)"}
    write_json(RESULTS, res)


# ---------------------------------------------------------------- the frozen P2 learners for the final file

FINAL = ROOT / "runs/final_nopub/final_p2"


def fit_final() -> None:
    """P2's ranking learners: the three learners x ten seeds fitted on all four development months of the no-published chain with the
    nine published columns dropped (tools/run_stack_weight.rank_fit's fit, ~fixed rows, the same day-of-month validation split), saved as
    LightGBM model files so the final chain only predicts (tools/final_early_chain.py --published-missing auto)."""
    import run_stack_resid as S
    import run_stack_weight as W
    if FINAL.exists():
        raise ValueError(f"{FINAL} exists.")
    meta = pd.read_parquet(OUT / "meta.parquet")
    xnp = pd.read_parquet(OUT / "xnp.parquet")
    cols = list(xnp.columns)
    fx, dom = meta["fx"].to_numpy(dtype=bool), meta["dom"].to_numpy()
    res = meta["res_np"].to_numpy(dtype=float)
    X = xnp.to_numpy(np.float32)
    FINAL.mkdir(parents=True)
    manifest = {"created_at_utc": utc_now(), "inputs": {q: sha256_file(OUT / q) for q in ["xnp.parquet", "meta.parquet"]}, "learners": {},
                "dropped_columns": PUB_COLS, "seeds": W.SEEDS10, "fit_rows": int((~fx).sum())}
    saved = list(S.SEEDS)
    try:
        for key, width in WIDTHS.items():
            pos = keep_positions(cols, width, "P2")
            manifest["learners"][key] = {"positions": pos, "columns": [cols[i] for i in pos], "models": []}
            for seed in W.SEEDS10:
                S.SEEDS = [seed]
                m = S.fit_bag(X[:, pos], res, ~fx, dom)[0]
                f = FINAL / f"{key}_{seed}.txt"
                m.save_model(str(f), num_iteration=m.best_iteration)
                manifest["learners"][key]["models"].append({"seed": seed, "file": f.name, "best_iteration": int(m.best_iteration), "sha256": sha256_file(f)})
                print("final", key, seed, "trees", m.best_iteration, flush=True)
    finally:
        S.SEEDS = saved
    manifest["code"] = {q: sha256_file(ROOT / q) for q in ["tools/final_nopub.py", "tools/run_stack_resid.py", "tools/run_stack_weight.py"]}
    write_json(FINAL / "manifest.json", manifest)


# ---------------------------------------------------------------- the third pass (v73) on the no-published chain

T3_AIRPORTS = ["EDDF", "EHAM", "LSZH"]   # configs/stack_third_v1.json t3_ro_3 (shipped)


def _apply(b: np.ndarray, p: np.ndarray) -> np.ndarray:
    n = b.copy()
    m = ~np.isnan(p)
    n[m] = np.maximum(b[m] + np.clip(0.5 * p[m], -120, 120), -12)
    return n


def pass3_build(base_variant: str) -> None:
    """The third pass's development table on the no-published chain with the shipped second pass (L1): p2 = L1's out-of-month bag,
    ship = v72 on that chain, d2 = v72 - v71, v71 as in pass2_<base>; raw and clock columns unchanged."""
    T = pd.read_parquet(ROOT / "runs/stack_third/dev_t3.parquet")
    ids = T[ID].to_numpy()
    T2 = pd.read_parquet(OUT / f"pass2_{base_variant}.parquet", columns=[ID, "v70", "v71"]).set_index(ID).reindex(ids)
    p = pd.concat([pd.read_parquet(OUT / f"pass2_preds_L1_{base_variant}_{m}.parquet") for m in MONTHS]).set_index(ID)["p"].reindex(ids).to_numpy()
    v71 = T2["v71"].to_numpy(dtype=float)
    v72 = _apply(v71, p)
    out = T.copy()
    out["v71"], out["p2"], out["ship"], out["d2"] = v71, p, v72, v72 - v71
    out["c_layers"] = v71 - T2["v70"].to_numpy(dtype=float)
    out.to_parquet(OUT / f"pass3_{base_variant}.parquet", index=False)
    print("pass3 table on", base_variant, "rows", len(out), "c_layers equal to the stored dev table:",
          float(np.nanmax(np.abs(out["c_layers"].to_numpy() - T["c_layers"].to_numpy()))), flush=True)


def pass3_fit(base_variant: str) -> None:
    """T1: the shipped third pass (month-blocked fits on the published chain's table, as shipped) predicting the held-out month of the
    no-published table."""
    import lightgbm as lgb
    import run_stack_third as S3
    T = pd.read_parquet(ROOT / "runs/stack_third/dev_t3.parquet")
    Tn = pd.read_parquet(OUT / f"pass3_{base_variant}.parquet")
    if not np.array_equal(T[ID].to_numpy(), Tn[ID].to_numpy()):
        raise ValueError("row order")
    F = S3.features(T)
    y = np.clip((T["label"] - T["ship"]).to_numpy(), -S3.CLIP_Y, S3.CLIP_Y)
    XT, XN = T[F].astype(np.float32), Tn[F].astype(np.float32)
    nf = (T["kind"] != 3).to_numpy()
    tr = T["airport"].isin(T3_AIRPORTS).to_numpy() & nf
    p = np.full(len(T), np.nan)
    for m in MONTHS:
        f, t = tr & (T["month"] != m).to_numpy(), tr & (T["month"] == m).to_numpy()
        acc = np.zeros(int(t.sum()))
        for sd in S3.SEEDS:
            mdl = lgb.train({**S3.PARAMS, "seed": sd}, lgb.Dataset(XT[f], y[f], categorical_feature=["kind", "ap"]), S3.ROUNDS)
            acc += mdl.predict(XN[t]) / len(S3.SEEDS)
        p[t] = acc
        print("pass3 T1", base_variant, m, flush=True)
    path = OUT / f"pass3_preds_T1_{base_variant}.parquet"
    pd.DataFrame({ID: T[ID].to_numpy(), "p": p}).to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "sha256": sha256_file(path), "features": len(F),
                                           "code": {q: sha256_file(ROOT / q) for q in ["tools/final_nopub.py", "tools/run_stack_third.py"]}})


def pass3_score(base_variant: str) -> None:
    import run_intruder_bound as R
    T = pd.read_parquet(ROOT / "runs/stack_third/dev_t3.parquet", columns=[ID, "month", "airport", "day", "label", "ship"])
    Tn = pd.read_parquet(OUT / f"pass3_{base_variant}.parquet", columns=[ID, "ship"])
    ref = _apply(T["ship"].to_numpy(dtype=float), pd.read_parquet(ROOT / "runs/stack_third/preds_dev.parquet").set_index(ID)["t3_ro_3"].reindex(T[ID]).to_numpy())
    base = Tn["ship"].to_numpy(dtype=float)
    p = pd.read_parquet(OUT / f"pass3_preds_T1_{base_variant}.parquet").set_index(ID)["p"].reindex(T[ID]).to_numpy()
    vals = {"T3": base, "T1": _apply(base, p)}
    rng = np.random.default_rng(20261004)
    out = {}
    tt = T[[ID, "month", "label", "day"]].copy()
    for name, against in [("T3", "ref"), ("T1", "ref"), ("T1", "T3")]:
        tt["cand"], tt["base"] = vals[name], (ref if against == "ref" else vals["T3"])
        months = {}
        for m in MONTHS:
            s = tt[tt["month"] == m]
            pct, lo, hi = R.interval(s.assign(value=s["base"]), "value", s["cand"].to_numpy(), rng)
            months[m] = {"pct": pct, "lo": lo, "hi": hi, "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2)))}
        eq = float(np.mean([months[m]["pct"] for m in EQUAL]))
        out[f"{name}_vs_{against}"] = {"months": months, "equal_footing_mean_pct": eq}
        print(f"pass3 on {base_variant}: {name} vs {against}:", " ".join(f"{m[5:]} {months[m]['pct']:+.4f}% [{months[m]['lo']:+.3f},{months[m]['hi']:+.3f}]" for m in MONTHS),
              f"| eq {eq:+.4f}%", flush=True)
    res = __import__("json").loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    res.setdefault("pass3", {})[base_variant] = {"readings": out, "note": "on the second pass L1 (shipped) of the same chain"}
    write_json(RESULTS, res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "fit", "score", "pass2-build", "pass2-fit", "pass2-score", "fit-final", "pass3-build", "pass3-fit", "pass3-score"])
    ap.add_argument("variant", nargs="?")
    ap.add_argument("month", nargs="?")
    ap.add_argument("--base", default="P2", help="the step-6 variant the second pass is read on")
    a = ap.parse_args()
    if a.command == "build":
        build()
    elif a.command == "fit":
        fit(a.variant, a.month)
    elif a.command == "score":
        score()
    elif a.command == "pass2-build":
        pass2_build(a.base)
    elif a.command == "pass2-fit":
        pass2_fit(a.variant, a.base, a.month)
    elif a.command == "fit-final":
        fit_final()
    elif a.command == "pass3-build":
        pass3_build(a.base)
    elif a.command == "pass3-fit":
        pass3_fit(a.base)
    elif a.command == "pass3-score":
        pass3_score(a.base)
    else:
        pass2_score(a.base)


if __name__ == "__main__":
    main()
