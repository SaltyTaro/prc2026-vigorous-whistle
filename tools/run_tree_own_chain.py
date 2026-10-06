"""Reading of the tree own-points screen (configs/tree_own_screen_v1.json, amended by configs/tree_own_screen_v1_amend1.json) inside v66's
development chain (v65's chain with the own-points member at weight 1.00).

  --check              control reproduction: every ctl fit against the saved leaf it re-fits (lomo months: runs/lomo_dev/learners; forward:
                       <leaf>/forward; jan_nov: model vs <leaf>/refit/model.txt and predictions vs runs/december/learners) -> runs/tree_own/check.json
  --tables             each development month's committee / LIRF table rebuilt from its saved leaves (frozen member rule) and checked against
                       runs/lomo_dev/rows.parquet; the declared candidates' tables (identity + five) -> runs/tree_own/tables.parquet
  --model-part C       v66's model part with candidate C's tables: v46 -> own-points member own_all at 1.00 (v66's member) -> masked member ->
                       LIRF v45 gate / v53m / v54m / v55m (tools/build_dev_model_part.py's arithmetic) -> runs/tree_own/chain/C/dmp.parquet;
                       C = identity must reproduce runs/ctx_ownpoints_weight/chain/own_all_w100/dmp.parquet within 1e-9 s
  --chain C            no-NM rematch, v48, extreme-day, variance re-allocation (shipped parameters; tools/run_ctx_ownpoints_chain.step_chain's
                       procedure; month runs under runs/daily_totals/tree_own_v66_<C>_<kind>)
  --score C            stack_resid_v3 refitted on the rebuilt components (committee, LIRF tree, mix, v30, v33 and v46..v55m of candidate C;
                       LIRF takes v2 refitted) -> intruder bound -> identity extension -> own-motion v1 -> own_motion_v2; against the v66
                       development proxy; the screen's GO bar -> <chain dir>/C/score.json. Unpinned, C = identity must reproduce the v66 proxy
                       within 1e-6 s. --pinned (the amendment's noise control): the variance re-allocation's LightGBM pinned (deterministic,
                       column-wise), outputs under runs/tree_own/chain_v66_pinned/, and a candidate reads v66 proxy + (pinned C - pinned identity).
No official score, no leaderboard, no submission file; nothing is uploaded.
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
from prc2026.features import ID  # noqa: E402
from prc2026_december.scoring import ensemble_prediction, leaf_members  # noqa: E402
from prc2026_final import lirf_gate as lg, network_member, share_gate  # noqa: E402
import build_dev_model_part as BD  # noqa: E402
import run_ctx_ownpoints_chain as CC  # noqa: E402
import run_network_member as rnm  # noqa: E402
from run_fair_weights import build  # noqa: E402

DECL = "configs/tree_own_screen_v1.json"
OUT = ROOT / "runs/tree_own"
FITS = OUT / "fits"
DOCS = ROOT / "docs/tree_own"
V66 = ("runs/ctx_ownpoints_weight/chain/own_all_w100/dev_v66_proxy.parquet", "5bc62cec0af5856de10a9efd195713bd99d73c3e9bacddb38b8bde81c1991c07")
V66_DMP = ("runs/ctx_ownpoints_weight/chain/own_all_w100/dmp.parquet", "5a5c3373dcda7f4442f04117ffbd8319e95e7eff6c981585bf206f7db1069d47")
MEMBER_DECL, MEMBER_CANDIDATE = "configs/ctx_ownpoints_weight_v1.json", "own_all_w100"
AMEND = "configs/tree_own_screen_v1_amend1.json"
PINNED = False
TABLES = {"committee": "runs/adsb_v20/ens_v20_bag_q_all", "lirf_tree": "runs/adsb_v22/ens_v22_lirf_q"}
GROUPS = {"lirf_tree": ["runs/adsb_v21/lgb_lirf_s31", "runs/adsb_v21/lgb_lirf_cap_s32", "runs/adsb_v21/lgb_lirf_nm_s33"],
          "committee": ["runs/adsb_v20/lgb_x19_s24", "runs/adsb_v20/lgb_x19_s25", "runs/adsb_v20/lgb_x19_s26"]}
FOLD_OF = {"2025-01": "lomo_2025-01", "2025-07": "lomo_2025-07", "2025-11": "forward", "2025-12": "jan_nov"}
CANDIDATES = {"identity": {}, "lirf_exact": {"lirf_tree": "exact"}, "lirf_fam100": {"lirf_tree": "fam"}, "com_exact": {"committee": "exact"},
              "com_fam100": {"committee": "fam"}, "both_fam100": {"lirf_tree": "fam", "committee": "fam"}}
V30 = ({"ltfm": 0.7, "matched_other": 0.7, "unmatched_other": 0.7},
       {"lirf_nonpop_matched": {"v9": 0.15, "net": 0.40, "tree": 0.45}, "lirf_nonpop_unmatched": {"v9": 0.40, "net": 0.35, "tree": 0.25}})
FLOOR = -12.0


def key_of(leaf: str) -> str:
    return leaf.replace("runs/", "").replace("/", "__")


def read(path: Path, ids: np.ndarray) -> np.ndarray:
    return pd.read_parquet(path).drop_duplicates(ID).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)


def saved_path(leaf: str, month: str) -> Path:
    if month in ("2025-01", "2025-07"):
        return ROOT / "runs/lomo_dev/learners" / key_of(leaf) / f"lomo_{month}" / "predictions.parquet"
    if month == "2025-11":
        return ROOT / leaf / "forward" / "predictions.parquet"
    return ROOT / "runs/december/learners" / (leaf.replace("/", "__") + ".parquet")


def fit_path(leaf: str, variant: str, month: str) -> Path:
    return FITS / key_of(leaf) / variant / FOLD_OF[month] / "predictions.parquet"


def lomo_rows() -> pd.DataFrame:
    return pd.read_parquet(ROOT / rnm.ROWS)


def chain_dir(candidate: str) -> Path:
    return OUT / ("chain_v66_pinned" if PINNED else "chain_v66") / candidate


# ---------------------------------------------------------------- check / tables

def step_check() -> None:
    rows = lomo_rows()
    out, ok = {"created_at_utc": utc_now(), "declaration_sha256": sha256_file(ROOT / DECL), "leaves": {}}, True
    for leaf in [*GROUPS["lirf_tree"], *GROUPS["committee"]]:
        out["leaves"][leaf] = {}
        for month, fold in FOLD_OF.items():
            ids = rows.loc[rows["month"] == month, ID].to_numpy()
            ref_ids = pd.read_parquet(saved_path(leaf, month), columns=[ID])[ID].to_numpy()
            fit_ids = pd.read_parquet(fit_path(leaf, "ctl", month), columns=[ID])[ID].to_numpy()
            saved, ctl = read(saved_path(leaf, month), ids), read(fit_path(leaf, "ctl", month), ids)
            both = np.isfinite(saved) & np.isfinite(ctl)
            diff = np.abs(saved[both] - ctl[both])
            e = {"rows": int(len(ids)), "fit_rows_cover_month": bool(set(ids) <= set(fit_ids)), "saved_rows_cover_month": bool(set(ids) <= set(ref_ids)),
                 "finite_pattern_equal": bool((np.isfinite(saved) == np.isfinite(ctl)).all()), "max_abs_s": float(diff.max()) if diff.size else 0.0,
                 "rows_differing": int((diff > 0).sum())}
            if fold == "jan_nov":
                refit = ROOT / leaf / "refit" / "model.txt"
                e["model_identical_to_refit"] = sha256_file(refit) == sha256_file(fit_path(leaf, "ctl", month).parent / "model.txt") if refit.exists() else None
                passed = bool(e["model_identical_to_refit"]) and e["finite_pattern_equal"]
            else:
                passed = e["max_abs_s"] == 0.0 and e["finite_pattern_equal"]
            e["passed"] = passed
            ok &= passed and e["fit_rows_cover_month"]
            out["leaves"][leaf][month] = e
            print(leaf, month, e, flush=True)
    out["passed"] = bool(ok)
    write_json(OUT / "check.json", out)
    print("control reproduction passed:", ok)
    if not ok:
        raise SystemExit("A control does not reproduce its saved leaf: the screen stops here.")


def step_tables() -> None:
    if not read_json(OUT / "check.json")["passed"]:
        raise ValueError("Control reproduction has not passed.")
    rows = lomo_rows()
    frames, record = [], {"created_at_utc": utc_now(), "months": {}}
    for month in FOLD_OF:
        f = rows[rows["month"] == month].reset_index(drop=True)
        ids = f[ID].to_numpy()
        out = pd.DataFrame({ID: ids, "month": month})
        rec = {}
        for table, run_dir in TABLES.items():
            members = leaf_members(ROOT, run_dir)
            saved = {leaf: read(saved_path(leaf, month), ids) for leaf in sorted(set(members))}
            rebuilt = ensemble_prediction(members, saved)
            stored = f[table].to_numpy(dtype=float)
            gap = float(np.nanmax(np.abs(rebuilt - stored)))
            if not gap <= 1e-9 or not (np.isfinite(rebuilt) == np.isfinite(stored)).all():
                raise ValueError(f"{month} {table}: the member rule does not reproduce the development table (max {gap}).")
            deltas = {leaf: read(fit_path(leaf, "own", month), ids) - read(fit_path(leaf, "ctl", month), ids) for leaf in GROUPS[table]}
            exact = dict(saved)
            for leaf, d in deltas.items():
                exact[leaf] = np.where(np.isfinite(d), saved[leaf] + d, saved[leaf])
            d_exact = ensemble_prediction(members, exact) - rebuilt
            d_exact = np.where(np.isfinite(d_exact), d_exact, 0.0)
            wide = np.column_stack(list(deltas.values()))
            with np.errstate(invalid="ignore"):
                fam = np.nanmean(wide, axis=1)
            fam = np.where(np.isfinite(fam), fam, 0.0)
            out[f"{table}_d_exact"], out[f"{table}_d_fam"] = d_exact, fam
            share = sum(members.count(leaf) for leaf in GROUPS[table]) / len(members)
            rec[table] = {"rebuild_max_abs_s": gap, "group_share": share, "members": len(members),
                          "d_exact_rms_s": float(np.sqrt(np.mean(d_exact ** 2))), "d_fam_rms_s": float(np.sqrt(np.mean(fam ** 2))),
                          "leaf_delta_rms_s": {leaf: float(np.sqrt(np.nanmean(d ** 2))) for leaf, d in deltas.items()}}
        record["months"][month] = rec
        frames.append(out)
        print(month, rec, flush=True)
    table = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT / "tables.parquet", index=False)
    record["sha256"] = sha256_file(OUT / "tables.parquet")
    write_json(OUT / "tables.json", record)


def candidate_rows(candidate: str) -> pd.DataFrame:
    """runs/lomo_dev/rows.parquet with candidate C's committee / LIRF tree, and the components the stacker reads that depend on them (mix, v30,
    v33) moved by the change of their own construction (the delta form keeps the identity exact)."""
    rows = lomo_rows()
    if candidate == "identity":
        return rows
    t = pd.read_parquet(OUT / "tables.parquet").drop_duplicates(ID).set_index(ID).reindex(rows[ID])
    new = rows.copy()
    for table, kind in CANDIDATES[candidate].items():
        new[table] = rows[table].to_numpy(dtype=float) + t[f"{table}_d_{kind}"].to_numpy(dtype=float)
    nets = rows["nets"].to_numpy(dtype=float)
    mix0, _ = share_gate.mix(rows["committee"].to_numpy(dtype=float), nets, share_gate.CONTROL_SHARE)
    mix1, _ = share_gate.mix(new["committee"].to_numpy(dtype=float), nets, share_gate.CONTROL_SHARE)
    new["mix"] = rows["mix"].to_numpy(dtype=float) + (mix1 - mix0)
    v33 = read_json(ROOT / "configs/fair_weights_v1.json")["control"]
    for col, (outer, lirf) in (("v30", V30), ("v33", (v33["outer"], v33["lirf"]))):
        new[col] = rows[col].to_numpy(dtype=float) + (build(new, outer, lirf) - build(rows, outer, lirf))
    return new


# ---------------------------------------------------------------- model part

def step_model_part(candidate: str) -> None:
    rows = candidate_rows(candidate)
    cs = rnm.scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = rnm.offsets(), rnm.offsets_all()
    seg = rnm.segments(rnm.MONTHS)
    k = read_json(ROOT / rnm.STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    member_decl = read_json(ROOT / MEMBER_DECL)
    spec, w = member_decl["member"], float(member_decl["candidates"][MEMBER_CANDIDATE]["member_weight"])
    mask = read_json(ROOT / BD.MASK)
    frames, checks = [], {}
    for month in rnm.MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        ids, y = f[ID].to_numpy(), f["label"].to_numpy(dtype=float)
        lirf = f["airport"].to_numpy() == "LIRF"
        committee, nets = f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float)

        def stack(table: np.ndarray) -> np.ndarray:
            mix, _ = share_gate.mix(committee, table, network_member.V46_SHARE, network_member.V46_GATE)
            return rnm.full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)

        v46 = stack(nets)
        member, _ = rnm.member_values(spec, month, ids)
        t51, _ = network_member.table_with_member(nets, np.where(lirf, np.nan, member), w)
        v51 = stack(t51)
        m52, _ = rnm.member_values(mask["member"], month, ids)
        t52, _ = network_member.table_with_member(t51, np.where(lirf, np.nan, m52), network_member.candidate_weight(mask, "w40"))
        v52 = stack(t52)
        v9, net, tree = (f[x].to_numpy(dtype=float) for x in ("v9", "lirf_net", "lirf_tree"))
        u, m = f["lirf_u"].to_numpy(dtype=bool), f["lirf_m"].to_numpy(dtype=bool)
        M = BD.lirf_member(month, ids)
        ctrl_u, _ = lg.gated(v9[u], net[u], tree[u], "unmatched", BD.rlm.GATE)
        cand_u, _ = lg.gated(v9[u], 0.45 * net[u] + 0.55 * M[u], tree[u], "unmatched", BD.rlm.GATE)
        v53 = v52.copy()
        v53[u] = np.maximum(v52[u] + (cand_u - ctrl_u), FLOOR)
        share = m.sum() / max(m.sum() + u.sum(), 1)
        v54 = v53.copy()
        if share < 0.05:
            s = {"v9": v9[u], "net": net[u], "tree": tree[u], "member": M[u]}
            dark, _ = BD.rld.values(s, 0.70, 0.75, 600.0)
            base, _ = BD.rld.values(s, BD.rld.CONTROL["member_weight"], BD.rld.CONTROL["network_share"], 600.0)
            v54[u] = np.maximum(v53[u] + (dark - base), FLOOR)
        v55 = v54.copy()
        if m.any():
            ctrl_m, _ = BD.rlc.values(v9[m], net[m], tree[m], None, 0.0)
            cand_m, _ = BD.rlc.values(v9[m], net[m], tree[m], M[m], 0.40)
            v55[m] = np.maximum(v54[m] + (cand_m - ctrl_m), FLOOR)
        out = {"v46": v46, "v51m": v51, "v52m": v52, "v53m": v53, "v54m": v54, "v55m": v55}
        checks[month] = {c: float(np.sqrt(np.mean((y - v) ** 2))) for c, v in out.items()}
        frames.append(pd.DataFrame({ID: ids, "month": month, **out}))
        print(candidate, month, {c: round(v, 4) for c, v in checks[month].items()}, flush=True)
    dmp = pd.concat(frames, ignore_index=True)
    if sha256_file(ROOT / V66_DMP[0]) != V66_DMP[1]:
        raise ValueError("The v66 model part differs from the declared file.")
    ref = pd.read_parquet(ROOT / V66_DMP[0]).set_index(ID).reindex(dmp[ID])
    pop = (rows.set_index(ID).reindex(dmp[ID])[["airport", "lirf_u", "lirf_m"]])
    pop = (pop["airport"].eq("LIRF") & ~pop["lirf_u"].astype(bool) & ~pop["lirf_m"].astype(bool)).to_numpy()
    gaps = {c: float(np.max(np.abs(dmp[c].to_numpy() - ref[c].to_numpy()))) for c in ["v46", "v51m", "v52m", "v53m", "v54m", "v55m"]}
    pop_gap = max(float(np.max(np.abs(dmp[c].to_numpy()[pop] - ref[c].to_numpy()[pop]))) for c in gaps)
    if pop_gap > 1e-9:
        raise ValueError(f"A LIRF population row changed ({pop_gap} s).")
    if candidate == "identity" and max(gaps.values()) > 1e-9:
        raise ValueError(f"identity failed: {gaps}")
    d = chain_dir(candidate)
    d.mkdir(parents=True, exist_ok=True)
    dmp.to_parquet(d / "dmp.parquet", index=False)
    write_json(d / "dmp.json", {"created_at_utc": utc_now(), "candidate": candidate, "max_abs_vs_v66_model_part_s": gaps, "population_rows_unchanged": True,
                                "month_rmse": checks, "sha256": sha256_file(d / "dmp.parquet"), "code": sha256_file(ROOT / "tools/run_tree_own_chain.py")})
    print(candidate, "vs v66 model part", gaps)


# ---------------------------------------------------------------- chain (tools/run_ctx_ownpoints_chain.step_chain's procedure)

def step_chain(candidate: str) -> None:
    import run_daily_totals as rd
    import run_rematch_nonm as RM
    import scout_extreme_shift as se
    import scout_var_alloc as sva
    from scout_refit_prep import takeoff, with_stack
    from screen_daily_totals import published
    d = chain_dir(candidate)
    dmp = pd.read_parquet(d / "dmp.parquet").drop_duplicates(ID).set_index(ID)
    m25 = pd.read_parquet(RM.OUT / "matches_2025.parquet")
    lab = RM.labels_2025()
    base = pd.read_parquet(ROOT / "runs/daily_totals/rows.parquet")
    ids = base[ID].to_numpy()
    v = dmp["v55m"].reindex(ids).to_numpy(dtype=float)
    v = np.where(np.isnan(v), base["stack"].to_numpy(dtype=float), v)
    offs = read_json(RM.OUT / "dev/offsets.json")
    cand, changed = v.copy(), np.zeros(len(v), dtype=bool)
    for m in RM.MONTHS:
        idx = np.flatnonzero(base["month"].eq(m).to_numpy())
        if RM.offsets(m25, lab, m) != offs[m]:
            raise ValueError("The rematch offsets differ from the shipped development run.")
        c, ch = RM.apply(v[idx], ids[idx], m25, offs[m])
        cand[idx], changed[idx] = c, ch
    tk, pub = takeoff(ids), published()
    with_stack(base, v, tk, pub).to_parquet(d / "rows_all.parquet", index=False)
    settings, tau_scale, prior = rd.settings_from(read_json(ROOT / "configs/daily_totals_v1.json"))
    for kind, col in [("ctl", v), ("cand", cand)]:
        new = with_stack(base, col, tk, pub)
        parts = []
        for m in RM.MONTHS:
            mix = pd.concat([base[base["month"] != m], new[new["month"] == m]], ignore_index=True)
            rows_path = d / f"rows_{kind}_{m}.parquet"
            mix.to_parquet(rows_path, index=False)
            tag = f"tree_own_v66_{candidate}_{kind}"
            if not (ROOT / f"runs/daily_totals/{tag}/{m}.parquet").exists():
                rd.run_month(m, tag, tau_scale, settings, rows=str(rows_path.relative_to(ROOT)).replace("\\", "/"), prior=prior)
            vd = d / f"v48_{kind}_{m}"
            vd.mkdir(parents=True, exist_ok=True)
            for kk in RM.MONTHS:
                src = ROOT / (f"runs/daily_totals/{tag}/{kk}.parquet" if kk == m else f"runs/daily_totals/daily_totals_v1/{kk}.parquet")
                shutil.copyfile(src, vd / f"{kk}.parquet")
            se.TABLES[f"tree_own_v66_{candidate}_{kind}_{m}"] = (rows_path, vd)
            t, ks = sva.layered(f"tree_own_v66_{candidate}_{kind}_{m}")
            parts.append(t.loc[t["month"] == m, [ID, "month", "airport", "bday", "label", "stack", "v48", "ext", "new"]].assign(k=ks[m]))
            print(candidate, kind, m, "k", ks[m], flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(d / f"{kind}.parquet", index=False)
    write_json(d / "chain.json", {"created_at_utc": utc_now(), "pinned_var_alloc": PINNED, "rematch_rows": int(changed.sum()),
                                  "files": {p.name: sha256_file(p) for p in sorted(d.glob("*.parquet"))}, "code": sha256_file(ROOT / "tools/run_tree_own_chain.py")})


# ---------------------------------------------------------------- score

def dev_table(candidate: str, d: Path) -> pd.DataFrame:
    """tools/run_ctx_ownpoints_chain.dev_table_from with the candidate's component rows in place of runs/lomo_dev/rows.parquet."""
    import run_stack_resid as S
    ctl = pd.read_parquet(d / "ctl.parquet", columns=[ID, "month", "airport", "bday", "label", "stack", "new"])
    rows = pd.read_parquet(d / "rows_all.parquet", columns=[ID, "bday", "a", "matched", "cls", "pub", "n_pub", "complete"])
    comp = candidate_rows(candidate)[[ID, "v9", "residual", "base", "mix", "lirf_tree", "lirf_net", "committee", "nets", "v30", "v33"]]
    dmp = pd.read_parquet(d / "dmp.parquet", columns=[ID, "v46", "v51m", "v52m", "v53m", "v54m", "v55m"])
    t = ctl.merge(rows, on=ID, suffixes=("", "_rows"), validate="one_to_one").merge(comp, on=ID, validate="one_to_one").merge(dmp, on=ID, validate="one_to_one")
    if not (t["bday"] == t["bday_rows"]).all() or np.abs(t["stack"] - t["v55m"]).max() > 1e-9:
        raise ValueError("The development tables disagree on the off-block day or on v55m.")
    t["v55m_day"] = t["v55m"]
    raw = S.takeoff(t[ID].to_numpy(), sorted((ROOT / "data/raw").glob("training_2025-*.parquet")))
    tk = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    t["takeoff_day"] = tk.dt.strftime("%Y-%m-%d").to_numpy()
    t["dom"] = tk.dt.day.to_numpy()
    return t.drop(columns=["bday_rows"])


def step_score(candidate: str) -> None:
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    import run_intruder_bound as R
    if sha256_file(ROOT / V66[0]) != V66[1]:
        raise ValueError("The v66 development proxy differs from the declared file.")
    decl = read_json(ROOT / DECL)
    d = chain_dir(candidate)
    dev = dev_table(candidate, d)
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    core = V3.own_points("dev", dev[ID].to_numpy())
    Pn = V3.new_columns(core, dev["new"].to_numpy(dtype=float))
    X65 = pd.concat([X46, Pn], axis=1)
    pr = V3.dev_predictions(dev, X46, X65, Pn, res)
    new, air = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy()
    s1 = pd.read_parquet(d / "cand.parquet").set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    keys, _ = O2.O.dev_table()
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
    t["s2"] = s1 + (np.where(air == "LIRF", pr["base2"], pr["base3"]) - new)
    _, c63 = V3.Chain().run(t, "s2")
    dev12, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    t["c63"] = c63
    t["run"] = CC.own_motion_v2(t, "c63", t12, trig)
    t["v66"] = pd.read_parquet(ROOT / V66[0]).set_index(ID).reindex(t[ID])["v66"].to_numpy()
    t[[ID, "month", "airport", "label", "cls", "v66", "run"]].to_parquet(d / "run.parquet", index=False)
    if candidate == "identity":
        gap = float(np.max(np.abs(t["run"].to_numpy() - t["v66"].to_numpy())))
        print("identity", "pinned" if PINNED else "unpinned", ": max |run - v66 proxy| =", gap)
        write_json(d / "score.json", {"created_at_utc": utc_now(), "candidate": candidate, "pinned_var_alloc": PINNED, "max_abs_vs_v66_proxy_s": gap,
                                      "run_sha256": sha256_file(d / "run.parquet")})
        if not PINNED and gap > 1e-6:
            raise ValueError("The pipeline does not reproduce the v66 proxy.")
        return
    if not PINNED:
        raise ValueError("Candidates are read pinned (configs/tree_own_screen_v1_amend1.json).")
    ident = pd.read_parquet(chain_dir("identity") / "run.parquet").set_index(ID).reindex(t[ID])["run"].to_numpy(dtype=float)
    t["cand"] = t["v66"].to_numpy() + (t["run"].to_numpy() - ident)
    pop = t["cls"].eq("LIRF_pop").to_numpy()
    pop_gap = float(np.max(np.abs(t["cand"].to_numpy()[pop] - t["v66"].to_numpy()[pop])))
    if pop_gap > 1e-6:
        raise ValueError(f"A LIRF population row changed ({pop_gap} s).")
    rng = np.random.default_rng(20261006)
    months = {}
    for m in R.DEV_MONTHS:
        s_ = t[t["month"] == m]
        pct, lo, hi = R.interval(s_, "v66", s_["cand"].to_numpy(), rng)
        yy = s_["label"].to_numpy()
        ch = s_["cand"].to_numpy() - s_["v66"].to_numpy()
        months[m] = {"control_rmse": float(np.sqrt(np.mean((yy - s_["v66"].to_numpy()) ** 2))), "rmse": float(np.sqrt(np.mean((yy - s_["cand"].to_numpy()) ** 2))),
                     "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float(np.sum(ch ** 2))}
        print(candidate, m, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in months[m].items()}, flush=True)
    eq = decl["reading"]["months"]["equal_footing"]
    mean = float(np.mean([months[m]["pct"] for m in eq]))
    below = sum(months[m]["hi"] < 0 for m in eq)
    go = {"equal_footing_mean_pct": mean, "mean_ok": mean <= -0.05, "no_month_worse_than_0.01pct": all(months[m]["pct"] <= 0.01 for m in months),
          "equal_intervals_below_zero": int(below), "intervals_ok": below >= 2}
    go["GO"] = bool(go["mean_ok"] and go["no_month_worse_than_0.01pct"] and go["intervals_ok"])
    t[[ID, "month", "airport", "label", "cls", "v66", "cand"]].to_parquet(d / "dev_proxy.parquet", index=False)
    by = t.assign(e0=(t["label"] - t["v66"]) ** 2, e1=(t["label"] - t["cand"]) ** 2).groupby(["month", "cls"])[["e0", "e1"]].sum()
    by = {f"{m}|{c}": float(r["e1"] - r["e0"]) for (m, c), r in by.iterrows() if abs(r["e1"] - r["e0"]) > 0}
    report = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "amendment": AMEND,
              "amendment_sha256": sha256_file(ROOT / AMEND), "candidate": candidate, "control": "v66 development proxy", "v66_proxy": V66,
              "reading": "v66 proxy + (pinned candidate run - pinned identity run)", "pinned_identity_max_abs_vs_v66_s": float(np.max(np.abs(ident - t["v66"].to_numpy()))),
              "months": months, "go_bar": go, "stacker_trees": pr["trees"], "squared_error_change_by_month_cell": by,
              "proxy": {"path": str((d / "dev_proxy.parquet").relative_to(ROOT)), "sha256": sha256_file(d / "dev_proxy.parquet")},
              "code": {c: sha256_file(ROOT / c) for c in ["tools/run_tree_own_chain.py", "tools/run_tree_own.py", "tools/run_stack_resid_v3.py", "tools/run_own_motion_v2.py"]}}
    write_json(d / "score.json", report)
    print(candidate, go)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--model-part", metavar="C")
    ap.add_argument("--chain", metavar="C")
    ap.add_argument("--score", metavar="C")
    ap.add_argument("--pinned", action="store_true", help="the variance re-allocation's LightGBM pinned (deterministic, column-wise)")
    a = ap.parse_args()
    global PINNED
    if a.pinned:
        from prc2026_scout import var_alloc
        var_alloc.PARAMS = {**var_alloc.PARAMS, "deterministic": True, "force_col_wise": True}
        PINNED = True
    for c in (a.model_part, a.chain, a.score):
        if c is not None and c not in CANDIDATES:
            raise ValueError(f"Unknown candidate {c}.")
    if a.check:
        step_check()
    if a.tables:
        step_tables()
    if a.model_part:
        step_model_part(a.model_part)
    if a.chain:
        step_chain(a.chain)
    if a.score:
        step_score(a.score)


if __name__ == "__main__":
    main()
