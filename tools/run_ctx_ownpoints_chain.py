"""Stage 2 of the own-points development round (configs/ctx_ownpoints_dev_v1.json, stage_2_in_chain): the selected member in v64's chain
on the four development months. Every step reuses the shipped tools' functions with their shipped parameters (imported, not changed).

  --model-part  the development model part with the member in place of the combination member (tools/build_dev_model_part.py arithmetic:
                v51m' = v46 stack with the member at its weight; v52m' = + the masked member at 0.40; the LIRF layers v53m-v55m touch LIRF rows
                only and keep their values) -> runs/ctx_ownpoints_dev/chain/<candidate>/dmp.parquet. --identity: the combination member at
                0.40 must reproduce runs/dev_model_part/rows.parquet exactly.
  --chain       v48 reconciliation -> extreme-day -> variance re-allocation with the shipped parameters (tools/run_rematch_nonm.step_dev's
                procedure, as tools/scout_refit_run.py builds runs/scout/refit/ctl_v55m.parquet), on v55m' (ctl) and on v55m' with the no-NM
                rematch (cand) -> chain/<candidate>/{ctl,cand}.parquet, rows_all.parquet
  --score       stack_resid_v3 refitted on the rebuilt components (tools/run_stack_resid_v3.dev_predictions, month-blocked), LIRF keeps v2;
                intruder bound -> identity extension -> own-motion v1 (run_stack_resid_v3.Chain); own_motion_v2 last (configs/own_motion_v2.json
                arithmetic, tools/run_own_motion_v2). Control: the v64 development proxy = own_motion_v2 on the v63 proxy. Rule: v57's clauses
                1-3 -> docs/ctx_ownpoints_dev/results_chain_v1.json, runs/ctx_ownpoints_dev/chain/<candidate>/dev_v65_proxy.parquet
                --identity: the shipped files in place of the rebuilt ones must reproduce the v63 proxy exactly (pipeline check)
No official score enters. Nothing is uploaded.
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
from prc2026_final import network_member, share_gate  # noqa: E402
import run_network_member as rnm  # noqa: E402
import run_ctx_ownpoints_dev as OD  # noqa: E402

DECLARATION = "configs/ctx_ownpoints_dev_v1.json"
OUT = ROOT / "runs/ctx_ownpoints_dev/chain"
V63 = ("runs/stack_resid/dev_v63_proxy.parquet", "1628a1d64d06d4988b0445eb0f2768f8503fabcde529501b084d233117d575c2")
MASK = "configs/adsb_ctx3_mask_dev_v1.json"
LAYERS = ["v53m", "v54m", "v55m"]


def member_spec(decl: dict, name: str) -> dict:
    return decl["member"] if name == "own_all" else OD.matched_spec(decl)


def step_model_part(decl: dict, candidate: str, identity: bool) -> None:
    rows = pd.read_parquet(ROOT / rnm.ROWS)
    cs = rnm.scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = rnm.offsets(), rnm.offsets_all()
    seg = rnm.segments(rnm.MONTHS)
    k = read_json(ROOT / rnm.STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    shipped = pd.read_parquet(ROOT / "runs/dev_model_part/rows.parquet").set_index(ID)
    mask = read_json(ROOT / MASK)
    if identity:
        spec, w = read_json(ROOT / OD.COMBO)["member"], OD.COMBO_W
    else:
        spec, w = member_spec(decl, decl["candidates"][candidate]["member"]), float(decl["candidates"][candidate]["member_weight"])
    frames, checks = [], {}
    for month in rnm.MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        ids, y = f[ID].to_numpy(), f["label"].to_numpy(dtype=float)
        lirf, nets, committee = f["airport"].to_numpy() == "LIRF", f["nets"].to_numpy(dtype=float), f["committee"].to_numpy(dtype=float)

        def stack(table):
            mix, _ = share_gate.mix(committee, table, network_member.V46_SHARE, network_member.V46_GATE)
            return rnm.full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)

        sh = shipped.reindex(ids)
        member, _ = rnm.member_values(spec, month, ids)
        t51, _ = network_member.table_with_member(nets, np.where(lirf, np.nan, member), w)
        v51 = stack(t51)
        m52, _ = rnm.member_values(mask["member"], month, ids)
        t52, _ = network_member.table_with_member(t51, np.where(lirf, np.nan, m52), network_member.candidate_weight(mask, "w40"))
        v52 = stack(t52)
        if np.abs(v52[lirf] - sh["v52m"].to_numpy()[lirf]).max(initial=0.0) > 1e-9:
            raise ValueError(f"{month}: a LIRF row of v52m changed.")
        out = {"v46": sh["v46"].to_numpy(), "v51m": v51, "v52m": v52}
        for layer in LAYERS:
            base = sh[layer].to_numpy()
            if np.abs(base[~lirf] - sh["v52m"].to_numpy()[~lirf]).max(initial=0.0) > 1e-9:
                raise ValueError(f"{month}: the shipped {layer} differs from v52m outside LIRF.")
            out[layer] = np.where(lirf, base, v52)
        if identity:
            dev_max = max(float(np.abs(out[c] - sh[c].to_numpy()).max()) for c in ["v51m", "v52m", "v55m"])
            if dev_max > 1e-9:
                raise ValueError(f"identity failed on {month}: max {dev_max}")
        checks[month] = {c: float(np.sqrt(np.mean((y - out[c]) ** 2))) for c in out}
        frames.append(pd.DataFrame({ID: ids, "month": month, **out}))
        print(month, {c: round(v, 4) for c, v in checks[month].items()}, flush=True)
    if identity:
        print("identity: the combination member at 0.40 reproduces runs/dev_model_part/rows.parquet exactly")
        return
    d = OUT / candidate
    d.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(d / "dmp.parquet", index=False)
    write_json(d / "dmp.json", {"created_at_utc": utc_now(), "candidate": candidate, "member_weight": w, "sha256": sha256_file(d / "dmp.parquet"),
                                "month_rmse": checks, "code": sha256_file(ROOT / "tools/run_ctx_ownpoints_chain.py")})


def step_chain(candidate: str) -> None:
    import run_daily_totals as rd
    import run_rematch_nonm as RM
    import scout_extreme_shift as se
    import scout_var_alloc as sva
    from scout_refit_prep import takeoff, with_stack
    from screen_daily_totals import published
    d = OUT / candidate
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
            tag = f"ctx_ownpts_{candidate}_{kind}"
            if not (ROOT / f"runs/daily_totals/{tag}/{m}.parquet").exists():
                rd.run_month(m, tag, tau_scale, settings, rows=str(rows_path.relative_to(ROOT)).replace("\\", "/"), prior=prior)
            vd = d / f"v48_{kind}_{m}"
            vd.mkdir(parents=True, exist_ok=True)
            for kk in RM.MONTHS:
                src = ROOT / (f"runs/daily_totals/{tag}/{kk}.parquet" if kk == m else f"runs/daily_totals/daily_totals_v1/{kk}.parquet")
                shutil.copyfile(src, vd / f"{kk}.parquet")
            se.TABLES[f"{candidate}_{kind}_{m}"] = (rows_path, vd)
            t, ks = sva.layered(f"{candidate}_{kind}_{m}")
            parts.append(t.loc[t["month"] == m, [ID, "month", "airport", "bday", "label", "stack", "v48", "ext", "new"]].assign(k=ks[m]))
            print(candidate, kind, m, "k", ks[m], flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(d / f"{kind}.parquet", index=False)
    write_json(d / "chain.json", {"created_at_utc": utc_now(), "rematch_rows": int(changed.sum()),
                                  "files": {p.name: sha256_file(p) for p in sorted(d.glob("*.parquet"))}, "code": sha256_file(ROOT / "tools/run_ctx_ownpoints_chain.py")})


def dev_table_from(ctl_path: Path, rows_path: Path, dmp_path: Path) -> pd.DataFrame:
    """tools/run_stack_resid.dev_table with the rebuilt files (same merges and checks)."""
    import run_stack_resid as S
    ctl = pd.read_parquet(ctl_path, columns=[ID, "month", "airport", "bday", "label", "stack", "new"])
    rows = pd.read_parquet(rows_path, columns=[ID, "bday", "a", "matched", "cls", "pub", "n_pub", "complete"])
    lomo = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet", columns=[ID, "v9", "residual", "base", "mix", "lirf_tree", "lirf_net", "committee", "nets", "v30", "v33"])
    dmp = pd.read_parquet(dmp_path, columns=[ID, "v46", "v51m", "v52m", "v53m", "v54m", "v55m"])
    t = ctl.merge(rows, on=ID, suffixes=("", "_rows"), validate="one_to_one").merge(lomo, on=ID, validate="one_to_one").merge(dmp, on=ID, validate="one_to_one")
    if not (t["bday"] == t["bday_rows"]).all() or np.abs(t["stack"] - t["v55m"]).max() > 1e-9:
        raise ValueError("The development tables disagree on the off-block day or on v55m.")
    t["v55m_day"] = t["v55m"]
    raw = S.takeoff(t[ID].to_numpy(), sorted((ROOT / "data/raw").glob("training_2025-*.parquet")))
    tk = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    t["takeoff_day"] = tk.dt.strftime("%Y-%m-%d").to_numpy()
    t["dom"] = tk.dt.day.to_numpy()
    return t.drop(columns=["bday_rows"])


def own_motion_v2(dev: pd.DataFrame, col: str, t12: pd.DataFrame, trig: pd.DataFrame) -> np.ndarray:
    """configs/own_motion_v2.json's candidate on column col of the development table, month-blocked (tools/run_own_motion_v2.score)."""
    import run_own_motion as O
    import run_own_motion_v2 as O2
    spec = read_json(O2.CONFIG)["candidate"]
    c12 = O2.candidates(t12, "after3", trig, spec)
    cdv = O2.candidates(dev, col, trig, spec)
    out = pd.Series(dev[col].to_numpy(dtype=float), index=dev[ID].to_numpy())
    for m in O2.R.DEV_MONTHS:
        c = O.values(cdv[cdv["month"] == m], col, O.fit(c12[c12["month"] != m]), spec["w"])
        out.loc[c[ID].to_numpy()] = c["value"].to_numpy()
    return out.loc[dev[ID].to_numpy()].to_numpy()


def step_score(decl: dict, candidate: str, identity: bool) -> None:
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    import run_intruder_bound as R
    if sha256_file(ROOT / V63[0]) != V63[1]:
        raise ValueError("The v63 development proxy differs from the declared file.")
    d = OUT / candidate
    if identity:
        files = (ROOT / "runs/scout/refit/ctl_v55m.parquet", ROOT / "runs/scout/refit/rows_v55m.parquet", ROOT / "runs/dev_model_part/rows.parquet",
                 ROOT / "runs/rematch_nonm/dev/cand.parquet")
    else:
        files = (d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet", d / "cand.parquet")
    dev = dev_table_from(*files[:3])
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    core = V3.own_points("dev", dev[ID].to_numpy())
    Pn = V3.new_columns(core, dev["new"].to_numpy(dtype=float))
    X65 = pd.concat([X46, Pn], axis=1)
    pr = V3.dev_predictions(dev, X46, X65, Pn, res)
    new, air = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy()
    s1 = pd.read_parquet(files[3]).set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    keys, _ = O2.O.dev_table()  # v61 proxy keys + take-off day (validated against census61)
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
    t["s2"] = s1 + (np.where(air == "LIRF", pr["base2"], pr["base3"]) - new)
    _, cand63 = V3.Chain().run(t, "s2")
    v63p = pd.read_parquet(ROOT / V63[0]).set_index(ID).reindex(t[ID])
    if identity:
        dev_max = float(np.abs(cand63 - v63p["v63"].to_numpy()).max())
        print("identity: the shipped files reproduce the v63 proxy, max |diff| =", dev_max)
        if dev_max > 1e-6:
            raise ValueError("The pipeline does not reproduce the v63 proxy.")
        return
    dev12, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    t["v63"] = v63p["v63"].to_numpy()
    t["c63"] = cand63
    ctl64 = own_motion_v2(t, "v63", t12, trig)
    cand65 = own_motion_v2(t, "c63", t12, trig)
    rule = read_json(ROOT / "configs/own_motion_v2.json")["rule"]
    rng = np.random.default_rng(20261004)
    t["v64"], t["cand"] = ctl64, cand65
    months = {}
    for m in R.DEV_MONTHS:
        s = t[t["month"] == m]
        pct, lo, hi = R.interval(s, "v64", s["cand"].to_numpy(), rng)
        yy = s["label"].to_numpy()
        months[m] = {"control_rmse": float(np.sqrt(np.mean((yy - s["v64"].to_numpy()) ** 2))), "rmse": float(np.sqrt(np.mean((yy - s["cand"].to_numpy()) ** 2))),
                     "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(s["cand"].to_numpy() - s["v64"].to_numpy()) > 1e-9).sum()),
                     "squared_change": float(np.sum((s["cand"].to_numpy() - s["v64"].to_numpy()) ** 2))}
        print(m, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in months[m].items()}, flush=True)
    eq = float(np.mean([months[m]["pct"] for m in rule["equal_footing"]]))
    checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= rule["max_month_worsening_pct"] for m in R.DEV_MONTHS),
              "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -rule["min_mean_improvement_pct"],
              "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in rule["equal_footing"])}
    checks["accepted"] = bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"])
    t[[ID, "month", "airport", "label", "cls", "v63", "v64", "cand"]].rename(columns={"cand": "v65"}).to_parquet(d / "dev_v65_proxy.parquet", index=False)
    report = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "candidate": candidate,
              "rule": "v57's clauses 1-3 (configs/own_motion_v2.json rule values); clause 4 not applicable (declared)", "control": "v64 development proxy (own_motion_v2 on the v63 proxy)",
              "inputs": {p.name: sha256_file(p) for p in files}, "v63_proxy": V63, "months": months, "checks": checks,
              "stacker_trees": pr["trees"], "proxy": {"path": str((d / "dev_v65_proxy.parquet").relative_to(ROOT)), "sha256": sha256_file(d / "dev_v65_proxy.parquet")},
              "code": {c: sha256_file(ROOT / c) for c in ["tools/run_ctx_ownpoints_chain.py", "tools/run_stack_resid_v3.py", "tools/run_own_motion_v2.py"]}}
    dest = ROOT / "docs/ctx_ownpoints_dev/results_chain_v1.json"
    if dest.exists():
        raise ValueError("Preserve the original stage-2 result.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, report)
    print(checks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--model-part", action="store_true")
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--identity", action="store_true")
    a = ap.parse_args()
    decl = read_json(ROOT / DECLARATION)
    if not a.identity:
        sel = read_json(ROOT / decl["selection"])
        if sel.get("selected_candidate") != a.candidate:
            raise ValueError(f"Stage 2 runs the stage-1 selection only ({sel.get('selected_candidate')}).")
    if a.model_part:
        step_model_part(decl, a.candidate, a.identity)
    if a.chain:
        step_chain(a.candidate)
    if a.score:
        step_score(decl, a.candidate or "identity", a.identity)


if __name__ == "__main__":
    main()
