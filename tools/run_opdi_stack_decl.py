"""Governing run and ranking dry run of configs/opdi_stack_v1.json (lever 7, docs/opdi_stack/README.md).

  --govern     tools/run_opdi_stack.py --score opdi recomputed from scratch into runs/opdi_stack/govern/ (the screen's code; its proxy must equal
               the screen's exactly), then clauses 1-3 against the pinned v66 development proxy -> docs/opdi_stack/results_v1.json
  --dry-run    identity: tools/export_ctx_ownpoints.step_stack (unchanged) on v66's shipped step-5 outputs, then the shipped chain
               (run_stack_resid_v3.chain_from_stack, run_own_motion_v2.dry_run) must reproduce the shipped stack step and the uploaded v66
               byte for byte; then the same path with the OPDI learner (declared candidate) -> runs/opdi_stack/rank/dry_v67/; clause 4
               (transfer guard) and the final acceptance -> docs/opdi_stack/results_v1.json. Never under runs/submissions; nothing uploaded.
No official score enters.
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
from prc2026_submission.contract import export_submission  # noqa: E402
import run_opdi_stack as O  # noqa: E402

DECL = "configs/opdi_stack_v1.json"
RESULTS = ROOT / "docs/opdi_stack/results_v1.json"
GOV = ROOT / "runs/opdi_stack/govern"
SH = ROOT / "runs/ctx_ownpoints_weight/rank/dry_v66/steps"
V66 = ("runs/submissions/vigorous-whistle_v66.parquet", "b4ea727e1644b6add104f44832898bfecfd3d5607f7d6a9ce0471efe581322e4")
CANDIDATE_CHAIN = "own_all_w100"
CODE = ["tools/run_opdi_stack_decl.py", "tools/run_opdi_stack.py", "tools/export_ctx_ownpoints.py", "tools/run_ownpoints_weight.py",
        "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py", "tools/run_ctx_ownpoints_chain.py", "tools/run_intruder_bound.py",
        "tools/run_intruder_identity.py", "tools/run_own_motion.py", "tools/run_own_motion_v2.py"]


def govern() -> None:
    import run_intruder_bound as R
    decl = read_json(ROOT / DECL)
    name = decl["candidate"]["name"]
    if (GOV / "chain" / name / "proxy.parquet").exists():
        raise ValueError("Preserve the governing run.")
    GOV.mkdir(parents=True, exist_ok=True)
    for f in ["features_dev.parquet", "features_dev.receipt.json", "features_rank.parquet", "features_rank.receipt.json"]:
        shutil.copyfile(O.OUT / f, GOV / f)
    screen_proxy = O.OUT / "chain" / name / "proxy.parquet"
    O.OUT = GOV
    O.score(name)
    mine = pd.read_parquet(GOV / "chain" / name / "proxy.parquet")
    scr = pd.read_parquet(screen_proxy)
    same = bool(mine[ID].equals(scr[ID]) and np.array_equal(mine["value"].to_numpy(), scr["value"].to_numpy()))
    ctl = pd.read_parquet(O.SHIP / "proxy.parquet")
    if sha256_file(O.SHIP / "proxy.parquet") != decl["control"]["development"].split("SHA-256 ")[1].split(";")[0]:
        raise ValueError("The pinned control differs from the declared file.")
    c = mine.set_index(ID).reindex(ctl[ID])
    t = ctl.rename(columns={"value": "ctl"}).assign(cand=c["value"].to_numpy(dtype=float), used=c["opdi_used"].to_numpy(dtype=bool))
    rule = decl["rule"]
    rng = np.random.default_rng(20261004)
    months = {}
    for m in O.MONTHS:
        s = t[t["month"] == m]
        pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
        ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
        months[m] = {"control_rmse": float(np.sqrt(np.mean((s["label"] - s["ctl"]) ** 2))), "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))),
                     "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum()),
                     "opdi_rows_used": int(s["used"].sum())}
    eq = float(np.mean([months[m]["pct"] for m in rule["equal_footing"]]))
    checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= rule["max_month_worsening_pct"] for m in O.MONTHS),
              "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -rule["min_mean_improvement_pct"],
              "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in rule["equal_footing"])}
    dev_delta = {}
    for a, g in t[t["used"]].groupby("airport"):
        dev_delta[a] = {"rows": int(len(g)), "rms_s": float(np.sqrt(np.mean((g["cand"] - g["ctl"]) ** 2)))}
    out = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "candidate": name,
           "control": "pinned v66 development proxy", "governing_proxy_equals_screen": same, "months": months, "checks_1_to_3": checks,
           "dev_opdi_change_by_airport": dev_delta, "official_scores_used": "none", "code": {p: sha256_file(ROOT / p) for p in CODE},
           "twelve_month_check": "none exists: the stacking components cover only the four development months (as stack_resid_v3); stated, not substituted"}
    write_json(RESULTS, out)
    print("governing proxy equals screen:", same, {m: (round(v["pct"], 4), round(v["lo"], 3), round(v["hi"], 3)) for m, v in months.items()}, checks, flush=True)


def step_stack_opdi(b: Path, c: Path, c_rows: Path, delta_path: Path, out: Path, version: int, airports: list[str]):
    """tools/export_ctx_ownpoints.step_stack's arithmetic (candidate own_all_w100) with the OPDI learner added (declaration: mechanism)."""
    import export_ctx_ownpoints as E
    import run_ctx_ownpoints_chain as CH
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    d = E.CHAIN / CANDIDATE_CHAIN
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
    air, dom, rair = dev["airport"].to_numpy(), dev["dom"].to_numpy(), rk["airport"].to_numpy()
    fxr = S.fixed_rows(rk)
    m2 = S.fit_bag(X46.to_numpy(np.float32), res, ~fx, dom)
    p2 = S.predict_bag(m2, X46r.to_numpy(np.float32))
    h2 = S.outside_support(X46[~fx], air[~fx], X46r, rair) & ~fxr
    p2[fxr], p2[h2] = 0.0, 0.0
    m3 = S.fit_bag(X65.to_numpy(np.float32), res, ~fx, dom)
    p3 = S.predict_bag(m3, X65r.to_numpy(np.float32))
    h3 = (h2 | V3.guard_new(Pn[~fx].reset_index(drop=True), air[~fx], Pr, rair)) & ~fxr
    p3[fxr], p3[h3] = 0.0, 0.0
    mo = S.fit_bag(X79.to_numpy(np.float32), res, ~fx, dom)
    po = S.predict_bag(mo, X79r.to_numpy(np.float32))
    ho = (h2 | V3.guard_new(Pg[~fx].reset_index(drop=True), air[~fx], Pgr, rair)) & ~fxr
    po[fxr], po[ho] = 0.0, 0.0
    use = (Por["o_has"].to_numpy() > 0) & ~ho & np.isin(rair, airports) & ~lirf & ~fxr
    v3vals = np.where(lirf, S.apply(new_r, p2, fxr), S.apply(new_r, p3, fxr))
    values = np.where(use, S.apply(new_r, po, fxr), v3vals)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"vigorous-whistle_v{version}.parquet"
    export_submission(pd.DataFrame({ID: rk[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", path,
                      {"dry_run": True, "step": "stack_resid_v3 refitted (LIRF: refitted v2) + OPDI learner on OPDI-matched rows (configs/opdi_stack_v1.json)",
                       "official_scores_used": "none", "audit_used": False})
    write_json(out / "stack.json", {"checks": checks, "trees_v2": [int(m.best_iteration) for m in m2], "trees_v3": [int(m.best_iteration) for m in m3],
                                    "trees_opdi": [int(m.best_iteration) for m in mo], "held_v3": int(h3.sum()), "held_opdi": int(ho.sum()),
                                    "opdi_rows_used": int(use.sum()), "sha256": sha256_file(path)})
    return path, rk, v3vals, values, use


def dry_run() -> None:
    import export_ctx_ownpoints as E
    import run_ownpoints_weight as W
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    import run_intruder_bound as R
    decl = read_json(ROOT / DECL)
    res = read_json(RESULTS)
    c13 = res["checks_1_to_3"]
    if not (c13["1_no_month_worse_than_0.005pct"] and c13["2_mean_ok"] and c13["3_every_equal_interval_below_zero"] and res["governing_proxy_equals_screen"]):
        raise ValueError("Clauses 1-3 do not hold; no ranking file.")
    if sha256_file(ROOT / V66[0]) != V66[1]:
        raise ValueError("The uploaded v66 file changed.")
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    b, c, c_rows = SH / "rematch/base/vigorous-whistle_v55m_rematch.parquet", SH / "Y/vigorous-whistle_v66.parquet", SH / "Y/rows.parquet"
    delta = SH / "model_part/member_delta.parquet"
    base = ROOT / "runs/opdi_stack/rank"
    # identity: the unchanged step and chain must give v66 byte for byte
    idd = base / "identity_v66"
    if not (idd / "vigorous-whistle_v66.parquet").exists():
        st = E.step_stack(CANDIDATE_CHAIN, b, c, c_rows, delta, idd / "steps/stack", 66)
        if sha256_file(st) != sha256_file(SH / "stack/vigorous-whistle_v66.parquet"):
            raise ValueError("identity: the stack step does not reproduce v66's shipped stack step.")
        v63like = V3.chain_from_stack(st, 66, idd / "steps")
        O2.dry_run(v63like, sha256_file(v63like), 66, idd / "steps/om2")
        shutil.copyfile(idd / "steps/om2/vigorous-whistle_v66.parquet", idd / "vigorous-whistle_v66.parquet")
    ident_ok = sha256_file(idd / "vigorous-whistle_v66.parquet") == V66[1]
    print("identity reproduces the uploaded v66:", ident_ok, flush=True)
    if not ident_ok:
        raise ValueError("identity: the chain does not reproduce v66.")
    # candidate
    out = base / "dry_v67"
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite a dry run.")
    st, rk, v3vals, values, use = step_stack_opdi(b, c, c_rows, delta, out / "steps/stack", 67, decl["candidate"]["airports"])
    shipped = pd.read_parquet(SH / "stack/vigorous-whistle_v66.parquet").set_index(ID)[TARGET].reindex(rk[ID]).to_numpy(dtype=float)
    v3_check = float(np.abs(np.rint(v3vals) - shipped).max())
    print("v3 values (rounded) vs the shipped stack step, max |diff|:", v3_check, flush=True)
    if v3_check != 0.0:
        raise ValueError("The v3 part of the OPDI stack step does not reproduce the shipped stack step.")
    v63like = V3.chain_from_stack(st, 67, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), 67, out / "steps/om2")
    final = out / "vigorous-whistle_v67.parquet"
    shutil.copyfile(out / "steps/om2/vigorous-whistle_v67.parquet", final)
    # clause 4: transfer guard
    rule = decl["rule"]
    lo, hi = rule["transfer_guard"]
    dd = res["dev_opdi_change_by_airport"]
    rk_delta = pd.DataFrame({"airport": rk["airport"].to_numpy(), "d": values - v3vals})[use]
    ratios = {}
    for a, g in rk_delta.groupby("airport"):
        if len(g) >= rule["transfer_guard_min_rows"] and a in dd and dd[a]["rows"] >= rule["transfer_guard_min_rows"] and dd[a]["rms_s"] > 0:
            ratios[a] = {"rank_rows": int(len(g)), "rank_rms_s": float(np.sqrt(np.mean(g["d"] ** 2))), "dev_rows": dd[a]["rows"], "dev_rms_s": dd[a]["rms_s"],
                         "ratio": float(np.sqrt(np.mean(g["d"] ** 2)) / dd[a]["rms_s"])}
    c4 = all(lo <= r["ratio"] <= hi for r in ratios.values())
    # report against v66
    v = pd.read_parquet(final)
    v66 = pd.read_parquet(ROOT / V66[0])
    if not v[ID].equals(v66[ID]):
        raise ValueError("Row order differs from v66.")
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v[ID])
    pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    lirf = raw["ADEP_mvt"].eq("LIRF").to_numpy()
    guard = v[ID].isin(R.GUARD).to_numpy()
    diff = v[TARGET].to_numpy(dtype=float) - v66[TARGET].to_numpy(dtype=float)
    ch = diff != 0
    by = pd.DataFrame({"airport": raw["ADEP_mvt"].to_numpy(), "month": raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy(), "d": diff})
    accepted = bool(c13["1_no_month_worse_than_0.005pct"] and c13["2_mean_ok"] and c13["3_every_equal_interval_below_zero"] and c4)
    rep = {"created_at_utc": utc_now(), "file": final.relative_to(ROOT).as_posix(), "sha256": sha256_file(final), "bytes": final.stat().st_size,
           "identity_reproduces_v66": ident_ok, "v3_part_vs_shipped_stack_max_abs": v3_check, "opdi_rows_used_ranking": int(use.sum()),
           "vs_v66": {"rows_changed": int(ch.sum()), "squared_change": float((diff ** 2).sum()), "rms_change_s": float(np.sqrt((diff ** 2).mean())),
                      "max_up_s": float(diff.max()), "max_down_s": float(diff.min()), "mean_change_s": float(diff.mean())},
           "lirf_rows_changed": int((ch & lirf).sum()), "lirf_population_rows_changed": int((ch & pop).sum()), "guard_rows_changed": int((ch & guard).sum()),
           "by_airport_month": {f"{a} {m}": {"rows_changed": int((g["d"] != 0).sum()), "squared_change": float((g["d"] ** 2).sum()),
                                             "rms_s": float(np.sqrt((g["d"] ** 2).mean())), "max_abs_s": float(g["d"].abs().max())}
                                for (a, m), g in by.groupby(["airport", "month"])},
           "steps": {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted((out / "steps").rglob("vigorous-whistle_v*.parquet"))}}
    if rep["lirf_population_rows_changed"] or rep["guard_rows_changed"]:
        raise ValueError("A LIRF population or guard row changed against v66.")
    write_json(out / "vigorous-whistle_v67.report.json", rep)
    res.update({"clause_4_transfer_guard": {"ratios": ratios, "ok": c4}, "accepted": accepted, "selected": decl["candidate"]["name"] if accepted else None,
                "ranking_dry_run": {k: rep[k] for k in ["file", "sha256", "identity_reproduces_v66", "opdi_rows_used_ranking", "vs_v66", "lirf_rows_changed",
                                                        "lirf_population_rows_changed", "guard_rows_changed"]}})
    write_json(RESULTS, res)
    if not accepted:
        final.rename(final.with_name("NOT_ACCEPTED_vigorous-whistle_v67.parquet"))
    print({k: rep[k] for k in ["sha256", "opdi_rows_used_ranking", "vs_v66", "lirf_rows_changed", "lirf_population_rows_changed", "guard_rows_changed"]},
          "clause 4", c4, {a: round(r["ratio"], 2) for a, r in ratios.items()}, "ACCEPTED", accepted, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--govern", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.govern:
        govern()
    if a.dry_run:
        dry_run()


if __name__ == "__main__":
    main()
