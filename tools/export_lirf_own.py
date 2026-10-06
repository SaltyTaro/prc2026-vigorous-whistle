"""Ranking dry run of the LIRF own-points round (configs/lirf_own_dev_v1.json): v65's chain with the masked LIRF member M replaced by
M' = (1 - s) M + s Mown on the ranking's own-route rows of non-dark months, every other step through the shipped tools with their
shipped parameters (tools/export_ctx_ownpoints.py's chain).

  1. model part M'': v65's model part (runs/ctx_ownpoints_dev/rank/build_v65/steps/model_part/vigorous-whistle_v65.parquet) with the
     LIRF own-route rows recomputed as the LIRF member exports do (tools/export_lirf_member.py, tools/export_lirf_combo.py):
     value = base + (value with M' - value with M), floored at -12 s; lirf_nonpop_matched: v34 matched blend + m600 on 0.60 net + 0.40 M;
     lirf_nonpop_unmatched of non-dark months: v34 unmatched blend + u600 on 0.45 net + 0.55 M; dark months (LIRF non-population match
     share below 5%: July 2026) keep v65's values. M = the 16-seed refit12 mean of lm25_32 (configs/adsb_ctx3_lirf_mask_dev_v1.json),
     Mown = the 16-seed refit12 mean of the declared variant. Route: p_first_mv not null in features_v19own (2026-01 / 2026-07).
  2-5. no-NM rematch -> v48 -> extreme-day -> variance re-allocation (tools/export_ctx_ownpoints.steps_2_to_5)
  6. stack_resid_v3 refitted on this round's rebuilt development components (runs/lirf_own/chain/<candidate>/), applied to the ranking
     table with v65's member deltas on non-LIRF rows (as v65) and the LIRF route delta on v53m / v54m of unmatched route rows; LIRF rows
     take the refitted v2 learner
  7-9. intruder bound + identity extension -> own-motion v1 -> own_motion_v2
  --identity   M' = M and the identity chain directory: must reproduce runs/submissions/vigorous-whistle_v65.parquet byte for byte
  --dry-run --version N   -> runs/lirf_own/rank/dry_vN/ (never under runs/submissions). Nothing is uploaded; no official score enters.
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
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_final import lirf_gate as lg  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
import run_lirf_combo as rlc  # noqa: E402
import run_lirf_member as rlm  # noqa: E402

DECL = "configs/lirf_own_dev_v1.json"
SHIPPED_MEMBER = "configs/adsb_ctx3_lirf_mask_dev_v1.json"
DETAIL = "runs/final_dryrun/run/recipe/detail.parquet"
V65_BUILD = ROOT / "runs/ctx_ownpoints_dev/rank/build_v65"
V65_MP = V65_BUILD / "steps/model_part/vigorous-whistle_v65.parquet"
V65_DELTA = V65_BUILD / "steps/model_part/member_delta.parquet"
V65 = ROOT / "runs/submissions/vigorous-whistle_v65.parquet"
V65_SHA = "45160819c67ed4827280d6968ca26ec3f09d915d31e83cee62090b7217bf3cc3"
OWN = ROOT / "data/external/adsblol/features_v19own"
OUT = ROOT / "runs/lirf_own/rank"
GUARD = [213142471.0, 213142469.0]


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


def refit12_mean(decl_fits: dict, ids: np.ndarray) -> tuple[np.ndarray, dict]:
    paths = [ROOT / decl_fits["outputs"].split("<variant>")[0] / f"{v}_s{s}" / "refit12" / "ranking_predictions.parquet"
             for v in decl_fits["variants"] for s in decl_fits["seeds"]]
    missing = [p.as_posix() for p in paths if not p.exists()]
    if missing:
        raise ValueError(f"refit12 predictions missing: {len(missing)}")
    v = np.mean([pd.read_parquet(p).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float) for p in paths], axis=0)
    if not np.isfinite(v).all():
        raise ValueError("A member does not cover every route row.")
    return v, {rel(p): sha256_file(p) for p in paths}


def step_model_part(decl: dict | None, candidate: str, out: Path, version: int) -> dict:
    share = 0.0 if decl is None else float(decl["candidates"][candidate]["own_share"])
    base = pd.read_parquet(V65_MP)
    ids = base[ID].to_numpy()
    values = base[TARGET].to_numpy(dtype=float).copy()
    detail = pd.read_parquet(ROOT / DETAIL, columns=[ID, "airport", "month", "scope", "v9", "ens_ctx_lirf", "ens_v22_lirf_q"])
    detail = pd.DataFrame({ID: ids}).merge(detail, on=ID, how="left", validate="one_to_one")
    if detail["scope"].isna().any():
        raise ValueError("The recipe detail does not cover every ranking row.")
    month = detail["month"].astype(str).to_numpy()
    own = pd.concat([pd.read_parquet(OWN / f"features_v19_{m}.parquet", columns=[ID, "p_first_mv"]) for m in ("2026-01", "2026-07")])
    route = own.set_index(ID)["p_first_mv"].reindex(ids).notna().to_numpy()
    km, ku = detail["scope"].eq("lirf_nonpop_matched").to_numpy(), detail["scope"].eq("lirf_nonpop_unmatched").to_numpy()
    dark = {}
    for m in sorted(set(month[km | ku])):
        a, b = int((km & (month == m)).sum()), int((ku & (month == m)).sum())
        dark[m] = a / max(a + b, 1) < 0.05
    live = np.array([not dark.get(m, True) for m in month])
    rm, ru = km & route & live, ku & route & live
    k = rm | ru
    mask_fits = read_json(ROOT / SHIPPED_MEMBER)["fits"]
    M, m_sources = refit12_mean(mask_fits, ids[k])
    if decl is None:
        Mo, o_sources = M.copy(), {}
    else:
        Mo, o_sources = refit12_mean(decl["fits"], ids[k])
    Mp = (1 - share) * M + share * Mo
    v9, net, tree = (detail.loc[k, c].to_numpy(dtype=float) for c in ("v9", "ens_ctx_lirf", "ens_v22_lirf_q"))
    sub_m, sub_u = rm[k], ru[k]
    delta = np.zeros(int(k.sum()))
    if sub_m.any():
        c0, _ = rlc.values(v9[sub_m], net[sub_m], tree[sub_m], M[sub_m], 0.40)
        c1, _ = rlc.values(v9[sub_m], net[sub_m], tree[sub_m], Mp[sub_m], 0.40)
        delta[sub_m] = c1 - c0
    if sub_u.any():
        c0, _ = lg.gated(v9[sub_u], 0.45 * net[sub_u] + 0.55 * M[sub_u], tree[sub_u], "unmatched", rlm.GATE)
        c1, _ = lg.gated(v9[sub_u], 0.45 * net[sub_u] + 0.55 * Mp[sub_u], tree[sub_u], "unmatched", rlm.GATE)
        delta[sub_u] = c1 - c0
    new = values.copy()
    new[k] = np.maximum(values[k] + delta, lg.FLOOR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"vigorous-whistle_v{version}.parquet"
    if path.exists():
        raise ValueError(f"{path} exists; never overwrite.")
    export_submission(pd.DataFrame({ID: ids, "prediction": new}), ROOT / "data/raw/submitting.parquet", path,
                      {"dry_run": True, "step": "v65's model part with the LIRF own-route member replaced", "candidate": candidate, "own_share": share,
                       "member_tables": {**m_sources, **o_sources}, "official_scores_used": "none", "audit_used": False})
    full_delta = np.zeros(len(ids))
    full_delta[k] = delta
    pd.DataFrame({ID: ids, "d_lirf": full_delta, "u_route": ru}).to_parquet(out / "lirf_delta.parquet", index=False)
    exported = pd.read_parquet(path).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - values
    if ((change != 0) & ~k).any():
        raise ValueError("A row outside the route changed.")
    info = {"file": rel(path), "sha256": sha256_file(path), "route_rows": {"matched": int(rm.sum()), "unmatched": int(ru.sum())}, "dark_months": dark,
            "rows_changed": int((change != 0).sum()), "squared_change": float((change ** 2).sum()), "max_abs_change_s": float(np.abs(change).max(initial=0.0)),
            "member_minus_own_rms_s": float(np.sqrt(np.mean((Mo - M) ** 2))) if k.any() else 0.0}
    print("model part", info, flush=True)
    return info


def step_stack(candidate: str, b: Path, c: Path, c_rows: Path, out: Path, version: int, lirf_delta: Path) -> Path:
    import run_ctx_ownpoints_chain as CH
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    d = ROOT / "runs/lirf_own/chain" / candidate
    dev = CH.dev_table_from(d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet")
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    core = V3.own_points("dev", dev[ID].to_numpy())
    Pn = V3.new_columns(core, dev["new"].to_numpy(dtype=float))
    X65 = pd.concat([X46, Pn], axis=1)
    fx = S.fixed_rows(dev)
    checks: dict = {}
    rk = S.ranking_table((rel(c), sha256_file(c)), checks, rel(b), rel(c_rows))
    delta = pd.read_parquet(V65_DELTA).set_index(ID).reindex(rk[ID])
    lirf = rk["airport"].eq("LIRF").to_numpy()
    for col, dc in [("v51m", "d51"), ("v52m", "d52"), ("v53m", "d52"), ("v54m", "d52")]:
        rk[col] = rk[col].to_numpy(dtype=float) + np.where(lirf, 0.0, delta[dc].to_numpy(dtype=float))
    ld = pd.read_parquet(lirf_delta).set_index(ID).reindex(rk[ID])
    du = np.where(ld["u_route"].to_numpy(dtype=bool), ld["d_lirf"].to_numpy(dtype=float), 0.0)
    for col in ("v53m", "v54m"):
        rk[col] = rk[col].to_numpy(dtype=float) + du
    X46r = S.features(rk).reset_index(drop=True)
    Pr = V3.new_columns(V3.own_points("rank", rk[ID].to_numpy()), rk["new"].to_numpy(dtype=float))
    X65r = pd.concat([X46r, Pr], axis=1)
    air, dom = dev["airport"].to_numpy(), dev["dom"].to_numpy()
    fxr = S.fixed_rows(rk)
    m2 = S.fit_bag(X46.to_numpy(np.float32), res, ~fx, dom)
    p2 = S.predict_bag(m2, X46r.to_numpy(np.float32))
    h2 = S.outside_support(X46[~fx], air[~fx], X46r, rk["airport"].to_numpy()) & ~fxr
    p2[fxr], p2[h2] = 0.0, 0.0
    m3 = S.fit_bag(X65.to_numpy(np.float32), res, ~fx, dom)
    p3 = S.predict_bag(m3, X65r.to_numpy(np.float32))
    h3 = (h2 | V3.guard_new(Pn[~fx].reset_index(drop=True), air[~fx], Pr, rk["airport"].to_numpy())) & ~fxr
    p3[fxr], p3[h3] = 0.0, 0.0
    base = rk["new"].to_numpy(dtype=float)
    values = np.where(lirf, S.apply(base, p2, fxr), S.apply(base, p3, fxr))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"vigorous-whistle_v{version}.parquet"
    export_submission(pd.DataFrame({ID: rk[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", path,
                      {"dry_run": True, "step": "stack_resid_v3 refitted (LIRF: refitted v2)", "official_scores_used": "none", "audit_used": False})
    write_json(out / "stack.json", {"checks": checks, "trees_v2": [int(m.best_iteration) for m in m2], "trees_v3": [int(m.best_iteration) for m in m3],
                                    "held_v3": int(h3.sum()), "sha256": sha256_file(path)})
    return path


def chain(decl: dict | None, candidate: str, out: Path, version: int) -> Path:
    import export_ctx_ownpoints as E
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    if sha256_file(V65) != V65_SHA:
        raise ValueError("v65 differs from its record.")
    steps = out / "steps"
    m = step_model_part(decl, candidate, steps / "model_part", version)
    if decl is None and sha256_file(ROOT / m["file"]) != sha256_file(V65_MP):
        raise ValueError("identity: M'' differs from v65's model part.")
    b, c, c_rows = E.steps_2_to_5(ROOT / m["file"], steps, version)
    stack = step_stack(candidate, b, c, c_rows, steps / "stack", version, steps / "model_part/lirf_delta.parquet")
    v63like = V3.chain_from_stack(stack, version, steps)
    O2.dry_run(v63like, sha256_file(v63like), version, steps / "om2")
    final = steps / "om2" / f"vigorous-whistle_v{version}.parquet"
    result = out / f"vigorous-whistle_v{version}.parquet"
    shutil.copyfile(final, result)
    write_json(out / "steps.json", {"created_at_utc": utc_now(), "candidate": candidate, "model_part": m,
                                    "steps": {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(steps.rglob("vigorous-whistle_v*.parquet"))},
                                    "final": sha256_file(result), "code": {q: sha256_file(ROOT / q) for q in ["tools/export_lirf_own.py", "tools/export_ctx_ownpoints.py"]}})
    return result


def compare(result: Path, ref: Path) -> dict:
    a, v = pd.read_parquet(ref), pd.read_parquet(result)
    if not v[ID].equals(a[ID]):
        raise ValueError("Row order differs.")
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v[ID])
    pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    d = v[TARGET].to_numpy(dtype=float) - a[TARGET].to_numpy(dtype=float)
    ch = d != 0
    guard = v[ID].isin(GUARD).to_numpy()
    rep = {"reference": {"file": rel(ref), "sha256": sha256_file(ref)}, "rows_changed": int(ch.sum()), "squared_change": float((d ** 2).sum()),
           "max_up_s": float(d.max()), "max_down_s": float(d.min()), "lirf_rows_changed": int((ch & raw["ADEP_mvt"].eq("LIRF").to_numpy()).sum()),
           "lirf_population_rows_changed": int((ch & pop).sum()), "guard_rows_changed": int((ch & guard).sum()),
           "by_airport": {k: {"rows_changed": int((g != 0).sum()), "squared_change": float((g ** 2).sum())} for k, g in pd.Series(d, index=raw["ADEP_mvt"].to_numpy()).groupby(level=0)}}
    if rep["lirf_population_rows_changed"] or rep["guard_rows_changed"]:
        raise ValueError("A LIRF population or guard row changed.")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--version", type=int, required=True)
    a = ap.parse_args()
    if a.identity:
        out = OUT / f"identity_v{a.version}"
        if out.exists():
            raise ValueError(f"{out} exists.")
        result = chain(None, "identity", out, a.version)
        same = sha256_file(result) == V65_SHA
        print("identity: the chain reproduces v65 byte for byte:", same, sha256_file(result))
        write_json(out / "identity.json", {"created_at_utc": utc_now(), "reproduces_v65": same, "sha256": sha256_file(result)})
        if not same:
            raise ValueError("identity: the chain does not reproduce v65.")
        return
    decl = read_json(ROOT / DECL)
    sel = read_json(ROOT / "runs/lirf_own/dev/selection.json")["selected_candidate"]
    stage2 = read_json(ROOT / "docs/lirf_own/results_chain_v1.json")
    if not sel or stage2.get("candidate") != sel or not stage2["checks"]["accepted"]:
        raise ValueError("Both declared stages must accept the candidate before a ranking dry run is written.")
    out = OUT / f"dry_v{a.version}"
    if out.exists():
        raise ValueError(f"{out} exists.")
    result = chain(decl, sel, out, a.version)
    rep = compare(result, V65)
    rep.update({"file": rel(result), "sha256": sha256_file(result), "candidate": sel, "declaration": {DECL: object_hash(decl)}})
    write_json(out / f"vigorous-whistle_v{a.version}.report.json", rep)
    print({k: v for k, v in rep.items() if k != "by_airport"})


if __name__ == "__main__":
    main()
