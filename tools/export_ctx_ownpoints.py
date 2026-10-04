"""Ranking file of the own-points development round (configs/ctx_ownpoints_dev_v1.json, stage_2_in_chain "build"): v64's chain with the
context-network member replaced, every step through the shipped tools with their shipped parameters.

  1. model part M': the shipped v55 model part (runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet) with the non-LIRF rows of the v52m
     step recomputed (tools/export_network_member.stacks on the ranking detail): table' = (1 - w) table + w member (twelve-month refit), then
     the masked member at 0.40; the candidate's float where it differs from the combination's float, the shipped integer elsewhere; the
     LIRF layers v53m-v55m touch LIRF rows only and keep their values
  2-5. no-NM rematch -> v48 reconciliation -> extreme-day -> variance re-allocation (tools/run_rematch_nonm.py --matches/--ranking-base,
     tools/export_daily_totals.py, tools/scout_extreme_shift.py, tools/scout_var_alloc.py; docs/final_file/v61_chain.md steps 2-5)
  6. stack_resid_v3 refitted on the rebuilt development components (runs/ctx_ownpoints_dev/chain/<candidate>/), applied to the ranking
     table of tools/run_stack_resid.ranking_table with the model part's v51m-v54m components moved by the member's delta; LIRF rows take the
     refitted v2 learner (as the development chain)
  7-9. intruder bound + identity extension -> own-motion v1 (tools/run_stack_resid_v3.chain_from_stack) -> own_motion_v2
     (tools/run_own_motion_v2.dry_run)
  --identity    the combination member and the shipped development tables in place of the candidate: steps 1-9 must give v64 byte for byte
                (runs/submissions/vigorous-whistle_v64.parquet), with the intermediate checks (M' = shipped v55 model part, step 5 = c67f20d6,
                step 6 = v63's stack step)
  --dry-run --version N    -> runs/ctx_ownpoints_dev/rank/dry_vN/
  --build --version N --user-decision TEXT    rerun into runs/ctx_ownpoints_dev/rank/build_vN, byte identity with the dry run, write
                runs/submissions/vigorous-whistle_vN.parquet (never overwrite) + docs/ctx_ownpoints_dev/export_vN.json
No official score enters. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_submission.contract import export_submission, validate_submission  # noqa: E402

DECLARATION = "configs/ctx_ownpoints_dev_v1.json"
COMBO, MASK = "configs/adsb_ctx3_combo_dev_v1.json", "configs/adsb_ctx3_mask_dev_v1.json"
SHIPPED_M = ("runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet", "266b8c93")
C67 = "c67f20d61f8a85770a68695268e9f49449cc0157970ffc2b9e83044c458201ca"
V63_STACK = "runs/stack_resid/dry_v63/steps/stack/vigorous-whistle_v63.parquet"
V64 = "runs/submissions/vigorous-whistle_v64.parquet"
PY = [sys.executable, "-X", "utf8", "-B"]
CHAIN = ROOT / "runs/ctx_ownpoints_dev/chain"


def run(args: list[str]) -> None:
    print(">", " ".join(args), flush=True)
    subprocess.run(PY + args, cwd=ROOT, check=True)


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


def step_model_part(spec: dict, w: float, out: Path, version: int) -> dict:
    """M' and the member's float deltas on v51m / v52m (zero for the combination itself)."""
    import run_network_member as rnm
    from export_disagreement_gate import rebuild_v34
    from export_network_member import stacks
    from prc2026_final import network_member
    rec, ids, dry, lfpg, _, v34, at34 = rebuild_v34()
    combo_decl, mask_decl = read_json(ROOT / COMBO), read_json(ROOT / MASK)
    combo, _ = rnm.member_values(combo_decl["member"], "ranking", ids)
    mask, _ = rnm.member_values(mask_decl["member"], "ranking", ids)
    member, sources = rnm.member_values(spec, "ranking", ids)
    wm = network_member.candidate_weight(mask_decl, "w40")
    k51 = stacks(dry, combo, 0.40, v34, at34, ids)[3]
    k52 = stacks(dry, mask, wm, v34, at34, ids, ((combo, 0.40),))[3]
    c51 = stacks(dry, member, w, v34, at34, ids)[3]
    c52 = stacks(dry, mask, wm, v34, at34, ids, ((member, w),))[3]
    shipped = pd.read_parquet(ROOT / SHIPPED_M[0]).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    if not sha256_file(ROOT / SHIPPED_M[0]).startswith(SHIPPED_M[1]):
        raise ValueError("The shipped v55 model part changed.")
    lirf = dry["airport"].to_numpy() == "LIRF"
    if np.abs((c52 - k52)[lirf]).max(initial=0.0) > 1e-9 or np.abs((c51 - k51)[lirf]).max(initial=0.0) > 1e-9:
        raise ValueError("A LIRF row depends on the member.")
    changed = (np.abs(c52 - k52) > 1e-9) & ~lirf
    values = np.where(changed, c52, shipped.astype(float))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"vigorous-whistle_v{version}.parquet"
    if path.exists():
        raise ValueError(f"{path} exists; never overwrite.")
    export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", path,
                      {"dry_run": True, "step": "model part with the member replaced", "member_tables": sources, "member_weight": w, "official_scores_used": "none", "audit_used": False})
    exported = pd.read_parquet(path).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    rows = int((exported != shipped).sum())
    delta = pd.DataFrame({ID: ids, "d51": c51 - k51, "d52": c52 - k52})
    delta.to_parquet(out / "member_delta.parquet", index=False)
    info = {"file": rel(path), "sha256": sha256_file(path), "rows_changed_vs_shipped_model_part": rows, "rows_float_changed": int(changed.sum()),
            "squared_change": float(((exported - shipped).astype(float) ** 2).sum()), "member_tables": sources}
    print("model part", {k: v for k, v in info.items() if k != "member_tables"}, flush=True)
    return info


def steps_2_to_5(m_path: Path, out: Path, version: int) -> tuple[Path, Path, Path]:
    m_sha = sha256_file(m_path)
    run(["tools/run_rematch_nonm.py", "--matches", "--out", rel(out / "rematch")])
    run(["tools/run_rematch_nonm.py", "--ranking-base", "--out", rel(out / "rematch"), "--base", rel(m_path), "--base-sha256", m_sha])
    b = out / "rematch/base/vigorous-whistle_v55m_rematch.parquet"
    run(["tools/export_daily_totals.py", "--base", rel(b), "--version", str(version), "--dry-run", rel(out / "R"), "--no-incumbent-report"])
    run(["tools/scout_extreme_shift.py", "--base", rel(b), "--recon", rel(out / "R"), "--version", str(version), "--dry-run", rel(out / "X")])
    x = out / "X" / f"vigorous-whistle_v{version}.parquet"
    run(["tools/scout_var_alloc.py", "--base", rel(b), "--recon", rel(out / "R"), "--version", str(version), "--previous", rel(x),
         "--previous-sha256", sha256_file(x), "--dry-run", rel(out / "Y")])
    return b, out / "Y" / f"vigorous-whistle_v{version}.parquet", out / "Y/rows.parquet"


def step_stack(candidate: str | None, b: Path, c: Path, c_rows: Path, delta_path: Path, out: Path, version: int) -> Path:
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_ctx_ownpoints_chain as CH
    if candidate is None:
        dev = S.dev_table()
    else:
        d = CHAIN / candidate
        dev = CH.dev_table_from(d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet")
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    core = V3.own_points("dev", dev[ID].to_numpy())
    Pn = V3.new_columns(core, dev["new"].to_numpy(dtype=float))
    X65 = pd.concat([X46, Pn], axis=1)
    fx = S.fixed_rows(dev)
    checks: dict = {}
    rk = S.ranking_table((rel(c), sha256_file(c)), checks, rel(b), rel(c_rows))
    delta = pd.read_parquet(delta_path).set_index(ID).reindex(rk[ID])
    lirf = rk["airport"].eq("LIRF").to_numpy()
    for col, dc in [("v51m", "d51"), ("v52m", "d52"), ("v53m", "d52"), ("v54m", "d52")]:
        rk[col] = rk[col].to_numpy(dtype=float) + np.where(lirf, 0.0, delta[dc].to_numpy(dtype=float))
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


def chain(candidate: str | None, out: Path, version: int, identity: bool) -> Path:
    import run_ctx_ownpoints_chain as CH
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    decl = read_json(ROOT / DECLARATION)
    if identity:
        spec, w = read_json(ROOT / COMBO)["member"], 0.40
    else:
        cand = decl["candidates"][candidate]
        spec, w = CH.member_spec(decl, cand["member"]), float(cand["member_weight"])
    steps = out / "steps"
    m = step_model_part(spec, w, steps / "model_part", version)
    if identity and sha256_file(ROOT / m["file"]) != sha256_file(ROOT / SHIPPED_M[0]):
        raise ValueError("identity: M' differs from the shipped v55 model part.")
    b, c, c_rows = steps_2_to_5(ROOT / m["file"], steps, version)
    if identity and sha256_file(c) != C67:
        raise ValueError("identity: step 5 does not reproduce c67f20d6.")
    stack = step_stack(None if identity else candidate, b, c, c_rows, steps / "model_part/member_delta.parquet", steps / "stack", version)
    if identity and sha256_file(stack) != sha256_file(ROOT / V63_STACK):
        raise ValueError("identity: the stack step does not reproduce v63's stack step.")
    v63like = V3.chain_from_stack(stack, version, steps)
    O2.dry_run(v63like, sha256_file(v63like), version, steps / "om2")
    final = steps / "om2" / f"vigorous-whistle_v{version}.parquet"
    shutil.copyfile(final, out / f"vigorous-whistle_v{version}.parquet")
    result = out / f"vigorous-whistle_v{version}.parquet"
    if identity:
        same = (ROOT / V64).exists() and sha256_file(result) == sha256_file(ROOT / V64)
        print("identity: the chain reproduces v64:", same, sha256_file(result))
        if (ROOT / V64).exists() and not same:
            raise ValueError("identity: the chain does not reproduce the built v64.")
    write_json(out / "steps.json", {"created_at_utc": utc_now(), "candidate": candidate if not identity else "identity (combination member)",
                                    "model_part": m, "steps": {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(steps.rglob("vigorous-whistle_v*.parquet"))},
                                    "final": sha256_file(result)})
    return result


def compare(result: Path, ref: Path, out: Path) -> dict:
    a = pd.read_parquet(ref)
    v = pd.read_parquet(result)
    if not v[ID].equals(a[ID]):
        raise ValueError("Row order differs.")
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v[ID])
    pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    d = v[TARGET].to_numpy(dtype=float) - a[TARGET].to_numpy(dtype=float)
    ch = d != 0
    guard = v[ID].isin([213142471.0, 213142469.0]).to_numpy()
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
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--user-decision", default=None)
    a = ap.parse_args()
    decl = read_json(ROOT / DECLARATION)
    base_dir = ROOT / "runs/ctx_ownpoints_dev/rank"
    if a.identity:
        out = base_dir / f"identity_v{a.version}"
        if out.exists():
            raise ValueError(f"{out} exists.")
        chain(None, out, a.version, True)
        return
    sel = read_json(ROOT / decl["selection"])["selected_candidate"]
    stage2 = read_json(ROOT / "docs/ctx_ownpoints_dev/results_chain_v1.json")
    if not sel or stage2.get("candidate") != sel or not stage2["checks"]["accepted"]:
        raise ValueError("Both declared stages must accept the candidate before a ranking file is written.")
    ref = sorted((ROOT / "runs/submissions").glob("vigorous-whistle_v*.parquet"), key=lambda p: int(p.stem.split("_v")[1]))[-1]
    if a.dry_run:
        out = base_dir / f"dry_v{a.version}"
        if out.exists():
            raise ValueError(f"{out} exists.")
        result = chain(sel, out, a.version, False)
        rep = compare(result, ref, out)
        rep.update({"file": rel(result), "sha256": sha256_file(result), "candidate": sel})
        write_json(out / f"vigorous-whistle_v{a.version}.report.json", rep)
        print({k: v for k, v in rep.items() if k != "by_airport"})
    if a.build:
        target = ROOT / f"runs/submissions/vigorous-whistle_v{a.version}.parquet"
        dry = base_dir / f"dry_v{a.version}/vigorous-whistle_v{a.version}.parquet"
        if target.exists() or not dry.exists() or not a.user_decision:
            raise ValueError("Never overwrite; run the dry run first; record the user's decision.")
        out = base_dir / f"build_v{a.version}"
        if out.exists():
            raise ValueError(f"{out} exists; never overwrite a build directory.")
        built = chain(sel, out, a.version, False)
        if sha256_file(built) != sha256_file(dry):
            raise ValueError("The rebuilt file differs from the dry run.")
        target.write_bytes(built.read_bytes())
        sub = validate_submission(target, ROOT / "data/raw/submitting.parquet")
        sub.update({"version": a.version, "provenance": {"experiment_id": decl["experiment_id"], "selected_candidate": sel, "selected_by": "both declared stages",
                                                         "user_decision": a.user_decision, "official_scores_used": "none", "audit_used": False,
                                                         "declaration": {DECLARATION: sha256_file(ROOT / DECLARATION)},
                                                         "code": {p: sha256_file(ROOT / p) for p in ["tools/export_ctx_ownpoints.py", "tools/run_ctx_ownpoints_chain.py", "tools/run_ctx_ownpoints_dev.py"]}}})
        write_json(target.with_suffix(".receipt.json"), sub)
        prev = [p for p in sorted((ROOT / "runs/submissions").glob("vigorous-whistle_v*.parquet"), key=lambda p: int(p.stem.split("_v")[1])) if p != target][-1]
        rep = compare(target, prev, out)
        rep.update({"kind": "submission build at the user's word", "file": rel(target), "sha256": sha256_file(target), "dry_run_sha256": sha256_file(dry), "candidate": sel})
        write_json(ROOT / f"docs/ctx_ownpoints_dev/export_v{a.version}.json", rep)
        print("built", target.name, rep["sha256"])


if __name__ == "__main__":
    main()
