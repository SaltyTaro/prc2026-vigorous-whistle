"""Development round of lever 12b (configs/lfpg_net_dev_v1.json): on LFPG rows with an ADS-B match only, the shipped member own_all is replaced
by 0.5 own_all + 0.5 lfpg_dl32own (own_all's matched-row network fitted on LFPG rows only); member weight 1.00; every other row keeps own_all.
Every arithmetic step is imported unchanged from the rounds that built v65-v69 (tools/run_ctx_own2_dev.py's stage 1, run_ctx_ownpoints_chain.py's
model part and chain, run_stack_weight.py's ten-seed stacking at weight 0.65, export_ctx_ownpoints.py's ranking steps 1-5); only the member, the
control and the output paths differ. The variance re-allocation's LightGBM is pinned (deterministic, column-wise) in every development chain run.

  --build-member      runs/lfpg_net/member/s<seed>/<leaf>.parquet: lfpg_dl32own's prediction on LFPG rows with adsb_matched = 1, NaN elsewhere
  --build-mix         runs/lfpg_net/member_mix/s<641+k>/<leaf>.parquet: own_all s<601+k>, LFPG matched rows 0.5 own_all + 0.5 lfpg_matched
  --stage1            member level (information): own_all at 1.00 (must reproduce docs/ctx_ownpoints_weight/results_stage1.json) vs lfpg_mix at 1.00
  --chain NAME        model part + chain (pinned) for own_all_w100 (identity: must equal runs/pubrefit/chain/ship byte for byte) or lfpg_mix_w100
  --dev-month M       the ten-seed month-blocked stacking predictions on the candidate's rebuilt chain files (run_stack_weight.dev_month unchanged)
  --score             identity (fork K's stored predictions on the ship files must give the v69 development proxy exactly), then the candidate;
                      v57's clauses 1-3; clause 4 (member-level transfer guard) -> docs/lfpg_net/results_v1.json
  --rank-steps NAME --pinned/--unpinned   ranking steps 1-5 (export_ctx_ownpoints.step_model_part + steps_2_to_5) into runs/lfpg_net/rank/<tag>/
  --rank-final NAME   the stacking layer (ten seeds, weight 0.65, OPDI at LSZH) fitted on the NAME chain's development tables and applied to the
                      pinned NAME steps, then chain_from_stack and own_motion_v2 -> runs/lfpg_net/rank/pinned_<NAME>/final
  --rank-dry          dry run = v69 + (pinned candidate - pinned identity), floor -12 s -> runs/lfpg_net/rank/dry_v70/ (only after acceptance)
No official score enters. Nothing is uploaded; nothing is written under runs/submissions.
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
from prc2026_scout import var_alloc as _VA  # noqa: E402

_VA.PARAMS = {**_VA.PARAMS, "deterministic": True, "force_col_wise": True}  # declared pinning (stage_2)

from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402

DECLARATION = "configs/lfpg_net_dev_v1.json"
OWN_DECL = "configs/ctx_ownpoints_dev_v1.json"
OUT = ROOT / "runs/lfpg_net"
DOCS = ROOT / "docs/lfpg_net"
WEIGHT_STAGE1 = "docs/ctx_ownpoints_weight/results_stage1.json"
SHIP = ROOT / "runs/pubrefit/chain/ship"
SHIP_SHA = {"ctl.parquet": "cc9f54173d59a1fbfa642bc40abcf09d8fb37b2e6aa48a1c9055e0c16a7870a1",
            "rows_all.parquet": "9389b3d9627667b1d6c9", "dmp.parquet": "5a5c3373dcda7f4442f0",
            "cand.parquet": "bdf959c953216e0dabad845d6f3cb016b0334ffbaeed019fc0a1ddad6474772f"}
V69 = ("runs/submissions/vigorous-whistle_v69.parquet", "192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665")
V66_STEPS = ROOT / "runs/ctx_ownpoints_weight/rank/dry_v66/steps"
ADSB_CACHE = ROOT / "runs/adsb_ctx2/inputs/adsb_features_arcw0"
GUARD = [213142471, 213142469]
W, K = 0.65, 10
CANDS = {"own_all_w100": {"member": "own_all", "member_weight": 1.0}, "lfpg_mix_w100": {"member": "lfpg_mix", "member_weight": 1.0}}
DEV_LEAVES = ["2025-01", "2025-07", "2025-11", "2025-12"]
CODE = ["tools/run_lfpg_net_dev.py", "tools/run_ctx_own2_dev.py", "tools/run_stack_weight.py", "tools/run_ctx_ownpoints_chain.py",
        "tools/export_ctx_ownpoints.py", "tools/run_ownpoints_weight.py", "tools/run_opdi_stack.py", "tools/run_opdi_stack_decl.py",
        "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py", "tools/run_own_motion_v2.py", "src/prc2026_scout/var_alloc.py"]


def declaration() -> dict:
    decl = read_json(ROOT / DECLARATION)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared or does not respect its own contract.")
    return decl


def specs(decl: dict) -> dict:
    return {"own_all": read_json(ROOT / OWN_DECL)["member"], "lfpg_matched": decl["member"],
            "lfpg_mix": {"name": "lfpg_mix", "column": "adsb_full", "tables": decl["members_derived"]["lfpg_mix"]["tables"]}}


def airports() -> pd.Series:
    """Departure airport by movement id over the twelve 2025 files and the ranking file (label-free)."""
    paths = sorted((ROOT / "data/raw").glob("training_2025-*.parquet")) + [ROOT / "data/raw/ranking.parquet"]
    raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", "ADEP_mvt"]) for p in paths], ignore_index=True)
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
    return raw.set_index(ID)["ADEP_mvt"]


# ----------------------------------------------------------------------------------------------------------------------------- member
def build_member(decl: dict) -> None:
    from build_combo_member import check_cache, matched_flags
    fits = decl["fits"]
    if sha256_file(ROOT / fits["network_config"]) != fits["network_config_sha256"]:
        raise ValueError("The network configuration differs from the declared one.")
    flags, cache_hashes = matched_flags(ADSB_CACHE)
    air = airports()
    built = {}
    for leaf, spec in fits["folds"].items():
        reference = None
        for seed, target in zip(fits["seeds"], decl["member"]["tables"][leaf]):
            if f"/s{seed}/" not in target:
                raise ValueError("The declared member tables do not follow the declared seeds.")
            directory = OUT / "nets" / f"lfpg_dl32own_s{seed}" / spec["fold"]
            record = read_json(directory / "run.json")
            if (record["config_sha256"] != fits["network_config_sha256"] or record["variant"] != "lfpg_dl32own" or record["seed"] != seed
                    or record["fold"] != spec["fold"] or record.get("smoke_limit_rows") is not None or record["inputs"]["device"] != "cuda"):
                raise ValueError(f"{directory} is not a declared fit.")
            check_cache(record, cache_hashes, directory)
            frame = pd.read_parquet(directory / spec["file"])
            ids = frame[ID].to_numpy()
            if reference is None:
                reference = ids
            elif not np.array_equal(reference, ids):
                raise ValueError(f"{leaf}: seed {seed} predicts different rows than the first seed.")
            matched = flags.reindex(ids)
            ap = air.reindex(ids)
            if matched.isna().any() or ap.isna().any():
                raise ValueError(f"{leaf}: rows without a match flag or an airport.")
            rows = matched.to_numpy(dtype=bool) & ap.eq("LFPG").to_numpy()
            values = np.where(rows, frame["adsb_full"].to_numpy(dtype=float), np.nan)
            if not np.isfinite(values[rows]).all():
                raise ValueError(f"{leaf} seed {seed}: non-finite prediction on an LFPG matched row.")
            out = ROOT / target
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: ids, "adsb_full": values}).to_parquet(out, index=False, compression="zstd")
            built[target] = {"rows": int(len(ids)), "lfpg_matched_rows": int(rows.sum()), "sha256": sha256_file(out),
                             "fit": {"dir": directory.relative_to(ROOT).as_posix(), "prediction_sha256": sha256_file(directory / spec["file"])}}
        print(leaf, "seeds", len(fits["seeds"]), "rows", len(reference), "LFPG matched", built[target]["lfpg_matched_rows"], flush=True)
    write_json(OUT / "member_build.json", {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
                                           "adsb_cache": {"path": ADSB_CACHE.relative_to(ROOT).as_posix(), "sha256": cache_hashes}, "tables": built,
                                           "tool_sha256": sha256_file(Path(__file__)), "audit_used": False})


def build_mix(decl: dict) -> None:
    sp = specs(decl)
    own, lf, mix = sp["own_all"]["tables"], sp["lfpg_matched"]["tables"], sp["lfpg_mix"]["tables"]
    built = {}
    for leaf in mix:
        if not (len(own[leaf]) == len(lf[leaf]) == len(mix[leaf]) == 16):
            raise ValueError(f"{leaf}: 16 seeds per member are declared.")
        for a_rel, b_rel, m_rel in zip(own[leaf], lf[leaf], mix[leaf]):
            a, b = pd.read_parquet(ROOT / a_rel), pd.read_parquet(ROOT / b_rel)
            if not np.array_equal(a[ID].to_numpy(), b[ID].to_numpy()):
                raise ValueError(f"{leaf}: {a_rel} and {b_rel} predict different rows.")
            va, vb = a["adsb_full"].to_numpy(dtype=float), b["adsb_full"].to_numpy(dtype=float)
            rows = np.isfinite(vb)
            if not np.isfinite(va[rows]).all():
                raise ValueError(f"{leaf}: own_all is not finite on every LFPG matched row of {b_rel}.")
            values = va.copy()
            values[rows] = 0.5 * va[rows] + 0.5 * vb[rows]
            out = ROOT / m_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: a[ID].to_numpy(), "adsb_full": values}).to_parquet(out, index=False, compression="zstd")
            same = np.array_equal(values[~rows], va[~rows], equal_nan=True)
            if not same:
                raise ValueError("A row outside the LFPG matched rows changed.")
            built[m_rel] = {"own_all": [a_rel, sha256_file(ROOT / a_rel)], "lfpg_matched": [b_rel, sha256_file(ROOT / b_rel)], "rows": int(len(a)),
                            "mixed_rows": int(rows.sum()), "sha256": sha256_file(out)}
        print(leaf, "mixed", len(mix[leaf]), "rows mixed", int(rows.sum()), flush=True)
    write_json(OUT / "member_mix_build.json", {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
                                               "tables": built, "tool_sha256": sha256_file(Path(__file__)), "audit_used": False})


# ----------------------------------------------------------------------------------------------------------------------------- stage 1
def stage1(decl: dict) -> None:
    """tools/run_ctx_own2_dev.stage1's arithmetic with this round's members and paths (information only)."""
    import run_ctx_own2_dev as L5
    out_path, doc_path = OUT / "dev/stage1.json", DOCS / "results_stage1.json"
    if out_path.exists() or doc_path.exists():
        raise ValueError("Preserve the original stage-1 result.")
    sp = specs(decl)
    shim = {"candidates": decl["candidates"], "stage_1": {"equal_footing_months": ["2025-01", "2025-07", "2025-12"],
                                                          "minimum_mean_improvement_fraction": 0.0002}}
    L5.specs = lambda _decl: sp
    L5.OUT, L5.DOCS, L5.DECLARATION = OUT, DOCS, DECLARATION
    L5.stage1({**decl, **shim})


# ----------------------------------------------------------------------------------------------------------------------------- stage 2
def patch_members(decl: dict) -> None:
    import run_ctx_ownpoints_chain as CH
    import scout_refit_prep as SRP
    SRP.ROOT, SRP.OUT = ROOT, ROOT / "runs/scout/refit"  # the module hard-codes the workstation's D:/ root (identical there; needed on the Linux VM)
    sp = specs(decl)
    CH.member_spec = lambda _decl, name: sp[name]
    CH.OUT = OUT / "chain"


def build_chain(decl: dict, name: str) -> None:
    import run_ctx_ownpoints_chain as CH
    patch_members(decl)
    d = CH.OUT / name
    if not (d / "cand.parquet").exists():
        CH.step_model_part({"candidates": CANDS}, name, False)
        CH.step_chain(name)
    got = {f: sha256_file(d / f) for f in SHIP_SHA}
    rec = {"created_at_utc": utc_now(), "name": name, "files": got, "var_alloc_params": _VA.PARAMS}
    if name == "own_all_w100":
        # amendment 1 (configs/lfpg_net_dev_v1.json): byte identity is reported; the identity is numeric (<= 1e-9 s on every column)
        rec["byte_identical_to_ship"] = {f: got[f].startswith(SHIP_SHA[f]) for f in SHIP_SHA}
        diffs = {}
        for f in SHIP_SHA:
            a, b = pd.read_parquet(d / f), pd.read_parquet(SHIP / f)
            if list(a.columns) != list(b.columns) or not a[ID].equals(b[ID]):
                raise ValueError(f"identity: {f} differs in layout from the ship file.")
            for c in a.columns:
                if a[c].equals(b[c]):
                    continue
                if not pd.api.types.is_numeric_dtype(a[c]) or not np.array_equal(a[c].isna().to_numpy(), b[c].isna().to_numpy()):
                    raise ValueError(f"identity: {f}:{c} differs beyond rounding.")
                diffs[f"{f}:{c}"] = float(np.nanmax(np.abs(a[c].to_numpy(dtype=float) - b[c].to_numpy(dtype=float))))
        rec["max_abs_diff_vs_ship"] = diffs
        rec["identity_equals_ship"] = bool(max(diffs.values(), default=0.0) <= 1e-9)
        print("identity: own_all_w100 chain files equal the ship files (bytes)", rec["byte_identical_to_ship"], "max |diff|", diffs,
              "->", rec["identity_equals_ship"], flush=True)
    write_json(d / "build.json", rec)
    if name == "own_all_w100" and not rec["identity_equals_ship"]:
        raise ValueError("identity: the rebuilt control chain differs from runs/pubrefit/chain/ship beyond 1e-9 s.")


def dev_month(name: str, m: str) -> None:
    """tools/run_stack_weight.dev_month unchanged, on the rebuilt chain files of NAME."""
    import run_opdi_stack as O
    import run_stack_weight as KW
    O.SHIP = OUT / "chain" / name
    KW.OUT = OUT / "stack" / name
    KW.dev_month(m)


def chain_value(chain_dir: Path, P: pd.DataFrame, chain, keys: pd.DataFrame, t12, trig) -> tuple[np.ndarray, pd.DataFrame]:
    """tools/run_stack_weight.dev_score's path for one candidate (weight 0.65, ten seeds) on the chain files in chain_dir."""
    import run_ctx_ownpoints_chain as CH
    import run_stack_weight as KW
    dev = CH.dev_table_from(chain_dir / "ctl.parquet", chain_dir / "rows_all.parquet", chain_dir / "dmp.parquet")
    P = P.set_index(ID).reindex(dev[ID]).reset_index()
    if P["p2_0"].isna().any():
        raise ValueError("The stored predictions do not cover the development rows.")
    s1 = pd.read_parquet(chain_dir / "cand.parquet").set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    k = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(k["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    value, use, _ = KW.dev_value(dev, P, W, K)
    t = k[[ID, "month", "airport", "label", "cls", "day"]].copy()
    t["s2"] = s1 + (value - dev["new"].to_numpy(dtype=float))
    _, cand63 = chain.run(t, "s2")
    t["c63"] = cand63
    out = t[[ID, "month", "airport", "label", "cls", "day"]].copy()
    out["value"] = CH.own_motion_v2(t, "c63", t12, trig)
    out["opdi_used"] = use
    return out["value"].to_numpy(dtype=float), out


def member_guard(decl: dict) -> dict:
    """Clause 4: RMS of (lfpg_mix - own_all) 16-seed means on LFPG matched ranking rows over that on LFPG matched development rows."""
    import run_network_member as rnm
    sp = specs(decl)
    rms = {}
    for leaf in [*DEV_LEAVES, "ranking"]:
        ids = pd.read_parquet(ROOT / sp["own_all"]["tables"][leaf][0], columns=[ID])[ID].to_numpy()
        a, _ = rnm.member_values(sp["own_all"], leaf, ids)
        b, _ = rnm.member_values(sp["lfpg_mix"], leaf, ids)
        d = b - a
        rows = np.abs(d) > 0
        rms[leaf] = {"rows": int(rows.sum()), "sum_sq": float(np.sum(d[rows] ** 2)), "rms_s": float(np.sqrt(np.mean(d[rows] ** 2))) if rows.any() else None}
    dev_rows = sum(rms[m]["rows"] for m in DEV_LEAVES)
    dev_rms = float(np.sqrt(sum(rms[m]["sum_sq"] for m in DEV_LEAVES) / dev_rows))
    ratio = rms["ranking"]["rms_s"] / dev_rms
    return {"by_leaf": rms, "development_rms_s": dev_rms, "ranking_rms_s": rms["ranking"]["rms_s"], "ratio": ratio, "ok": bool(0.5 <= ratio <= 2.0)}


def score(decl: dict) -> None:
    import run_intruder_bound as R
    import run_own_motion_v2 as O2
    import run_stack_resid_v3 as V3
    dest = DOCS / "results_v1.json"
    if dest.exists():
        raise ValueError("Preserve the original result.")
    ctl_path, ctl_sha = decl["stage_2"]["control"]
    if sha256_file(ROOT / ctl_path) != ctl_sha:
        raise ValueError("The control differs from the declared file.")
    for f in SHIP_SHA:
        if not sha256_file(SHIP / f).startswith(SHIP_SHA[f]):
            raise ValueError(f"The ship file {f} changed.")
    ident = read_json(OUT / "chain/own_all_w100/build.json")
    if not ident.get("identity_equals_ship"):
        raise ValueError("Identity (a) has not passed.")
    ctl = pd.read_parquet(ROOT / ctl_path)
    keys, _ = O2.O.dev_table()
    chain = V3.Chain()
    _, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    months = ["2025-01", "2025-07", "2025-11", "2025-12"]
    pk = pd.concat([pd.read_parquet(ROOT / f"runs/stack_weight/dev/preds_{m}.parquet") for m in months], ignore_index=True)
    vi, ti = chain_value(SHIP, pk, chain, keys, t12, trig)
    c = ctl.set_index(ID)["value"].reindex(ti[ID]).to_numpy(dtype=float)
    imax = float(np.abs(vi - c).max())
    print("identity (b): w65_bag10 on the ship files vs the v69 development proxy max |diff|", imax, flush=True)
    res = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "official_scores_used": "none",
           "audit_used": False, "control": {"path": ctl_path, "sha256": ctl_sha}, "var_alloc_params": _VA.PARAMS,
           "identity": {"a_chain_files_equal_ship": True, "b_max_abs_vs_control": imax}, "code": {p: sha256_file(ROOT / p) for p in CODE}}
    if imax != 0.0:
        write_json(OUT / "dev/identity_failed.json", res)
        raise ValueError("identity (b) failed; nothing is read.")
    # amendment 1: the control rebuilt through this VM path (rebuilt chain files, stacking predictions refitted on them)
    pr = pd.concat([pd.read_parquet(OUT / f"stack/own_all_w100/dev/preds_{m}.parquet") for m in months], ignore_index=True)
    vr, tr = chain_value(OUT / "chain/own_all_w100", pr, chain, keys, t12, trig)
    if not tr[ID].equals(ti[ID]):
        raise ValueError("Row order differs between the rebuilt control and the control.")
    rmax = float(np.abs(vr - c).max())
    res["identity"]["c_rebuilt_control_max_abs_vs_control"] = rmax
    res["identity"]["c_rebuilt_control_rows_differing"] = int((np.abs(vr - c) > 0).sum())
    print("identity (c): the control rebuilt on this path vs the v69 development proxy max |diff|", rmax, flush=True)
    if rmax > 1e-6:
        write_json(OUT / "dev/identity_failed.json", res)
        raise ValueError("identity (c) failed; nothing is read.")
    c_declared = c
    c = vr  # governing comparison against the same-path control (amendment 1); the declared control is reported beside it
    cd = OUT / "chain/lfpg_mix_w100"
    pc = pd.concat([pd.read_parquet(OUT / f"stack/lfpg_mix_w100/dev/preds_{m}.parquet") for m in months], ignore_index=True)
    vc, tc = chain_value(cd, pc, chain, keys, t12, trig)
    if not tc[ID].equals(ti[ID]):
        raise ValueError("Row order differs between control and candidate.")
    proxy = OUT / "dev/lfpg_mix_w100_proxy.parquet"
    proxy.parent.mkdir(parents=True, exist_ok=True)
    tc.to_parquet(proxy, index=False)
    tt = ti[[ID, "month", "airport", "label", "day"]].assign(ctl=c, cand=vc)
    rng = np.random.default_rng(20261004)
    rule = read_json(ROOT / "configs/own_motion_v2.json")["rule"]
    out = {}
    for m in months:
        s = tt[tt["month"] == m]
        pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
        ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
        lf = s["airport"].eq("LFPG").to_numpy()
        out[m] = {"control_rmse": float(np.sqrt(np.mean((s["label"] - s["ctl"]) ** 2))), "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))),
                  "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum()),
                  "lfpg_rows_changed": int((np.abs(ch[lf]) > 1e-9).sum()), "lfpg_squared_change": float((ch[lf] ** 2).sum()),
                  "lfpg_change_pct": float(100 * (np.sqrt(np.mean((s["label"].to_numpy()[lf] - s["cand"].to_numpy()[lf]) ** 2))
                                                  / np.sqrt(np.mean((s["label"].to_numpy()[lf] - s["ctl"].to_numpy()[lf]) ** 2)) - 1))}
        print("lfpg_mix_w100", m, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in out[m].items()}, flush=True)
    eq = float(np.mean([out[m]["pct"] for m in rule["equal_footing"]]))
    checks = {"1_no_month_worse_than_0.005pct": all(out[m]["pct"] <= rule["max_month_worsening_pct"] for m in months),
              "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -rule["min_mean_improvement_pct"],
              "3_every_equal_interval_below_zero": all(out[m]["hi"] < 0 for m in rule["equal_footing"])}
    guard = member_guard(decl)
    checks["4_member_transfer_guard"] = guard["ok"]
    checks["accepted"] = bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"] and guard["ok"])
    vs_declared = {}
    for m in months:
        mm = tt["month"].eq(m).to_numpy()
        y = tt["label"].to_numpy()[mm]
        vs_declared[m] = float(100 * (np.sqrt(np.mean((y - vc[mm]) ** 2)) / np.sqrt(np.mean((y - c_declared[mm]) ** 2)) - 1))
    res["candidate_vs_declared_control_pct"] = vs_declared
    fx = ti["cls"].eq("LIRF_pop").to_numpy() | ti[ID].isin(GUARD).to_numpy()
    res.update({"months": out, "checks": checks, "transfer_guard_member": guard, "fixed_rows_moved": int((np.abs(vc - c)[fx] > 1e-9).sum()),
                "selected_candidate": "lfpg_mix_w100" if checks["accepted"] else None, "proxy": {"path": proxy.relative_to(ROOT).as_posix(), "sha256": sha256_file(proxy)},
                "identity_proxy_rows": int(len(ti)), "environment": environment(ROOT)})
    write_json(dest, res)
    write_json(OUT / "dev/selection.json", {"created_at_utc": utc_now(), "selected_candidate": res["selected_candidate"], "results": dest.relative_to(ROOT).as_posix(),
                                            "results_sha256": sha256_file(dest)})
    print(checks, {"guard_ratio": round(guard["ratio"], 3), "selected": res["selected_candidate"]}, flush=True)


# ----------------------------------------------------------------------------------------------------------------------------- ranking
def export_module(decl: dict, pinned: bool):
    import export_ctx_ownpoints as E
    patch_members(decl)
    if pinned:
        def pinned_run(args: list[str]) -> None:
            print("> (pinned)", " ".join(args), flush=True)
            subprocess.run([sys.executable, "-X", "utf8", "-B", "tools/lirf_own_pinned.py", *args], cwd=ROOT, check=True)
        E.run = pinned_run
    return E


def rank_steps(decl: dict, name: str, pinned: bool) -> None:
    """Steps 1-5 of v66 / v69's export chain (version 66 file names, as runs/ctx_ownpoints_weight/rank/dry_v66/steps)."""
    E = export_module(decl, pinned)
    tag = f"{'pinned' if pinned else 'unpinned'}_{name}"
    steps = OUT / "rank" / tag / "steps"
    if steps.exists():
        raise ValueError(f"{steps} exists; never overwrite.")
    cand = CANDS[name]
    spec = specs(decl)[cand["member"]]
    m = E.step_model_part(spec, float(cand["member_weight"]), steps / "model_part", 66)
    b, c, c_rows = E.steps_2_to_5(ROOT / m["file"], steps, 66)
    files = {"model_part/member_delta.parquet": steps / "model_part/member_delta.parquet", "rematch/base/vigorous-whistle_v55m_rematch.parquet": b,
             "Y/vigorous-whistle_v66.parquet": c, "Y/rows.parquet": c_rows, "model_part/vigorous-whistle_v66.parquet": steps / "model_part/vigorous-whistle_v66.parquet"}
    got = {k: sha256_file(p) for k, p in files.items()}
    rec = {"created_at_utc": utc_now(), "name": name, "pinned": pinned, "files": got}
    if name == "own_all_w100" and not pinned:
        # every exported step file byte for byte; Y/rows.parquet (float side table) to 1e-9 s (it was built on Windows, these on Linux)
        ref = {k: sha256_file(V66_STEPS / k) for k in files}
        exact = {k: got[k] == ref[k] for k in files}
        ra, rb = pd.read_parquet(files["Y/rows.parquet"]), pd.read_parquet(V66_STEPS / "Y/rows.parquet")
        same_frame = list(ra.columns) == list(rb.columns) and ra[ID].equals(rb[ID])
        rows_diff = {c: float(np.nanmax(np.abs(ra[c].to_numpy(dtype=float) - rb[c].to_numpy(dtype=float)), initial=0.0))
                     for c in ra.columns if c != ID and pd.api.types.is_numeric_dtype(ra[c])}
        nonnum_equal = all(ra[c].equals(rb[c]) for c in ra.columns if c != ID and not pd.api.types.is_numeric_dtype(ra[c]))
        nan_equal = all(np.array_equal(ra[c].isna().to_numpy(), rb[c].isna().to_numpy()) for c in ra.columns)
        rec["v66_steps"], rec["exact"], rec["y_rows_max_abs_diff"] = ref, exact, rows_diff
        rec["identity_equals_v66_steps"] = bool(all(v for k, v in exact.items() if k != "Y/rows.parquet") and same_frame and nonnum_equal and nan_equal
                                                and max(rows_diff.values(), default=0.0) <= 1e-9)
        print("identity: unpinned steps 1-5 with own_all equal v66's (v69's) shipped steps:", rec["identity_equals_v66_steps"], exact, rows_diff, flush=True)
    write_json(OUT / "rank" / tag / "steps.json", rec)
    if name == "own_all_w100" and not pinned and not rec["identity_equals_v66_steps"]:
        raise ValueError("identity: the ranking steps 1-5 differ from v66's shipped steps.")


def rank_identity_final() -> None:
    """The unpinned identity's remainder: fork K's stored ranking predictions (fitted on these very inputs) at w65_bag10 -> chain -> v69."""
    import run_own_motion_v2 as O2
    import run_stack_resid_v3 as V3
    import run_stack_weight as KW
    from prc2026_submission.contract import export_submission
    if not read_json(OUT / "rank/unpinned_own_all_w100/steps.json").get("identity_equals_v66_steps"):
        raise ValueError("Run --rank-steps own_all_w100 --unpinned first.")
    out = OUT / "rank/unpinned_own_all_w100/final"
    if out.exists():
        raise ValueError(f"{out} exists.")
    P = pd.read_parquet(ROOT / "runs/stack_weight/rank/preds.parquet")
    values, _, _ = KW.rank_value(P, W, K)
    st = out / "steps/stack/vigorous-whistle_v69.parquet"
    export_submission(pd.DataFrame({ID: P[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", st,
                      {"dry_run": True, "step": "identity: w65_bag10 stacking value", "official_scores_used": "none", "audit_used": False})
    v63like = V3.chain_from_stack(st, 69, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), 69, out / "steps/om2")
    final = out / "steps/om2/vigorous-whistle_v69.parquet"
    same = sha256_file(final) == V69[1]
    write_json(out / "identity.json", {"created_at_utc": utc_now(), "final_sha256": sha256_file(final), "reproduces_v69": same})
    print("identity: the ranking path reproduces v69 byte for byte:", same, flush=True)
    if not same:
        raise ValueError("identity: the ranking path does not reproduce v69.")


def rank_final(decl: dict, name: str) -> None:
    """Pinned NAME: the stacking layer (run_stack_weight.rank_inputs / rank_fit / rank_value unchanged) trained on the NAME chain's development
    tables (own_all_w100: the ship files) and applied to the pinned NAME steps; then chain_from_stack and own_motion_v2."""
    import export_ctx_ownpoints as E
    import run_opdi_stack_decl as D
    import run_own_motion_v2 as O2
    import run_stack_resid_v3 as V3
    import run_stack_weight as KW
    from prc2026_submission.contract import export_submission
    tag = f"pinned_{name}"
    base = OUT / "rank" / tag
    out = base / "final"
    if out.exists():
        raise ValueError(f"{out} exists.")
    import run_ownpoints_weight as W
    W.OUT = OUT  # run_stack_weight.rank_inputs sets E.CHAIN = W.OUT / "chain": the training tables are this round's chain files
    E.CHAIN = OUT / "chain"
    D.CANDIDATE_CHAIN = name
    D.SH = base / "steps"
    KW.OUT = base / "stackfit"
    KW.rank_fit()
    P = pd.read_parquet(base / "stackfit/rank/preds.parquet")
    values, use, _ = KW.rank_value(P, W, K)
    st = out / "steps/stack/vigorous-whistle_v70.parquet"
    export_submission(pd.DataFrame({ID: P[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", st,
                      {"dry_run": True, "step": f"pinned {name}: w65_bag10 stacking value", "official_scores_used": "none", "audit_used": False})
    v63like = V3.chain_from_stack(st, 70, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), 70, out / "steps/om2")
    final = out / "steps/om2/vigorous-whistle_v70.parquet"
    write_json(out / "final.json", {"created_at_utc": utc_now(), "name": name, "final_sha256": sha256_file(final), "opdi_rows_used": int(use.sum()),
                                    "stack_sha256": sha256_file(st)})
    print(tag, "final", sha256_file(final), flush=True)


def rank_dry(decl: dict) -> None:
    from prc2026_submission.contract import export_submission, validate_submission
    sel = read_json(OUT / "dev/selection.json")["selected_candidate"]
    if sel != "lfpg_mix_w100":
        raise ValueError("The declared rule must accept the candidate before a ranking dry run.")
    if not read_json(OUT / "rank/unpinned_own_all_w100/final/identity.json")["reproduces_v69"]:
        raise ValueError("The ranking identity has not passed.")
    ref = ROOT / V69[0]
    if sha256_file(ref) != V69[1]:
        raise ValueError("v69 changed.")
    out = OUT / "rank/dry_v70"
    if out.exists():
        raise ValueError(f"{out} exists.")
    pid = OUT / "rank/pinned_own_all_w100/final/steps/om2/vigorous-whistle_v70.parquet"
    pc = OUT / "rank/pinned_lfpg_mix_w100/final/steps/om2/vigorous-whistle_v70.parquet"
    v69, a, b = pd.read_parquet(ref), pd.read_parquet(pid), pd.read_parquet(pc)
    if not (v69[ID].equals(a[ID]) and v69[ID].equals(b[ID])):
        raise ValueError("Row order differs.")
    base = v69[TARGET].to_numpy(dtype=np.int64)
    delta = b[TARGET].to_numpy(dtype=np.int64) - a[TARGET].to_numpy(dtype=np.int64)
    values = np.maximum(base + delta, -12)
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
    dep = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(v69[ID])
    pop = (dep["ADEP_mvt"].eq("LIRF") & dep["AOBT_3_flt"].isna() & dep["LOBT_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    guard = v69[ID].isin(GUARD).to_numpy()
    changed = values != base
    if (changed & (pop | guard)).any():
        raise ValueError("A LIRF population row or a guard row changed.")
    out.mkdir(parents=True)
    path = out / "vigorous-whistle_v70.parquet"
    export_submission(pd.DataFrame({ID: v69[ID].to_numpy(), "prediction": values.astype(float)}), ROOT / "data/raw/submitting.parquet", path,
                      {"dry_run": True, "step": "v69 + (pinned lfpg_mix_w100 chain - pinned own_all_w100 chain)", "official_scores_used": "none", "audit_used": False})
    check = validate_submission(path, ROOT / "data/raw/submitting.parquet")
    d = pd.read_parquet(path)[TARGET].to_numpy(dtype=float) - base.astype(float)
    air = dep["ADEP_mvt"].to_numpy()
    month = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m").to_numpy()
    res = read_json(DOCS / "results_v1.json")
    dev_lfpg = np.sqrt(sum(res["months"][m]["lfpg_squared_change"] for m in DEV_LEAVES) / max(1, sum(res["months"][m]["lfpg_rows_changed"] for m in DEV_LEAVES)))
    lf = (air == "LFPG") & (d != 0)
    rep = {"created_at_utc": utc_now(), "file": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "validation": check,
           "reference": {"file": V69[0], "sha256": V69[1]},
           "pinned_identity": {"file": pid.relative_to(ROOT).as_posix(), "sha256": sha256_file(pid), "rows_differing_from_v69": int((a[TARGET].to_numpy() != base).sum())},
           "pinned_candidate": {"file": pc.relative_to(ROOT).as_posix(), "sha256": sha256_file(pc)},
           "rows_changed": int((d != 0).sum()), "squared_change": float((d ** 2).sum()), "rms_s": float(np.sqrt(np.mean(d ** 2))), "mean_s": float(d.mean()),
           "max_up_s": float(d.max()), "max_down_s": float(d.min()), "lirf_rows_changed": int(((d != 0) & (air == "LIRF")).sum()),
           "lirf_population_rows_changed": 0, "guard_rows_changed": 0,
           "lfpg_final_change_rms_ranking_s": float(np.sqrt(np.mean(d[lf] ** 2))) if lf.any() else None, "lfpg_final_change_rms_dev_s": float(dev_lfpg),
           "by_airport_month": {f"{k} {mm}": {"rows_changed": int(((d != 0) & (air == k) & (month == mm)).sum()),
                                              "squared_change": float((d[(air == k) & (month == mm)] ** 2).sum())}
                                for k in sorted(set(air)) for mm in sorted(set(month))}, "code": {p: sha256_file(ROOT / p) for p in CODE}}
    write_json(out / "vigorous-whistle_v70.report.json", rep)
    write_json(DOCS / "dry_run_v70.json", rep)
    print({k: v for k, v in rep.items() if k not in ("by_airport_month", "code", "validation")}, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-member", action="store_true")
    ap.add_argument("--build-mix", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--chain")
    ap.add_argument("--dev-month", nargs=2, metavar=("NAME", "MONTH"))
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--rank-steps")
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--rank-identity-final", action="store_true")
    ap.add_argument("--rank-final")
    ap.add_argument("--rank-dry", action="store_true")
    a = ap.parse_args()
    decl = declaration()
    if a.build_member:
        build_member(decl)
    if a.build_mix:
        build_mix(decl)
    if a.stage1:
        stage1(decl)
    if a.chain:
        build_chain(decl, a.chain)
    if a.dev_month:
        dev_month(*a.dev_month)
    if a.score:
        score(decl)
    if a.rank_steps:
        rank_steps(decl, a.rank_steps, a.pinned)
    if a.rank_identity_final:
        rank_identity_final()
    if a.rank_final:
        rank_final(decl, a.rank_final)
    if a.rank_dry:
        rank_dry(decl)


if __name__ == "__main__":
    main()
