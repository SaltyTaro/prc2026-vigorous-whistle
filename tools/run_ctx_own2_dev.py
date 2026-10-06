"""Development round of lever 5 (configs/ctx_own2_dev_v1.json): mix_own = 0.5 own_all + 0.5 ref_own replaces own_all at member weight 1.00.
Every arithmetic step is imported unchanged from the rounds that built v65 / v66 (tools/run_ownpoints_weight.py, run_ctx_ownpoints_chain.py,
run_network_member.py, export_ctx_ownpoints.py); only the members, the control and the output paths differ. The variance re-allocation's
LightGBM is pinned (deterministic, column-wise) in every chain run of this tool, control and candidate alike (docs/lirf_own/README.md).

  --build-mix                 runs/ctx_own2/member_mix/s<621+k>/<leaf>.parquet = 0.5 own_all s<601+k> + 0.5 ref_own s<621+k> (after
                              tools/build_combo_member.py --declaration configs/ctx_own2_dev_v1.json has written ref_own's tables)
  --stage1                    member level: own_all at 1.00 (must reproduce docs/ctx_ownpoints_weight/results_stage1.json) against mix_own at 1.00
  --stage2                    v66's chain, pinned: control own_all_w100 and candidate mix_own_w100; v57's clauses 1-3
  --export-identity --version N   the unpinned export chain with own_all at 1.00 must give v66 byte for byte
  --dry-run --version N       pinned identity and pinned candidate through the export chain; dry run = v66 + (candidate - identity), floor -12 s
No official score enters. Nothing is uploaded and nothing is written under runs/submissions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026_scout import var_alloc as _VA  # noqa: E402

_VA.PARAMS = {**_VA.PARAMS, "deterministic": True, "force_col_wise": True}  # declared pinning (stage_2.var_alloc_pinning)

from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_final import network_member, share_gate  # noqa: E402
import run_network_member as rnm  # noqa: E402
import run_ctx_ownpoints_dev as OD  # noqa: E402
import run_ctx_ownpoints_chain as CH  # noqa: E402
import run_ownpoints_weight as W  # noqa: E402

DECLARATION = "configs/ctx_own2_dev_v1.json"
OWN_DECL = "configs/ctx_ownpoints_dev_v1.json"
OUT = ROOT / "runs/ctx_own2"
DOCS = ROOT / "docs/ctx_own2"
WEIGHT_STAGE1 = "docs/ctx_ownpoints_weight/results_stage1.json"
V66_PROXY = ROOT / "runs/ctx_ownpoints_weight/chain/own_all_w100/dev_v66_proxy.parquet"
V66_FILE = ("runs/submissions/vigorous-whistle_v66.parquet", "b4ea727e1644b6add104f44832898bfecfd3d5607f7d6a9ce0471efe581322e4")
GUARD = [213142471, 213142469]
CODE = ["tools/run_ctx_own2_dev.py", "tools/run_ownpoints_weight.py", *W.CODE[1:], "src/prc2026_scout/var_alloc.py"]
CANDS = {"own_all_w100": {"member": "own_all", "member_weight": 1.0}, "mix_own_w100": {"member": "mix_own", "member_weight": 1.0}}


def declaration() -> dict:
    decl = read_json(ROOT / DECLARATION)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared or does not respect its own contract.")
    return decl


def specs(decl: dict) -> dict:
    mix = decl["members_derived"]["mix_own"]
    return {"own_all": read_json(ROOT / OWN_DECL)["member"], "ref_own": decl["member"],
            "mix_own": {"name": "mix_own", "column": "adsb_full", "tables": mix["tables"]}}


# ----------------------------------------------------------------------------------------------------------------------------- mix
def build_mix(decl: dict) -> None:
    sp = specs(decl)
    own, ref, mix = sp["own_all"]["tables"], sp["ref_own"]["tables"], sp["mix_own"]["tables"]
    built = {}
    for leaf in mix:
        if not (len(own[leaf]) == len(ref[leaf]) == len(mix[leaf]) == 16):
            raise ValueError(f"{leaf}: 16 seeds per member are declared.")
        for a_rel, b_rel, m_rel in zip(own[leaf], ref[leaf], mix[leaf]):
            a, b = pd.read_parquet(ROOT / a_rel), pd.read_parquet(ROOT / b_rel)
            if not np.array_equal(a[ID].to_numpy(), b[ID].to_numpy()):
                raise ValueError(f"{leaf}: {a_rel} and {b_rel} predict different rows.")
            va, vb = a["adsb_full"].to_numpy(dtype=float), b["adsb_full"].to_numpy(dtype=float)
            if not np.array_equal(np.isfinite(va), np.isfinite(vb)):
                raise ValueError(f"{leaf}: {a_rel} and {b_rel} are finite on different rows.")
            out = ROOT / m_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: a[ID].to_numpy(), "adsb_full": 0.5 * va + 0.5 * vb}).to_parquet(out, index=False, compression="zstd")
            built[m_rel] = {"own_all": [a_rel, sha256_file(ROOT / a_rel)], "ref_own": [b_rel, sha256_file(ROOT / b_rel)], "rows": int(len(a)),
                            "finite": int(np.isfinite(va).sum()), "sha256": sha256_file(out)}
        print(leaf, "mixed", len(mix[leaf]), flush=True)
    write_json(OUT / "member_mix_build.json", {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
                                               "tables": built, "tool_sha256": sha256_file(ROOT / "tools/run_ctx_own2_dev.py"), "audit_used": False})


# ----------------------------------------------------------------------------------------------------------------------------- stage 1
def stage1(decl: dict) -> None:
    out_path, doc_path = OUT / "dev/stage1.json", DOCS / "results_stage1.json"
    if out_path.exists() or doc_path.exists():
        raise ValueError("Preserve the original stage-1 result.")
    s1 = decl["stage_1"]
    equal, minimum = list(s1["equal_footing_months"]), float(s1["minimum_mean_improvement_fraction"])
    sp = specs(decl)
    rows = pd.read_parquet(ROOT / rnm.ROWS)
    cs = rnm.scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = rnm.offsets(), rnm.offsets_all()
    seg = rnm.segments(rnm.MONTHS)
    k = read_json(ROOT / rnm.STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    reread = read_json(ROOT / rnm.REREAD)["months"]
    combo_decl, combo_rec = read_json(ROOT / OD.COMBO), read_json(ROOT / OD.COMBO_RESULTS)
    wrec = read_json(ROOT / WEIGHT_STAGE1)["candidates"]["own_all_w100"]["months"]

    def stack(f: pd.DataFrame, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mix, fire = share_gate.mix(f["committee"].to_numpy(dtype=float), table, network_member.V46_SHARE, network_member.V46_GATE)
        return rnm.full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec), fire

    report = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
              "declaration_object_hash": object_hash(decl), "audit_used": False, "official_scores_used": "none", "rows_sha256": sha256_file(ROOT / rnm.ROWS),
              "equal_footing_months": equal, "minimum_mean_improvement_fraction": minimum, "control": "own_all at 1.00 (v66's member weight)",
              "reproductions": {}, "member_sources": {}, "months": {}, "candidates": {}}
    entries = {name: {**spec, "months": {}} for name, spec in decl["candidates"].items()}
    for month in rnm.MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y, ids, days = f["label"].to_numpy(dtype=float), f[ID].to_numpy(), f["day"].to_numpy()
        lirf, nets = f["airport"].to_numpy() == "LIRF", f["nets"].to_numpy(dtype=float)
        v46, _ = stack(f, nets)
        if abs(rnm.rmse(y, v46) - reread[month]["candidate_rmse"]) > 1e-9:
            raise ValueError(f"{month}: the v46 stack does not reproduce its re-read.")
        combo, _ = rnm.member_values(combo_decl["member"], month, ids)
        tcombo, _ = network_member.table_with_member(nets, np.where(lirf, np.nan, combo), OD.COMBO_W)
        c51, _ = stack(f, tcombo)
        if abs(rnm.rmse(y, c51) - combo_rec["candidates"]["w40"]["months"][month]["rmse"]) > 1e-9:
            raise ValueError(f"{month}: v51's model part does not reproduce the combination's w40.")
        own, found = rnm.member_values(sp["own_all"], month, ids)
        report["member_sources"].update(found)
        tinc, inside_inc = network_member.table_with_member(nets, np.where(lirf, np.nan, own), 1.0)
        control, _ = stack(f, tinc)
        if abs(rnm.rmse(y, control) - wrec[month]["rmse"]) > 1e-9:
            raise ValueError(f"{month}: own_all at 1.00 does not reproduce {WEIGHT_STAGE1}: {rnm.rmse(y, control)} against {wrec[month]['rmse']}.")
        report["reproductions"][month] = {"v46": rnm.rmse(y, v46), "v51_model_part": rnm.rmse(y, c51), "own_all_w100": rnm.rmse(y, control)}
        report["months"][month] = {"rows": int(len(f)), "control_rmse": rnm.rmse(y, control)}
        print(month, "controls reproduce", report["reproductions"][month], flush=True)
        for name, spec in decl["candidates"].items():
            member, found = rnm.member_values(sp[spec["member"]], month, ids)
            report["member_sources"].update(found)
            member = np.where(lirf, np.nan, member)
            table, inside = network_member.table_with_member(nets, member, float(spec["member_weight"]))
            values, fire = stack(f, table)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            if (changed & lirf).any():
                raise ValueError("A LIRF row changed.")
            if (changed & ~(inside | inside_inc)).any():
                raise ValueError("A row outside the member's support changed.")
            d = values - control
            entries[name]["months"][month] = {"rmse": rnm.rmse(y, values), "ratio_to_control": rnm.rmse(y, values) / rnm.rmse(y, control),
                                              "change_pct": 100 * (rnm.rmse(y, values) / rnm.rmse(y, control) - 1), "rows_changed": int(changed.sum()),
                                              "gate_rows": int((fire & ~lirf).sum()), "sum_squared_change": float(np.sum(d ** 2)),
                                              "alignment": float(-np.sum(d * (control - y)) / np.sum(d ** 2)) if changed.any() else None,
                                              "by_scope_change_pct": {s: 100 * (rnm.rmse(y[m], values[m]) / rnm.rmse(y[m], control[m]) - 1) for s, m in rnm.slices(f).items()},
                                              "paired_days_vs_control": rnm.paired_days(days, y, values, control, rnm.SEED)}
        print(month, "scored", flush=True)
    eligible, selected = rnm.select(entries, equal, minimum)
    report.update({"candidates": entries, "eligible": eligible, "stage_1_selected": selected,
                   "code": {p: sha256_file(ROOT / p) for p in CODE}, "environment": environment(ROOT)})
    for path in [out_path, doc_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    for name, e in entries.items():
        line = " ".join(f"{m}:{e['months'][m]['change_pct']:+.4f}% [{e['months'][m]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},"
                        f"{e['months'][m]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}] a={e['months'][m]['alignment']:.2f}" for m in rnm.MONTHS)
        print(f"{name:14s} mean {100 * (e['mean_ratio_equal_footing'] - 1):+.4f}% | {line} | eligible={e['eligible']}", flush=True)
    print({"eligible": eligible, "selected": selected})


# ----------------------------------------------------------------------------------------------------------------------------- stage 2
def patch_members(decl: dict) -> None:
    sp = specs(decl)
    CH.member_spec = lambda _decl, name: sp[name]
    CH.OUT = OUT / "chain"


def stage2(decl: dict) -> None:
    import run_intruder_bound as R
    dest = DOCS / "results_v1.json"
    if dest.exists():
        raise ValueError("Preserve the original stage-2 result.")
    s1 = read_json(OUT / "dev/stage1.json")
    if "mix_own_w100" not in s1["eligible"]:
        raise ValueError("Stage 1 did not find mix_own_w100 eligible; stage 2 is not run.")
    patch_members(decl)
    rule = read_json(ROOT / "configs/own_motion_v2.json")["rule"]
    cdecl = {"candidates": CANDS}
    vals = {}
    for name in ["own_all_w100", "mix_own_w100"]:
        if not (CH.OUT / name / "cand.parquet").exists():
            CH.step_model_part(cdecl, name, False)
            CH.step_chain(name)
        t, values, inputs, trees = W.score_chain(CH.OUT / name)
        vals[name] = (t, values, inputs, trees)
        print(name, "chain scored", flush=True)
    t = vals["own_all_w100"][0].copy()
    if not t[ID].equals(vals["mix_own_w100"][0][ID]):
        raise ValueError("Row order differs between control and candidate.")
    t["ctl"], t["cand"] = vals["own_all_w100"][1], vals["mix_own_w100"][1]
    stored = pd.read_parquet(V66_PROXY).set_index(ID)["v66"].reindex(t[ID]).to_numpy(dtype=float)
    unpinned = {m: {"stored_v66_rmse": float(np.sqrt(np.mean((t.loc[t["month"] == m, "label"].to_numpy() - stored[t["month"].to_numpy() == m]) ** 2))),
                    "pinned_control_rmse": float(np.sqrt(np.mean((t.loc[t["month"] == m, "label"].to_numpy() - t.loc[t["month"] == m, "ctl"].to_numpy()) ** 2)))}
                for m in R.DEV_MONTHS}
    for m in unpinned:
        unpinned[m]["pinned_vs_stored_pct"] = 100 * (unpinned[m]["pinned_control_rmse"] / unpinned[m]["stored_v66_rmse"] - 1)
    rng = np.random.default_rng(20261004)
    months = {}
    for m in R.DEV_MONTHS:
        s = t[t["month"] == m]
        pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
        yy = s["label"].to_numpy()
        months[m] = {"control_rmse": float(np.sqrt(np.mean((yy - s["ctl"].to_numpy()) ** 2))), "rmse": float(np.sqrt(np.mean((yy - s["cand"].to_numpy()) ** 2))),
                     "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(s["cand"].to_numpy() - s["ctl"].to_numpy()) > 1e-9).sum()),
                     "squared_change": float(np.sum((s["cand"].to_numpy() - s["ctl"].to_numpy()) ** 2))}
        print("mix_own_w100", m, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in months[m].items()}, flush=True)
    eq = float(np.mean([months[m]["pct"] for m in rule["equal_footing"]]))
    checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= rule["max_month_worsening_pct"] for m in R.DEV_MONTHS),
              "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -rule["min_mean_improvement_pct"],
              "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in rule["equal_footing"])}
    checks["accepted"] = bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"])
    proxy = CH.OUT / "mix_own_w100" / "dev_v67_proxy.parquet"
    t[[ID, "month", "airport", "label", "cls", "ctl", "cand"]].to_parquet(proxy, index=False)
    report = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "official_scores_used": "none",
              "stage_1": {"path": "docs/ctx_own2/results_stage1.json", "sha256": sha256_file(DOCS / "results_stage1.json")},
              "control": "own_all_w100 through the pinned chain", "var_alloc_params": _VA.PARAMS, "pinned_control_vs_stored_unpinned_v66_proxy": unpinned,
              "rule": decl["stage_2"]["rule"], "months": months, "checks": checks,
              "selected_candidate": "mix_own_w100" if checks["accepted"] else None,
              "inputs": {n: v[2] for n, v in vals.items()}, "stacker_trees": {n: v[3] for n, v in vals.items()},
              "proxy": {"path": str(proxy.relative_to(ROOT)), "sha256": sha256_file(proxy)}, "code": {p: sha256_file(ROOT / p) for p in CODE}}
    write_json(dest, report)
    write_json(OUT / "dev/selection.json", {"created_at_utc": utc_now(), "selected_candidate": report["selected_candidate"], "results": str(dest.relative_to(ROOT)),
                                            "results_sha256": sha256_file(dest)})
    print(checks, {"selected": report["selected_candidate"]})


# ----------------------------------------------------------------------------------------------------------------------------- export
def export_setup(decl: dict, pinned: bool, chain_dir: Path):
    import export_ctx_ownpoints as E
    patch_members(decl)
    path = OUT / "export_candidates.json"
    write_json(path, {"note": "runtime candidate list for tools/export_ctx_ownpoints.chain (lever 5)", "candidates": CANDS})
    E.DECLARATION, E.CHAIN = str(path.relative_to(ROOT)), chain_dir
    if pinned:
        def pinned_run(args: list[str]) -> None:
            print("> (pinned)", " ".join(args), flush=True)
            subprocess.run([sys.executable, "-X", "utf8", "-B", "tools/lirf_own_pinned.py", *args], cwd=ROOT, check=True)
        E.run = pinned_run
    return E


def export_identity(decl: dict, version: int) -> None:
    out = OUT / "rank" / f"identity_v{version}"
    if out.exists():
        raise ValueError(f"{out} exists.")
    E = export_setup(decl, False, ROOT / "runs/ctx_ownpoints_weight/chain")
    result = E.chain("own_all_w100", out, version, False)
    same = sha256_file(result) == V66_FILE[1] == sha256_file(ROOT / V66_FILE[0])
    print("export identity (unpinned, own_all at 1.00) reproduces v66:", same, sha256_file(result), flush=True)
    write_json(out / "identity.json", {"created_at_utc": utc_now(), "result": sha256_file(result), "v66": V66_FILE, "byte_identical": same})
    if not same:
        raise ValueError("The export chain does not reproduce v66.")


def dry_run(decl: dict, version: int) -> None:
    from prc2026_submission.contract import validate_submission
    import export_ctx_ownpoints as E0
    sel = read_json(OUT / "dev/selection.json")["selected_candidate"]
    if sel != "mix_own_w100":
        raise ValueError("Both declared stages must accept the candidate before a ranking dry run.")
    ident = OUT / "rank" / f"identity_v{version}" / "identity.json"
    if not ident.exists() or not read_json(ident)["byte_identical"]:
        raise ValueError("Run --export-identity first.")
    ref = ROOT / V66_FILE[0]
    if sha256_file(ref) != V66_FILE[1]:
        raise ValueError("v66 changed.")
    out = OUT / "rank" / f"dry_v{version}"
    if out.exists():
        raise ValueError(f"{out} exists.")
    E = export_setup(decl, True, OUT / "chain")
    pid = E.chain("own_all_w100", OUT / "rank" / f"pinned_identity_v{version}", version, False)
    pc = E.chain("mix_own_w100", OUT / "rank" / f"pinned_candidate_v{version}", version, False)
    v66, a, b = pd.read_parquet(ref), pd.read_parquet(pid), pd.read_parquet(pc)
    if not (v66[ID].equals(a[ID]) and v66[ID].equals(b[ID])):
        raise ValueError("Row order differs.")
    base = v66[TARGET].to_numpy(dtype=np.int64)
    delta = b[TARGET].to_numpy(dtype=np.int64) - a[TARGET].to_numpy(dtype=np.int64)
    values = np.maximum(base + delta, -12)
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
    dep = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(v66[ID])
    pop = (dep["ADEP_mvt"].eq("LIRF") & dep["AOBT_3_flt"].isna() & dep["LOBT_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    guard = v66[ID].isin(GUARD).to_numpy()
    changed = values != base
    if (changed & (pop | guard)).any():
        raise ValueError("A LIRF population row or a guard row changed.")
    out.mkdir(parents=True)
    path = out / f"vigorous-whistle_v{version}.parquet"
    E0.export_submission(pd.DataFrame({ID: v66[ID].to_numpy(), "prediction": values.astype(float)}), ROOT / "data/raw/submitting.parquet", path,
                         {"dry_run": True, "step": "v66 + (pinned mix_own_w100 chain - pinned own_all_w100 chain)", "official_scores_used": "none", "audit_used": False})
    check = validate_submission(path, ROOT / "data/raw/submitting.parquet")
    d = (pd.read_parquet(path)[TARGET].to_numpy(dtype=float) - base.astype(float))
    air = dep["ADEP_mvt"].to_numpy()
    rep = {"created_at_utc": utc_now(), "file": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "validation": check,
           "reference": {"file": V66_FILE[0], "sha256": V66_FILE[1]}, "pinned_identity": {"file": str(pid.relative_to(ROOT)), "sha256": sha256_file(pid),
           "rows_differing_from_v66": int((a[TARGET].to_numpy() != base).sum())}, "pinned_candidate": {"file": str(pc.relative_to(ROOT)), "sha256": sha256_file(pc)},
           "rows_changed": int((d != 0).sum()), "squared_change": float((d ** 2).sum()), "rms_s": float(np.sqrt(np.mean(d ** 2))), "mean_s": float(d.mean()),
           "max_up_s": float(d.max()), "max_down_s": float(d.min()), "lirf_rows_changed": int(((d != 0) & (air == "LIRF")).sum()),
           "lirf_population_rows_changed": 0, "guard_rows_changed": 0,
           "by_airport": {k: {"rows_changed": int(((d != 0) & (air == k)).sum()), "rms_s": float(np.sqrt(np.mean(d[air == k] ** 2))), "mean_s": float(d[air == k].mean())}
                          for k in sorted(set(air))}, "code": {p: sha256_file(ROOT / p) for p in CODE}}
    write_json(out / f"vigorous-whistle_v{version}.report.json", rep)
    write_json(DOCS / f"dry_run_v{version}.json", rep)
    print({k: v for k, v in rep.items() if k not in ("by_airport", "code", "validation")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-mix", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--export-identity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--version", type=int)
    a = ap.parse_args()
    decl = declaration()
    if a.build_mix:
        build_mix(decl)
    if a.stage1:
        stage1(decl)
    if a.stage2:
        stage2(decl)
    if a.export_identity:
        export_identity(decl, a.version)
    if a.dry_run:
        dry_run(decl, a.version)


if __name__ == "__main__":
    main()
