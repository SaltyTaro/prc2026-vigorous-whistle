"""Weight round for v65's own-points context-network member (configs/ctx_ownpoints_weight_v1.json). Every arithmetic step is imported
unchanged from the round that built v65 (tools/run_ctx_ownpoints_dev.py, tools/run_ctx_ownpoints_chain.py, tools/export_ctx_ownpoints.py);
only the control (own_all_w40 / the v65 development proxy), the output paths and the candidate weights differ.

  --stage1                     member level: the four candidates against own_all_w40 (control reproductions first)
  --stage2-identity            the model-part step at w 0.40 reproduces runs/ctx_ownpoints_dev/chain/own_all_w40/dmp.parquet, and the stage-2
                               scorer on the own_all_w40 chain files reproduces the v65 development proxy exactly
  --stage2                     the declared stage-2 order: eligible stage-1 candidates by mean ratio, the first that passes is selected
  --export-identity            the export chain with own_all_w40 and the original tables reproduces v65 byte for byte
  --dry-run / --build --version N [--user-decision TEXT]
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
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_final import network_member, share_gate  # noqa: E402
import run_network_member as rnm  # noqa: E402
import run_ctx_ownpoints_dev as OD  # noqa: E402
import run_ctx_ownpoints_chain as CH  # noqa: E402

DECLARATION = "configs/ctx_ownpoints_weight_v1.json"
OUT = ROOT / "runs/ctx_ownpoints_weight"
DOCS = ROOT / "docs/ctx_ownpoints_weight"
ORIG_CHAIN = ROOT / "runs/ctx_ownpoints_dev/chain"
V65_PROXY = ORIG_CHAIN / "own_all_w40/dev_v65_proxy.parquet"
V65_FILE = ("runs/submissions/vigorous-whistle_v65.parquet", "45160819c67ed4827280d6968ca26ec3f09d915d31e83cee62090b7217bf3cc3")
ORIG_RESULTS = "docs/ctx_ownpoints_dev/results_v1.json"
CODE = ["tools/run_ownpoints_weight.py", "tools/run_ctx_ownpoints_dev.py", "tools/run_ctx_ownpoints_chain.py", "tools/export_ctx_ownpoints.py",
        "tools/run_network_member.py", "src/prc2026_final/network_member.py", "src/prc2026_final/share_gate.py", "tools/run_stack_resid_v3.py",
        "tools/run_own_motion_v2.py"]


def declaration() -> dict:
    decl = read_json(ROOT / DECLARATION)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared or does not respect its own contract.")
    return decl


def with_incumbent(decl: dict) -> dict:
    inc = decl["incumbent"]
    return {**decl, "candidates": {**decl["candidates"], inc["name"]: {"member": inc["member"], "member_weight": inc["member_weight"]}}}


# ----------------------------------------------------------------------------------------------------------------------------- stage 1
def stage1(decl: dict) -> None:
    out_path, doc_path = OUT / "stage1.json", DOCS / "results_stage1.json"
    if out_path.exists() or doc_path.exists():
        raise ValueError("Preserve the original stage-1 result.")
    s1 = decl["stage_1"]
    equal, minimum = list(s1["equal_footing_months"]), float(s1["minimum_mean_improvement_fraction"])
    rows = pd.read_parquet(ROOT / rnm.ROWS)
    cs = rnm.scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = rnm.offsets(), rnm.offsets_all()
    seg = rnm.segments(rnm.MONTHS)
    k = read_json(ROOT / rnm.STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    reread = read_json(ROOT / rnm.REREAD)["months"]
    combo_decl, combo_rec = read_json(ROOT / OD.COMBO), read_json(ROOT / OD.COMBO_RESULTS)
    orig = read_json(ROOT / ORIG_RESULTS)
    inc = decl["incumbent"]

    def stack(f: pd.DataFrame, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mix, fire = share_gate.mix(f["committee"].to_numpy(dtype=float), table, network_member.V46_SHARE, network_member.V46_GATE)
        return rnm.full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec), fire

    report = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
              "declaration_object_hash": object_hash(decl), "audit_used": False, "official_scores_used": "none", "rows_sha256": sha256_file(ROOT / rnm.ROWS),
              "equal_footing_months": equal, "minimum_mean_improvement_fraction": minimum, "control": f"{inc['name']} (v65's member weight)",
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
        member, found = rnm.member_values(decl["member"], month, ids)
        report["member_sources"].update(found)
        member = np.where(lirf, np.nan, member)
        tinc, inside_inc = network_member.table_with_member(nets, member, float(inc["member_weight"]))
        control, _ = stack(f, tinc)
        want = orig["candidates"][inc["name"]]["months"][month]["rmse"]
        if abs(rnm.rmse(y, control) - want) > 1e-9:
            raise ValueError(f"{month}: {inc['name']} does not reproduce {ORIG_RESULTS}: {rnm.rmse(y, control)} against {want}.")
        report["reproductions"][month] = {"v46": rnm.rmse(y, v46), "v51_model_part": rnm.rmse(y, c51), inc["name"]: rnm.rmse(y, control)}
        report["months"][month] = {"rows": int(len(f)), "control_rmse": rnm.rmse(y, control)}
        print(month, "controls reproduce", report["reproductions"][month], flush=True)
        for name, spec in decl["candidates"].items():
            w = float(spec["member_weight"])
            table, inside = network_member.table_with_member(nets, member, w)
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
    order = sorted(eligible, key=lambda n: (entries[n]["mean_ratio_equal_footing"], n))
    report.update({"candidates": entries, "eligible": eligible, "stage_2_order": order, "stage_1_best": selected,
                   "code": {p: sha256_file(ROOT / p) for p in CODE}, "environment": environment(ROOT)})
    for path in [out_path, doc_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    for name, e in entries.items():
        line = " ".join(f"{m}:{e['months'][m]['change_pct']:+.4f}% [{e['months'][m]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},"
                        f"{e['months'][m]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}] a={e['months'][m]['alignment']:.2f}" for m in rnm.MONTHS)
        print(f"{name:14s} mean {100 * (e['mean_ratio_equal_footing'] - 1):+.4f}% | {line} | eligible={e['eligible']}", flush=True)
    print({"eligible": eligible, "stage_2_order": order})


# ----------------------------------------------------------------------------------------------------------------------------- stage 2
def score_chain(d: Path) -> tuple[pd.DataFrame, np.ndarray, dict, list]:
    """tools/run_ctx_ownpoints_chain.step_score's arithmetic on the chain files in d; returns the table, the candidate's v65-like column,
    the input hashes and the stacker trees."""
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    if sha256_file(ROOT / CH.V63[0]) != CH.V63[1]:
        raise ValueError("The v63 development proxy differs from the declared file.")
    files = (d / "ctl.parquet", d / "rows_all.parquet", d / "dmp.parquet", d / "cand.parquet")
    dev = CH.dev_table_from(*files[:3])
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    core = V3.own_points("dev", dev[ID].to_numpy())
    Pn = V3.new_columns(core, dev["new"].to_numpy(dtype=float))
    X65 = pd.concat([X46, Pn], axis=1)
    pr = V3.dev_predictions(dev, X46, X65, Pn, res)
    new, air = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy()
    s1 = pd.read_parquet(files[3]).set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    keys, _ = O2.O.dev_table()
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
    t["s2"] = s1 + (np.where(air == "LIRF", pr["base2"], pr["base3"]) - new)
    _, cand63 = V3.Chain().run(t, "s2")
    dev12, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    t["c63"] = cand63
    c65 = CH.own_motion_v2(t, "c63", t12, trig)
    return t, c65, {p.name: sha256_file(p) for p in files}, pr["trees"]


def v65_proxy(t: pd.DataFrame) -> np.ndarray:
    p = pd.read_parquet(V65_PROXY).set_index(ID).reindex(t[ID])
    if p["v65"].isna().any():
        raise ValueError("The v65 proxy misses rows.")
    return p["v65"].to_numpy(dtype=float)


def stage2_identity(decl: dict) -> None:
    dest = OUT / "stage2_identity.json"
    if dest.exists():
        raise ValueError("Preserve the original identity record.")
    CH.OUT = OUT / "chain_identity"
    CH.step_model_part(with_incumbent(decl), decl["incumbent"]["name"], False)
    mine = pd.read_parquet(CH.OUT / decl["incumbent"]["name"] / "dmp.parquet")
    orig = pd.read_parquet(ORIG_CHAIN / "own_all_w40/dmp.parquet")
    if list(mine.columns) != list(orig.columns) or not mine[ID].equals(orig[ID]):
        raise ValueError("identity: the model-part tables differ in layout.")
    dmp_max = max(float(np.abs(mine[c].to_numpy(dtype=float) - orig[c].to_numpy(dtype=float)).max()) for c in mine.columns if c not in (ID, "month"))
    print("identity: model part at w 0.40 vs the v65 round's dmp.parquet, max |diff| =", dmp_max, flush=True)
    if dmp_max != 0.0:
        raise ValueError("identity: the model-part step does not reproduce the v65 round.")
    t, c65, inputs, trees = score_chain(ORIG_CHAIN / "own_all_w40")
    proxy_max = float(np.abs(c65 - v65_proxy(t)).max())
    print("identity: stage-2 scorer on the own_all_w40 chain files vs the v65 proxy, max |diff| =", proxy_max, flush=True)
    if proxy_max != 0.0:
        raise ValueError("identity: the stage-2 scorer does not reproduce the v65 development proxy.")
    write_json(dest, {"created_at_utc": utc_now(), "model_part_max_abs_diff": dmp_max, "v65_proxy_max_abs_diff": proxy_max, "inputs": inputs,
                      "v65_proxy": {"path": str(V65_PROXY.relative_to(ROOT)), "sha256": sha256_file(V65_PROXY)}, "stacker_trees": trees,
                      "code": {p: sha256_file(ROOT / p) for p in CODE}})
    shutil.rmtree(CH.OUT / decl["incumbent"]["name"])


def stage2(decl: dict) -> None:
    import run_intruder_bound as R
    dest = DOCS / "results_stage2.json"
    if dest.exists():
        raise ValueError("Preserve the original stage-2 result.")
    if not (OUT / "stage2_identity.json").exists():
        raise ValueError("Run --stage2-identity first.")
    s1 = read_json(OUT / "stage1.json")
    order = s1["stage_2_order"]
    rule = read_json(ROOT / "configs/own_motion_v2.json")["rule"]
    report = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "official_scores_used": "none",
              "control": "v65 development proxy (runs/ctx_ownpoints_dev/chain/own_all_w40/dev_v65_proxy.parquet, column v65)",
              "v65_proxy_sha256": sha256_file(V65_PROXY), "stage_1_order": order, "rule": decl["stage_2"]["rule"], "tried": {}, "selected_candidate": None}
    CH.OUT = OUT / "chain"
    for name in order:
        CH.step_model_part(with_incumbent(decl), name, False)
        CH.step_chain(name)
        t, c65, inputs, trees = score_chain(CH.OUT / name)
        t["v65"], t["cand"] = v65_proxy(t), c65
        rng = np.random.default_rng(20261004)
        months = {}
        for m in R.DEV_MONTHS:
            s = t[t["month"] == m]
            pct, lo, hi = R.interval(s, "v65", s["cand"].to_numpy(), rng)
            yy = s["label"].to_numpy()
            months[m] = {"control_rmse": float(np.sqrt(np.mean((yy - s["v65"].to_numpy()) ** 2))), "rmse": float(np.sqrt(np.mean((yy - s["cand"].to_numpy()) ** 2))),
                         "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(s["cand"].to_numpy() - s["v65"].to_numpy()) > 1e-9).sum()),
                         "squared_change": float(np.sum((s["cand"].to_numpy() - s["v65"].to_numpy()) ** 2))}
            print(name, m, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in months[m].items()}, flush=True)
        eq = float(np.mean([months[m]["pct"] for m in rule["equal_footing"]]))
        checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= rule["max_month_worsening_pct"] for m in R.DEV_MONTHS),
                  "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -rule["min_mean_improvement_pct"],
                  "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in rule["equal_footing"])}
        checks["accepted"] = bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"])
        proxy = CH.OUT / name / "dev_v66_proxy.parquet"
        t[[ID, "month", "airport", "label", "cls", "v65", "cand"]].rename(columns={"cand": "v66"}).to_parquet(proxy, index=False)
        report["tried"][name] = {"months": months, "checks": checks, "inputs": inputs, "stacker_trees": trees,
                                 "proxy": {"path": str(proxy.relative_to(ROOT)), "sha256": sha256_file(proxy)}}
        print(name, checks, flush=True)
        if checks["accepted"]:
            report["selected_candidate"] = name
            break
    report["code"] = {p: sha256_file(ROOT / p) for p in CODE}
    DOCS.mkdir(parents=True, exist_ok=True)
    write_json(dest, report)
    write_json(OUT / "selection.json", {"created_at_utc": utc_now(), "selected_candidate": report["selected_candidate"], "results": str(dest.relative_to(ROOT)),
                                        "results_sha256": sha256_file(dest)})
    print({"selected": report["selected_candidate"]})


# ----------------------------------------------------------------------------------------------------------------------------- export
def export_identity(version: int) -> None:
    import export_ctx_ownpoints as E
    out = OUT / "rank" / f"identity_v{version}"
    if out.exists():
        raise ValueError(f"{out} exists.")
    E.DECLARATION, E.CHAIN = "configs/ctx_ownpoints_dev_v1.json", ORIG_CHAIN
    result = E.chain("own_all_w40", out, version, False)
    same = sha256_file(result) == V65_FILE[1] == sha256_file(ROOT / V65_FILE[0])
    print("export identity: own_all_w40 through the export chain reproduces v65:", same, sha256_file(result), flush=True)
    write_json(out / "identity.json", {"created_at_utc": utc_now(), "result": sha256_file(result), "v65": V65_FILE, "byte_identical": same})
    if not same:
        raise ValueError("The export chain does not reproduce v65.")


def export(version: int, build: bool, decision: str | None) -> None:
    import export_ctx_ownpoints as E
    from prc2026_submission.contract import validate_submission
    sel = read_json(OUT / "selection.json")["selected_candidate"]
    stage2_rec = read_json(DOCS / "results_stage2.json")
    if not sel or stage2_rec["selected_candidate"] != sel or not stage2_rec["tried"][sel]["checks"]["accepted"]:
        raise ValueError("Both declared stages must accept the candidate before a ranking file is written.")
    if not (OUT / "rank" / f"identity_v{version}" / "identity.json").exists() or not read_json(OUT / "rank" / f"identity_v{version}" / "identity.json")["byte_identical"]:
        raise ValueError("Run --export-identity first.")
    E.CHAIN = OUT / "chain"
    E.DECLARATION = DECLARATION  # its candidates name the member ('own_all') and the weight; E.chain reads both through run_ctx_ownpoints_chain.member_spec
    ref = ROOT / V65_FILE[0]
    if sha256_file(ref) != V65_FILE[1]:
        raise ValueError("v65 changed.")
    if not build:
        out = OUT / "rank" / f"dry_v{version}"
        if out.exists():
            raise ValueError(f"{out} exists.")
        result = E.chain(sel, out, version, False)
        rep = E.compare(result, ref, out)
        rep.update({"file": E.rel(result), "sha256": sha256_file(result), "candidate": sel})
        write_json(out / f"vigorous-whistle_v{version}.report.json", rep)
        print({k: v for k, v in rep.items() if k != "by_airport"})
        return
    target = ROOT / f"runs/submissions/vigorous-whistle_v{version}.parquet"
    dry = OUT / "rank" / f"dry_v{version}" / f"vigorous-whistle_v{version}.parquet"
    if target.exists() or not dry.exists() or not decision:
        raise ValueError("Never overwrite; run the dry run first; record the user's decision.")
    out = OUT / "rank" / f"build_v{version}"
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite a build directory.")
    built = E.chain(sel, out, version, False)
    if sha256_file(built) != sha256_file(dry):
        raise ValueError("The rebuilt file differs from the dry run.")
    target.write_bytes(built.read_bytes())
    sub = validate_submission(target, ROOT / "data/raw/submitting.parquet")
    decl = declaration()
    sub.update({"version": version, "provenance": {"experiment_id": decl["experiment_id"], "selected_candidate": sel, "selected_by": "both declared stages",
                                                   "user_decision": decision, "official_scores_used": "none", "audit_used": False,
                                                   "declaration": {DECLARATION: sha256_file(ROOT / DECLARATION)},
                                                   "code": {p: sha256_file(ROOT / p) for p in CODE}}})
    write_json(target.with_suffix(".receipt.json"), sub)
    rep = E.compare(target, ref, out)
    rep.update({"kind": "submission build at the user's word", "file": E.rel(target), "sha256": sha256_file(target), "dry_run_sha256": sha256_file(dry), "candidate": sel})
    write_json(DOCS / f"export_v{version}.json", rep)
    print("built", target.name, rep["sha256"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2-identity", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--export-identity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--version", type=int)
    ap.add_argument("--user-decision", default=None)
    a = ap.parse_args()
    decl = declaration()
    if a.stage1:
        stage1(decl)
    if a.stage2_identity:
        stage2_identity(decl)
    if a.stage2:
        stage2(decl)
    if a.export_identity:
        export_identity(a.version)
    if a.dry_run:
        export(a.version, False, None)
    if a.build:
        export(a.version, True, a.user_decision)


if __name__ == "__main__":
    main()
