"""Final verification of the final-phase file before its upload (13 October 2026).

Checks the recommended final file (candidate 6) from the supplied files, without the code that built its last two steps:
  inputs     the organiser files and the EUROCONTROL workbook carry the hashes the chain recorded
  contract   670,790 rows in final_submitting order, the template's dtypes, unique IDs, every ID a departure of final_ranking
  lineage    every stage file named in docs/final_file/final_submission.md has its recorded hash; receipts link base -> out
  cap        the LIRF cap recomputed from the late-chain output (population from the raw columns) equals candidate 4 on every row
  pm         the published-means shift recomputed from candidate 4 (receipt shifts, declared row rule) equals candidate 6 on every row
  fixed      LIRF fill population, guard rows and v40's nine pin rows: values through the stages
  v76        January / July rows against the uploaded v76 (pins on, ranking.parquet times): size and location of the differences
  months     per airport-month level of the file against the same calendar months' 2025 labels (label-free on the 2026 side)
  code       git HEAD and a clean working tree for every tool and declaration of the chain
  rerun      (--rerun-dir) the cap and published-means tools re-run from the committed code reproduce candidates 4 and 6 byte for byte

  python tools/verify_final_file.py [--rerun-dir DIR] [--out docs/final_file/final_verification.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
RAW = ROOT / "data/raw/final_ranking.parquet"
TEMPLATE = ROOT / "data/raw/final_submitting.parquet"
WORKBOOK = ROOT / "data/external/eurocontrol_aiu/Taxi-Out_Additional_Time.xlsx"
V76 = ROOT / "runs/submissions/vigorous-whistle_v76.parquet"
RANKING = ROOT / "data/raw/ranking.parquet"
EXPECTED = {
    "final_ranking": "4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f",
    "final_submitting": "b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924",
    "workbook": "82d119ed3cfaf7a3c2f29b27ce942f0a36ea599b3af76e777e4eb354a7a4adb8",
}
STAGES = {  # docs/final_file/final_submission.md
    "model_part": ("runs/final_blind/run_final4m/final/vigorous-whistle_v9001.parquet", "cdb81e08b845a2436ffb3e6b3b40d4463164c5a234294c5282c71a5ab2ad969b"),
    "early_chain": ("runs/final_blind/early_9002/vigorous-whistle_v9002.parquet", "4c3937691bc51b17e5c25379c88f419f7057c94138725ac9aac8f5f5bcde9f79"),
    "early_c4": ("runs/final_blind/early_c4/vigorous-whistle_v9002.parquet", "d5834fcbbd9293c06a4e36ccefe8b8d35f5cde11056e07eebee3e2c9dce6b068"),
    "late_c4": ("runs/final_blind/late_c4/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet", "ab25a196e74a325618eda095918b653aadedef3479267a83e06e46b2c56ab849"),
    "candidate4": ("runs/final_blind/final_candidate4/chain_out_cap.parquet", "8fc8828d1f5338bda08972f16af61c0f5f288b5d50992c1750d4d768d89b0a42"),
    "candidate6": ("runs/final_blind/final_candidate6/chain_out_pm.parquet", "a7ac1f60c5ddd79ef6927887b50471b3160748c386ed8186d99095bee6b63a81"),
}
CAP_U, FLOOR, PM_CAP = 88392.0, -12.0, 7200.0
GUARD = [213142471, 213142469]
PINS = ["WMTT3508", "TAY9LA", "WMT402", "WMTMT4SG", "RYR4852", "TARAR753", "ITYTY674", "NOSOS431", "QFAA7006"]  # configs/fill_pin_v2.json
CHAIN_PATHS = [  # tools and declarations of the chain (docs/final_file/final_submission.md)
    "tools/final_predict.py", "src/prc2026_final", "tools/final_own_member.py", "tools/final_early_chain.py", "tools/final_early_resume6.py",
    "tools/final_late_chain.py", "tools/run_sixmonth.py", "tools/run_lirf_cap.py", "tools/run_published_means_v4.py",
    "tools/run_published_means_v3.py", "tools/prc_submit.py", "src/prc2026_submission", "src/prc2026_lirf_mixture",
    "configs/lirf_cap_v1.json", "configs/published_means_v4.json", "configs/published_means_v3.json", "configs/sixmonth_refit_v1.json",
    "configs/sixmonth_override_v1.json", "configs/sixmonth_devmonths_v1.json", "configs/fill_pin_v2.json",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(cond: bool, what: str, failures: list) -> bool:
    if not cond:
        failures.append(what)
    return bool(cond)


def values(path: str, ids: pd.Series) -> np.ndarray:
    t = pd.read_parquet(ROOT / path)
    if not t[ID].equals(ids):
        raise ValueError(f"{path}: ID order differs from the template")
    return t[TARGET].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun-dir", type=Path, help="re-run the cap and published-means tools into this empty directory")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/final_file/final_verification.json")
    args = ap.parse_args()
    fail: list[str] = []
    rep: dict = {"checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "file": STAGES["candidate6"][0]}

    # inputs
    got = {"final_ranking": sha(RAW), "final_submitting": sha(TEMPLATE), "workbook": sha(WORKBOOK)}
    rep["inputs"] = {k: {"sha256": v, "as_recorded": need(v == EXPECTED[k], f"input {k} hash", fail)} for k, v in got.items()}

    # contract
    tpl = pq.read_table(TEMPLATE)
    c6 = pq.read_table(ROOT / STAGES["candidate6"][0])
    ids = tpl.column(ID).to_pandas()
    raw = pd.read_parquet(RAW)
    dep = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID)
    v6 = c6.column(TARGET).to_pandas().to_numpy()
    rep["contract"] = {
        "rows": c6.num_rows, "template_rows": tpl.num_rows,
        "columns": c6.column_names, "schema_equals_template": need(c6.schema.remove_metadata().equals(tpl.schema.remove_metadata()), "schema", fail),
        "ids_equal_template_order": need(c6.column(ID).to_pandas().equals(ids), "ID order", fail),
        "ids_unique": need(not ids.duplicated().any(), "unique IDs", fail),
        "every_id_a_departure": need(bool(ids.isin(dep.index).all()), "IDs are departures", fail),
        "departures_without_prediction": int((~dep.index.isin(ids)).sum()),
        "nulls": int(c6.column(TARGET).null_count),
        "min": int(v6.min()), "max": int(v6.max()),
        "below_floor": int((v6 < FLOOR).sum()), "nonpositive": int((v6 <= 0).sum()),
        "above_7200": int((v6 > 7200).sum()), "above_20000": int((v6 > 20000).sum()),
    }
    need(c6.num_rows == tpl.num_rows == 670790, "row count", fail)
    need(rep["contract"]["nulls"] == 0, "no nulls", fail)
    need(rep["contract"]["below_floor"] == 0, "floor -12 s", fail)
    need(rep["contract"]["departures_without_prediction"] == 0, "every departure predicted", fail)

    # lineage
    rep["lineage"] = {}
    for k, (p, h) in STAGES.items():
        g = sha(ROOT / p)
        rep["lineage"][k] = {"path": p, "sha256": g, "as_recorded": need(g == h, f"stage {k} hash", fail)}
    r4 = json.loads((ROOT / STAGES["candidate4"][0]).with_suffix(".receipt.json").read_text(encoding="utf-8"))
    r6 = json.loads((ROOT / STAGES["candidate6"][0]).with_suffix(".receipt.json").read_text(encoding="utf-8"))
    rep["lineage"]["receipt_links"] = {
        "candidate4_base_is_late_c4": need(r4["base_sha256"] == STAGES["late_c4"][1] and r4["out_sha256"] == STAGES["candidate4"][1], "cap receipt link", fail),
        "candidate6_base_is_candidate4": need(r6["base_sha256"] == STAGES["candidate4"][1] and r6["out_sha256"] == STAGES["candidate6"][1], "pm receipt link", fail),
        "receipts_read_final_ranking": need(r4["movements_sha256"] == r6["movements_sha256"] == EXPECTED["final_ranking"], "receipt movements", fail),
        "pm_receipt_workbook": need(r6["workbook_sha256"] == EXPECTED["workbook"], "pm receipt workbook", fail),
    }
    stage = {k: values(p, ids) for k, (p, _) in STAGES.items()}

    # population and row attributes from the raw columns
    d = dep.reindex(ids)
    s = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()
    pop = (d["ADEP_mvt"].eq("LIRF") & d["AOBT_3_flt"].isna() & d["LOBT_flt"].isna() & d["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    air, mon = d["ADEP_mvt"].to_numpy(), d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()
    guard = ids.isin(GUARD).to_numpy()

    # cap: candidate 4 = min(late, max(s, U)) on population rows with a finite limit
    late = stage["late_c4"].astype(float)
    limit = np.maximum(s, CAP_U)
    capped = np.where(pop & np.isfinite(limit) & (late > limit), limit, late)
    c4 = stage["candidate4"]
    diff_cap = np.flatnonzero(np.rint(capped) != c4)
    moved_cap = np.flatnonzero(stage["late_c4"] != c4)
    rep["cap"] = {"population_rows": int(pop.sum()), "rows_moved": int(len(moved_cap)),
                  "moved": [{"MVT_ID": int(ids[i]), "flight": d["FLIGHT_mvt"].iloc[i], "s": float(s[i]), "late": int(stage["late_c4"][i]),
                             "candidate4": int(c4[i])} for i in moved_cap],
                  "recomputed_equals_candidate4": need(len(diff_cap) == 0, "cap recomputation", fail)}
    need(pop.sum() == 619, "619 population rows", fail)

    # published means: candidate 6 = round(max(c4 - shift, -12)) on non-fixed rows with c4 in (0, 7200] of an eligible cell
    shifts = {k: v["shift_s"] for k, v in r6["cells"].items() if v["eligible"]}
    fixed = pop | guard
    c4f = c4.astype(float)
    new = c4f.copy()
    for cell, sh in shifts.items():
        a, m = cell.split()
        sel = (air == a) & (mon == m) & (c4f > 0) & (c4f <= PM_CAP) & ~fixed
        new[sel] = np.maximum(c4f[sel] - sh, FLOOR)
    c6v = stage["candidate6"]
    diff_pm = np.flatnonzero(np.rint(new).astype(np.int64) != c6v)
    ch = c6v.astype(np.int64) - c4.astype(np.int64)
    by_cell = {}
    for cell in shifts:
        a, m = cell.split()
        k = (air == a) & (mon == m) & (ch != 0)
        by_cell[cell] = {"shift_s": shifts[cell], "rows_moved": int(k.sum()), "changes_s": sorted(set(ch[k].tolist()))}
    outside = (ch != 0) & ~np.isin(np.char.add(np.char.add(air.astype(str), " "), mon.astype(str)), list(shifts))
    rep["pm"] = {"eligible_cells": by_cell, "rows_changed": int((ch != 0).sum()), "squared_change": float((ch.astype(float) ** 2).sum()),
                 "changed_outside_cells": int(outside.sum()), "fixed_rows_changed": int(((ch != 0) & fixed).sum()),
                 "recomputed_equals_candidate6": need(len(diff_pm) == 0, "pm recomputation", fail)}
    need(rep["pm"]["changed_outside_cells"] == 0 and rep["pm"]["fixed_rows_changed"] == 0, "pm scope", fail)

    # fixed rows through the stages
    def trace(mask):
        return {k: v[mask].tolist() for k, v in stage.items()}
    pins = (pd.Series(pop) & d["FLIGHT_mvt"].isin(PINS).to_numpy()).to_numpy()
    v76 = pd.read_parquet(V76).set_index(ID)[TARGET]
    rep["fixed"] = {
        "population_changed_after_late_chain": int((stage["candidate6"][pop] != stage["late_c4"][pop]).sum()),
        "guard_rows": {"MVT_ID": ids[guard].astype(int).tolist(), **trace(guard), "v76": v76.reindex(ids[guard]).tolist()},
        "pin_rows": [{"flight": d["FLIGHT_mvt"].iloc[i], "MVT_ID": int(ids[i]), "month": mon[i], "s": float(s[i]),
                      "population": bool(pop[i]), "candidate6": int(c6v[i]),
                      "v76_pinned": None if pd.isna(v76.get(ids[i])) else int(v76.get(ids[i])),
                      "candidate6_equals_s": bool(abs(c6v[i] - s[i]) <= 1)} for i in np.flatnonzero(pins)],
    }
    need(rep["fixed"]["population_changed_after_late_chain"] == len(moved_cap), "population moved only by the cap", fail)
    need((stage["candidate6"][guard] == stage["late_c4"][guard]).all(), "guard rows keep the chain value", fail)

    # January / July against the uploaded v76
    shared = ids.isin(v76.index).to_numpy()
    dv = c6v[shared].astype(float) - v76.reindex(ids[shared]).to_numpy(dtype=float)
    absd = np.abs(dv)
    jj = pd.DataFrame({"cell": np.char.add(np.char.add(air[shared].astype(str), " "), mon[shared].astype(str)), "d": dv})
    big = np.flatnonzero(shared)[absd > 1000]
    rep["v76"] = {"shared_rows": int(shared.sum()), "v76_rows": int(len(v76)),
                  "abs_diff_percentiles_s": {str(q): float(np.percentile(absd, q)) for q in (50, 90, 99, 99.9)},
                  "rms_diff_s": float(np.sqrt(np.mean(dv ** 2))), "mean_diff_s": float(dv.mean()),
                  "by_cell": jj.groupby("cell")["d"].agg(["mean", lambda x: float(np.sqrt(np.mean(x ** 2)))]).rename(columns={"<lambda_0>": "rms"}).round(3).to_dict("index"),
                  "rows_over_1000_s": [{"MVT_ID": int(ids[i]), "airport": air[i], "flight": d["FLIGHT_mvt"].iloc[i], "population": bool(pop[i]),
                                        "pin_flight": bool(pins[i]), "candidate6": int(c6v[i]), "v76": int(v76[ids[i]])} for i in big]}

    # months: 2026 prediction level against the same calendar month's 2025 labels (ordinary rows, (0, 3600] s)
    tr = []
    for m in ("01", "02", "06", "07"):
        t = pd.read_parquet(next((ROOT / "data/raw").glob(f"training_2025-{m}-01_*.parquet")), columns=["PHASE_mvt", "ADEP_mvt", TARGET])
        t = t[t["PHASE_mvt"].eq("DEP") & (t[TARGET] > 0) & (t[TARGET] <= 3600)]
        tr.append(t.groupby("ADEP_mvt")[TARGET].mean().rename(m))
    lab = pd.concat(tr, axis=1)
    ok = (c6v > 0) & (c6v <= 3600) & ~pop
    pr = pd.DataFrame({"a": air[ok], "m": [x[-2:] for x in mon[ok]], "v": c6v[ok]}).groupby(["a", "m"])["v"].mean().unstack()
    gap = (pr - lab).round(1)
    rep["months"] = {"pred_2026_mean_s": pr.round(1).to_dict("index"), "label_2025_mean_s": lab.round(1).to_dict("index"),
                     "gap_s": gap.to_dict("index"),
                     "feb_jun_minus_jan_jul_gap_s": ((gap["02"] + gap["06"]) / 2 - (gap["01"] + gap["07"]) / 2).round(1).to_dict()}

    # code state
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CHAIN_PATHS], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    untracked = [p for p in CHAIN_PATHS if subprocess.run(["git", "ls-files", "--error-unmatch", p], cwd=ROOT, capture_output=True).returncode]
    rep["code"] = {"head": head, "chain_paths": CHAIN_PATHS, "uncommitted": dirty, "not_tracked": untracked}
    need(not dirty and not untracked, "chain code committed and clean", fail)

    # re-run the last two tools from the committed code
    if args.rerun_dir:
        out = args.rerun_dir.resolve()
        out.mkdir(parents=True, exist_ok=False)
        py = [sys.executable, "-X", "utf8", "-B"]
        o4, o6 = out / "chain_out_cap.parquet", out / "chain_out_pm.parquet"
        subprocess.run(py + ["tools/run_lirf_cap.py", "apply", "--base", STAGES["late_c4"][0], "--movements", "data/raw/final_ranking.parquet",
                             "--out", str(o4)], cwd=ROOT, check=True)
        subprocess.run(py + ["tools/run_published_means_v4.py", "apply", "--base", str(o4), "--out", str(o6)], cwd=ROOT, check=True)
        rep["rerun"] = {"candidate4": {"sha256": sha(o4), "equal": need(sha(o4) == STAGES["candidate4"][1], "cap tool rerun", fail)},
                        "candidate6": {"sha256": sha(o6), "equal": need(sha(o6) == STAGES["candidate6"][1], "pm tool rerun", fail)}}

    rep["failures"] = fail
    rep["verdict"] = "PASS" if not fail else "FAIL"
    args.out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"verdict": rep["verdict"], "failures": fail, "contract": rep["contract"], "cap": rep["cap"]["rows_moved"],
                      "pm_rows": rep["pm"]["rows_changed"], "v76": rep["v76"]["abs_diff_percentiles_s"],
                      "feb_jun_minus_jan_jul_gap_s": rep["months"]["feb_jun_minus_jan_jul_gap_s"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
