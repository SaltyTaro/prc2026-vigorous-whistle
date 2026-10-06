"""LSZH-only variant of the OPDI stacker layer (configs/opdi_stack_lszh_v1.json): development reading by row splice."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
import run_intruder_bound as R  # noqa: E402

ID = "MVT_ID_mvt"
DECL = "configs/opdi_stack_lszh_v1.json"
CTL = ROOT / "runs/pubrefit/chain/ship/proxy.parquet"
CAND = ROOT / "runs/opdi_stack/govern/chain/opdi/proxy.parquet"
OUT = ROOT / "docs/opdi_stack_lszh/results_v1.json"
MONTHS, EQ = ["2025-01", "2025-07", "2025-11", "2025-12"], ["2025-01", "2025-07", "2025-12"]


def main() -> None:
    if OUT.exists():
        raise ValueError("Preserve the reading.")
    if sha256_file(CTL) != "cf5a4e8e1cc6733e7e8a95fd6dcebb21764fc0b6f1e573a6ad1c38a1cc7dcea3":
        raise ValueError("The pinned control differs from the declared file.")
    ctl = pd.read_parquet(CTL)
    cand = pd.read_parquet(CAND).set_index(ID).reindex(ctl[ID])
    lszh = ctl["airport"].eq("LSZH").to_numpy() & cand["opdi_used"].to_numpy(dtype=bool)
    value = np.where(lszh, cand["value"].to_numpy(dtype=float), ctl["value"].to_numpy(dtype=float))
    t = ctl.rename(columns={"value": "ctl"}).assign(cand=value)
    rng = np.random.default_rng(20261004)
    months = {}
    for m in MONTHS:
        s = t[t["month"] == m]
        pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
        ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
        months[m] = {"control_rmse": float(np.sqrt(np.mean((s["label"] - s["ctl"]) ** 2))),
                     "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))), "pct": pct, "lo": lo, "hi": hi,
                     "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum())}
    eq = float(np.mean([months[m]["pct"] for m in EQ]))
    checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= 0.005 for m in MONTHS), "2_equal_footing_mean_pct": eq,
              "2_mean_ok": eq <= -0.02, "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in EQ),
              "4_transfer_guard_LSZH": "1.0524 (docs/opdi_stack/results_v1.json clause_4_transfer_guard, same rows and change)"}
    res = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "control_sha256": sha256_file(CTL),
           "candidate_source_sha256": sha256_file(CAND), "lszh_rows_changed_dev": int(lszh.sum()), "months": months, "checks": checks,
           "accepted": bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"]),
           "official_scores_used": "none", "code_sha256": sha256_file(Path(__file__))}
    write_json(OUT, res)
    print({m: (round(v["pct"], 4), round(v["lo"], 3), round(v["hi"], 3), v["rows_changed"]) for m, v in months.items()}, checks, "ACCEPTED", res["accepted"])


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def rank() -> None:
    """Ranking dry run on v66: lever 7's stack step restricted to LSZH, then v66's chain; plus an independent row splice check."""
    import shutil
    import export_ctx_ownpoints as E
    import run_ownpoints_weight as W
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    import run_opdi_stack_decl as D
    from prc2026.artifacts import read_json
    res = read_json(OUT)
    if not res["accepted"]:
        raise ValueError("Clauses 1-3 do not hold; no ranking file.")
    if sha256_file(ROOT / D.V66[0]) != D.V66[1]:
        raise ValueError("The uploaded v66 file changed.")
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    b, c, c_rows = D.SH / "rematch/base/vigorous-whistle_v55m_rematch.parquet", D.SH / "Y/vigorous-whistle_v66.parquet", D.SH / "Y/rows.parquet"
    delta = D.SH / "model_part/member_delta.parquet"
    out = ROOT / "runs/opdi_stack_lszh/rank/dry_v68"
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite a dry run.")
    st, rk, v3vals, values, use = D.step_stack_opdi(b, c, c_rows, delta, out / "steps/stack", 68, ["LSZH"])
    shipped = pd.read_parquet(D.SH / "stack/vigorous-whistle_v66.parquet").set_index(ID)["TAXITIME_SEC_mvt"].reindex(rk[ID]).to_numpy(dtype=float)
    v3_check = float(np.abs(np.rint(v3vals) - shipped).max())
    if v3_check != 0.0:
        raise ValueError("The v3 part does not reproduce the shipped stack step.")
    v63like = V3.chain_from_stack(st, 68, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), 68, out / "steps/om2")
    final = out / "vigorous-whistle_v68.parquet"
    shutil.copyfile(out / "steps/om2/vigorous-whistle_v68.parquet", final)
    v = pd.read_parquet(final)
    v66 = pd.read_parquet(ROOT / D.V66[0])
    full = pd.read_parquet(ROOT / "runs/opdi_stack/rank/dry_v67/NOT_ACCEPTED_vigorous-whistle_v67.parquet")
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v66[ID])
    is_lszh = raw["ADEP_mvt"].eq("LSZH").to_numpy()
    splice = np.where(is_lszh, full["TAXITIME_SEC_mvt"].to_numpy(), v66["TAXITIME_SEC_mvt"].to_numpy())
    splice_equal = bool(np.array_equal(splice, v["TAXITIME_SEC_mvt"].to_numpy()))
    diff = v["TAXITIME_SEC_mvt"].to_numpy(dtype=float) - v66["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    ch = diff != 0
    rep = {"created_at_utc": utc_now(), "file": final.relative_to(ROOT).as_posix(), "sha256": sha256_file(final), "v3_part_max_abs": v3_check,
           "order_equals_v66": bool(v[ID].equals(v66[ID])), "splice_of_lever7_file_equal": splice_equal,
           "rows_changed": int(ch.sum()), "rows_changed_outside_lszh": int((ch & ~is_lszh).sum()), "squared_change": float((diff ** 2).sum()),
           "rms_change_s": float(np.sqrt((diff ** 2).mean())), "max_up_s": float(diff.max()), "max_down_s": float(diff.min()),
           "lirf_population_rows_changed": int((ch & pop).sum()), "guard_rows_changed": int((ch & v[ID].isin(R.GUARD).to_numpy()).sum()),
           "min_value": int(v["TAXITIME_SEC_mvt"].min()), "opdi_rows_used_ranking": int(use.sum())}
    if rep["lirf_population_rows_changed"] or rep["guard_rows_changed"] or rep["rows_changed_outside_lszh"]:
        raise ValueError("A protected or non-LSZH row changed.")
    write_json(out / "vigorous-whistle_v68.report.json", rep)
    print(rep)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "rank":
    rank()
