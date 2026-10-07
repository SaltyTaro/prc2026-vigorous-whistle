"""OPDI stacker change at the guard-passing airports EGLL / EHAM / LFPG on top of v68 (configs/opdi_stack_guard3_v1.json): row splice."""
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

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DECL = "configs/opdi_stack_guard3_v1.json"
CTL = ROOT / "runs/opdi_stack_lszh/dev_v68_proxy.parquet"
CAND = ROOT / "runs/opdi_stack/govern/chain/opdi/proxy.parquet"
FULL = ROOT / "runs/opdi_stack/rank/dry_v67/NOT_ACCEPTED_vigorous-whistle_v67.parquet"
V68 = (ROOT / "runs/submissions/vigorous-whistle_v68.parquet", "d41d8705e75a71e865fe21a8d6ad2f62044a5ed07f125fd208067914b2b10eab")
AIR = ["EGLL", "EHAM", "LFPG"]
OUT = ROOT / "docs/opdi_stack_guard3/results_v1.json"
DRY = ROOT / "runs/opdi_stack_guard3/rank/dry_v69/vigorous-whistle_v69.parquet"
MONTHS, EQ = ["2025-01", "2025-07", "2025-11", "2025-12"], ["2025-01", "2025-07", "2025-12"]


def main() -> None:
    if OUT.exists():
        raise ValueError("Preserve the reading.")
    if sha256_file(CTL) != "938a91cd84a9e6c0773afb72dec490fdf30c6762ee3f505469b3a26e3ab56aca":
        raise ValueError("The v68 proxy differs from the declared file.")
    ctl = pd.read_parquet(CTL)
    cand = pd.read_parquet(CAND).set_index(ID).reindex(ctl[ID])
    sel = ctl["airport"].isin(AIR).to_numpy() & cand["opdi_used"].to_numpy(dtype=bool)
    t = ctl.rename(columns={"value": "ctl"}).assign(cand=np.where(sel, cand["value"].to_numpy(dtype=float), ctl["value"].to_numpy(dtype=float)))
    rng = np.random.default_rng(20261004)
    months = {}
    for m in MONTHS:
        s = t[t["month"] == m]
        pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
        ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
        by = {a: float(np.sum((g["label"] - g["cand"]) ** 2) - np.sum((g["label"] - g["ctl"]) ** 2)) for a, g in s[s["airport"].isin(AIR)].groupby("airport")}
        months[m] = {"control_rmse": float(np.sqrt(np.mean((s["label"] - s["ctl"]) ** 2))), "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))),
                     "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum()),
                     "sse_change_by_airport": by}
    eq = float(np.mean([months[m]["pct"] for m in EQ]))
    checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= 0.005 for m in MONTHS), "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -0.02,
              "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in EQ), "4_guard": {"EGLL": 0.7395, "EHAM": 0.8856, "LFPG": 0.9270}}
    acc = bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"])
    res = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "control_sha256": sha256_file(CTL),
           "candidate_source_sha256": sha256_file(CAND), "months": months, "checks": checks, "accepted": acc, "official_scores_used": "none"}
    if acc:
        v68 = pd.read_parquet(V68[0])
        if sha256_file(V68[0]) != V68[1]:
            raise ValueError("v68 changed.")
        full = pd.read_parquet(FULL)
        raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
        raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v68[ID])
        if not (full[ID].equals(v68[ID])):
            raise ValueError("Row order differs.")
        isa = raw["ADEP_mvt"].isin(AIR).to_numpy()
        vals = np.where(isa, full[T].to_numpy(), v68[T].to_numpy())
        DRY.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow as pa
        import pyarrow.parquet as pq
        tab = pq.read_table(V68[0])
        tab = tab.set_column(tab.schema.get_field_index(T), tab.schema.field(T), pa.array(vals.astype(np.int32), type=pa.int32()))
        pq.write_table(tab, DRY, compression="snappy")
        diff = vals.astype(float) - v68[T].to_numpy(dtype=float)
        pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
        ch = diff != 0
        rep = {"file": DRY.relative_to(ROOT).as_posix(), "sha256": sha256_file(DRY), "rows_changed": int(ch.sum()),
               "rows_changed_by_airport": pd.Series(raw["ADEP_mvt"].to_numpy()[ch]).value_counts().to_dict(),
               "squared_change": float((diff ** 2).sum()), "max_up_s": float(diff.max()), "max_down_s": float(diff.min()),
               "lirf_population_rows_changed": int((ch & pop).sum()), "guard_rows_changed": int((ch & v68[ID].isin(R.GUARD).to_numpy()).sum()),
               "min_value": int(vals.min())}
        if rep["lirf_population_rows_changed"] or rep["guard_rows_changed"]:
            raise ValueError("Protected row changed.")
        res["ranking_dry_run"] = rep
    write_json(OUT, res)
    print({m: (round(v["pct"], 4), round(v["lo"], 3), round(v["hi"], 3)) for m, v in months.items()}, checks, "ACCEPTED", acc, res.get("ranking_dry_run"))


if __name__ == "__main__":
    main()
