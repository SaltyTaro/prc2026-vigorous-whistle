"""Aggregate allowed development-label diagnostics, without publishing raw rows."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import object_hash, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_phase3.bundle import DEFAULT_PARENT, load_parent
from prc2026_phase4.features import INPUT_COLUMNS, quality_features


def main():
    root = Path(__file__).resolve().parents[1]
    original, catalog = load_parent(root, root / DEFAULT_PARENT)
    allowed = catalog[TIME].dt.strftime("%Y-%m").isin([f"2025-{m:02}" for m in range(1, 12)])
    parts = []
    for source in sorted(catalog.loc[allowed, "source_file"].unique()):
        raw = pd.read_parquet(root / "data/raw" / source, columns=[*INPUT_COLUMNS, TARGET])
        raw = raw.loc[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").isin([f"2025-{m:02}" for m in range(1, 12)])].reset_index(drop=True)
        quality = quality_features(raw)
        data = raw[[ID, TIME, TARGET, "ADEP_mvt"]].copy()
        data["nm_missing"] = raw["ADEP_flt"].isna()
        data["aircraft_missing"] = raw["AIRCRAFT_TYPE_mvt"].isna()
        data["schedule_interval"] = (raw[TIME]-raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
        data["target_minus_schedule_interval"] = data[TARGET]-data["schedule_interval"]
        for name in ["lirf_missing_record", "flight_prefix_length", "flight_standard_shape"]:
            data[name] = quality[name]
        parts.append(data)
    frame = pd.concat(parts, ignore_index=True)
    gate = frame.loc[frame["lirf_missing_record"].eq(1)]
    tails = frame.loc[frame[TARGET] > 10800]
    groups = []
    for (airport, nm_missing, aircraft_missing), group in frame.groupby(["ADEP_mvt", "nm_missing", "aircraft_missing"]):
        delta = group["target_minus_schedule_interval"].to_numpy(dtype=float)
        finite = np.isfinite(delta)
        groups.append({"airport": airport, "nm_missing": bool(nm_missing), "aircraft_missing": bool(aircraft_missing),
                       "rows": len(group), "finite_schedule_rows": int(finite.sum()),
                       "fraction_target_within_10_seconds_of_schedule_interval": float((np.abs(delta[finite]) <= 10).mean()),
                       "schedule_interval_rmse_seconds": float(np.sqrt(np.mean(delta[finite]**2)))})
    monthly = []
    for month, group in gate.groupby(gate[TIME].dt.strftime("%Y-%m")):
        monthly.append({"month": month, "rows": len(group),
                        "fraction_target_within_10_seconds_of_schedule_interval": float(group["target_minus_schedule_interval"].abs().le(10).mean())})
    tail_groups = [{"airport": a, "nm_missing": bool(n), "rows": len(g), "maximum_target_seconds": float(g[TARGET].max())}
                   for (a, n), g in tails.groupby(["ADEP_mvt", "nm_missing"])]
    report = {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
              "parent_bundle_id": original["bundle_id"], "development_rows": len(frame), "audit_used": False,
              "scope": "Exploratory January-November diagnostics, including reused development labels; no December labels or hidden ranking labels",
              "mechanism": "TAXITIME minus (movement minus schedule) equals schedule minus block by target definition. Block is not an input feature.",
              "missing_lirf_rows": len(gate), "above_three_hour_rows": len(tails),
              "above_three_hour_slices": tail_groups, "missingness_slices": groups, "missing_lirf_months": monthly,
              "input_projection": [*INPUT_COLUMNS, TARGET], "external_data": []}
    write_json(root / "docs/phase4/diagnostics.json", report)
    print({"development_rows": len(frame), "missing_lirf_rows": len(gate), "above_three_hour_rows": len(tails), "audit_used": False})


if __name__ == "__main__":
    main()
