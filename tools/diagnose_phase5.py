"""Describe remaining clock regimes using only the intersection of fitting IDs."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_phase4.bundle import load_bundle


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "data/phase4/2ceb3449c26767a97311"
    manifest, parent, original, catalog = load_bundle(root, base)
    folds = [pd.read_parquet(parent / "splits" / f"{fold}.parquet") for fold in ["seasonal", "forward"]]
    fitting = set.intersection(*[set(f.loc[f["partition"].eq("fit"), ID]) for f in folds])
    protected = set.union(*[set(f.loc[~f["partition"].eq("fit"), ID]) for f in folds])
    if fitting & protected:
        raise ValueError("The diagnostic would consume a held-out or reserved ID.")
    assignment = catalog[[ID]].copy()
    assignment["partition"] = np.where(assignment[ID].isin(fitting), "fit", "excluded")
    names = ["airport", "nm_missing", "aircraft_missing", "schedule_to_movement_s",
             "movement_minus_eobt1_s", "movement_minus_iobt_s", "eobt_minus_schedule_s",
             "iobt_minus_schedule_s", "schedule_day_offset", "flight_prefix_length",
             "flight_standard_shape", "lirf_missing_record"]
    features, truth = load_partition(base, parent, catalog, assignment, "fit", names)
    data = features.merge(truth[[ID, TIME, TARGET]], on=ID, validate="one_to_one")
    rules = {"all_fitting_rows": np.ones(len(data), dtype=bool),
             "existing_lirf_missing_gate": data["lirf_missing_record"].eq(1)}
    for threshold in [10800, 21600, 43200, 86400]:
        large = data["schedule_to_movement_s"].gt(threshold)
        for airport in ["ALL", "LIRF"]:
            for missing in [0, 1]:
                mask = large & data["nm_missing"].eq(missing)
                if airport != "ALL":
                    mask = mask & data["airport"].eq(airport)
                rules[f"{airport}_nm_missing_{missing}_schedule_gt_{threshold}"] = mask
    slices = {}
    for name, mask in rules.items():
        group = data.loc[mask]
        if group.empty:
            slices[name] = {"rows": 0}
            continue
        clocks = {}
        for column in ["schedule_to_movement_s", "movement_minus_eobt1_s", "movement_minus_iobt_s"]:
            delta = (group[TARGET]-group[column]).to_numpy(dtype=float)
            delta = delta[np.isfinite(delta)]
            clocks[column] = {"finite_rows": len(delta), "rmse_seconds": None if not len(delta) else float(np.sqrt(np.mean(delta**2))),
                              "within_10_seconds_fraction": None if not len(delta) else float((np.abs(delta) <= 10).mean())}
        slices[name] = {"rows": len(group), "above_three_hour_rows": int(group[TARGET].gt(10800).sum()),
                        "above_one_day_rows": int(group[TARGET].gt(86400).sum()),
                        "median_target_seconds": float(group[TARGET].median()), "maximum_target_seconds": float(group[TARGET].max()),
                        "clock_proxies": clocks}
    report = {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "bundle_id": manifest["bundle_id"],
              "fitting_rows": len(data), "fitting_ids_sha256": identity_hash(data[ID]),
              "scope": "Intersection of seasonal and forward fitting partitions; no scoring, stopping, December, or hidden ranking labels",
              "actual_months": sorted(data[TIME].dt.strftime("%Y-%m").unique().tolist()),
              "audit_used": False, "feature_columns": names, "slices": slices}
    write_json(root / "docs/phase5/fitting_diagnostics.json", report)
    print({"fitting_rows": len(data), "fitting_ids_sha256": report["fitting_ids_sha256"]})
    for name, values in slices.items():
        if name != "all_fitting_rows":
            print(name, values)


if __name__ == "__main__":
    main()
