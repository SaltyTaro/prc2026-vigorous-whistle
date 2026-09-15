"""Independently verify a post-processed export against the raw data and frozen parents.

Recomputes every changed row from raw ranking columns and the frozen v7/v5
ranking predictions without importing the rule package, and checks the template
contract. Writes a verification report; never uploads.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_submission.contract import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/postprocess_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/postprocess/verification.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    receipt = read_json(output.with_suffix(".receipt.json"))
    if receipt["sha256"] != sha256_file(output) or receipt["provenance"]["config_sha256"] != object_hash(config):
        raise ValueError("Export receipt does not match the file or declaration.")
    contract = validate_submission(output, ROOT / "data/raw/submitting.parquet")
    params = receipt["provenance"]["parameters"]
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
    raw = raw.loc[raw[PHASE].eq("DEP")].set_index(ID)
    v8 = pd.read_parquet(output).set_index(ID)[TARGET]
    v7 = pd.read_parquet(ROOT / "runs/submissions/vigorous-whistle_v7.parquet").set_index(ID)[TARGET]
    v7_raw = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"]
    v5_raw = pd.read_parquet(ROOT / config["parents"]["fallback_ranking_predictions"]["path"]).set_index(ID)["prediction"]
    for name, parent in config["parents"].items():
        if sha256_file(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError(f"Parent changed: {name}")
    ids = v8.index.to_numpy()
    raw, v7, v7_raw, v5_raw = raw.reindex(ids), v7.reindex(ids), v7_raw.reindex(ids), v5_raw.reindex(ids)
    # Independent rule recomputation from raw columns.
    clock = raw["AOBT_3_flt"].where(raw["AOBT_3_flt"].notna(), raw["LOBT_flt"])
    proxy = (raw[TIME] - clock).dt.total_seconds()
    schedule = (raw[TIME] - raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    route = proxy.notna() & (proxy < 0)
    hedge = raw["ADEP_mvt"].eq("LIRF") & clock.isna() & raw["AIRCRAFT_TYPE_mvt"].isna() & (schedule > params["schedule_interval_threshold_seconds"])
    expected = v7_raw.to_numpy(dtype=float).copy()
    expected[route.to_numpy()] = v5_raw.to_numpy(dtype=float)[route.to_numpy()]
    expected[hedge.to_numpy()] = ((1 - params["one_day_share"]) * schedule + params["one_day_share"] * (86400 + params["one_day_taxi_median_seconds"])).to_numpy()[hedge.to_numpy()]
    floored = expected < params["floor_seconds"]
    expected[floored] = params["floor_seconds"]
    expected_int = np.rint(expected).astype(np.int32)
    if not np.array_equal(expected_int, v8.to_numpy()):
        raise ValueError("Independent recomputation differs from the export.")
    changed = expected_int != v7.to_numpy()
    untouched = ~(route | hedge).to_numpy() & ~floored
    if not np.array_equal(v8.to_numpy()[untouched], v7.to_numpy()[untouched]):
        raise ValueError("Rows outside the declared rules differ from v7.")
    # Fitting-only parameter recomputation from the refit months.
    months = config["refit_months"]
    if "2025-12" in months:
        raise ValueError("December remains reserved.")
    frames = []
    for month in months:
        m = int(month[-2:])
        path = ROOT / "data/raw" / f"training_2025-{m:02}-01_2025-{m + 1:02}-01.parquet"
        frame = pd.read_parquet(path, columns=[PHASE, TIME, "SCHED_TIME_UTC_mvt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt", "LOBT_flt", TARGET])
        frames.append(frame.loc[frame[PHASE].eq("DEP")])
    fit = pd.concat(frames, ignore_index=True)
    y = fit[TARGET].to_numpy(dtype=float)
    pop = fit["ADEP_mvt"].eq("LIRF") & fit["AOBT_3_flt"].isna() & fit["LOBT_flt"].isna() & fit["AIRCRAFT_TYPE_mvt"].isna()
    s = (fit[TIME] - fit["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    one_day = pop & (y >= 82800) & (y <= 90000)
    group = pop & (s > params["schedule_interval_threshold_seconds"])
    recomputed = {"floor_seconds": float(y.min()), "one_day_share": float((group & one_day).sum() / group.sum()),
                  "one_day_taxi_median_seconds": float(np.median(y[one_day] - 86400)), "long_schedule_group_rows": int(group.sum()),
                  "long_schedule_group_ordinary_rows": int((group & (y < 3600)).sum())}
    for key, value in recomputed.items():
        if abs(value - params[key]) > 1e-12:
            raise ValueError(f"Parameter {key} does not reproduce from fitting labels.")
    report = {"verified_at_utc": utc_now(), "file": output.relative_to(ROOT).as_posix(), "sha256": receipt["sha256"], "contract": contract,
              "independent_recomputation_matches": True, "untouched_rows_equal_v7": True, "parameters_reproduce_from_fitting_labels": True,
              "recomputed_parameters": recomputed, "changed_rows": int(changed.sum()), "routed_rows": int(route.sum()), "hedged_rows": int(hedge.sum()),
              "floored_rows": int(floored.sum()), "rows_below_zero": int((v8 < 0).sum()), "v7_rows_below_zero": int((v7 < 0).sum()),
              "changed_by_airport": {k: int(v) for k, v in raw.loc[changed, "ADEP_mvt"].value_counts().items()},
              "largest_changes": [{"id": float(i), "airport": raw.at[i, "ADEP_mvt"], "v7": int(v7.at[i]), "v8": int(v8.at[i])}
                                  for i in v8.index[changed][np.argsort(-np.abs((v8 - v7).to_numpy()[changed]))[:15]]]}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["changed_rows", "routed_rows", "hedged_rows", "floored_rows", "rows_below_zero", "v7_rows_below_zero", "changed_by_airport"]})


if __name__ == "__main__":
    main()
