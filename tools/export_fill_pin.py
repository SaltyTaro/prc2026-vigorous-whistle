"""Export v39 = the uploaded v38 with configs/fill_pin_v1.json's fill pins (src/prc2026_final/fill_pin.py) on the ranking file.

Construction: the uploaded v38 integers (hash-checked) -> the declared candidate's pinned rows take their schedule interval -> every other
row is unchanged -> export_submission (template order, nearest-integer rounding). The report lists every changed row. Nothing is uploaded.

  --dry-run DIR [--candidate all_twelve|dominant_three]
  --version 39 --user-override TEXT [--candidate ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026_final.fill_pin import COLUMNS, apply_pins, fill_pins  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402

DECL = "configs/fill_pin_v1.json"
V38_FILE, V38_SHA, V38_RMSE = "runs/submissions/vigorous-whistle_v38.parquet", "61264cdb7f0a37a8e5cf943f8e950441df38b826132cfc1097f00ca9d0519fe4", 226.8053
RAW, TEMPLATE = "data/raw/ranking.parquet", "data/raw/submitting.parquet"
EXTRA = ["FLIGHT_mvt", "STAND_mvt"]


def candidate_pins(pins: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    keep = pins["scope"].isin(candidate["scopes"]) & ~(pins["scope"].eq("population") & (pins["interval_s"] < float(candidate["population_interval_s_min"])))
    return pins.loc[keep].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", default="all_twelve")
    parser.add_argument("--version", type=int, default=39)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc") or args.candidate not in decl["candidates"]:
        raise ValueError("The declaration is not recorded or the candidate is not declared.")
    if args.dry_run is None:
        if not args.user_override:
            raise ValueError("No rule selects these rows; the real export needs --user-override with the user's words.")
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/fill_pin/export_v{args.version}.json"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    if sha256_file(ROOT / V38_FILE) != V38_SHA:
        raise ValueError("The uploaded v38 changed.")

    v38 = pd.read_parquet(ROOT / V38_FILE)
    ids = v38[ID].to_numpy()
    base = v38[TARGET].to_numpy(dtype=float)
    raw = pd.read_parquet(ROOT / RAW, columns=COLUMNS + EXTRA)
    pins_all = fill_pins(raw[COLUMNS])
    counts = pins_all["scope"].value_counts().to_dict()
    expected = {k: v for k, v in decl["expected_ranking_scope"].items() if k != "note"}
    if counts != expected:
        raise ValueError(f"The ranking's fill-pin scope is {counts}, not the declared {expected}.")
    pins = candidate_pins(pins_all, decl["candidates"][args.candidate])
    airport = pd.DataFrame({ID: ids}).merge(raw[[ID, "ADEP_mvt"]], on=ID, how="left", validate="one_to_one")["ADEP_mvt"].to_numpy()
    values, pinned = apply_pins(base, ids, pins, airport)
    provenance = {"experiment_id": decl["experiment_id"], "candidate": args.candidate, "dry_run": args.dry_run is not None,
                  "selected_by": f"the user's decision on the declared candidates: {args.user_override}" if args.user_override else "dry run",
                  "recipe": f"v38 (uploaded integers) + fill pins ({args.candidate}: {decl['candidates'][args.candidate]['scopes']}) at the schedule interval",
                  "incumbent": {"submission": Path(V38_FILE).name, "official_rmse_seconds": V38_RMSE, "sha256": V38_SHA},
                  "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
                  "code": {p: sha256_file(ROOT / p) for p in ["src/prc2026_final/fill_pin.py", "tools/export_fill_pin.py"]},
                  "inputs": {RAW: sha256_file(ROOT / RAW)}, "audit_used": False, "external_data": "none for this layer"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / TEMPLATE, output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - base
    if ((change != 0) & ~pinned).any():
        raise ValueError("Rows outside the pins changed.")
    if not np.array_equal(exported[pinned], np.rint(pins.set_index(ID)["pin_s"].reindex(ids[pinned]).to_numpy())):
        raise ValueError("A pinned row is not at its schedule interval.")
    meta = raw.set_index(ID).reindex(ids[pinned])
    rows = [{"MVT_ID_mvt": float(i), "airport": str(meta.at[i, "ADEP_mvt"]), "takeoff_utc": str(meta.at[i, TIME]), "flight": meta.at[i, "FLIGHT_mvt"],
             "stand": meta.at[i, "STAND_mvt"], "scope": str(p.scope), "interval_s": float(p.interval_s), "nm_proxy_s": (None if not np.isfinite(p.proxy_s) else float(p.proxy_s)),
             "v38": int(base[k]), "v39": int(exported[k]), "change_s": float(change[k])}
            for i, k, p in zip(ids[pinned], np.flatnonzero(pinned), pins.set_index(ID).reindex(ids[pinned]).itertuples())]
    n = len(ids)
    sse38 = n * V38_RMSE ** 2
    gain = float(np.sum(change ** 2))
    report = {"created_at_utc": utc_now(), "version": args.version, "candidate": args.candidate, "dry_run": args.dry_run is not None, "output": output.as_posix(),
              "receipt_sha256": receipt.get("sha256"), "scope_counts": counts, "rows_changed": int(pinned.sum()), "rows": rows,
              "sum_squared_change": gain, "if_every_pinned_row_is_a_fill": {"sse_removed": gain, "share_of_v38_sse": gain / sse38,
                                                                            "rmse": float(np.sqrt((sse38 - gain) / n)), "rmse_change_s": float(np.sqrt((sse38 - gain) / n) - V38_RMSE)},
              "if_one_population_row_is_a_one_day_error": "see configs/fill_pin_v1.json evidence.stake_on_v38 (about -1.09 s net for all_twelve)",
              "user_override": args.user_override}
    write_json(report_path, report)
    print({k: report[k] for k in ["rows_changed", "scope_counts", "sum_squared_change", "if_every_pinned_row_is_a_fill"]}, receipt.get("sha256"))


if __name__ == "__main__":
    main()
