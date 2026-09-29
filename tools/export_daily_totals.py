"""Export v46 reconciled with the published daily totals of positive off-block delay (configs/daily_totals_v1.json).

Construction, on the ranking rows:
  1. predictions = the uploaded v46 values; a = takeoff - schedule from data/raw/ranking.parquet; class from the final-file dry run's
     scope (matched / unmatched, LIRF fill population); off-block day = the day of takeoff - prediction.
  2. cells, the published figure's accuracy and the day-shift scale tau per airport from the four development months (the v46
     development stack, runs/daily_totals/rows.parquet); nothing from the scored months except the published totals themselves.
  3. per airport and off-block day read (rows cover at least 90% of the departures behind the published figure):
     src/prc2026_daily/reconcile.prepare_day / solve_day with the declaration's settings; heavy rows keep their prediction.
  4. export = the reconciled value (nearest integer, floor -12 s) where it differs from v46, v46 elsewhere.
Nothing is uploaded.

  --dry-run DIR
  --version 48 [--user-override TEXT]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_daily.reconcile import prepare_day, solve_day  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
import export_v33  # noqa: E402
from export_share_gate import uploaded  # noqa: E402
from run_daily_totals import Cells, add_keys, rows_table, scales, settings_from  # noqa: E402
from screen_daily_totals import BANDS, published  # noqa: E402

DECLARATION = "configs/daily_totals_v1.json"
V46 = ("runs/submissions/vigorous-whistle_v46.parquet", "8abdd87ad5098d7753927f407ffd7e8cd342aa84a2864392e1b64c0dde015af9", 223.9339)
WORKBOOK = "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.xlsx"
RANKING = "data/raw/ranking.parquet"
CODE = ["tools/export_daily_totals.py", "tools/run_daily_totals.py", "src/prc2026_daily/reconcile.py", "tools/screen_daily_totals.py"]
FLOOR = -12.0
SCOPES = {"matched_no_lirf": True, "all_no_lirf": False, "lirf_nonpop_matched": True, "lirf_nonpop_unmatched": False, "fixed": False}


def ranking_table() -> pd.DataFrame:
    detail = pd.read_parquet(ROOT / export_v33.DRY, columns=[ID, "airport", "month", "scope"])
    ids = detail[ID].to_numpy()
    if set(detail["scope"]) - set(SCOPES):
        raise ValueError("Unknown scope in the ranking detail.")
    raw = pd.read_parquet(ROOT / RANKING, columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(ids)
    if raw["MVT_TIME_UTC_mvt"].isna().any() or raw["SCHED_TIME_UTC_mvt"].isna().any() or not (raw["ADEP_mvt"].to_numpy() == detail["airport"].to_numpy()).all():
        raise ValueError("The ranking clocks do not cover the ranking rows.")
    takeoff, schedule = (pd.to_datetime(raw[c], utc=True).reset_index(drop=True) for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    t = pd.DataFrame({ID: ids, "airport": detail["airport"].to_numpy(), "month": detail["month"].to_numpy(), "stack": uploaded(V46[0], V46[1], ids).astype(float),
                      "matched": detail["scope"].map(SCOPES).to_numpy(dtype=bool)})
    t["a"] = (takeoff - schedule).dt.total_seconds().to_numpy()
    t["bday"] = (takeoff - pd.to_timedelta(t["stack"], unit="s")).dt.strftime("%Y-%m-%d").to_numpy()
    t["cls"] = np.where(detail["scope"].eq("fixed").to_numpy(), "LIRF_pop", np.where(t["matched"], t["airport"] + "_m", t["airport"] + "_u"))
    t["band"] = pd.cut(t["a"] / 3600, BANDS).astype(str)
    t = t.join(published(), on=["airport", "bday"])
    t["complete"] = (t["pub"].notna() & (t.groupby(["airport", "bday"])["a"].transform("size") >= 0.9 * t["n_pub"])).to_numpy()
    return add_keys(t)


def reconcile(t: pd.DataFrame, settings, tau_scale: float) -> tuple[np.ndarray, pd.DataFrame, dict]:
    fit = add_keys(rows_table())
    cells = Cells(fit)
    noise, wide, tau = scales(fit)
    new = t["stack"].to_numpy(dtype=float).copy()
    heavy = np.zeros(len(t), dtype=bool)
    days = []
    use = t[t["complete"]]
    for (airport, bday), day in use.groupby(["airport", "bday"], sort=True):
        positions = day.index.to_numpy()
        day = day.reset_index(drop=True)
        p = prepare_day(day["a"].to_numpy(dtype=float), day["stack"].to_numpy(dtype=float), cells.resolve(day), settings)
        values, info = solve_day(p, float(day["pub"].iat[0]), noise[airport], wide[airport], 0.0, tau_scale * tau[airport], settings)
        new[positions] = np.maximum(values, FLOOR)
        heavy[positions] = p.is_heavy
        days.append({"airport": airport, "bday": bday, "month": day["month"].iat[0], **info})
    return new, pd.DataFrame(days), {"noise_sd_s": noise, "noise_wide_share": wide, "tau_s": tau, "heavy_rows": int(heavy.sum()), "fit_rows": int(len(fit))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, default=48)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    results = read_json(ROOT / decl["report"])
    if results.get("declaration_sha256") != object_hash(decl):
        raise ValueError("The recorded results are not for this declaration.")
    if not results.get("accepted") and not args.user_override:
        raise ValueError("The governing run did not accept the procedure; shipping it needs --user-override with the user's words.")
    if sha256_file(ROOT / WORKBOOK) != decl["external_data"]["sha256"]:
        raise ValueError("The workbook changed.")
    if args.dry_run is None:
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/daily_totals/export_v{args.version}.json"
        detail_path = ROOT / f"runs/daily_totals/export_v{args.version}"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path, detail_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json", out_dir
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    settings, tau_scale, prior = settings_from(decl)
    if prior != "fixed" or settings.heavy_updates:
        raise ValueError("Only the declared configuration (fixed prior, heavy rows kept) is wired for the ranking rows.")

    t = ranking_table()
    ids = t[ID].to_numpy()
    v46 = t["stack"].to_numpy().astype(np.int64)
    new, days, fitted = reconcile(t, settings, tau_scale)
    if not np.isfinite(new).all():
        raise ValueError("Non-finite values.")
    changed = np.rint(new).astype(np.int64) != v46
    if (changed & ~t["complete"].to_numpy()).any():
        raise ValueError("A row outside a read airport-day changed.")
    values = np.where(changed, new, v46.astype(float))
    provenance = {"experiment_id": decl["experiment_id"], "dry_run": args.dry_run is not None,
                  "selected_by": "dry run" if args.dry_run is not None else (f"the user's override: {args.user_override}" if args.user_override else "the declared development rule (accepted)"),
                  "recipe": "v46 reconciled with EUROCONTROL's published daily all-cause pre-departure delay per airport: every row of a read airport-day that is not heavy moves by the "
                            "posterior day shift plus a first-order tilt; heavy rows and rows of airport-days not read keep v46; floor -12 s",
                  "incumbent": {"submission": Path(V46[0]).name, "official_rmse_seconds": V46[2], "sha256": V46[1],
                                "note": "the clean model line's best; v47 is a separate file on the same base"},
                  "declaration": {DECLARATION: {"sha256": sha256_file(ROOT / DECLARATION), "object_hash": object_hash(decl)}},
                  "development_results": {decl["report"]: sha256_file(ROOT / decl["report"])},
                  "code": {p: sha256_file(ROOT / p) for p in CODE},
                  "inputs": {p: sha256_file(ROOT / p) for p in [export_v33.DRY, V46[0], RANKING, "runs/daily_totals/rows.parquet"]},
                  "external_data": {"adsb": "adsb.lol globe_history (ODbL) surface positions, as in v46", "published_totals": decl["external_data"]},
                  "user_decision": decl["user_decision"], "audit_used": False, "floor_seconds": FLOOR, "official_scores_used": "none"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - v46.astype(float)
    if ((change != 0) & ~changed).any() or not np.array_equal(exported[changed], np.rint(new[changed]).astype(np.int64)):
        raise ValueError("The exported file is not the reconciled v46.")
    detail_path.mkdir(parents=True, exist_ok=True)
    days.to_parquet(detail_path / "days.parquet", index=False)
    pd.DataFrame({ID: ids, "v46": v46, "reconciled": new, "exported": exported}).to_parquet(detail_path / "values.parquet", index=False)

    n = len(ids)
    t["change"] = change
    cells = {}
    for (airport, month), g in t.groupby(["airport", "month"]):
        d = days[(days["airport"] == airport) & (days["month"] == month)]
        cells[f"{airport}|{month}"] = {"rows": int(len(g)), "rows_in_read_days": int(g["complete"].sum()), "rows_changed": int((g["change"] != 0).sum()), "days_read": int(d["used"].sum()),
                                       "days_without_use": int((~d["used"]).sum()), "mean_change_s": float(g["change"].mean()), "mean_abs_change_s": float(g["change"].abs().mean()),
                                       "max_abs_change_s": float(g["change"].abs().max()), "day_shift_mean_s": float(d["shift_s"].mean()) if len(d) else 0.0,
                                       "day_shift_sd_s": float(d["shift_s"].std()) if len(d) > 1 else 0.0, "day_shift_min_s": float(d["shift_s"].min()) if len(d) else 0.0,
                                       "day_shift_max_s": float(d["shift_s"].max()) if len(d) else 0.0, "sum_squared_change": float((g["change"] ** 2).sum())}
    counts = t[t["complete"]].groupby(["airport", "bday"]).agg(n=("a", "size"), n_pub=("n_pub", "first"))
    report = {"created_at_utc": utc_now(), "version": args.version, "base": 46, "dry_run": args.dry_run is not None, "output": output.as_posix(), "sha256": sha256_file(output),
              "receipt": receipt, "rows": n, "declaration_object_hash": object_hash(decl), "fitted_from_development_months": fitted,
              "checks": {"rows_changed_outside_read_days": 0, "exported_equals_reconciled_on_changed_rows": True, "heavy_rows_changed": 0},
              "airport_days": {"read": int(days["used"].sum()), "prepared_without_use": int((~days["used"]).sum()), "rows_outside_read_days": int((~t["complete"]).sum()),
                               "row_count_over_published_count": {"median": float((counts["n"] / counts["n_pub"]).median()), "min": float((counts["n"] / counts["n_pub"]).min()),
                                                                  "max": float((counts["n"] / counts["n_pub"]).max())}},
              "rows_changed": int((change != 0).sum()), "mean_change_s": float(change.mean()), "mean_abs_change_s": float(np.abs(change).mean()), "max_abs_change_s": float(np.abs(change).max()),
              "rows_changed_over_60s": int((np.abs(change) > 60).sum()), "sum_squared_change": float((change ** 2).sum()), "cells": cells,
              "rmse_if_every_changed_label_equals_the_new_value": float(np.sqrt(max(n * V46[2] ** 2 - (change ** 2).sum(), 0.0) / n)),
              "rmse_if_the_changes_are_noise": float(np.sqrt((n * V46[2] ** 2 + (change ** 2).sum()) / n)),
              "rmse_at_the_development_alignment": {f"{b}": float(np.sqrt(max(n * V46[2] ** 2 - (2 * b - 1) * (change ** 2).sum(), 0.0) / n)) for b in (0.78, 0.84, 1.0, 1.17)}}
    if int((change[np.flatnonzero(t["cls"].eq("LIRF_pop").to_numpy())] != 0).sum()):
        report["checks"]["lirf_population_rows_changed"] = int((change[np.flatnonzero(t["cls"].eq("LIRF_pop").to_numpy())] != 0).sum())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print({k: report[k] for k in ["sha256", "rows_changed", "mean_change_s", "mean_abs_change_s", "max_abs_change_s", "rows_changed_over_60s", "sum_squared_change", "airport_days",
                                  "rmse_if_every_changed_label_equals_the_new_value", "rmse_if_the_changes_are_noise", "rmse_at_the_development_alignment"]})
    print(pd.DataFrame(cells).T[["rows_changed", "days_read", "mean_change_s", "day_shift_mean_s", "day_shift_sd_s", "day_shift_min_s", "day_shift_max_s", "max_abs_change_s"]].round(1).to_string())


if __name__ == "__main__":
    main()
