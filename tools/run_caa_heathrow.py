"""Heathrow reconciliation with the UK CAA punctuality statistics (src/prc2026_caa/heathrow.py).

  --verify                       the 2025 check of the published figures against the supplied clocks (labels of 2025 only) and the
                                 definitional offset / noise per month -> docs/caa_heathrow/verification.json
  --score --declaration CONFIG   governing run on the four development months against v48's procedure (runs/daily_totals/daily_totals_v1)
                                 -> the declaration's report; placebo (another month's figures) reported
  --export --version N [--base FILE] [--dry-run DIR] [--user-override TEXT]
                                 the procedure on the ranking rows of a submission-format base (default: the uploaded v48)
Nothing is uploaded. Mapping tables: docs/caa_heathrow/destinations.csv, airlines.csv (built from 2025 monthly counts).
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
from prc2026_caa.heathrow import FLOOR, ROUNDING_VAR, Settings, reconcile_month  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_daily_totals import SEED, rows_table  # noqa: E402

CAA_DIR = ROOT / "data/external/caa_punctuality"
MAP_DIR = ROOT / "docs/caa_heathrow"
VERIFICATION = MAP_DIR / "verification.json"
V48 = ("runs/submissions/vigorous-whistle_v48.parquet", "714ce4257b9b224ee7720e7bcae5c9240bc829935a05aa541049314ab8e51e0e", 222.6042)
CONTROL = ROOT / "runs/daily_totals/daily_totals_v1"
DEV_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
RANKING_MONTHS = ["2026-01", "2026-07"]
BANDS = ["flights_more_than_15_minutes_early_percent", "flights_15_minutes_early_to_1_minute_early_percent", "flights_0_to_15_minutes_late_percent",
         "flights_between_16_and_30_minutes_late_percent", "flights_between_31_and_60_minutes_late_percent", "flights_between_61_and_120_minutes_late_percent",
         "flights_between_121_and_180_minutes_late_percent", "flights_between_181_and_360_minutes_late_percent", "flights_more_than_360_minutes_late_percent"]
CODE = ["tools/run_caa_heathrow.py", "src/prc2026_caa/heathrow.py", "docs/caa_heathrow/destinations.csv", "docs/caa_heathrow/airlines.csv"]


def caa_file(month: str) -> Path:
    return CAA_DIR / f"{month.replace('-', '')}_arrdep.csv"


def published(month: str) -> pd.DataFrame:
    """Heathrow departures of the month: index (airline, destination), n_pub and the flight-weighted mean of the published averages
    over the scheduled / charter rows (whole minutes each)."""
    d = pd.read_csv(caa_file(month), encoding="utf-8-sig")
    d = d[(d["reporting_airport"] == "HEATHROW") & (d["arrival_departure"] == "D") & (d["number_flights_matched"] > 0)].copy()
    d["w"] = d["number_flights_matched"] * d["average_delay_mins"]
    g = d.groupby(["airline_name", "origin_destination"]).agg(n_pub=("number_flights_matched", "sum"), w=("w", "sum"))
    g["delay_min"] = g["w"] / g["n_pub"]
    g.index = g.index.set_names(["airline", "dest"])
    return g[["n_pub", "delay_min"]]


def maps() -> tuple[dict, dict]:
    a = pd.read_csv(MAP_DIR / "airlines.csv")
    d = pd.read_csv(MAP_DIR / "destinations.csv")
    return dict(zip(a["prefix"], a["caa_airline"])), dict(zip(d["icao"], d["caa"]))


def with_groups(t: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Attach the CAA airline / destination names (NaN when unmapped or cargo) and the class key to rows indexed by MVT_ID."""
    amap, dmap = maps()
    r = raw.set_index(ID).reindex(t[ID])
    prefix = r["FLIGHT_mvt"].astype(str).str.extract(r"^([A-Z]{3})")[0]
    t = t.copy()
    t["airline"] = prefix.map(amap).to_numpy()
    t["dest"] = r["ADES_mvt"].map(dmap).to_numpy()
    t.loc[r["MARKET_SEGMENT_flt"].eq("Cargo").to_numpy(), ["airline", "dest"]] = np.nan
    return t


def raw_columns(files: list[Path]) -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(f, columns=[ID, "PHASE_mvt", "FLIGHT_mvt", "ADES_mvt", "MARKET_SEGMENT_flt"]) for f in files])
    return raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)


def training_file(month: str) -> Path:
    y, m = int(month[:4]), int(month[5:])
    nxt = f"{y + 1}-01" if m == 12 else f"{y}-{m + 1:02d}"
    return ROOT / f"data/raw/training_{month}-01_{nxt}-01.parquet"


def dev_rows(month: str) -> pd.DataFrame:
    """Heathrow development rows of the month with v48's value as the prediction."""
    t = rows_table()
    t = t[t["airport"].eq("EGLL") & t["month"].eq(month)].reset_index(drop=True)
    new = pd.read_parquet(CONTROL / f"{month}.parquet", columns=[ID, "new"]).set_index(ID)["new"]
    t["pred"] = t[ID].map(new).fillna(t["stack"]).to_numpy(dtype=float)
    t["cls"] = np.where(t["matched"], "m", "u")
    return with_groups(t, raw_columns([training_file(month)]))


def sd_by_class(months: list[str]) -> dict:
    """Row error sd of v48's values at Heathrow by class, ordinary rows, over the given development months."""
    frames = [dev_rows(m) for m in months]
    t = pd.concat(frames, ignore_index=True)
    r = t["label"].to_numpy(dtype=float) - t["pred"].to_numpy(dtype=float)
    ok = np.abs(r) < 3600
    return {c: float(np.sqrt(np.mean(r[ok & t["cls"].eq(c).to_numpy()] ** 2))) for c in ("m", "u")}


def verify() -> dict:
    """2025 labels against the published figures, month by month: the definitional offset (published minus label mean) and the noise
    beyond the rounding, on read-eligible groups."""
    out = {"created_at_utc": utc_now(), "files": {caa_file(f"2025-{m:02d}").name: sha256_file(caa_file(f"2025-{m:02d}")) for m in range(1, 13)}, "months": {}}
    settings = Settings()
    for m in range(1, 13):
        month = f"2025-{m:02d}"
        raw = pd.read_parquet(training_file(month), columns=[ID, "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "ADES_mvt", "MARKET_SEGMENT_flt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
        raw = raw[raw["PHASE_mvt"].eq("DEP") & raw["ADEP_mvt"].eq("EGLL")].drop_duplicates(ID)
        t = with_groups(raw[[ID]].reset_index(drop=True), raw)
        delay = (pd.to_datetime(raw["BLOCK_TIME_UTC_mvt"], utc=True) - pd.to_datetime(raw["SCHED_TIME_UTC_mvt"], utc=True)).dt.total_seconds().to_numpy() / 60.0
        t["pos"] = np.clip(delay, 0, None)
        t["late"] = delay
        pub = published(month)
        g = t.dropna(subset=["airline", "dest"]).groupby(["airline", "dest"]).agg(n_ours=("pos", "size"), label_mean=("pos", "mean")).join(pub, how="inner")
        g = g[(g["n_pub"] >= settings.min_group) & ((g["n_ours"] - g["n_pub"]).abs() <= np.maximum(2, settings.count_tolerance * g["n_pub"]))]
        e = (g["delay_min"] - g["label_mean"]).to_numpy()
        w = g["n_pub"].to_numpy(dtype=float)
        delta = float(np.average(e, weights=w))
        mad = float(np.median(np.abs(e - np.median(e))) * 1.4826)
        out["months"][month] = {"groups_read": int(len(g)), "flights_in_read_groups": int(w.sum()), "heathrow_departures": int(len(t)),
                                "published_minus_label_mean_min": delta, "robust_sd_min": mad, "definitional_var_min2": max(mad ** 2 - ROUNDING_VAR, 0.0),
                                "share_within_half_minute": float(np.mean(np.abs(e - delta) <= 0.5)), "groups_off_by_more_than_a_minute": int(np.sum(np.abs(e - delta) > 1.0))}
        print(month, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["months"][month].items()})
    write_json(VERIFICATION, out)
    return out


def offsets(exclude: list[str]) -> tuple[float, float]:
    v = read_json(VERIFICATION)["months"]
    use = [m for m in v if m not in exclude]
    return float(np.mean([v[m]["published_minus_label_mean_min"] for m in use])), float(np.mean([v[m]["definitional_var_min2"] for m in use]))


def placebo_month(month: str) -> str:
    m = int(month[5:])
    return f"2025-{(m + 5) % 12 + 1:02d}"


def score(declaration: str) -> dict:
    decl = read_json(ROOT / declaration)
    settings = Settings(**decl["procedure"]["settings"])
    out = {"created_at_utc": utc_now(), "declaration_sha256": object_hash(decl), "control": "v48's procedure on the development stack (runs/daily_totals/daily_totals_v1)", "months": {}}
    full = rows_table()
    for month in DEV_MONTHS:
        others = [m for m in DEV_MONTHS if m != month]
        sd = sd_by_class(others)
        delta, def_var = offsets([month])
        t = dev_rows(month)
        result = {}
        for kind, pub_month in (("read", month), ("placebo", placebo_month(month))):
            new, table, summary = reconcile_month(t, published(pub_month), sd, delta, def_var, settings)
            y, before = t["label"].to_numpy(dtype=float), t["pred"].to_numpy(dtype=float)
            # the complete month: every airport, Heathrow rows replaced
            fm = full[full["month"].eq(month)]
            base_all = fm[ID].map(pd.read_parquet(CONTROL / f"{month}.parquet", columns=[ID, "new"]).set_index(ID)["new"]).fillna(fm["stack"]).to_numpy(dtype=float)
            new_all = base_all.copy()
            pos = pd.Series(np.arange(len(fm)), index=fm[ID].to_numpy()).reindex(t[ID].to_numpy()).to_numpy()
            new_all[pos] = new
            y_all = fm["label"].to_numpy(dtype=float)
            result[kind] = {"published_month": pub_month, **summary, "heathrow_rmse_before": rmse(y, before), "heathrow_rmse_after": rmse(y, new),
                            "heathrow_change_pct": 100 * (rmse(y, new) / rmse(y, before) - 1), "rmse_before": rmse(y_all, base_all), "rmse_after": rmse(y_all, new_all),
                            "change_pct": 100 * (rmse(y_all, new_all) / rmse(y_all, base_all) - 1),
                            "alignment": float(-np.sum((new - before) * (before - y)) / max(np.sum((new - before) ** 2), 1e-9)),
                            "groups_with_a_gain": int(sum(((before[t.index.isin(g)] - y[t.index.isin(g)]) ** 2).sum() > ((new[t.index.isin(g)] - y[t.index.isin(g)]) ** 2).sum()
                                                         for g in [t[(t["airline"] == r.airline) & (t["dest"] == r.dest)].index for r in table.itertuples()])) if len(table) else 0,
                            "paired_days": paired_days(fm["bday"].to_numpy(), y_all, new_all, base_all, SEED)}
            if kind == "read":
                (ROOT / "runs/caa_heathrow").mkdir(parents=True, exist_ok=True)
                table.to_parquet(ROOT / f"runs/caa_heathrow/groups_{month}.parquet", index=False)
                pd.DataFrame({ID: t[ID].to_numpy(), "control": before, "new": new}).to_parquet(ROOT / f"runs/caa_heathrow/{month}.parquet", index=False)
        result["row_sd_s"], result["delta_min"], result["definitional_var_min2"] = sd, delta, def_var
        out["months"][month] = result
        r = result["read"]
        print(month, f"groups {r['groups_read']}, rows {r['rows_moved']}, prior sd {r['prior_sd_s']:.1f} s, mean |shift| {r['mean_abs_shift_s']:.1f} s | Heathrow {r['heathrow_change_pct']:+.3f}% | month {r['change_pct']:+.4f}%",
              "interval", [round(v, 3) for v in r["paired_days"]["rmse_difference_95pct_interval_seconds"]], f"| alignment {r['alignment']:.2f} | placebo {result['placebo']['change_pct']:+.4f}%")
    rule = decl["rule"]
    changes = [out["months"][m]["read"]["change_pct"] for m in EQUAL]
    worse = [m for m in EQUAL if out["months"][m]["read"]["change_pct"] > 0]
    out.update({"mean_change_pct_equal_footing": float(np.mean(changes)), "equal_footing_months_worse": worse,
                "accepted": bool(not worse and -np.mean(changes) >= rule["min_mean_improvement_pct_equal_footing"])})
    write_json(ROOT / decl["report"], out)
    print({"accepted": out["accepted"], "mean_change_pct_equal_footing": round(out["mean_change_pct_equal_footing"], 4), "worse": worse})
    return out


def export(declaration: str, version: int, base: Path | None, dry_run: Path | None, user_override: str | None) -> None:
    import export_daily_totals as edt
    decl = read_json(ROOT / declaration)
    results = read_json(ROOT / decl["report"])
    if results.get("declaration_sha256") != object_hash(decl):
        raise ValueError("The recorded results are not for this declaration.")
    if not results.get("accepted") and not user_override:
        raise ValueError("The governing run did not accept the procedure; shipping it needs --user-override with the user's words.")
    settings = Settings(**decl["procedure"]["settings"])
    base_path = ROOT / V48[0] if base is None else base
    if base is None and sha256_file(base_path) != V48[1]:
        raise ValueError("The uploaded v48 changed.")
    if dry_run is None:
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{version}.parquet", ROOT / f"docs/caa_heathrow/export_v{version}.json"
    else:
        out_dir = dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{version}.parquet", out_dir / f"export_v{version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    base_path = base_path.resolve()
    t = edt.ranking_table((base_path.relative_to(ROOT).as_posix(), sha256_file(base_path)))   # the values of the base file, hash-checked
    ids = t[ID].to_numpy()
    base_values = t["stack"].to_numpy(dtype=float)
    if not np.array_equal(base_values, np.rint(base_values)):
        raise ValueError("The base must hold integer predictions.")
    raw = raw_columns([ROOT / edt.RANKING])
    sd = sd_by_class(DEV_MONTHS)
    delta, def_var = offsets([])
    new = base_values.copy()
    tables, summaries = {}, {}
    for month in RANKING_MONTHS:
        e = t[t["airport"].eq("EGLL") & t["month"].eq(month)].copy()
        e["pred"] = e["stack"].to_numpy(dtype=float)
        e["cls"] = np.where(e["matched"], "m", "u")
        e = with_groups(e, raw)
        pos = e.index.to_numpy()
        e = e.reset_index(drop=True)
        values, table, summary = reconcile_month(e, published(month), sd, delta, def_var, settings)
        new[pos] = values
        tables[month], summaries[month] = table, summary
    changed = np.rint(new).astype(np.int64) != base_values.astype(np.int64)
    if (changed & ~t["airport"].eq("EGLL").to_numpy()).any():
        raise ValueError("A row outside Heathrow changed.")
    values = np.where(changed, new, base_values)
    provenance = {"experiment_id": decl["experiment_id"], "dry_run": dry_run is not None,
                  "selected_by": "dry run" if dry_run is not None else (f"the user's override: {user_override}" if user_override else "the declared development rule (accepted)"),
                  "recipe": f"{base_path.name} with Heathrow rows reconciled, per airline x destination group of the month, with the UK CAA's published average departure delay; "
                            "level error of the group's implied mean of max(delay, 0) against the published figure, shrunk by the month's own dispersion of the errors; floor -12 s",
                  "base": {"file": base_path.as_posix(), "sha256": sha256_file(base_path)},
                  "incumbent": {"submission": Path(V48[0]).name, "official_rmse_seconds": V48[2], "sha256": V48[1]},
                  "declaration": {declaration: {"sha256": sha256_file(ROOT / declaration), "object_hash": object_hash(decl)}},
                  "development_results": {decl["report"]: sha256_file(ROOT / decl["report"])}, "verification": {VERIFICATION.relative_to(ROOT).as_posix(): sha256_file(VERIFICATION)},
                  "code": {p: sha256_file(ROOT / p) for p in CODE},
                  "inputs": {p: sha256_file(ROOT / p) for p in [edt.export_v33.DRY, edt.RANKING, "runs/daily_totals/rows.parquet"]},
                  "external_data": {"adsb": "adsb.lol globe_history (ODbL) surface positions, as in v46", "published_totals": "EUROCONTROL daily pre-departure delay, as in v48 (the base)",
                                    "caa": {**decl["external_data"], "files": {caa_file(m).name: sha256_file(caa_file(m)) for m in RANKING_MONTHS}}},
                  "user_decision": decl["user_decision"], "audit_used": False, "floor_seconds": FLOOR, "official_scores_used": "none"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - base_values
    if ((change != 0) & ~changed).any():
        raise ValueError("The exported file changed a row the reconciliation did not.")
    n = len(ids)
    report = {"created_at_utc": utc_now(), "version": version, "base": provenance["base"], "dry_run": dry_run is not None, "output": output.as_posix(), "sha256": sha256_file(output),
              "receipt": receipt, "rows": n, "declaration_object_hash": object_hash(decl), "row_sd_s": sd, "delta_min": delta, "definitional_var_min2": def_var,
              "months": {m: {**summaries[m], "rows_changed": int((change != 0)[(t["airport"].eq("EGLL") & t["month"].eq(m)).to_numpy()].sum()),
                             "sum_squared_change": float((change[(t["airport"].eq("EGLL") & t["month"].eq(m)).to_numpy()] ** 2).sum()),
                             "largest_shifts": tables[m].reindex(tables[m]["shift_s"].abs().sort_values(ascending=False).index).head(8)[["airline", "dest", "n_pub", "published_min", "implied_min", "level_error_min", "correction_min", "shift_s"]].to_dict("records")}
                         for m in RANKING_MONTHS},
              "rows_changed": int((change != 0).sum()), "mean_change_s": float(change.mean()), "mean_abs_change_s": float(np.abs(change).mean()), "max_abs_change_s": float(np.abs(change).max()),
              "sum_squared_change": float((change ** 2).sum()),
              "stake_if_the_base_is_v48": {"rmse_if_the_changes_are_noise": float(np.sqrt((n * V48[2] ** 2 + (change ** 2).sum()) / n)),
                                           "rmse_if_every_changed_label_equals_the_new_value": float(np.sqrt(max(n * V48[2] ** 2 - (change ** 2).sum(), 0.0) / n)),
                                           "note": "arithmetic on v48's published score for the size of the stake only; meaningful only when the base is the uploaded v48"}}
    for m in RANKING_MONTHS:
        tables[m].to_parquet(report_path.parent / f"groups_v{version}_{m}.parquet", index=False) if dry_run is not None else None
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print({k: report[k] for k in ["sha256", "rows_changed", "mean_change_s", "mean_abs_change_s", "max_abs_change_s", "sum_squared_change", "stake_if_the_base_is_v48"]})
    for m in RANKING_MONTHS:
        print(m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in summaries[m].items()}, "rows changed", report["months"][m]["rows_changed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--declaration", default="configs/caa_heathrow_v1.json")
    parser.add_argument("--version", type=int)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    if args.verify:
        verify()
    if args.score:
        score(args.declaration)
    if args.export:
        if args.version is None:
            raise ValueError("--export needs --version.")
        export(args.declaration, args.version, args.base, args.dry_run, args.user_override)


if __name__ == "__main__":
    main()
