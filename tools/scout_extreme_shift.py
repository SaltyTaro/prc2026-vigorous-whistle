"""Extreme-day calibration of the reconciliation's day shift (src/prc2026_scout/extreme_shift.py, configs/scout_extreme_shift_v1.json).

  --score --declaration configs/scout_extreme_shift_v1.json
        the governing run: the four development months (v46 development stack, control = the declared v48 run
        runs/daily_totals/daily_totals_v1) and the twelve-month check (v23-recipe predictor, control = v48's procedure on that table,
        runs/scout/daily_ifr/v48_twelve, identical to runs/daily_totals/check12f); fitting side of each month from the other months;
        the declared rule -> the declaration's "report"
  --base FILE --recon DIR --version N --dry-run OUT [--declaration ...]
        the ranking: FILE = the model part (before the reconciliation), DIR = v48's procedure on FILE (tools/export_daily_totals.py
        --base FILE --dry-run DIR: values.parquet, days.parquet); parameters from all four development months; the layer on top
  --base FILE --recon DIR --version N --declaration ... [--user-override TEXT]   a real export under runs/submissions (never uploads)
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
from prc2026_scout import extreme_shift, published_figure  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402

SEED = 20260929
RULES = ROOT / "runs/scout/daily_ifr/rules.parquet"
TABLES = {"stack": (ROOT / "runs/daily_totals/rows.parquet", ROOT / "runs/daily_totals/daily_totals_v1"),
          "twelve": (ROOT / "runs/daily_totals/rows12.parquet", ROOT / "runs/scout/daily_ifr/v48_twelve")}
CODE = ["tools/scout_extreme_shift.py", "src/prc2026_scout/extreme_shift.py", "src/prc2026_scout/published_figure.py", "tools/scout_daily_ifr.py",
        "tools/export_daily_totals.py", "src/prc2026_daily/reconcile.py", "tools/run_daily_totals.py"]
KEYS = ["month", "airport", "bday"]


def training_rules() -> pd.Series:
    if not RULES.exists():
        from scout_daily_ifr import rule_table
        return rule_table()
    return pd.read_parquet(RULES).set_index(ID)["ifr"]


def dev_rows(kind: str) -> pd.DataFrame:
    path, run = TABLES[kind]
    t = pd.read_parquet(path)
    if "extra" not in t:
        t["extra"] = False
    months = sorted(t.loc[~t["extra"], "month"].unique())
    v48 = pd.concat([pd.read_parquet(run / f"{m}.parquet", columns=[ID, "new", "shift"]) for m in months], ignore_index=True)
    t = t.merge(v48.rename(columns={"new": "v48", "shift": "v48_shift"}), on=ID, how="left")
    t["moved"] = t["v48"].notna() & ((t["v48"] - t["stack"]).abs() > 1e-9)
    t["v48"] = t["v48"].fillna(t["stack"])
    t["in_figure"] = published_figure.in_figure(t["airport"].to_numpy(), training_rules().reindex(t[ID]).fillna(True).to_numpy(dtype=bool))
    return t


def read_table(kind: str) -> dict:
    t = dev_rows(kind)
    days = extreme_shift.day_frame(t, KEYS)
    months = sorted(t.loc[~t["extra"], "month"].unique())
    new = t["v48"].to_numpy(dtype=float).copy()
    changed, per_month = np.zeros(len(t), dtype=bool), {}
    for m in months:
        params = extreme_shift.fit(days[days.index.get_level_values("month") != m].droplevel("month"))
        use = days[days.index.get_level_values("month") == m]
        ch = extreme_shift.shifts(use.droplevel("month"), params)
        ch.index = pd.MultiIndex.from_tuples([(m, a, b) for a, b in ch.index], names=KEYS)
        values, mask = extreme_shift.apply(t[t["month"].eq(m)], ch, KEYS)
        idx = np.flatnonzero(t["month"].eq(m).to_numpy())
        new[idx], changed[idx] = values, mask
        per_month[m] = {"calibrated_airports": params["beta"], "days_changed": [{"airport": a, "bday": b, **{k: float(v) for k, v in r.items()}} for (_, a, b), r in ch.iterrows()]}
    t["new"] = new
    t = t[~t["extra"]]
    out = {"months": {}, "airports": {}}
    for m, g in t.groupby("month"):
        y, c, n = g["label"].to_numpy(dtype=float), g["v48"].to_numpy(dtype=float), g["new"].to_numpy(dtype=float)
        out["months"][m] = {"control_rmse": rmse(y, c), "rmse": rmse(y, n), "change_pct": 100 * (rmse(y, n) / rmse(y, c) - 1), "rows_changed": int((np.abs(n - c) > 1e-9).sum()),
                            "paired_days": paired_days(g["bday"].to_numpy(), y, n, c, SEED), **per_month[m]}
    se_c, se_n = (t["v48"] - t["label"]) ** 2, (t["new"] - t["label"]) ** 2
    by = pd.DataFrame({"airport": t["airport"], "month": t["month"], "c": se_c, "n": se_n}).groupby(["airport", "month"]).sum()
    pct = (100 * (by["n"] / by["c"] - 1)).unstack("month")
    out["airports"] = {a: {m: float(v) for m, v in r.items()} for a, r in pct.iterrows()}
    return out


def govern(path: str) -> None:
    decl = read_json(ROOT / path)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared, or does not respect its own contract.")
    target = ROOT / decl["report"]
    if target.exists():
        raise ValueError("Preserve the original result.")
    rule = decl["rule"]
    result = {"created_at_utc": utc_now(), "declaration": path, "declaration_sha256": object_hash(decl), "audit_used": False, "official_scores_used": "none",
              "code": {c: sha256_file(ROOT / c) for c in CODE}, "inputs": {str(p.relative_to(ROOT)): sha256_file(p) for p, _ in TABLES.values()}}
    for kind in ("stack", "twelve"):
        r = read_table(kind)
        result[kind] = r
        for m, e in r["months"].items():
            print(kind, m, f"{e['control_rmse']:.4f} -> {e['rmse']:.4f} {e['change_pct']:+.4f}%", "rows", e["rows_changed"], "interval",
                  [round(v, 3) for v in e["paired_days"]["rmse_difference_95pct_interval_seconds"]], "days changed", len(e["days_changed"]))
    dev = result["stack"]["months"]
    equal = rule["equal_footing"]
    worst_dev = max(e["change_pct"] for e in dev.values())
    mean_equal = float(np.mean([dev[m]["change_pct"] for m in equal]))
    best = min(dev, key=lambda m: dev[m]["change_pct"])
    best_interval_below = dev[best]["paired_days"]["rmse_difference_95pct_interval_seconds"][1] < 0
    twelve = result["twelve"]["months"]
    worst_twelve = max(e["change_pct"] for e in twelve.values())
    moved = [e for e in twelve.values() if e["rows_changed"] > 0]
    better_share = float(np.mean([e["change_pct"] < 0 for e in moved])) if moved else 0.0
    checks = {"no_development_month_worse_than": worst_dev <= rule["max_month_worsening_pct"], "equal_footing_mean": mean_equal <= -rule["min_mean_improvement_pct"],
              "largest_gain_interval_below_zero": bool(best_interval_below), "twelve_no_month_worse_than": worst_twelve <= rule["max_month_worsening_pct"],
              "twelve_changed_months_better_share": better_share >= rule["twelve_min_better_share"]}
    result.update({"rule_values": {"worst_development_month_pct": worst_dev, "equal_footing_mean_pct": mean_equal, "largest_gain_month": best,
                                   "worst_twelve_month_pct": worst_twelve, "twelve_changed_months": len(moved), "twelve_changed_months_better_share": better_share},
                   "checks": checks, "accepted": bool(all(checks.values()))})
    write_json(target, result)
    print({"accepted": result["accepted"], **result["rule_values"]})


def ranking_rows(base: str, recon: Path) -> pd.DataFrame:
    from export_daily_totals import ranking_table
    t = ranking_table((base, sha256_file(ROOT / base)))
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "FLIGHT_RULE_mvt"])
    rule = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID)["FLIGHT_RULE_mvt"].reindex(t[ID])
    t["in_figure"] = published_figure.in_figure(t["airport"].to_numpy(), (rule.isna() | rule.eq("I")).to_numpy())
    vals = pd.read_parquet(recon / "values.parquet").set_index(ID).reindex(t[ID])
    if vals["reconciled"].isna().any() or not np.array_equal(vals.iloc[:, 0].to_numpy().astype(np.int64), t["stack"].to_numpy().astype(np.int64)):
        raise ValueError("The reconciliation directory is not v48's procedure on this base.")
    t["v48"], t["v48_exported"] = vals["reconciled"].to_numpy(dtype=float), vals["exported"].to_numpy().astype(np.int64)
    t["moved"] = np.abs(t["v48"] - t["stack"]) > 1e-9
    days = pd.read_parquet(recon / "days.parquet").set_index(["airport", "bday"])["shift_s"]
    t["v48_shift"] = days.reindex(pd.MultiIndex.from_frame(t[["airport", "bday"]])).to_numpy(dtype=float)
    t["extra"] = False
    return t


def export(args) -> None:
    decl = read_json(ROOT / args.declaration) if args.declaration else None
    if args.dry_run is None:
        if decl is None:
            raise ValueError("A real export needs the declaration.")
        result = read_json(ROOT / decl["report"])
        if result.get("declaration_sha256") != object_hash(decl):
            raise ValueError("The recorded result is not for this declaration.")
        if not result.get("accepted") and not args.user_override:
            raise ValueError("The governing run did not accept; shipping needs --user-override with the user's words.")
        output, report_path, detail = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/scout/export_v{args.version}.json", ROOT / f"runs/scout/extreme_export_v{args.version}"
    else:
        out = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out, *out.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path, detail = out / f"vigorous-whistle_v{args.version}.parquet", out / f"export_v{args.version}.json", out
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    dev = dev_rows("stack")
    params = extreme_shift.fit(extreme_shift.day_frame(dev, KEYS).droplevel("month"))
    t = ranking_rows(Path(args.base).as_posix(), ROOT / args.recon)
    days = extreme_shift.day_frame(t, ["airport", "bday"])
    ch = extreme_shift.shifts(days, params)
    new, mask = extreme_shift.apply(t, ch, ["airport", "bday"])
    values = np.where(mask, np.rint(new), t["v48_exported"].to_numpy(dtype=float))
    provenance = {"experiment": "scout_extreme_shift_v1", "dry_run": args.dry_run is not None, "base": {"path": args.base, "sha256": sha256_file(ROOT / args.base)},
                  "reconciliation": {"dir": args.recon, "values_sha256": sha256_file(ROOT / args.recon / "values.parquet"), "days_sha256": sha256_file(ROOT / args.recon / "days.parquet")},
                  "selected_by": "dry run" if args.dry_run is not None else (f"the user's override: {args.user_override}" if args.user_override else "the declared rule (accepted)"),
                  "recipe": "the base, v48's reconciliation with the published daily totals, then on extreme days at calibrated airports the day shift beta_a * I_d in place of v48's",
                  "declaration": {args.declaration: sha256_file(ROOT / args.declaration)} if args.declaration else None, "code": {c: sha256_file(ROOT / c) for c in CODE},
                  "official_scores_used": "none", "audit_used": False}
    receipt = export_submission(pd.DataFrame({ID: t[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(t[ID]).to_numpy().astype(np.int64)
    change = exported - t["v48_exported"].to_numpy()
    if ((change != 0) & ~mask).any():
        raise ValueError("A row outside the layer changed.")
    detail.mkdir(parents=True, exist_ok=True)
    ch.reset_index().to_parquet(detail / "days_changed.parquet", index=False)
    S = float((change.astype(float) ** 2).sum())
    rep = {"created_at_utc": utc_now(), "version": args.version, "output": output.as_posix(), "sha256": sha256_file(output), "receipt": receipt, "parameters": params,
           "days_changed": [{"airport": a, "bday": b, **{k: float(v) for k, v in r.items()}} for (a, b), r in ch.iterrows()],
           "rows_changed": int((change != 0).sum()), "mean_change_s": float(change[change != 0].mean()) if (change != 0).any() else 0.0, "sum_squared_change_against_reconciled_base": S,
           "by_airport": {a: {"rows_changed": int(((change != 0) & (t["airport"] == a).to_numpy()).sum()), "sum_squared_change": float((change[(t["airport"] == a).to_numpy()].astype(float) ** 2).sum())}
                          for a in sorted(t["airport"].unique())}}
    write_json(report_path, rep)
    print({k: rep[k] for k in ["sha256", "rows_changed", "mean_change_s", "sum_squared_change_against_reconciled_base"]})
    print(ch.round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--declaration")
    parser.add_argument("--base")
    parser.add_argument("--recon")
    parser.add_argument("--version", type=int)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    if args.score:
        govern(args.declaration)
    else:
        export(args)


if __name__ == "__main__":
    main()
