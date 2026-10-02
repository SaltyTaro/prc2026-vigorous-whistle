"""Variance re-allocation of the airport-day's published discrepancy (src/prc2026_scout/var_alloc.py, configs/scout_var_alloc_v1.json).

  --score --declaration configs/scout_var_alloc_v1.json
        the governing run: the four development months (v46 development stack) and the twelve-month check (v23-recipe predictor); in
        both, the control is the extreme-day layer on v48's procedure, fitted month-blocked as its own governing run; this layer's
        models and k are fitted without the scored month (month-blocked); the declared rule -> the declaration's "report"
  --base FILE --recon DIR --version N --dry-run OUT [--declaration ...]
        the ranking: FILE = the model part, DIR = v48's procedure on FILE; the extreme-day layer re-applied and checked byte for byte
        against the previous step's file (--previous PREV --previous-sha256 SHA: tools/scout_extreme_shift.py's output on the same
        FILE and DIR; default the uploaded v55, whose chain is the documented one); models and k from the four development months;
        the layer on top. In the final-file chain: runner -> tools/export_daily_totals.py --base -> tools/scout_extreme_shift.py ->
        this tool with --previous set to the extreme-day step's file
  --base FILE --recon DIR --version N --declaration ... [--user-override TEXT]   a real export under runs/submissions (never uploads)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_scout import extreme_shift, var_alloc  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from scout_extreme_shift import KEYS, SEED, TABLES, dev_rows, ranking_rows  # noqa: E402
from scout_extreme_shift import CODE as EXTREME_CODE  # noqa: E402

V55 = ROOT / "runs/submissions/vigorous-whistle_v55.parquet"
V55_SHA = "aa02ff597a619f223888370726cdcbf5c6d055720ef2f95b0c93818133fd1ece"
GUARD = (213142471.0, 213142469.0)
CODE = ["tools/scout_var_alloc.py", "src/prc2026_scout/var_alloc.py", *EXTREME_CODE, "src/prc2026_scout/extreme_shift.py"]


def departures(paths: list[str]) -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(p, columns=var_alloc.RAW_COLUMNS) for p in paths], ignore_index=True)
    return raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID)


def implied(t: pd.DataFrame, days: pd.DataFrame, c0: dict, keys: list[str]) -> np.ndarray:
    """I_d of each row's airport-day: days with a published total and n_eff >= 1, clipped; 0 elsewhere and on incomplete rows."""
    d = days[(days["n_eff"] >= 1) & days["pub"].notna()]
    d = d.assign(I=extreme_shift.implied_shift(d, pd.Series(c0)).to_numpy())
    I = t.join(d["I"], on=keys)["I"].fillna(0.0).clip(-var_alloc.CLIP_S, var_alloc.CLIP_S).to_numpy(dtype=float)
    return np.where(t["complete"].astype(bool).to_numpy(), I, 0.0)


def layered(kind: str) -> tuple[pd.DataFrame, dict]:
    """The table with the control (ext: the extreme-day layer, month-blocked) and this layer's out-of-fold values (new)."""
    t = dev_rows(kind)
    days = extreme_shift.day_frame(t, KEYS)
    months = sorted(t.loc[~t["extra"], "month"].unique())
    ext, I = t["v48"].to_numpy(dtype=float).copy(), np.zeros(len(t))
    for m in months:
        params = extreme_shift.fit(days[days.index.get_level_values("month") != m].droplevel("month"))
        use = days[days.index.get_level_values("month") == m]
        ch = extreme_shift.shifts(use.droplevel("month"), params)
        ch.index = pd.MultiIndex.from_tuples([(m, a, b) for a, b in ch.index], names=KEYS)
        idx = np.flatnonzero(t["month"].eq(m).to_numpy())
        ext[idx] = extreme_shift.apply(t.iloc[idx], ch, KEYS)[0]
        I[idx] = implied(t.iloc[idx], use.droplevel("month"), params["c0"], ["airport", "bday"])
    t["ext"], t["I"] = ext, I
    t["r"] = t["label"] - t["ext"]
    elig = var_alloc.eligible(t)
    f = var_alloc.features(t, departures(sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))), KEYS)
    a, label, r = t["a"].to_numpy(dtype=float), t["label"].to_numpy(dtype=float), t["r"].to_numpy(dtype=float)
    month, extra = t["month"].to_numpy(), t["extra"].to_numpy(dtype=bool)
    w = np.full(len(t), np.nan)
    for m in months:
        models = var_alloc.fit_models(f, a, label, r, elig & (month != m))
        ev = month == m
        w[ev] = var_alloc.weights(models, f.loc[ev], elig[ev])
        print(kind, m, "models fitted without the month", flush=True)
    base = var_alloc.basis(t, w, I, KEYS)
    new, ks = ext.copy(), {}
    for m in months:
        ks[m] = var_alloc.slope(r, base, elig & (month != m) & ~extra)
        ev = month == m
        new[ev] = var_alloc.update(ext[ev], ks[m], base[ev], elig[ev])
    t["w"], t["base"], t["new"] = w, base, new
    return t, ks


def read_table(kind: str) -> dict:
    t, ks = layered(kind)
    t = t[~t["extra"]]
    out = {"months": {}, "airports": {}}
    for m, g in t.groupby("month"):
        y, c, n = g["label"].to_numpy(dtype=float), g["ext"].to_numpy(dtype=float), g["new"].to_numpy(dtype=float)
        d = np.sort((y - n) ** 2 - (y - c) ** 2)
        out["months"][m] = {"control_rmse": rmse(y, c), "rmse": rmse(y, n), "change_pct": 100 * (rmse(y, n) / rmse(y, c) - 1), "k": ks[m],
                            "rows_changed": int((np.abs(n - c) > 1e-9).sum()), "sum_squared_change": float(((n - c) ** 2).sum()),
                            "best_ten_rows_share_of_gain_pct": float(100 * d[:10].sum() / d.sum()) if d.sum() < 0 else None,
                            "paired_days": paired_days(g["bday"].to_numpy(), y, n, c, SEED)}
    se_c, se_n = (t["ext"] - t["label"]) ** 2, (t["new"] - t["label"]) ** 2
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
              "evidence_note": "the four development months' readings of this procedure were seen in exploration before the declaration "
                               "(prior_exploration_disclosed); the twelve-month check had not been read and is the independent evidence",
              "code": {c: sha256_file(ROOT / c) for c in CODE}, "inputs": {str(p.relative_to(ROOT)): sha256_file(p) for p, _ in TABLES.values()}}
    for kind in ("stack", "twelve"):
        r = read_table(kind)
        result[kind] = r
        for m, e in r["months"].items():
            print(kind, m, f"{e['control_rmse']:.4f} -> {e['rmse']:.4f} {e['change_pct']:+.4f}%", "k", round(e["k"], 4), "rows", e["rows_changed"], "interval",
                  [round(v, 3) for v in e["paired_days"]["rmse_difference_95pct_interval_seconds"]], flush=True)
    dev, twelve = result["stack"]["months"], result["twelve"]["months"]
    equal = rule["equal_footing"]
    worst_dev = max(e["change_pct"] for e in dev.values())
    mean_equal = float(np.mean([dev[m]["change_pct"] for m in equal]))
    intervals_below = {m: dev[m]["paired_days"]["rmse_difference_95pct_interval_seconds"][1] < 0 for m in equal}
    worst_twelve = max(e["change_pct"] for e in twelve.values())
    better_share = float(np.mean([e["change_pct"] < 0 for e in twelve.values()]))
    checks = {"no_development_month_worse_than": worst_dev <= rule["max_month_worsening_pct"], "equal_footing_mean": mean_equal <= -rule["min_mean_improvement_pct"],
              "equal_footing_intervals_below_zero": bool(all(intervals_below.values())), "twelve_no_month_worse_than": worst_twelve <= rule["max_month_worsening_pct"],
              "twelve_better_share": better_share >= rule["twelve_min_better_share"]}
    result.update({"rule_values": {"worst_development_month_pct": worst_dev, "equal_footing_mean_pct": mean_equal, "equal_footing_intervals_below_zero": intervals_below,
                                   "worst_twelve_month_pct": worst_twelve, "twelve_better_share": better_share,
                                   "twelve_mean_pct": float(np.mean([e["change_pct"] for e in twelve.values()]))},
                   "checks": checks, "accepted": bool(all(checks.values()))})
    write_json(target, result)
    print({"accepted": result["accepted"], **result["rule_values"]})


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
        output, report_path, detail = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/scout/export_v{args.version}.json", ROOT / f"runs/scout/var_alloc_export_v{args.version}"
    else:
        out = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out, *out.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path, detail = out / f"vigorous-whistle_v{args.version}.parquet", out / f"export_v{args.version}.json", out
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    previous, previous_sha = (ROOT / args.previous, args.previous_sha256) if args.previous else (V55, V55_SHA)
    if not previous_sha or sha256_file(previous) != previous_sha:
        raise ValueError("The previous step's file does not have the expected SHA-256.")
    # development side: the control and the out-of-fold weights (for k), then the models on all four months
    dev, _ = layered("stack")
    elig_d = var_alloc.eligible(dev)
    k = var_alloc.slope(dev["r"].to_numpy(dtype=float), dev["base"].to_numpy(dtype=float), elig_d)
    f_d = var_alloc.features(dev, departures(sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))), KEYS)
    models = var_alloc.fit_models(f_d, dev["a"].to_numpy(dtype=float), dev["label"].to_numpy(dtype=float), dev["r"].to_numpy(dtype=float), elig_d)
    params = extreme_shift.fit(extreme_shift.day_frame(dev, KEYS).droplevel("month"))
    # ranking: the chain up to the extreme-day layer, checked against the previous step's file, then the layer
    t = ranking_rows(Path(args.base).as_posix(), ROOT / args.recon)
    days = extreme_shift.day_frame(t, ["airport", "bday"])
    ext, mask = extreme_shift.apply(t, extreme_shift.shifts(days, params), ["airport", "bday"])
    v55 = np.where(mask, np.rint(ext), t["v48_exported"].to_numpy(dtype=float)).astype(np.int64)
    uploaded = pd.read_parquet(previous).set_index(ID)[TARGET].reindex(t[ID]).to_numpy().astype(np.int64)
    if not np.array_equal(v55, uploaded):
        raise ValueError("The chain up to the extreme-day layer does not reproduce the previous step's file.")
    t["month"] = t["bday"].str[:7]
    I = implied(t, days, params["c0"], ["airport", "bday"])
    elig = var_alloc.eligible(t, GUARD)
    f = var_alloc.features(t, departures([str(ROOT / "data/raw/ranking.parquet")]), KEYS)
    w = var_alloc.weights(models, f, elig)
    base = var_alloc.basis(t, w, I, KEYS)
    new = var_alloc.update(ext, k, base, elig)
    values = np.where(elig, np.rint(new), v55)
    provenance = {"experiment_id": "scout_var_alloc_v1", "dry_run": args.dry_run is not None, "base": {"path": args.base, "sha256": sha256_file(ROOT / args.base)},
                  "reconciliation": {"dir": args.recon, "values_sha256": sha256_file(ROOT / args.recon / "values.parquet"), "days_sha256": sha256_file(ROOT / args.recon / "days.parquet")},
                  "previous_step": {"path": str(previous.relative_to(ROOT)) if previous.is_relative_to(ROOT) else str(previous), "sha256": previous_sha, "reproduced": True},
                  "selected_by": "dry run" if args.dry_run is not None else (f"the user's override: {args.user_override}" if args.user_override else "the declared rule (accepted)"),
                  "recipe": "v55's chain (the base, v48's reconciliation, the extreme-day layer), then the airport-day's discrepancy re-allocated by a learned per-row error variance (cap 100 s; heavy rows, the LIRF population and the two 4-5 July guard rows unchanged)",
                  "declaration": {args.declaration: sha256_file(ROOT / args.declaration)} if args.declaration else None, "code": {c: sha256_file(ROOT / c) for c in CODE},
                  "official_scores_used": "none", "audit_used": False}
    receipt = export_submission(pd.DataFrame({ID: t[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(t[ID]).to_numpy().astype(np.int64)
    change = exported - v55
    if ((change != 0) & ~elig).any():
        raise ValueError("A row outside the layer changed.")
    if (change[t[ID].isin(list(GUARD)).to_numpy() | t["cls"].eq("LIRF_pop").to_numpy()] != 0).any():
        raise ValueError("A guard row or a LIRF population row changed.")
    detail.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: t[ID].to_numpy(), "airport": t["airport"].to_numpy(), "bday": t["bday"].to_numpy(), "v55": v55, "value": exported, "I": I, "w": w, "base": base}).to_parquet(detail / "rows.parquet", index=False)
    S = float((change.astype(float) ** 2).sum())
    by = pd.DataFrame({"airport": t["airport"], "month": t["month"], "ch": change.astype(float)})
    rep = {"created_at_utc": utc_now(), "version": args.version, "output": output.as_posix(), "sha256": sha256_file(output), "receipt": receipt, "k": k,
           "rows_changed": int((change != 0).sum()), "max_abs_change_s": float(np.abs(change).max()), "sum_squared_change_against_v55": S,
           "rms_change_s": float(np.sqrt((change.astype(float) ** 2).mean())),
           "by_airport_month": {f"{a} {m}": {"rows_changed": int((g["ch"] != 0).sum()), "sum_squared_change": float((g["ch"] ** 2).sum()), "mean_change_s": float(g["ch"].mean())}
                                for (a, m), g in by.groupby(["airport", "month"]) if (g["ch"] != 0).any()}}
    write_json(report_path, rep)
    print({x: rep[x] for x in ["sha256", "k", "rows_changed", "max_abs_change_s", "sum_squared_change_against_v55", "rms_change_s"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--declaration")
    parser.add_argument("--base")
    parser.add_argument("--recon")
    parser.add_argument("--version", type=int)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    parser.add_argument("--previous", help="the extreme-day step's file on the same base (default: the uploaded v55)")
    parser.add_argument("--previous-sha256")
    args = parser.parse_args()
    if args.score:
        govern(args.declaration)
    else:
        export(args)


if __name__ == "__main__":
    main()
