"""Dry run (and, on a declaration that accepts, export) of the reconciliation variant of tools/scout_daily_ifr.py on the ranking rows.

Construction as tools/export_daily_totals.py (v48's export): the base file's values as predictions, a = takeoff - schedule, class from
the final-file dry run's scope, off-block day = the day of takeoff - prediction; cells, tau, the published-figure model and the shift
mixture from the four development months (runs/daily_totals/rows.parquet). Only the reconciliation's per-day noise terms, the rows that
enter the figure (IFR only at EHAM / LSZH / LEBL) and the shift prior differ, per the variant. Rows outside the figure, heavy rows and
rows of airport-days not read keep the base value. Floor -12 s, nearest integer.

  --base runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet --variant figure_mix --dry-run runs/scout/export_dry --version 56
  [--compare runs/daily_totals/dry_v54/vigorous-whistle_v54.parquet]   the same base with v48's procedure, for the size of the change
A real export (no --dry-run) needs --declaration with an accepted governing result, or --user-override with the user's words.
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
from prc2026_scout import published_figure, shift_prior  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from export_daily_totals import ranking_table  # noqa: E402
from run_daily_totals import Cells, add_keys, rows_table, scales, settings_from  # noqa: E402
from scout_daily_ifr import MIX_CLIP_S, TAU_SCALE, V48_DECLARATION, with_figure  # noqa: E402

FLOOR = -12.0
CODE = ["tools/scout_export_daily.py", "tools/scout_daily_ifr.py", "src/prc2026_scout/published_figure.py", "src/prc2026_scout/shift_prior.py",
        "src/prc2026_daily/reconcile.py", "tools/run_daily_totals.py", "tools/export_daily_totals.py"]


def ranking_rules(ids: np.ndarray) -> np.ndarray:
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "FLIGHT_RULE_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID)["FLIGHT_RULE_mvt"].reindex(ids)
    return raw.eq("I").fillna(True).to_numpy(dtype=bool) | raw.isna().to_numpy()


def reconcile(t: pd.DataFrame, variant: str) -> tuple[np.ndarray, pd.DataFrame, dict]:
    settings, tau_scale, prior = settings_from(read_json(ROOT / V48_DECLARATION))
    tau_scale *= TAU_SCALE.get(variant, 1.0)
    fit = with_figure(add_keys(rows_table()).assign(extra=False))
    cells = Cells(fit)
    noise, wide, tau = scales(fit)
    model = published_figure.fit(published_figure.fitting_days(fit[fit["in_figure"]])) if variant.startswith("figure") else None
    mixture = shift_prior.fit_mixture(shift_prior.day_means(fit, MIX_CLIP_S), tau) if "mix" in variant else None
    t = t.assign(in_figure=published_figure.in_figure(t["airport"].to_numpy(), ranking_rules(t[ID].to_numpy())))
    use = t[t["complete"] & (t["in_figure"] if model is not None else True)]
    new = t["stack"].to_numpy(dtype=float).copy()
    heavy = np.zeros(len(t), dtype=bool)
    days = []
    for (airport, bday), day in use.groupby(["airport", "bday"], sort=True):
        positions = day.index.to_numpy()
        day = day.reset_index(drop=True)
        p = prepare_day(day["a"].to_numpy(dtype=float), day["stack"].to_numpy(dtype=float), cells.resolve(day), settings)
        total, sd, share, dn = float(day["pub"].iat[0]), noise[airport], wide[airport], float(day["n_pub"].iat[0]) - len(day)
        if model is not None:
            offset, sd, share = published_figure.day_terms(model, airport, dn)
            total -= offset
        if mixture is None:
            values, info = solve_day(p, total, sd, share, 0.0, tau_scale * tau[airport], settings)
        else:
            values, info = shift_prior.solve_day_mix(p, total, sd, share, tau_scale * tau[airport], mixture["pi"], mixture["k"], settings)
        new[positions] = np.maximum(values, FLOOR)
        heavy[positions] = p.is_heavy
        days.append({"airport": airport, "bday": bday, "month": day["month"].iat[0], "dn": dn, "noise_sd_s": sd, "wide_share": share, **info})
    fitted = {"v48_noise_sd_s": noise, "v48_noise_wide_share": wide, "tau_s": tau, "tau_scale": tau_scale, "shift_mixture": mixture,
              "published_figure_model": {a: {**{k: v for k, v in m.items() if k != "sign"}, "sign": {str(s): v for s, v in m["sign"].items()}} for a, m in model.items()} if model else None}
    return new, pd.DataFrame(days), {**fitted, "heavy": heavy, "in_figure": t["in_figure"].to_numpy()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--compare")
    parser.add_argument("--declaration")
    parser.add_argument("--user-override")
    args = parser.parse_args()
    if args.dry_run is None:
        if not args.declaration:
            raise ValueError("A real export needs a declaration.")
        decl = read_json(ROOT / args.declaration)
        result = read_json(ROOT / decl["report"])
        if result.get("declaration_sha256") != object_hash(decl):
            raise ValueError("The recorded result is not for this declaration.")
        if not result.get("accepted") and not args.user_override:
            raise ValueError("The governing run did not accept; shipping needs --user-override with the user's words.")
        if decl["candidate"]["variant"] != args.variant:
            raise ValueError("The variant is not the declared candidate.")
        output, report_path, detail = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/scout/export_v{args.version}.json", ROOT / f"runs/scout/export_v{args.version}"
    else:
        out = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out, *out.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path, detail = out / f"vigorous-whistle_v{args.version}.parquet", out / f"export_v{args.version}.json", out
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    base = (Path(args.base).as_posix(), sha256_file(ROOT / args.base))
    t = ranking_table(base)
    ids = t[ID].to_numpy()
    v = t["stack"].to_numpy().astype(np.int64)
    new, days, fitted = reconcile(t, args.variant)
    heavy, in_figure = fitted.pop("heavy"), fitted.pop("in_figure")
    changed = np.rint(new).astype(np.int64) != v
    if (changed & heavy).any() or (changed & ~in_figure).any() or (changed & ~t["complete"].to_numpy()).any():
        raise ValueError("A heavy row, a row outside the figure or a row of an unread day changed.")
    values = np.where(changed, new, v.astype(float))
    provenance = {"experiment": "scout_daily_ifr", "variant": args.variant, "dry_run": args.dry_run is not None, "base": {"path": base[0], "sha256": base[1]},
                  "selected_by": "dry run" if args.dry_run is not None else (f"the user's override: {args.user_override}" if args.user_override else "the declared development rule (accepted)"),
                  "recipe": "the base reconciled with EUROCONTROL's published daily all-cause pre-departure delay (applied last, in place of v48's procedure): rows that enter the "
                            "figure on read airport-days move by the posterior day shift plus a first-order tilt; heavy rows, rows outside the figure and rows of unread days keep the base",
                  "declaration": {args.declaration: sha256_file(ROOT / args.declaration)} if args.declaration else None,
                  "code": {c: sha256_file(ROOT / c) for c in CODE}, "official_scores_used": "none", "audit_used": False, "floor_seconds": FLOOR}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - v.astype(float)
    detail.mkdir(parents=True, exist_ok=True)
    days.to_parquet(detail / "days.parquet", index=False)
    pd.DataFrame({ID: ids, "base": v, "reconciled": new, "exported": exported}).to_parquet(detail / "values.parquet", index=False)
    rep = {"created_at_utc": utc_now(), "variant": args.variant, "version": args.version, "output": output.as_posix(), "sha256": sha256_file(output), "receipt": receipt,
           "fitted_from_development_months": fitted, "rows_changed": int((change != 0).sum()), "mean_change_s": float(change.mean()),
           "mean_abs_change_s": float(np.abs(change).mean()), "sum_squared_change": float((change ** 2).sum())}
    if args.compare:
        c = pd.read_parquet(ROOT / args.compare).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(float)
        d = exported - c
        rep["against_compare"] = {"path": args.compare, "sha256": sha256_file(ROOT / args.compare), "rows_that_differ": int((d != 0).sum()), "sum_squared_change": float((d ** 2).sum()),
                                  "by_airport_month": {f"{a}|{m}": {"rows_that_differ": int((g["d"] != 0).sum()), "mean_change_s": float(g["d"].mean()), "sum_squared_change": float((g["d"] ** 2).sum())}
                                                       for (a, m), g in t.assign(d=d).groupby(["airport", "month"])}}
    write_json(report_path, rep)
    print({k: rep[k] for k in ["sha256", "rows_changed", "mean_change_s", "mean_abs_change_s", "sum_squared_change"]})
    if args.compare:
        print("against compare: rows", rep["against_compare"]["rows_that_differ"], "S", f"{rep['against_compare']['sum_squared_change']:.3e}")
        print(pd.DataFrame(rep["against_compare"]["by_airport_month"]).T.round(1).to_string())


if __name__ == "__main__":
    main()
