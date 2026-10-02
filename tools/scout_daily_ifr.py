"""The v48 reconciliation with the published figure's own definition as its noise model (src/prc2026_scout/published_figure.py).

Everything else is v48's procedure (configs/daily_totals_v1.json, src/prc2026_daily/reconcile.py, tools/run_daily_totals.py): the v46
development stack on the four development months, cells / tau / settings from the other three months, heavy rows keep their prediction,
an airport-day is read when its rows cover at least 90% of the published count. What changes, per airport-day:

  - the rows that enter the day's arithmetic are those that enter the figure (IFR only at EHAM / LSZH / LEBL); the others keep their
    prediction
  - the published total is compared with the own total plus the expected offset c0 + mu |dn| (dn = published count - the day's rows),
    with core noise sd sqrt(s0^2 + |dn| s^2) and the airport's wide share, all from the other months (published_figure.fit)

  --month 2025-07 --variant figure [--rows twelve]     -> runs/scout/daily_ifr/<variant>[_twelve]/<month>.parquet
  --variant v48 reproduces runs/daily_totals/daily_totals_v1 (four months) / the twelve-month check of v48's procedure
  --report --variant figure [--rows twelve]            against v48's procedure on the same rows
  --score --declaration configs/scout_daily_ifr_v1.json   the governing run (four months) and the declared rule -> its "report" path
  --check12 --declaration configs/scout_daily_ifr_v1.json the independent twelve-month reading                -> its "check12" path
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_daily.reconcile import prepare_day, solve_day  # noqa: E402
from prc2026_scout import published_figure, shift_prior  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_daily_totals import MONTHS, SEED, Cells, add_keys, rows12_table, rows_table, scales, settings_from  # noqa: E402

MONTHS12 = [f"2025-{m:02}" for m in range(1, 13)]
VARIANTS = ("v48", "figure", "mix", "figure_mix", "mix0", "figure_t15", "figure_mix_t15")   # mix0: the mixture code at pi = 0, k = 1 (must reproduce v48)
MIX_CLIP_S = 1800.0   # v48's own threshold between ordinary rows and big errors; at 600 s the rows that make a disruption day are cut away
TAU_SCALE = {"figure_t15": 1.5, "figure_mix_t15": 1.5}
OUT = ROOT / "runs/scout/daily_ifr"
RULES = OUT / "rules.parquet"
V48_DECLARATION = "configs/daily_totals_v1.json"
V48_RUN = ROOT / "runs/daily_totals/daily_totals_v1"
CODE = ["tools/scout_daily_ifr.py", "src/prc2026_scout/published_figure.py", "src/prc2026_scout/shift_prior.py", "src/prc2026_daily/reconcile.py", "tools/run_daily_totals.py", "tools/screen_daily_totals.py"]


def rule_table() -> pd.Series:
    """IFR flag of every 2025 departure (FLIGHT_RULE_mvt == "I"), from the supplied training files."""
    if RULES.exists():
        return pd.read_parquet(RULES).set_index(ID)["ifr"]
    raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", "FLIGHT_RULE_mvt"]) for p in sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))], ignore_index=True)
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
    s = pd.DataFrame({ID: raw[ID].to_numpy(), "ifr": raw["FLIGHT_RULE_mvt"].eq("I").to_numpy()})
    RULES.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(RULES, index=False)
    return s.set_index(ID)["ifr"]


def with_figure(t: pd.DataFrame) -> pd.DataFrame:
    ifr = rule_table().reindex(t[ID]).fillna(True).to_numpy(dtype=bool)
    return t.assign(in_figure=published_figure.in_figure(t["airport"].to_numpy(), ifr))


def table(rows: str) -> pd.DataFrame:
    t = add_keys(rows12_table() if rows == "twelve" else rows_table())
    if "extra" not in t:
        t["extra"] = False
    return with_figure(t)


def tag_of(variant: str, rows: str) -> str:
    return variant + ("_twelve" if rows == "twelve" else "")


def run_month(month: str, variant: str, rows: str = "stack") -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Variant: one of {VARIANTS}.")
    settings, tau_scale, prior = settings_from(read_json(ROOT / V48_DECLARATION))
    tau_scale *= TAU_SCALE.get(variant, 1.0)
    if prior != "fixed":
        raise ValueError("v48's procedure has the fixed prior.")
    t = table(rows)
    fit, use = t[t["month"].ne(month) & ~t["extra"]], t[t["month"].eq(month) & t["complete"]]
    cells = Cells(fit)
    noise, wide, tau = scales(fit)
    model, mixture = None, None
    if variant.startswith("figure"):
        model = published_figure.fit(published_figure.fitting_days(fit[fit["in_figure"]]))
        use = use[use["in_figure"]]
    if variant in ("mix", "figure_mix", "figure_mix_t15"):
        mixture = shift_prior.fit_mixture(shift_prior.day_means(fit, MIX_CLIP_S), tau)
    elif variant == "mix0":
        mixture = {"pi": 0.0, "k": 1.0}
    out, infos = [], []
    started = time.time()
    for airport, month_rows in use.groupby("airport", sort=True):
        for bday, day in month_rows.groupby("bday", sort=True):
            day = day.reset_index(drop=True)
            p = prepare_day(day["a"].to_numpy(dtype=float), day["stack"].to_numpy(dtype=float), cells.resolve(day), settings)
            total, sd, share, dn = float(day["pub"].iat[0]), noise[airport], wide[airport], float(day["n_pub"].iat[0]) - len(day)
            if model is not None:
                offset, sd, share = published_figure.day_terms(model, airport, dn)
                total -= offset
            if mixture is None:
                new, info = solve_day(p, total, sd, share, 0.0, tau_scale * tau[airport], settings)
            else:
                new, info = shift_prior.solve_day_mix(p, total, sd, share, tau_scale * tau[airport], mixture["pi"], mixture["k"], settings)
            out.append(pd.DataFrame({ID: day[ID].to_numpy(), "new": np.maximum(new, -12.0), "shift": info["shift_s"]}))
            infos.append({"airport": airport, "bday": bday, "dn": dn, "noise_sd_s": sd, "wide_share": share, **info})
    target = OUT / tag_of(variant, rows)
    target.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(target / f"{month}.parquet", index=False)
    pd.DataFrame(infos).to_parquet(target / f"{month}_days.parquet", index=False)
    write_json(target / f"{month}_settings.json", {"created_at_utc": utc_now(), "variant": variant, "rows": rows, "month": month, "v48_noise_sd_s": noise, "v48_noise_wide_share": wide, "tau_s": tau,
                                                   "published_figure_model": {a: {**{k: v for k, v in m.items() if k != "sign"}, "sign": {str(s): v for s, v in m["sign"].items()}}
                                                                              for a, m in model.items()} if model else None,
                                                   "shift_mixture": mixture, "seconds": time.time() - started})
    print(tag_of(variant, rows), month, "airport-days", len(infos), "seconds", round(time.time() - started, 1), flush=True)
    return target / f"{month}.parquet"


def values(source: Path, months: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(source / f"{m}.parquet", columns=[ID, "new"]) for m in months], ignore_index=True)


def report(candidate: Path, control: Path, rows: str = "stack") -> dict:
    """Changes of the candidate run against the control run on the same rows (both applied to the same stack)."""
    months = MONTHS12 if rows == "twelve" else MONTHS
    t = rows12_table() if rows == "twelve" else rows_table()
    if "extra" in t:
        t = t[~t["extra"]].reset_index(drop=True)
    t = t.merge(values(control, months).rename(columns={"new": "ctl"}), on=ID, how="left").merge(values(candidate, months), on=ID, how="left")
    t["ctl"], t["new"] = t["ctl"].fillna(t["stack"]), t["new"].fillna(t["stack"])
    out = {"created_at_utc": utc_now(), "candidate": candidate.relative_to(ROOT).as_posix(), "control": control.relative_to(ROOT).as_posix(), "rows": rows, "months": {}, "airports": {}}
    for month, g in t.groupby("month"):
        y, c, n = g["label"].to_numpy(dtype=float), g["ctl"].to_numpy(dtype=float), g["new"].to_numpy(dtype=float)
        o = ((g["label"] > 0) & (g["label"] <= 7200)).to_numpy()
        out["months"][month] = {"control_rmse": rmse(y, c), "rmse": rmse(y, n), "change_pct": 100 * (rmse(y, n) / rmse(y, c) - 1),
                                "ordinary_rows_change_pct": 100 * (rmse(y[o], n[o]) / rmse(y[o], c[o]) - 1), "rows_changed_over_1s": int((np.abs(n - c) > 1).sum()),
                                "alignment": float(-np.sum((n - c) * (c - y)) / max(np.sum((n - c) ** 2), 1e-9)),
                                "paired_days": paired_days(g["bday"].to_numpy(), y, n, c, SEED)}
        e = out["months"][month]
        print(month, round(e["control_rmse"], 4), "->", round(e["rmse"], 4), f"{e['change_pct']:+.4f}%", "ordinary", f"{e['ordinary_rows_change_pct']:+.4f}%",
              "interval", [round(v, 3) for v in e["paired_days"]["rmse_difference_95pct_interval_seconds"]], "days better", e["paired_days"]["days_with_lower_squared_error"])
    t["se_c"], t["se_n"] = (t["ctl"] - t["label"]) ** 2, (t["new"] - t["label"]) ** 2
    by = t.groupby(["airport", "month"]).agg(c=("se_c", "sum"), n=("se_n", "sum"))
    pct = (100 * (by["n"] / by["c"] - 1)).unstack("month")
    out["airports"] = {a: {m: float(v) for m, v in r.items()} for a, r in pct.iterrows()}
    out["airport_month_cells"] = {"better": int((pct < 0).sum().sum()), "worse": int((pct > 0).sum().sum())}
    print(pct.round(3).to_string())
    return out


def govern(path: str) -> None:
    decl = read_json(ROOT / path)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared, or does not respect its own contract.")
    target = ROOT / decl["report"]
    if target.exists():
        raise ValueError("Preserve the original result.")
    read = decl["development_read"]
    control = ROOT / read["control_run"]
    if not all((control / f"{m}.parquet").exists() for m in MONTHS):
        raise ValueError("The control run is not complete.")
    for m in MONTHS:
        run_month(m, "figure", "stack")
    r = report(OUT / "figure", control, "stack")
    equal = read["equal_footing"]
    changes = [r["months"][m]["change_pct"] for m in equal]
    below = [m for m in equal if r["months"][m]["paired_days"]["rmse_difference_95pct_interval_seconds"][1] < 0]
    accepted = bool(all(c < 0 for c in changes) and float(np.mean(changes)) <= -100 * read["minimum_mean_improvement_fraction"] and len(below) == len(equal))
    r.update({"declaration": path, "declaration_sha256": object_hash(decl), "audit_used": False, "official_scores_used": "none", "equal_footing_months": equal,
              "mean_change_pct_equal_footing": float(np.mean(changes)), "intervals_below_zero": below, "accepted": accepted,
              "control_files": {m: sha256_file(control / f"{m}.parquet") for m in MONTHS}, "candidate_files": {m: sha256_file(OUT / "figure" / f"{m}.parquet") for m in MONTHS},
              "settings_per_month": {m: read_json(OUT / "figure" / f"{m}_settings.json") for m in MONTHS},
              "workbook_sha256": sha256_file(ROOT / "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.xlsx"), "rows_sha256": sha256_file(ROOT / "runs/daily_totals/rows.parquet"),
              "code": {c: sha256_file(ROOT / c) for c in CODE}})
    write_json(target, r)
    print({"accepted": accepted, "mean_change_pct_equal_footing": round(r["mean_change_pct_equal_footing"], 4), "intervals_below_zero": below})


def check12(path: str) -> None:
    decl = read_json(ROOT / path)
    target = ROOT / decl["check12"]
    if target.exists():
        raise ValueError("Preserve the original result.")
    for variant in ("v48", "figure"):
        for m in MONTHS12:
            if not (OUT / tag_of(variant, "twelve") / f"{m}.parquet").exists():
                run_month(m, variant, "twelve")
    r = report(OUT / tag_of("figure", "twelve"), OUT / tag_of("v48", "twelve"), "twelve")
    r.update({"declaration": path, "declaration_sha256": object_hash(decl), "months_better": int(sum(v["change_pct"] < 0 for v in r["months"].values())),
              "mean_change_pct": float(np.mean([v["change_pct"] for v in r["months"].values()])), "official_scores_used": "none",
              "rows12_sha256": sha256_file(ROOT / "runs/daily_totals/rows12.parquet"), "code": {c: sha256_file(ROOT / c) for c in CODE}})
    write_json(target, r)
    print({"months_better": r["months_better"], "mean_change_pct": round(r["mean_change_pct"], 4)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month")
    parser.add_argument("--variant", default="figure")
    parser.add_argument("--rows", default="stack", choices=["stack", "twelve"])
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--check12", action="store_true")
    parser.add_argument("--declaration")
    args = parser.parse_args()
    if args.score:
        govern(args.declaration)
    elif args.check12:
        check12(args.declaration)
    elif args.report:
        control = V48_RUN if args.rows == "stack" else OUT / tag_of("v48", "twelve")
        report(OUT / tag_of(args.variant, args.rows), control, args.rows)
    else:
        run_month(args.month, args.variant, args.rows)


if __name__ == "__main__":
    main()
