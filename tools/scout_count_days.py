"""Count-day layer: the published daily departure counts as a constraint on each departure's off-block day
(src/prc2026_scout/count_days.py, configs/scout_count_days_v1.json).

  --exactness                 2025 count exactness per airport -> docs/scout/count_exactness.json
  --dry-dev                   label-free run on the four development months (candidates, posteriors, moves; no score) -> runs/scout/count_days/dry_dev/
  --score --declaration FILE  the governing run on the development months against v48's procedure (runs/daily_totals/daily_totals_v1)
  --base FILE --out DIR       the layer on a ranking file (a dry run; never under runs/submissions)

Development month m: base values = v48's procedure (the declared run's values, the stack where it did not read the day); cells, the
published figure's noise and tau from the other three development months (as run_daily_totals.run_month). Ranking: cells and scales
from the four development months (as export_daily_totals).
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
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_daily.reconcile import Settings  # noqa: E402
from prc2026_scout import count_days as cd  # noqa: E402
from run_daily_totals import Cells, add_keys, rows_table, scales  # noqa: E402

TABLE = ROOT / "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.parquet"
IFR_ONLY = ("EHAM", "LSZH", "LEBL")
DEV = ["2025-01", "2025-07", "2025-11", "2025-12"]
V48_DEV = ROOT / "runs/daily_totals/daily_totals_v1"
OUT = ROOT / "runs/scout/count_days"


def published_maps() -> tuple[dict, dict]:
    p = pd.read_parquet(TABLE)
    p["day"] = pd.to_datetime(p["FLT_DATE"]).dt.strftime("%Y-%m-%d")
    n = {(a, d): float(v) for a, d, v in zip(p["APT_ICAO"], p["day"], p["FLT_DEP_IFR_2"])}
    s = {(a, d): float(v) * 60 for a, d, v in zip(p["APT_ICAO"], p["day"], p["DLY_ALL_PRE_2"])}
    return n, s


def raw_clocks(files: list[Path]) -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(f, columns=[ID, "PHASE_mvt", "ADEP_mvt", "FLIGHT_RULE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]) for f in files], ignore_index=True)
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
    raw["in_figure"] = ~(raw["ADEP_mvt"].isin(IFR_ONLY) & raw["FLIGHT_RULE_mvt"].ne("I"))
    return raw.set_index(ID)


def exactness() -> pd.DataFrame:
    raw = raw_clocks(sorted((ROOT / "data/raw").glob("training_2025-*.parquet")))
    p = pd.read_parquet(TABLE)
    p["day"] = pd.to_datetime(p["FLT_DATE"]).dt.strftime("%Y-%m-%d")
    published = p.set_index(["APT_ICAO", "day"])["FLT_DEP_IFR_2"].astype(float).rename_axis(["airport", "day"])
    dep = pd.DataFrame({"airport": raw["ADEP_mvt"].to_numpy(), "block": raw["BLOCK_TIME_UTC_mvt"].to_numpy(), "in_figure": raw["in_figure"].to_numpy()})
    return cd.count_exactness(dep, published)


def month_days(month: str) -> list[str]:
    start = pd.Timestamp(month + "-01")
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D")]


def run_frame(t: pd.DataFrame, cells: Cells, noise: dict, wide: dict, tau: dict, exact: pd.DataFrame, settings: Settings) -> tuple[np.ndarray, list]:
    """t: one month's rows with takeoff, a, value, in_figure, airport, month and the cell keys. New values (aligned with t) and reports."""
    n_pub, pub_total = published_maps()
    new = t["value"].to_numpy(dtype=float).copy()
    reports = []
    for airport in sorted(set(t["airport"]) & set(exact.index[exact["exact"]])):
        for month in sorted(set(t.loc[t["airport"].eq(airport), "month"])):
            sel = np.flatnonzero((t["airport"].eq(airport) & t["month"].eq(month) & t["in_figure"]).to_numpy())
            f = add_keys(t.iloc[sel].assign(stack=lambda x: x["value"]).reset_index(drop=True))
            days = month_days(month)
            npub = {d: n_pub[(airport, d)] for d in days if (airport, d) in n_pub}
            ptot = {d: pub_total[(airport, d)] for d in days if (airport, d) in pub_total}
            complete = {d for d in days[:-1] if d in npub}
            started = time.time()
            values, rep = cd.solve_month(f, cells.resolve(f), days, npub, complete, float(exact.loc[airport, "eps"]), noise[airport], wide[airport], tau[airport],
                                         cells.resolve, ptot, settings)
            new[sel] = values
            for m in rep.get("flagged_candidates", []):
                r = f.iloc[m["position"]]
                m.update({ID: r[ID], "takeoff": str(r["takeoff"]), "early_day": days[m["k"] - 1] if m["k"] > 0 else None, "late_day": days[m["k"]] if m["k"] < len(days) else None})
            for m in rep.get("moved", []):
                r = f.iloc[m["position"]]
                m.update({ID: r[ID], "takeoff": str(r["takeoff"]), "a": float(r["a"]), "value": float(r["value"]), "new": float(values[m["position"]]), "cls": r["cls"]})
            reports.append({"airport": airport, "month": month, "seconds": round(time.time() - started, 1), **rep})
            print(airport, month, rep["status"], "candidates", rep.get("candidates"), "heavy", rep.get("heavy"), "changed", int((np.abs(values - f["value"].to_numpy()) > 0.5).sum()), flush=True)
    return new, reports


def dev_tables(month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows_table()
    fit = add_keys(rows[rows["month"].ne(month)])
    t = rows[rows["month"].eq(month)].reset_index(drop=True)
    v48 = pd.read_parquet(V48_DEV / f"{month}.parquet").set_index(ID)["new"]
    t["value"] = v48.reindex(t[ID]).fillna(pd.Series(t["stack"].to_numpy(), index=t[ID])).to_numpy()
    raw = raw_clocks(sorted((ROOT / "data/raw").glob(f"training_{month}-*.parquet"))).reindex(t[ID])
    t["takeoff"], t["in_figure"] = raw["MVT_TIME_UTC_mvt"].to_numpy(), raw["in_figure"].to_numpy(dtype=bool)
    return t, fit


def dev_run(tag: str, settings: Settings) -> dict:
    exact = exactness()
    out = {"exactness": exact.reset_index().to_dict(orient="records"), "months": {}}
    target = OUT / tag
    target.mkdir(parents=True, exist_ok=True)
    for month in DEV:
        t, fit = dev_tables(month)
        cells = Cells(fit)
        noise, wide, tau = scales(fit)
        new, reports = run_frame(t, cells, noise, wide, tau, exact, settings)
        pd.DataFrame({ID: t[ID], "base": t["value"], "new": new}).to_parquet(target / f"{month}.parquet", index=False)
        out["months"][month] = reports
    write_json(target / "reports.json", out)
    return out


def score(declaration: str) -> None:
    decl = read_json(ROOT / declaration)
    settings = Settings()
    run = dev_run("governing", settings)
    rows = rows_table()
    rng = np.random.default_rng(20260929)
    result = {"declaration": declaration, "declaration_sha256": object_hash(decl), "created_at_utc": utc_now(), "months": {}}
    for month in DEV:
        v = pd.read_parquet(OUT / "governing" / f"{month}.parquet")
        t = rows[rows["month"].eq(month)][[ID, "label", "airport", "bday"]].merge(v, on=ID)
        se0, se1 = (t["label"] - t["base"]) ** 2, (t["label"] - np.round(t["new"])) ** 2
        rmse0, rmse1 = float(np.sqrt(se0.mean())), float(np.sqrt(se1.mean()))
        day = t.assign(d0=se0, d1=se1).groupby("bday")[["d0", "d1"]].sum()
        boot = []
        for _ in range(2000):
            s = day.iloc[rng.integers(0, len(day), len(day))]
            boot.append(100 * (np.sqrt(s["d1"].sum() / s["d0"].sum()) - 1))
        changed = (np.abs(t["new"] - t["base"]) > 0.5)
        result["months"][month] = {"rows": int(len(t)), "rmse_v48": rmse0, "rmse_layer": rmse1, "change_pct": 100 * (rmse1 / rmse0 - 1),
                                   "interval_pct": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "rows_changed": int(changed.sum()),
                                   "sse_change": float(se1.sum() - se0.sum()), "by_airport": {a: float(((g["label"] - np.round(g["new"])) ** 2).sum() - ((g["label"] - g["base"]) ** 2).sum())
                                                                                          for a, g in t[changed].groupby("airport")}}
        print(month, result["months"][month]["change_pct"], result["months"][month]["interval_pct"], "changed", int(changed.sum()), flush=True)
    # calibration: every candidate of a flagged midnight, the posterior's choice of day against the recorded (minute-rounded) off-block day
    raw = raw_clocks(sorted(p for m in DEV for p in (ROOT / "data/raw").glob(f"training_{m}-*.parquet")))
    cal = []
    for month, reports in run["months"].items():
        for rep in reports:
            for c in rep.get("flagged_candidates", []):
                block = raw.loc[c[ID], "BLOCK_TIME_UTC_mvt"]
                true_day = cd.minute_day(pd.Series([block]))[0]
                cal.append({**{k: c[k] for k in ["p_early_prior", "p_early_post", "heavy", "big", "value_early", "early_day", "late_day"]}, "airport": rep["airport"], "month": month,
                            "true_early": true_day == c["early_day"], "true_late": true_day == c["late_day"]})
    cal = pd.DataFrame(cal)
    if len(cal):
        cal["choice_early"] = cal["p_early_post"] > 0.5
        cal["confidence"] = np.where(cal["choice_early"], cal["p_early_post"], 1 - cal["p_early_post"])
        cal["right"] = np.where(cal["choice_early"], cal["true_early"], cal["true_late"])
        cal["crossed"] = cal["choice_early"] != cal["value_early"]
        cal["band"] = pd.cut(cal["confidence"], [0.5, 0.7, 0.9, 0.99, 1.0001], include_lowest=True).astype(str)
        result["calibration"] = {"candidates": int(len(cal)), "neither_day": int((~cal["true_early"] & ~cal["true_late"]).sum()),
                                 "by_confidence": cal.groupby("band").agg(n=("right", "size"), right=("right", "mean"), mean_conf=("confidence", "mean")).reset_index().to_dict(orient="records"),
                                 "crossed": {"n": int(cal["crossed"].sum()), "right": int(cal.loc[cal["crossed"], "right"].sum()),
                                             "big_n": int((cal["crossed"] & cal["big"]).sum()), "big_right": int(cal.loc[cal["crossed"] & cal["big"], "right"].sum())},
                                 "by_airport": cal.groupby("airport").agg(n=("right", "size"), right=("right", "mean"), crossed=("crossed", "sum")).reset_index().to_dict(orient="records")}
        print("calibration", result["calibration"]["by_confidence"], result["calibration"]["crossed"], flush=True)
    write_json(ROOT / decl["report"], result)


def ranking(base: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
        raise ValueError("A dry run never writes under runs/submissions.")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = pd.read_parquet(base)
    raw = raw_clocks([ROOT / "data/raw/ranking.parquet"])
    import export_daily_totals as edt
    t = edt.ranking_table((Path(base).resolve().relative_to(ROOT).as_posix(), sha256_file(base)))
    t["value"] = t["stack"].to_numpy(dtype=float)
    r = raw.reindex(t[ID])
    t["takeoff"], t["in_figure"] = r["MVT_TIME_UTC_mvt"].to_numpy(), r["in_figure"].to_numpy(dtype=bool)
    fit = add_keys(rows_table())
    cells = Cells(fit)
    noise, wide, tau = scales(fit)
    new, reports = run_frame(t, cells, noise, wide, tau, exactness(), Settings())
    values = pd.Series(np.round(new), index=t[ID])
    out = sub.copy()
    out[TARGET] = values.reindex(out[ID]).to_numpy()
    changed = out[TARGET].to_numpy() != sub[TARGET].to_numpy()
    path = out_dir / base.name.replace(".parquet", "_count_days.parquet")
    out.to_parquet(path, index=False)
    write_json(out_dir / "count_days_ranking.json", {"created_at_utc": utc_now(), "base": str(base), "base_sha256": sha256_file(base), "output": str(path), "output_sha256": sha256_file(path),
                                                       "rows_changed": int(changed.sum()), "sum_sq_change": float(((out[TARGET] - sub[TARGET])[changed] ** 2).sum()), "reports": reports})
    print("rows changed", int(changed.sum()), "sum of squared change", float(((out[TARGET] - sub[TARGET])[changed] ** 2).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exactness", action="store_true")
    ap.add_argument("--dry-dev", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--declaration")
    ap.add_argument("--base", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.exactness:
        e = exactness()
        print(e.round(4).to_string())
        write_json(ROOT / "docs/scout/count_exactness.json", {"created_at_utc": utc_now(), "rule": f"share >= {cd.EXACT_MIN_SHARE}", "airports": e.reset_index().to_dict(orient="records")})
    if args.dry_dev:
        dev_run("dry_dev", Settings())
    if args.score:
        score(args.declaration)
    if args.base is not None:
        ranking(args.base, args.out)


if __name__ == "__main__":
    main()
