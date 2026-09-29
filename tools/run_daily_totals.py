"""Reconciliation of the v46 development stack with the published daily total of positive off-block delay (src/prc2026_daily).

Development months only (2025-01, 2025-07, 2025-11, 2025-12). For the month read, the cells (recorded outcomes of comparable rows), the
published figure's accuracy per airport and the day-shift scale tau per airport come from the OTHER three months.

  cell        class (airport x matched; LIRF fill population) x schedule-interval band x prediction band; pooled over airports, then over
              prediction bands, where a cell holds fewer than MIN_CELL rows; at most MAX_SAMPLES outcomes per cell (seeded subsample)
  airport-day read only when our rows cover at least 90% of the departures behind the published figure

  --month 2025-07 [--tag NAME] [--tau-scale 1.0] [--heavy 900]   -> runs/daily_totals/<tag>/<month>.parquet
  --rows twelve   the independent check: all twelve months of 2025 with the out-of-sample v9 + residual stage (v23's recipe,
                  scratch/v24 + runs/adsb_v25/dev) in place of the v46 stack; cells from the other eleven months. The cross-fit tables
                  leave out 14,092 departures of the eight non-recipe months (0.7%, mean positive delay 66 min); they enter their
                  airport-day with the airport-month's median prediction ("extra" rows), are never scored and never enter a cell.
  --components tracked   two day shifts per airport-day, one for the tracked rows (ADS-B surface match) and one for the others, jointly
                  Gaussian with the airport's scales and correlation from the fitting months (component_scales); one published total
  --centres PATH  a JSON table {"AIRPORT|MONTH": {"mean_s", "sigma_s"}}: in those airport-months the day shift has the prior
                  N(mean_s, tau_within^2 + sigma_s^2), tau_within = the day shift's sd about its month's mean from the fitting months
  --report [--tag NAME] [--control TAG]                          -> prints month / airport changes; runs/daily_totals/<tag>/report.json
                  with --control the changes are against the control run's values, not against the stack
  --score --declaration configs/daily_totals_v1.json             the governing run of a declaration -> its "report" path
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
from prc2026_daily.reconcile import Settings, month_prior, prepare_day, shift_estimate, solve_day  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402

MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
ROWS = ROOT / "runs/daily_totals/rows.parquet"
PRED_BANDS = [-1e9, 1800.0, 3600.0, 7200.0, 1e9]
MIN_CELL, MAX_SAMPLES = 150, 1500
SEED = 20260929
COMPONENT_MIN_ROWS, COMPONENT_MIN_DAYS, COMPONENT_MIN_TAU_S, COMPONENT_MAX_CORRELATION = 10, 20, 5.0, 0.95


def rows_table() -> pd.DataFrame:
    """Development rows with the stack's value, clocks and the published total; built once."""
    if ROWS.exists():
        return pd.read_parquet(ROWS)
    from screen_daily_totals import published, stack_rows
    t = stack_rows().join(published(), on=["airport", "bday"])
    t["complete"] = (t["pub"].notna() & (t.groupby(["airport", "bday"])["a"].transform("size") >= 0.9 * t["n_pub"])).to_numpy()
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(ROWS, index=False)
    return t


def rows12_table() -> pd.DataFrame:
    """All twelve months of 2025 with the out-of-sample v9 + mean(adj2, cat) residual as the prediction; built once."""
    path = ROOT / "runs/daily_totals/rows12.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from screen_daily_totals import BANDS, published
    frames = []
    for m in range(1, 13):
        month = f"2025-{m:02}"
        base = pd.read_parquet(ROOT / f"scratch/v24/{month}.parquet", columns=[ID, "airport", "TAXITIME_SEC_mvt", "v9", "population"])
        res = pd.read_parquet(ROOT / f"runs/adsb_v25/dev/predictions_{month}.parquet", columns=[ID, "residual:adj2", "residual:cat"]).set_index(ID).reindex(base[ID])
        residual = 0.5 * (res["residual:adj2"].to_numpy(dtype=float) + res["residual:cat"].to_numpy(dtype=float))
        matched = np.isfinite(residual) & (residual != 0) & base["airport"].ne("LIRF").to_numpy()
        raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
                         for p in sorted((ROOT / "data/raw").glob(f"training_{month}-*.parquet"))], ignore_index=True)
        raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(base[ID])
        clocks = {c: pd.to_datetime(raw[c], utc=True).reset_index(drop=True) for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]}
        f = pd.DataFrame({ID: base[ID].to_numpy(), "month": month, "airport": base["airport"].to_numpy(), "label": base["TAXITIME_SEC_mvt"].to_numpy(dtype=float), "matched": matched,
                          "stack": np.maximum(base["v9"].to_numpy(dtype=float) + np.where(matched, residual, 0.0), -12.0), "population": base["population"].to_numpy(dtype=bool)})
        f["a"] = (clocks["MVT_TIME_UTC_mvt"] - clocks["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()
        f["block_day"] = clocks["BLOCK_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy()
        f["bday"] = (clocks["MVT_TIME_UTC_mvt"] - pd.to_timedelta(f["stack"], unit="s")).dt.strftime("%Y-%m-%d").to_numpy()
        frames.append(f)
    t = pd.concat(frames, ignore_index=True).dropna(subset=["a", "label"]).reset_index(drop=True)
    t["extra"] = False
    raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
                     for p in sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))], ignore_index=True)
    raw = raw[raw["PHASE_mvt"].eq("DEP") & ~raw[ID].isin(t[ID])].drop_duplicates(ID).reset_index(drop=True)
    clocks = {c: pd.to_datetime(raw[c], utc=True) for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]}
    x = pd.DataFrame({ID: raw[ID].to_numpy(), "month": clocks["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy(), "airport": raw["ADEP_mvt"].to_numpy(),
                      "label": raw["TAXITIME_SEC_mvt"].to_numpy(dtype=float), "matched": False, "population": False, "extra": True})
    x["stack"] = pd.MultiIndex.from_frame(x[["airport", "month"]]).map(t.groupby(["airport", "month"])["stack"].median()).to_numpy(dtype=float)
    x["a"] = (clocks["MVT_TIME_UTC_mvt"] - clocks["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()
    x["block_day"] = clocks["BLOCK_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy()
    x["bday"] = (clocks["MVT_TIME_UTC_mvt"] - pd.to_timedelta(x["stack"], unit="s")).dt.strftime("%Y-%m-%d").to_numpy()
    t = pd.concat([t, x.dropna(subset=["a", "label", "stack"])], ignore_index=True)
    t["pos_true"], t["pos_hat"] = np.maximum(t["a"] - t["label"], 0.0), np.maximum(t["a"] - t["stack"], 0.0)
    t["r"] = t["label"] - t["stack"]
    t["se"] = t["r"] ** 2
    t["cls"] = np.where(t["airport"].eq("LIRF") & t["population"], "LIRF_pop", np.where(t["matched"], t["airport"] + "_m", t["airport"] + "_u"))
    t["band"] = pd.cut(t["a"] / 3600, BANDS).astype(str)
    t = t.join(published(), on=["airport", "bday"])
    t["complete"] = (t["pub"].notna() & (t.groupby(["airport", "bday"])["a"].transform("size") >= 0.9 * t["n_pub"])).to_numpy()
    t.to_parquet(path, index=False)
    return t


def table(rows: str) -> pd.DataFrame:
    """stack (the development stack, four months), twelve (the independent check's table) or the path of a table of that form."""
    if rows.endswith(".parquet"):
        return pd.read_parquet(ROOT / rows)
    return rows12_table() if rows == "twelve" else rows_table()


def add_keys(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["group"] = np.where(t["cls"].eq("LIRF_pop"), "LIRF_pop", np.where(t["airport"].eq("LIRF"), "LIRF", "other") + np.where(t["matched"], "_m", "_u"))
    t.loc[t["cls"].eq("LIRF_pop"), "group"] = "LIRF_pop"
    t["pband"] = pd.cut(t["stack"], PRED_BANDS).astype(str)
    return t


class Cells:
    """Recorded outcomes by cell, with the pooling chain."""

    LEVELS = (["cls", "band", "pband"], ["group", "band", "pband"], ["group", "band"], ["group"])

    def __init__(self, fit: pd.DataFrame):
        rng = np.random.default_rng(SEED)
        self.tables = []
        for level in self.LEVELS:
            table = {}
            for key, g in fit.groupby(level, sort=False):
                if len(g) < MIN_CELL and level != self.LEVELS[-1]:
                    continue
                take = g if len(g) <= MAX_SAMPLES else g.iloc[np.sort(rng.choice(len(g), MAX_SAMPLES, replace=False))]
                table[key if isinstance(key, tuple) else (key,)] = (take["a"].to_numpy(dtype=float), take["stack"].to_numpy(dtype=float), take["label"].to_numpy(dtype=float))
            self.tables.append(table)

    def resolve(self, day: pd.DataFrame) -> list[tuple[np.ndarray, tuple]]:
        """Partition of the day's row positions by the first level of the chain that holds their cell."""
        left = np.arange(len(day))
        groups = []
        for level, table in zip(self.LEVELS, self.tables):
            if not len(left):
                break
            keys = list(map(tuple, day.iloc[left][level].to_numpy()))
            found: dict[tuple, list[int]] = {}
            rest = []
            for position, key in zip(left, keys):
                if key in table:
                    found.setdefault(key, []).append(position)
                else:
                    rest.append(position)
            groups.extend((np.array(v), table[k]) for k, v in found.items())
            left = np.array(rest, dtype=int)
        if len(left):
            raise ValueError("A row has no cell at any level.")
        return groups


def robust_sd(v: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def scales(fit: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per airport, from the fitting months: the published figure's robust error sd (s), the share of airport-days off by more than
    ten minutes (the wide noise components' share, at least 0.02) and the day-shift scale tau (s)."""
    own = fit.groupby(["airport", "block_day"]).agg(true=("pos_true", "sum"), rows=("a", "size")).rename_axis(["airport", "bday"])
    pub = fit.groupby(["airport", "bday"]).agg(pub=("pub", "first"), n_pub=("n_pub", "first"))
    own = own.join(pub, how="inner").dropna()
    own = own[own["rows"] >= 0.9 * own["n_pub"]]
    noise = {a: max(robust_sd((g["pub"] - g["true"]).to_numpy()), 30.0) for a, g in own.groupby(level="airport")}
    wide = {a: max(float(((g["pub"] - g["true"]).abs() > 600).mean()), 0.02) for a, g in own.groupby(level="airport")}
    o = fit[fit["r"].abs() <= 600]
    day = o.groupby(["airport", "bday"])["r"].agg(["mean", "var", "size"])
    day = day[day["size"] >= 100]
    tau = {a: float(np.sqrt(max(g["mean"].var() - (g["var"] / g["size"]).mean(), 25.0))) for a, g in day.groupby(level="airport")}
    return noise, wide, tau


def within_month_tau(fit: pd.DataFrame) -> dict:
    """Per airport, from the fitting months: the sd (s) of the day shift about its month's mean; as tau, the months' means removed."""
    o = fit[fit["r"].abs() <= 600]
    day = o.groupby(["airport", "month", "bday"])["r"].agg(["mean", "var", "size"])
    day = day[day["size"] >= 100]
    out = {}
    for airport, g in day.groupby(level="airport"):
        centred = g["mean"] - g.groupby(level="month")["mean"].transform("mean")
        freedom = len(g) - g.index.get_level_values("month").nunique()
        out[airport] = float(np.sqrt(max(float((centred ** 2).sum()) / max(freedom, 1) - float((g["var"] / g["size"]).mean()), 25.0)))
    return out


def component_scales(fit: pd.DataFrame, pooled: dict) -> dict:
    """Per airport, from the fitting months: the covariance (s^2) of the day shifts of the tracked rows (component 0) and of the
    others (component 1). Scales as the pooled tau (airport-day means of residuals within 600 s, sampling variance removed, at least
    5 s), over the airport-days that hold at least COMPONENT_MIN_ROWS rows of the component; the covariance from the airport-days
    that hold both, the correlation kept inside [0, 0.95]. A component with fewer than COMPONENT_MIN_DAYS such days takes the
    airport's pooled tau; without enough days that hold both, the correlation is the median of the airports that have one."""
    o = fit[fit["r"].abs() <= 600]
    found = {}
    for airport, g in o.groupby("airport"):
        d = g.groupby(["bday", "matched"])["r"].agg(["mean", "var", "size"])
        e = {}
        for k, flag in enumerate((True, False)):
            x = d.xs(flag, level="matched") if flag in d.index.get_level_values("matched") else d.iloc[:0]
            x = x[x["size"] >= COMPONENT_MIN_ROWS]
            e[k] = x
        tau = [float(np.sqrt(max(x["mean"].var() - (x["var"] / x["size"]).mean(), COMPONENT_MIN_TAU_S ** 2))) if len(x) >= COMPONENT_MIN_DAYS else None for x in (e[0], e[1])]
        both = e[0][["mean"]].join(e[1][["mean"]], how="inner", lsuffix="_0", rsuffix="_1")
        rho = None
        if len(both) >= COMPONENT_MIN_DAYS and tau[0] is not None and tau[1] is not None:
            rho = float(np.clip(np.cov(both["mean_0"], both["mean_1"])[0, 1] / (tau[0] * tau[1]), 0.0, COMPONENT_MAX_CORRELATION))
        found[airport] = {"tau_s": tau, "correlation": rho, "days": [int(len(e[0])), int(len(e[1])), int(len(both))]}
    known = [v["correlation"] for v in found.values() if v["correlation"] is not None]
    fallback = float(np.median(known)) if known else 0.0
    out = {}
    for airport, v in found.items():
        tau = [pooled[airport] if s is None else s for s in v["tau_s"]]
        rho = fallback if v["correlation"] is None else v["correlation"]
        out[airport] = {"tau_s": tau, "correlation": rho, "days": v["days"], "estimated": [s is not None for s in v["tau_s"]] + [v["correlation"] is not None],
                        "cov": [[tau[0] ** 2, rho * tau[0] * tau[1]], [rho * tau[0] * tau[1], tau[1] ** 2]]}
    return out


def run_month(month: str, tag: str, tau_scale: float, settings: Settings, rows: str = "stack", prior: str = "fixed", components: str = "one", centres: str | None = None,
              airports: list[str] | None = None, placebo: int | None = None) -> None:
    t = add_keys(table(rows))
    if "extra" not in t:
        t["extra"] = False
    fit, use = t[t["month"].ne(month) & ~t["extra"]], t[t["month"].eq(month) & t["complete"]]
    if airports:  # the fitting side is unchanged; only these airports' days are read
        use = use[use["airport"].isin(airports)]
    cells = Cells(fit)
    noise, wide, tau = scales(fit)
    if components not in ("one", "tracked") or (components == "tracked" and prior != "fixed"):
        raise ValueError("Components: one or tracked; tracked with the fixed prior only.")
    by_component = component_scales(fit, tau) if components == "tracked" else {}
    if centres is not None and (components != "one" or prior != "fixed"):
        raise ValueError("Centres: with one component and the fixed prior only.")
    centre = read_json(Path(centres)) if centres is not None else {}
    within = within_month_tau(fit) if centre else {}
    out, infos = [], []
    started = time.time()
    priors = {}
    for airport, month_rows in use.groupby("airport", sort=True):
        days = []
        for bday, day in month_rows.groupby("bday", sort=True):
            day = day.reset_index(drop=True)
            split = {"component": (~day["matched"].to_numpy(dtype=bool)).astype(np.int64), "components": 2} if components == "tracked" else {}
            days.append((bday, day[ID].to_numpy(), float(day["pub"].iat[0]), prepare_day(day["a"].to_numpy(dtype=float), day["stack"].to_numpy(dtype=float), cells.resolve(day), settings, **split)))
        if placebo is not None:  # each day reads another day's discrepancy (total minus the total its predictions imply)
            order = np.random.default_rng(placebo).permutation(len(days))
            delta = [days[j][2] - days[j][3].reference_total for j in order]
            days = [(bday, ids, p.reference_total + d, p) for (bday, ids, _, p), d in zip(days, delta)]
        mean, sd = 0.0, tau_scale * tau[airport]
        if prior == "adaptive":  # from the month's own published totals and predictions; no label
            mean, sd = month_prior(np.array([shift_estimate(p, total) for _, _, total, p in days]),
                                   np.array([(p.var_bulk + noise[airport] ** 2) / max(p.n_eff, 1.0) ** 2 for _, _, _, p in days]))
            sd *= tau_scale
        if f"{airport}|{month}" in centre:
            c = centre[f"{airport}|{month}"]
            mean, sd = float(c["mean_s"]), tau_scale * float(np.sqrt(within[airport] ** 2 + float(c["sigma_s"]) ** 2))
        priors[airport] = {"mean_s": mean, "sd_s": sd}
        cov = tau_scale ** 2 * np.array(by_component[airport]["cov"]) if components == "tracked" else None
        for bday, ids, total, p in days:
            new, info = solve_day(p, total, noise[airport], wide[airport], mean, sd, settings, shift_cov=cov)
            shifts = info.pop("shifts_s", None)
            if shifts is not None:
                info.update({"shift_tracked_s": shifts[0], "shift_untracked_s": shifts[1], "n_eff_tracked": float(p.n_eff_k[0]), "n_eff_untracked": float(p.n_eff_k[1])})
            out.append(pd.DataFrame({ID: ids, "new": np.maximum(new, -12.0), "shift": info["shift_s"] if shifts is None else np.asarray(shifts)[p.component]}))
            infos.append({"airport": airport, "bday": bday, **info})
    target = ROOT / f"runs/daily_totals/{tag}"
    target.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(target / f"{month}.parquet", index=False)
    pd.DataFrame(infos).to_parquet(target / f"{month}_days.parquet", index=False)
    write_json(target / f"{month}_settings.json", {"created_at_utc": utc_now(), "month": month, "tau_scale": tau_scale, "settings": settings.__dict__, "prior": prior, "shift_priors": priors, "noise_sd_s": noise, "noise_wide_share": wide, "tau_s": tau,
                                                   "components": components, "component_scales": by_component, "centres": centres, "tau_within_s": within, "airports": airports, "placebo": placebo,
                                                   "min_cell": MIN_CELL, "max_samples": MAX_SAMPLES, "seconds": time.time() - started})
    print(month, "airport-days", len(infos), "seconds", round(time.time() - started, 1), flush=True)


def settings_from(declaration: dict, candidate: str | None = None) -> tuple[Settings, float, str]:
    """The declaration's settings; with a candidate, its declared values replace the common ones."""
    spec = dict(declaration["procedure"]["settings"])
    if candidate is not None:
        extra = declaration["candidates"][candidate]
        if set(extra) - {"heavy_temper", "note"}:
            raise ValueError("A candidate may set heavy_temper only.")
        spec.update({k: v for k, v in extra.items() if k != "note"})
    if spec["grid_s"] != 30.0 or spec["min_cell"] != MIN_CELL or spec["max_samples"] != MAX_SAMPLES or spec["seed"] != SEED or spec["complete_share"] != 0.9:
        raise ValueError("The declaration's constants differ from the code's.")
    return (Settings(heavy_sd_s=float(spec["heavy_sd_s"]), max_heavy=int(spec["max_heavy"]), heavy_updates=bool(spec["heavy_updates"]), tilt_sd_s=float(spec["tilt_sd_s"]),
                     noise_wide_sd_s=tuple(float(v) for v in spec["noise_wide_sd_s"]), heavy_temper=float(spec.get("heavy_temper", 1.0)), heavy_prior=str(spec.get("heavy_prior", "tilt")),
                     pin_s=float(spec.get("pin_s", 0.0))), float(spec["tau_scale"]), str(spec["prior"]))


def _month(args: tuple) -> None:
    run_month(*args)


def govern_candidates(path: str) -> None:
    """The governing run of a declaration with candidates: each candidate on the four development months against the control run,
    the declared rule per candidate, the selection among those that pass."""
    from concurrent.futures import ProcessPoolExecutor
    decl = read_json(ROOT / path)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared, or does not respect its own contract.")
    out = ROOT / decl["report"]
    if out.exists():
        raise ValueError("Preserve the original result.")
    read = decl["development_read"]
    control, equal = read["control_run"], read["equal_footing"]
    if not all((ROOT / f"runs/daily_totals/{control}/{m}.parquet").exists() for m in MONTHS):
        raise ValueError("The control run is not complete.")
    rows_table()
    jobs = []
    for name in decl["candidates"]:
        settings, tau_scale, prior = settings_from(decl, name)
        jobs += [(m, f"{decl['experiment_id']}_{name}", tau_scale, settings, "stack", prior) for m in MONTHS]
    with ProcessPoolExecutor(max_workers=int(decl.get("execution", {}).get("parallel_months", 2))) as pool:
        list(pool.map(_month, jobs))
    result = {"created_at_utc": utc_now(), "declaration": path, "declaration_sha256": object_hash(decl), "audit_used": False, "official_scores_used": "none", "equal_footing_months": equal,
              "control_run": {"tag": control, "files": {m: sha256_file(ROOT / f"runs/daily_totals/{control}/{m}.parquet") for m in MONTHS}}, "candidates": {},
              "workbook_sha256": sha256_file(ROOT / "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.xlsx"), "rows_sha256": sha256_file(ROWS),
              "code": {c: sha256_file(ROOT / c) for c in ["tools/run_daily_totals.py", "src/prc2026_daily/reconcile.py", "tools/screen_daily_totals.py", "tools/run_share_gate_reread.py"]}}
    for name in decl["candidates"]:
        tag = f"{decl['experiment_id']}_{name}"
        r = report(tag, "stack", control)
        changes = [r["months"][m]["change_pct"] for m in equal]
        below = [m for m in equal if r["months"][m]["paired_days"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        r.update({"mean_change_pct_equal_footing": float(np.mean(changes)), "intervals_below_zero": below,
                  "passes": bool(all(c < 0 for c in changes) and float(np.mean(changes)) <= -100 * read["minimum_mean_improvement_fraction"] and len(below) == len(equal)),
                  "settings_per_month": {m: read_json(ROOT / f"runs/daily_totals/{tag}/{m}_settings.json") for m in MONTHS}})
        result["candidates"][name] = r
    passing = [n for n, r in result["candidates"].items() if r["passes"]]
    result["selected"] = min(passing, key=lambda n: (result["candidates"][n]["mean_change_pct_equal_footing"], n)) if passing else None
    result["accepted"] = result["selected"] is not None
    write_json(out, result)
    print({"selected": result["selected"], "candidates": {n: {"mean_change_pct_equal_footing": round(r["mean_change_pct_equal_footing"], 4), "months": {m: round(r["months"][m]["change_pct"], 4) for m in MONTHS},
                                                                "intervals_below_zero": r["intervals_below_zero"], "passes": r["passes"]} for n, r in result["candidates"].items()}})


def govern(path: str) -> None:
    """The governing run of a declaration on the four development months, and its rule."""
    from concurrent.futures import ProcessPoolExecutor
    decl = read_json(ROOT / path)
    if decl.get("candidates"):
        return govern_candidates(path)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared, or does not respect its own contract.")
    out = ROOT / decl["report"]
    if out.exists():
        raise ValueError("Preserve the original result.")
    settings, tau_scale, prior = settings_from(decl)
    tag = decl["experiment_id"]
    control = decl["development_read"].get("control_run")   # a saved run (the declared procedure before this one); absent: the stack
    if control is not None and not all((ROOT / f"runs/daily_totals/{control}/{m}.parquet").exists() for m in MONTHS):
        raise ValueError("The control run is not complete.")
    rows_table()
    with ProcessPoolExecutor(max_workers=int(decl.get("execution", {}).get("parallel_months", 4))) as pool:
        list(pool.map(_month, [(m, tag, tau_scale, settings, "stack", prior) for m in MONTHS]))
    result = report(tag, "stack", control)
    equal = decl["development_read"]["equal_footing"]
    changes = [result["months"][m]["change_pct"] for m in equal]
    below = [m for m in equal if result["months"][m]["paired_days"]["rmse_difference_95pct_interval_seconds"][1] < 0]
    accepted = bool(all(c < 0 for c in changes) and float(np.mean(changes)) <= -100 * decl["development_read"]["minimum_mean_improvement_fraction"] and len(below) == len(equal))
    result.update({"declaration": path, "declaration_sha256": object_hash(decl), "audit_used": False, "official_scores_used": "none", "equal_footing_months": equal,
                   "mean_change_pct_equal_footing": float(np.mean(changes)), "intervals_below_zero": below, "accepted": accepted,
                   "workbook_sha256": sha256_file(ROOT / "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.xlsx"), "rows_sha256": sha256_file(ROWS),
                   "settings_per_month": {m: read_json(ROOT / f"runs/daily_totals/{tag}/{m}_settings.json") for m in MONTHS},
                   "control_run": {"tag": control, "files": {m: sha256_file(ROOT / f"runs/daily_totals/{control}/{m}.parquet") for m in MONTHS}} if control is not None else None,
                   "code": {c: sha256_file(ROOT / c) for c in ["tools/run_daily_totals.py", "src/prc2026_daily/reconcile.py", "tools/screen_daily_totals.py", "tools/run_share_gate_reread.py"]}})
    write_json(out, result)
    print({"accepted": accepted, "mean_change_pct_equal_footing": round(result["mean_change_pct_equal_footing"], 4), "intervals_below_zero": below})


def report(tag: str, rows: str = "stack", control: str | None = None) -> dict:
    """Changes of the run `tag` against the stack, or against the run `control` (then "stack" below reads "control")."""
    t = table(rows)
    if "extra" in t:
        t = t[~t["extra"]].reset_index(drop=True)
    months = sorted(t["month"].unique())
    target = ROOT / f"runs/daily_totals/{tag}"
    if control is not None:
        source = ROOT / f"runs/daily_totals/{control}"
        if any((target / f"{m}.parquet").exists() and not (source / f"{m}.parquet").exists() for m in months):
            raise ValueError("The control run lacks a month of the run.")
        base = pd.concat([pd.read_parquet(source / f"{m}.parquet", columns=[ID, "new"]) for m in months if (source / f"{m}.parquet").exists()], ignore_index=True).rename(columns={"new": "control"})
        t = t.merge(base, on=ID, how="left")
        t["stack"] = t["control"].fillna(t["stack"])
        t["se"] = (t["stack"] - t["label"]) ** 2
    new = pd.concat([pd.read_parquet(target / f"{m}.parquet") for m in months if (target / f"{m}.parquet").exists()], ignore_index=True)
    t = t.merge(new, on=ID, how="left")
    t["new"] = t["new"].fillna(t["stack"])
    t["se_new"] = (t["new"] - t["label"]) ** 2
    out = {"created_at_utc": utc_now(), "tag": tag, "control": control or "the stack", "months": {}, "airports": {}}
    for month, g in t.groupby("month"):
        if not (target / f"{month}.parquet").exists():
            continue
        y, before, after = g["label"].to_numpy(dtype=float), g["stack"].to_numpy(dtype=float), g["new"].to_numpy(dtype=float)
        o = ((g["label"] > 0) & (g["label"] <= 7200)).to_numpy()
        big = (np.abs(before - y) > 1800)
        out["months"][month] = {"stack_rmse": rmse(y, before), "rmse": rmse(y, after), "change_pct": 100 * (rmse(y, after) / rmse(y, before) - 1),
                                "ordinary_rows_change_pct": 100 * (rmse(y[o], after[o]) / rmse(y[o], before[o]) - 1),
                                "rows_over_1800s_error_sse_change_pct": 100 * (float(((after - y)[big] ** 2).sum()) / float(((before - y)[big] ** 2).sum()) - 1),
                                "rows_changed_over_60s": int((np.abs(after - before) > 60).sum()), "alignment": float(-np.sum((after - before) * (before - y)) / max(np.sum((after - before) ** 2), 1e-9)),
                                "paired_days": paired_days(g["bday"].to_numpy(), y, after, before, SEED)}
        e = out["months"][month]
        print(month, round(e["stack_rmse"], 3), "->", round(e["rmse"], 3), f"{e['change_pct']:+.3f}%", "| ordinary", f"{e['ordinary_rows_change_pct']:+.3f}%", "| big rows SSE", f"{e['rows_over_1800s_error_sse_change_pct']:+.2f}%",
              "| alignment", round(e["alignment"], 3), "| interval", [round(v, 3) for v in e["paired_days"]["rmse_difference_95pct_interval_seconds"]])
    by = t.groupby(["airport", "month"]).agg(before=("se", "sum"), after=("se_new", "sum"))
    pct = (100 * (by["after"] / by["before"] - 1)).unstack("month")
    out["airports"] = {a: {m: float(v) for m, v in r.items()} for a, r in pct.iterrows()}
    out["airport_month_cells"] = {"better": int((pct < 0).sum().sum()), "worse": int((pct > 0).sum().sum())}
    print(pct.round(2).to_string())
    write_json(target / ("report.json" if control is None else f"report_against_{control}.json"), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month")
    parser.add_argument("--tag", default="explore1")
    parser.add_argument("--tau-scale", type=float, default=1.0)
    parser.add_argument("--heavy", type=float, default=900.0)
    parser.add_argument("--no-heavy-updates", action="store_true")
    parser.add_argument("--heavy-temper", type=float, default=1.0)
    parser.add_argument("--heavy-tilt", action="store_true")
    parser.add_argument("--heavy-prior", choices=["tilt", "groups", "bounded"], default="tilt")
    parser.add_argument("--pin", type=float, default=0.0)
    parser.add_argument("--placebo", type=int, help="exploration: the airport's published totals permuted over the month's days, this seed")
    parser.add_argument("--score-clip", type=float, default=3.0)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rows", default="stack", help="stack, twelve or the path of a table of that form")
    parser.add_argument("--prior", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--components", choices=["one", "tracked"], default="one")
    parser.add_argument("--control")
    parser.add_argument("--centres")
    parser.add_argument("--airports", nargs="*")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--declaration")
    args = parser.parse_args()
    if args.score:
        govern(args.declaration)
    elif args.report:
        report(args.tag, args.rows, args.control)
    else:
        run_month(args.month, args.tag, args.tau_scale, Settings(heavy_sd_s=args.heavy, heavy_updates=not args.no_heavy_updates, heavy_temper=args.heavy_temper, heavy_tilt=args.heavy_tilt,
                                                                 score_clip=args.score_clip, heavy_prior=args.heavy_prior, pin_s=args.pin), args.rows, args.prior, args.components, args.centres, args.airports, args.placebo)


if __name__ == "__main__":
    main()
