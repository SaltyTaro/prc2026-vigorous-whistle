"""Screens after v48 (descriptive, 2025 months only): three uses of the published figures beyond the declared daily procedure.

  1. tracked   two day shifts per airport-day, one for the rows with an ADS-B surface match and one for the others
               (tools/run_daily_totals.py --components tracked), tags fu1_tracked (development stack) and fu1_tracked12 (twelve months)
  2. centre    the monthly taxi-out source (configs/published_means_v2.json: mapping, gate, shrinkage) as the centre of the day shift
               in the airport-months that pass its gate: as the prior mean (run_daily_totals.py --centres, tags fu2_centre and
               fu2_centre12) and, stronger, by recentring the control's day shifts on it (computed here)
  3. heavy     one tilt per airport-month of the heavy rows' scenario weights, read from the month's daily totals jointly

Control everywhere: the declared procedure of configs/daily_totals_v1.json (tags daily_totals_v1 and check12f).

  --centres                 -> docs/daily_totals/centres_stack.json, centres_twelve.json
  --heavy AIRPORT           -> docs/daily_totals/followup_heavy_<AIRPORT>.json (development stack, four months)
  --summary                 -> docs/daily_totals/followups.json (needs the four tagged runs and the centre tables)
Nothing is declared, selected or exported.
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
from prc2026.artifacts import read_json, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_daily import reconcile as rec  # noqa: E402
from prc2026_daily.reconcile import GRID_S, N, Settings, prepare_day  # noqa: E402
import run_daily_totals as runner  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402

DOCS = ROOT / "docs/daily_totals"
CONTROLS = {"stack": "daily_totals_v1", "twelve": "check12f"}
MAX_SIGMA = 5.0
THETAS, MUS = np.linspace(-1.5, 1.5, 31), np.linspace(-40.0, 40.0, 41)


def centres() -> None:
    """The monthly source's gated, shrunk level shift per airport-month, by the procedure of configs/published_means_v2.json."""
    from run_published_means import fit_mapping, shift_for
    from screen_published_means import CAP, label_means, published
    res = read_json(ROOT / "docs/published_means/results_v2.json")
    four = {f"{a}|{m}": {"mean_s": -c["shift_s"], "sigma_s": c["sigma_s"], "true_level_error_s": c["true_level_error_s"]}
            for m, v in res["development"].items() for a, c in v["cells"].items() if c["eligible"]}
    write_json(DOCS / "centres_stack.json", four)
    pub = published()
    table = label_means().merge(pub, on=["airport", "month"], how="inner", validate="one_to_one")
    t = runner.rows12_table()
    t = t[~t["extra"]]
    months = sorted(t["month"].unique())
    level = {}
    for (airport, month), g in t.groupby(["airport", "month"]):
        p, y = g["stack"].to_numpy(dtype=float), g["label"].to_numpy(dtype=float)
        inside = (p > 0) & (p <= CAP)
        ordinary = inside & (y > 0) & (y <= CAP)
        q = pub[(pub["airport"] == airport) & (pub["month"] == month)].iloc[0]
        mp = fit_mapping(table, airport, month)
        level[(airport, month)] = {"estimate": float(p[inside].mean() - (q["P"] + mp["function"](q["u"]))), "true": float(p[ordinary].mean() - y[ordinary].mean()),
                                   "sd_map": mp["sd_map"], "u": float(q["u"]), "u_max": mp["u_max"]}
    twelve = {}
    for (airport, month), e in level.items():
        d = np.array([level[(airport, m)]["estimate"] - level[(airport, m)]["true"] for m in months if m != month])
        bias, noise = float(d.mean()), float(np.sqrt(np.mean((d - d.mean()) ** 2)))
        shift, record = shift_for(e["estimate"], bias, max(e["sd_map"], noise), e["u"], e["u_max"], MAX_SIGMA)
        if record["eligible"]:
            twelve[f"{airport}|{month}"] = {"mean_s": -shift, "sigma_s": record["sigma_s"], "true_level_error_s": e["true"]}
    write_json(DOCS / "centres_twelve.json", twelve)
    for name, c in (("stack", four), ("twelve", twelve)):
        left = np.array([v["true_level_error_s"] + v["mean_s"] for v in c.values()])
        true = np.array([v["true_level_error_s"] for v in c.values()])
        print(name, "eligible cells", len(c), "rms level error", round(float(np.sqrt(np.mean(true ** 2))), 2), "rms left after the monthly shift", round(float(np.sqrt(np.mean(left ** 2))), 2))


def with_control(rows: str) -> pd.DataFrame:
    t = runner.table(rows)
    if "extra" in t:
        t = t[~t["extra"]].reset_index(drop=True)
    source = ROOT / f"runs/daily_totals/{CONTROLS[rows]}"
    new = pd.concat([pd.read_parquet(source / f"{m}.parquet") for m in sorted(t["month"].unique())], ignore_index=True)
    t = t.merge(new, on=ID, how="left")
    t["moved"] = t["new"].notna() & (t["new"] != t["stack"])
    t["new"], t["shift"] = t["new"].fillna(t["stack"]), t["shift"].fillna(0.0)
    return t


def recentred(rows: str, keep) -> dict:
    """The control's day shifts recentred, in the cells of the centre table that `keep` admits, so that their mean over the cell's
    moved rows is the monthly source's shift; the day-to-day part and the tilt stay. Label-free; labels score it."""
    t = with_control(rows)
    table = {k: v for k, v in read_json(DOCS / f"centres_{rows}.json").items() if keep(v)}
    t["candidate"] = t["new"]
    cells = {}
    for key, v in table.items():
        airport, month = key.split("|")
        k = (t["airport"].eq(airport) & t["month"].eq(month) & t["moved"]).to_numpy()
        centre = float(t.loc[k, "shift"].mean())
        t.loc[k, "candidate"] = np.maximum(t.loc[k, "new"] - centre + v["mean_s"], -12.0)
        g = t[k]
        cells[key] = {"control_mean_shift_s": centre, "monthly_shift_s": v["mean_s"], "rows": int(k.sum()),
                      "cell_squared_error_change_pct": 100 * (float(((g["candidate"] - g["label"]) ** 2).sum()) / float(((g["new"] - g["label"]) ** 2).sum()) - 1)}
    months = {}
    for month, g in t.groupby("month"):
        if not any(k.endswith(month) for k in table):
            continue
        y = g["label"].to_numpy(dtype=float)
        months[month] = {"change_pct": 100 * (rmse(y, g["candidate"].to_numpy(dtype=float)) / rmse(y, g["new"].to_numpy(dtype=float)) - 1),
                         "interval_s": paired_days(g["bday"].to_numpy(), y, g["candidate"].to_numpy(dtype=float), g["new"].to_numpy(dtype=float), runner.SEED)["rmse_difference_95pct_interval_seconds"]}
    return {"months": months, "cells_better": int(sum(c["cell_squared_error_change_pct"] < 0 for c in cells.values())), "cells": cells}


def tilted(e: np.ndarray, w0: np.ndarray, theta: float) -> np.ndarray:
    x = theta * e / 3600.0
    w = w0 * np.exp(x - x.max())
    return w / w.sum()


def heavy(airport: str) -> None:
    """One tilt theta per airport-month of the heavy rows' scenario weights, w ~ w0 exp(theta e / 3600), and a month-level shift mu
    of the ordinary labels: likelihood of (theta, mu) from the month's daily totals by convolution; posterior mean under three priors."""
    settings = Settings(heavy_updates=False)
    t = runner.add_keys(runner.rows_table())
    out = {"created_at_utc": utc_now(), "airport": airport, "thetas": [float(THETAS[0]), float(THETAS[-1]), len(THETAS)], "months": {}}
    for month in runner.MONTHS:
        fit, use = t[t["month"].ne(month)], t[t["month"].eq(month) & t["complete"] & t["airport"].eq(airport)]
        cells = runner.Cells(fit)
        noise, wide, tau = runner.scales(fit)
        share = wide[airport]
        noise_cf = (1 - share) * rec._gauss_cf(noise[airport]) + sum(0.5 * share * rec._gauss_cf(s) for s in settings.noise_wide_sd_s)
        ll = np.zeros((len(THETAS), len(MUS)))
        rows = []
        for _, day in use.groupby("bday", sort=True):
            day = day.reset_index(drop=True)
            p = prepare_day(day["a"].to_numpy(dtype=float), day["stack"].to_numpy(dtype=float), cells.resolve(day), settings)
            delta = (float(day["pub"].iat[0]) - p.reference_total) / GRID_S
            if not np.isfinite(delta) or abs(delta) >= N // 2 - 1:
                continue
            found = {}
            for positions, y, e, w in p.store:
                for local, row in enumerate(positions):
                    if p.is_heavy[row]:
                        found[int(row)] = (y[local].astype(float), e[local].astype(float), w[local].astype(float))
            base = p.bulk_cf * noise_cf * rec._gauss_cf(p.n_eff * tau[airport])
            for k, theta in enumerate(THETAS):
                cf = base.copy()
                for y, e, w in found.values():
                    cf = cf * np.fft.rfft(rec._pmf(e[None, :], tilted(e, w, theta)[None, :])[0])
                q = np.maximum(np.fft.irfft(cf, n=N), 0.0)
                ll[k] += np.log(rec._at(q, delta + p.n_eff * MUS / GRID_S) + 1e-300)
            for row, (y, e, w) in found.items():
                rows.append({"error": float(day["stack"].iat[row] - day["label"].iat[row]), "prior": float((w * y).sum()), "by_theta": [float((tilted(e, w, th) * y).sum()) for th in THETAS]})
        if not rows:
            continue
        error = np.array([v["error"] for v in rows])
        moves = np.array([v["by_theta"] for v in rows]) - np.array([v["prior"] for v in rows])[:, None]
        base_sse = float((error ** 2).sum())
        sse = ((error[:, None] + moves) ** 2).sum(axis=0)
        record = {"heavy_rows": len(rows), "heavy_squared_error": base_sse, "sum_of_errors_h": float(error.sum() / 3600), "best_theta_with_labels": float(THETAS[int(np.argmin(sse))]),
                  "best_change_pct_with_labels": 100 * (float(sse.min()) / base_sse - 1), "one_common_shift_with_labels_pct": -100 * float(error.sum() ** 2 / len(error)) / base_sse, "label_free": {}}
        for name, s_theta, s_mu in (("flat", 1e3, 1e3), ("theta 0.5, mu 10 s", 0.5, 10.0), ("theta 0.3, mu 10 s", 0.3, 10.0)):
            lp = ll - 0.5 * (THETAS[:, None] / s_theta) ** 2 - 0.5 * (MUS[None, :] / s_mu) ** 2
            w = np.exp(lp - lp.max())
            w /= w.sum()
            by_theta = w.sum(axis=1)
            record["label_free"][name] = {"theta": float(by_theta @ THETAS), "mu_s": float(w.sum(axis=0) @ MUS), "moved_h": float((moves @ by_theta).sum() / 3600),
                                          "heavy_squared_error_change_pct": 100 * (float(((error + moves @ by_theta) ** 2).sum()) / base_sse - 1)}
        out["months"][month] = record
        print(month, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items() if k != "label_free"},
              {k: (round(v["theta"], 3), round(v["heavy_squared_error_change_pct"], 2)) for k, v in record["label_free"].items()}, flush=True)
    write_json(DOCS / f"followup_heavy_{airport}.json", out)


def summary() -> None:
    out = {"created_at_utc": utc_now(), "control": CONTROLS, "audit_used": False, "official_scores_used": "none", "runs": {}, "recentred": {}}
    for tag, rows in (("fu1_tracked", "stack"), ("fu1_tracked12", "twelve"), ("fu2_centre", "stack"), ("fu2_centre12", "twelve")):
        report = runner.report(tag, rows, CONTROLS[rows])
        changes = {m: v["change_pct"] for m, v in report["months"].items()}
        out["runs"][tag] = {"rows": rows, "months": {m: {"change_pct": v["change_pct"], "interval_s": v["paired_days"]["rmse_difference_95pct_interval_seconds"]} for m, v in report["months"].items()},
                            "mean_change_pct": float(np.mean(list(changes.values()))), "months_better": int(sum(c < 0 for c in changes.values())), "airport_month_cells": report["airport_month_cells"]}
        if rows == "stack":
            out["runs"][tag]["mean_change_pct_equal_footing"] = float(np.mean([changes[m] for m in ("2025-01", "2025-07", "2025-12")]))
    for rows in ("stack", "twelve"):
        for name, keep in (("all eligible cells", lambda v: True), ("sigma up to 1 s", lambda v: v["sigma_s"] <= 1.0), ("sigma above 1 s", lambda v: v["sigma_s"] > 1.0)):
            out["recentred"][f"{rows} | {name}"] = recentred(rows, keep)
            print(rows, name, {m: round(v["change_pct"], 4) for m, v in out["recentred"][f"{rows} | {name}"]["months"].items()})
    write_json(DOCS / "followups.json", out)
    print({k: (round(v["mean_change_pct"], 4), v["months_better"]) for k, v in out["runs"].items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--centres", action="store_true")
    parser.add_argument("--heavy")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.centres:
        centres()
    elif args.heavy:
        heavy(args.heavy)
    elif args.summary:
        summary()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
