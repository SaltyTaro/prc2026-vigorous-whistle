"""Read-only audit of the v22 recipe's out-of-sample development predictions, December first (22 September 2026).

Inputs are the adsb_v24_v1 development outputs (runs/adsb_v24/dev/predictions_<month>.parquet: the selected candidate v7x_r12x_lirf25
is the v22 recipe, `baseline` the v21 recipe), the twelve-month scratch frames (labels, cross-fit v9, clocks), the cached residual
feature tables (match class) and the declared committee tables rebuilt exactly as the development tool did (January-November refit
leaves on December, fold predictions on the other months). Nothing is fitted; every number is descriptive on reused development data.
The residual stage's value is backed out of the final prediction (final = 0.3 * (v9 + r) + 0.7 * committee on non-LIRF rows) and the
recipe is rebuilt with the frozen blend arithmetic as a reconstruction check before anything is reported.
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
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from run_adsb_v11 import SCOPES, candidate_values, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, month_table  # noqa: E402

CANDIDATE = "v7x_r12x_lirf25"
MONTHS = ["2025-12", "2025-11", "2025-07", "2025-01"]
LONG = 21600.0


def classes(t: pd.DataFrame) -> pd.Series:
    lirf = t["airport"].eq("LIRF")
    pop = t["population"].astype(bool)
    matched = t["adsb_matched"].eq(1)
    gate = t["adsb_gate"].eq(1)
    out = pd.Series("", index=t.index)
    out[~lirf & ~matched] = "unmatched"
    out[~lirf & matched & ~gate] = "dark"
    out[~lirf & matched & gate] = "gate"
    out[lirf & pop] = "lirf_population"
    out[SCOPES["lirf_nonpop_matched"](t)] = "lirf_nonpop_matched"
    out[SCOPES["lirf_nonpop_unmatched"](t)] = "lirf_nonpop_unmatched"
    out[lirf & ~pop & out.eq("")] = "lirf_nonpop_long_clock"
    return out


def summary(label: np.ndarray, columns: dict[str, np.ndarray], mask: np.ndarray, total_sse: float) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"rows": 0}
    res = label[mask] - columns["final"][mask]
    out = {"rows": n, "sse_share_pct": float(100 * np.square(res).sum() / total_sse),
           "bias_s": float(res.mean()), "median_residual_s": float(np.median(res))}
    for name, values in columns.items():
        v = values[mask]
        if np.isfinite(v).all():
            out[f"rmse_{name}"] = rmse(label[mask], v)
    return out


def load_month(config: dict, month: str, cache: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = pd.read_parquet(ROOT / config["cross_fit"]["scratch"] / f"{month}.parquet")
    frame = frame[[ID, TIME, "airport", TARGET, "v9", "floor", "proxy_s", "schedule_interval_s", "clock_available", "population"]]
    table = month_table(frame, adsb_features(month, ROOT / config["matches_dirs"]["v11"], cache), month)
    for name, values in extra_columns(config, month, table[ID].to_numpy()).items():
        table[name] = values
    preds = pd.read_parquet(ROOT / "runs/adsb_v24/dev" / f"predictions_{month}.parquet").set_index(ID).reindex(table[ID])
    if preds[CANDIDATE].isna().any():
        raise ValueError(f"Development predictions do not cover {month}.")
    floor = float(table["floor"].iloc[0])
    label = table[TARGET].to_numpy(dtype=float)
    v9 = table["v9"].to_numpy(dtype=float)
    final = preds[CANDIDATE].to_numpy(dtype=float)
    committee = table["ens_v20_bag_q_all"].to_numpy(dtype=float)
    lirf_q = table["ens_v22_lirf_q"].to_numpy(dtype=float)
    # Back out the residual stage on its scope (matched non-LIRF rows) from the final value; rows at the floor cannot be inverted.
    in_scope = SCOPES["matched_no_lirf"](table).to_numpy()
    r = np.zeros(len(table))
    invertible = in_scope & (final > floor + 1e-9) & np.isfinite(committee)
    r[invertible] = (final[invertible] - 0.7 * committee[invertible]) / 0.3 - v9[invertible]
    spec = {"weight": 1.0, "scope": "matched_no_lirf", "blend_with": config["blend_with"],
            "extra_blends": [*config["extra_blends"], *config["candidates"][CANDIDATE].get("extra_blends", [])]}
    rebuilt, _ = candidate_values(table, v9, r, spec, floor)
    if np.abs(rebuilt - final).max() > 1e-6:
        raise ValueError(f"The v22 recipe does not rebuild from its components on {month}: max difference {np.abs(rebuilt - final).max()}.")
    # Descriptive ablations with the same frozen arithmetic.
    no_residual, _ = candidate_values(table, v9, np.zeros(len(table)), spec, floor)
    no_committee, _ = candidate_values(table, v9, r, {**spec, "blend_with": {**config["blend_with"], "weight": 0.0}}, floor)
    columns = {"final": final, "v21": preds["baseline"].to_numpy(dtype=float), "v9": v9, "v9_plus_residual": v9 + r,
               "committee": committee, "lirf_q": lirf_q, "without_residual": no_residual, "without_committee": no_committee}
    table["class"] = classes(table)
    table["day"] = table[TIME].dt.strftime("%Y-%m-%d")
    table["hour_of_day"] = table[TIME].dt.hour
    table["minute_of_day"] = table[TIME].dt.hour * 60 + table[TIME].dt.minute
    return table, columns


def audit_month(config: dict, month: str, cache: Path) -> dict:
    table, columns = load_month(config, month, cache)
    label = table[TARGET].to_numpy(dtype=float)
    final = columns["final"]
    res = label - final
    sse = float(np.square(res).sum())
    report = {"rows": int(len(table)), "rmse": rmse(label, final), "rmse_v21": rmse(label, columns["v21"]), "sse": sse,
              "ablations": {name: {"rmse": rmse(label, columns[name]), "change_pct": float(100 * (rmse(label, columns[name]) / rmse(label, final) - 1))}
                            for name in ["v9", "without_residual", "without_committee", "v21"]}}
    airports = table["airport"].to_numpy()
    cls = table["class"].to_numpy()
    long = label > LONG
    report["long_labels"] = {"rows": int(long.sum()), "sse_share_pct": float(100 * np.square(res[long]).sum() / sse),
                             "by_airport": {a: int((long & (airports == a)).sum()) for a in sorted(set(airports)) if (long & (airports == a)).any()}}
    ordinary = ~long
    sse_ordinary = float(np.square(res[ordinary]).sum())
    report["ordinary_rows"] = {"rows": int(ordinary.sum()), "rmse": rmse(label[ordinary], final[ordinary]), "sse_share_pct": float(100 * sse_ordinary / sse)}
    report["by_class"] = {c: summary(label, columns, cls == c, sse) for c in sorted(set(cls))}
    report["by_class_ordinary"] = {c: summary(label, columns, (cls == c) & ordinary, sse_ordinary) for c in sorted(set(cls))}
    report["by_airport"] = {a: summary(label, columns, airports == a, sse) for a in sorted(set(airports))}
    report["by_airport_ordinary"] = {a: summary(label, columns, (airports == a) & ordinary, sse_ordinary) for a in sorted(set(airports))}
    report["by_airport_class_ordinary"] = {f"{a}/{c}": summary(label, columns, (airports == a) & (cls == c) & ordinary, sse_ordinary)
                                           for a in sorted(set(airports)) for c in sorted(set(cls)) if ((airports == a) & (cls == c) & ordinary).sum() >= 50}
    # Top rows.
    order = np.argsort(-np.square(res))
    for k in [20, 200, 1000]:
        top = order[:k]
        report[f"top_{k}_rows"] = {"sse_share_pct": float(100 * np.square(res[top]).sum() / sse),
                                   "classes": {c: int((cls[top] == c).sum()) for c in sorted(set(cls[top]))},
                                   "airports": {a: int((airports[top] == a).sum()) for a in sorted(set(airports[top]))},
                                   "labels_above_6h": int((label[top] > LONG).sum()), "under_predicted": int((res[top] > 0).sum())}
    top = order[:20]
    sched = table["schedule_interval_s"].to_numpy(dtype=float)
    proxy = table["proxy_s"].to_numpy(dtype=float)
    report["top_20_detail"] = [{"airport": str(airports[i]), "class": str(cls[i]), "day": str(table["day"].iloc[i]), "label_s": float(label[i]), "final_s": float(final[i]),
                                "v9_s": float(columns["v9"][i]), "committee_s": (float(columns["committee"][i]) if np.isfinite(columns["committee"][i]) else None),
                                "proxy_s": (float(proxy[i]) if np.isfinite(proxy[i]) else None), "schedule_interval_s": (float(sched[i]) if np.isfinite(sched[i]) else None),
                                "label_minus_schedule_s": (float(label[i] - sched[i]) if np.isfinite(sched[i]) else None), "sse_share_pct": float(100 * res[i] ** 2 / sse)}
                               for i in top]
    # Days.
    days = table["day"].to_numpy()
    by_day = pd.DataFrame({"day": days, "sq": np.square(res), "res": res}).groupby("day").agg(rows=("sq", "size"), sse=("sq", "sum"), bias=("res", "mean"))
    by_day["rmse"] = np.sqrt(by_day["sse"] / by_day["rows"])
    by_day["sse_share_pct"] = 100 * by_day["sse"] / sse
    report["days"] = {"count": int(len(by_day)), "top_5": by_day.sort_values("sse", ascending=False).head(5).round(2).to_dict("index"),
                      "daily_rmse_quantiles": {q: float(by_day["rmse"].quantile(float(q))) for q in ["0.1", "0.5", "0.9"]}}
    # Midnight rows (day-file boundary at the match stage).
    minute = table["minute_of_day"].to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    non_lirf = airports != "LIRF"
    windows = {"00:00-00:10": minute < 10, "00:10-00:20": (minute >= 10) & (minute < 20), "00:20-01:00": (minute >= 20) & (minute < 60), "01:00-24:00": minute >= 60}
    report["midnight"] = {w: {"rows_non_lirf": int((m & non_lirf).sum()), "matched_share_non_lirf": float(matched[m & non_lirf].mean()),
                              "sse_share_pct": float(100 * np.square(res[m]).sum() / sse), "rmse": rmse(label[m], final[m])} for w, m in windows.items()}
    # Hour-of-day bias by airport on ordinary rows (cells with at least 200 rows).
    hours = table["hour_of_day"].to_numpy()
    cells = {}
    for a in sorted(set(airports)):
        for h in range(24):
            m = (airports == a) & (hours == h) & ordinary
            if m.sum() >= 200:
                cells[f"{a}/{h:02d}"] = {"rows": int(m.sum()), "bias_s": float(res[m].mean()), "se_s": float(res[m].std(ddof=1) / np.sqrt(m.sum()))}
    report["hour_bias_ordinary"] = cells
    return report


def consistent_bias(reports: dict[str, dict], key: str) -> dict:
    """Cells whose ordinary-row bias has the same sign in every month, with the squared-error share a perfect additive correction would remove."""
    out = {}
    cells = set.intersection(*[set(r[key]) for r in reports.values()])
    for cell in sorted(cells):
        biases = [reports[m][key][cell]["bias_s"] for m in reports]
        if all(b > 0 for b in biases) or all(b < 0 for b in biases):
            removable = {m: 100 * reports[m][key][cell]["rows"] * reports[m][key][cell]["bias_s"] ** 2 / reports[m]["sse"] for m in reports}
            out[cell] = {"bias_by_month_s": {m: round(b, 1) for m, b in zip(reports, biases)}, "rows_by_month": {m: reports[m][key][cell]["rows"] for m in reports},
                         "sse_share_removable_pct_by_month": {m: round(v, 3) for m, v in removable.items()}}
    return out


def long_population_rows(config: dict) -> dict:
    """LIRF population rows with schedule interval above 43,200 s (the rows the v9 mixture leaves at v7): honest out-of-sample
    predictions on the four development months, and the shipped classifier's label-free reading of the ranking's rows."""
    frames = {"2025-01": "scratch/seasonal.parquet", "2025-07": "scratch/seasonal.parquet", "2025-11": "scratch/forward.parquet", "2025-12": "scratch/december.parquet"}
    labelled = []
    for month, path in frames.items():
        f = pd.read_parquet(ROOT / path)
        f = f[f[TIME].dt.strftime("%Y-%m").eq(month) & f["airport"].eq("LIRF") & f["population"].astype(bool) & f["schedule_interval_s"].gt(43200)]
        for _, r in f.iterrows():
            labelled.append({"month": month, "label_s": float(r[TARGET]), "schedule_interval_s": float(r["schedule_interval_s"]), "prediction_s": float(r["v9"]),
                             "label_minus_schedule_s": float(r[TARGET] - r["schedule_interval_s"]), "prediction_minus_schedule_s": float(r["v9"] - r["schedule_interval_s"]),
                             "mode": "fill" if abs(r[TARGET] - r["schedule_interval_s"]) <= 10 else ("one_day" if 86000 <= r[TARGET] <= 90000 else "other")})
    pop = pd.read_parquet(ROOT / "runs/lirf_mixture/export_v9/population_predictions.parquet")
    values = pd.read_parquet(ROOT / "runs/adsb_v24/export_v22/ranking_values.parquet").set_index(ID)["v22"]
    pop["v22"] = values.reindex(pop[ID]).to_numpy()
    long = pop[pop["schedule_interval_s"].gt(43200)]
    total_sse = 344841 * 239.1335 ** 2
    ranking = [{"schedule_interval_s": float(r["schedule_interval_s"]), "v22_s": float(r["v22"]), "v22_minus_schedule_s": float(r["v22"] - r["schedule_interval_s"]),
                "classifier_p_fill": float(r["p_fill"]), "classifier_p_one_day": float(r["p_one_day"])} for _, r in long.iterrows()]
    return {"note": "descriptive; the v9 declaration left these rows at v7 and the v8 result is not a rule input (docs/postprocess/README.md)",
            "labelled_out_of_sample_rows": labelled,
            "labelled_mode_counts": {m: int(sum(1 for r in labelled if r["mode"] == m)) for m in ["fill", "one_day", "other"]},
            "ranking_rows": ranking, "ranking_row_count": int(len(long)),
            "ranking_sse_share_pct_if_every_row_were_a_fill": float(100 * np.square(long["v22"] - long["schedule_interval_s"]).sum() / total_sse),
            "ranking_total_sse_at_official_rmse": total_sse}


def ranking_composition() -> dict:
    """Label-free match-class shares per airport in the ranking months (matches_v11 tables), for reading the development shares."""
    rank = pd.read_parquet(ROOT / "runs/adsb_v24/export_v22/ranking_values.parquet")[[ID, "airport"]]
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, TIME])
    rank = rank.merge(raw, on=ID)
    rank["month"] = rank[TIME].dt.strftime("%Y-%m")
    out = {}
    for month in sorted(set(rank["month"])):
        matches = pd.read_parquet(ROOT / "data/external/adsblol/matches_v11" / f"matches2_{month}.parquet")[[ID, "gate"]]
        r = rank[rank["month"].eq(month)].merge(matches, on=ID, how="left")
        r["class"] = np.where(r["gate"].isna(), "unmatched", np.where(r["gate"].fillna(False).astype(bool), "gate", "dark"))
        table = r.groupby(["airport", "class"]).size().unstack(fill_value=0)
        out[month] = {a: {"rows": int(table.loc[a].sum()), **{c: float(100 * table.loc[a].get(c, 0) / table.loc[a].sum()) for c in ["gate", "dark", "unmatched"]}} for a in table.index}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/adsb_v24_v1.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/adsb_v24/features")
    parser.add_argument("--months", nargs="+", default=MONTHS)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/audit_20260922/audit.json")
    args = parser.parse_args()
    config = read_json(args.config)
    reports = {}
    for month in args.months:
        reports[month] = audit_month(config, month, args.cache)
        print(month, {"rmse": round(reports[month]["rmse"], 4), "long_label_sse_pct": round(reports[month]["long_labels"]["sse_share_pct"], 1),
                      "top_200_pct": round(reports[month]["top_200_rows"]["sse_share_pct"], 1)}, flush=True)
    result = {"created_at_utc": utc_now(), "candidate": CANDIDATE, "audit_used": False, "note": "descriptive on reused development data; nothing fitted or selected",
              "inputs": {"config": sha256_file(args.config), "predictions": {m: sha256_file(ROOT / "runs/adsb_v24/dev" / f"predictions_{m}.parquet") for m in args.months}},
              "months": reports,
              "consistent_bias_airport": consistent_bias(reports, "by_airport_ordinary"),
              "consistent_bias_airport_class": consistent_bias(reports, "by_airport_class_ordinary"),
              "consistent_bias_airport_hour": consistent_bias(reports, "hour_bias_ordinary"),
              "long_schedule_lirf_population": long_population_rows(config),
              "ranking_match_composition_pct": ranking_composition()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print("written", args.output)


if __name__ == "__main__":
    main()
