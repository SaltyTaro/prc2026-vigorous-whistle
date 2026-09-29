"""Screen (descriptive): what are EUROCONTROL's published airport-month taxi-out aggregates worth as a level check of the stack?

Source: the Performance Review Unit's "Taxi-Out Additional Time" workbook (ansperformance.eu/data, data/external/eurocontrol_aiu/): per
airport and month the departures (TF), the departures with valid data (VALID_FL) and, for those with a reference time, the total
reference and additional taxi-out minutes. P = (reference + additional) / flights is the published mean taxi-out time of the valid
flights; u = 1 - VALID_FL / TF is the share the unit leaves out (mostly winter months: de-iced flights).

Mapping (2025 labels only): per airport, D = mean label (labels in (0, 7200] s) - P is a constant, or a line in u where u varies by more
than 0.03 over 2025. Development read, month-blocked: for each development month the mapping is fitted on the other eleven months of
2025, the target mean is P + D(u), the stack's level error is mean(prediction, predictions in (0, 7200]) - target, and every such
prediction of the airport-month is shifted by -w x level error. Control: the v46 stack on runs/lomo_dev/rows.parquet.

The ranking months' implied level errors are reported from the uploaded v46 (no label involved). Nothing is declared, selected or
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_final import share_gate  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402
from run_share_gate_reread import full_stack  # noqa: E402

WORKBOOK = "data/external/eurocontrol_aiu/Taxi-Out_Additional_Time.xlsx"
ROWS = "runs/lomo_dev/rows.parquet"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
CAP = 7200.0
SEED = 20260929
SLOPE_RANGE = 0.03
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def published() -> pd.DataFrame:
    pub = pd.read_excel(ROOT / WORKBOOK, sheet_name="DATA")
    pub = pub[pub["APT_ICAO"].isin(AIRPORTS) & pub["YEAR"].isin([2025, 2026])].copy()
    pub["month"] = pub["YEAR"].astype(int).astype(str) + "-" + pub["MONTH_NUM"].astype(int).astype(str).str.zfill(2)
    pub["P"] = 60.0 * (pub["TOTAL_REF_TIME_MIN"] + pub["TOTAL_ADD_TIME_MIN"]) / pub["TOTAL_REF_NB_FL"]
    pub["u"] = 1.0 - pub["VALID_FL"] / pub["TF"]
    return pub.rename(columns={"APT_ICAO": "airport"})[["airport", "month", "TF", "VALID_FL", "NO_REF", "P", "u"]]


def label_means() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))):
        d = pd.read_parquet(path, columns=["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", TARGET])
        d = d[d["PHASE_mvt"].eq("DEP")]
        month = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
        for (a, m), g in d.groupby([d["ADEP_mvt"], month]):
            y = g[TARGET].to_numpy(dtype=float)
            if len(y) > 100 and m.startswith("2025"):
                rows.append({"airport": a, "month": m, "n": len(y), "label_mean": float(y[(y > 0) & (y <= CAP)].mean())})
    return pd.DataFrame(rows)


def mapping(table: pd.DataFrame, airport: str, exclude: str | None):
    """D(u) for one airport from its 2025 months except `exclude`: (function, uses_slope, months)."""
    g = table[(table["airport"] == airport) & (table["month"] != exclude)]
    x, y = g["u"].to_numpy(), (g["label_mean"] - g["P"]).to_numpy()
    if x.max() - x.min() > SLOPE_RANGE:
        b = np.polyfit(x, y, 1)
        return (lambda q: float(np.polyval(b, q))), True, len(g)
    m = float(y.mean())
    return (lambda q: m), False, len(g)


def main() -> None:
    pub = published()
    table = label_means().merge(pub, on=["airport", "month"], how="inner", validate="one_to_one")
    report = {"created_at_utc": utc_now(), "workbook": {"path": WORKBOOK, "sha256": sha256_file(ROOT / WORKBOOK), "source": "https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx",
                                                        "terms": "EUROCONTROL: may be copied with the source named, not for commercial purposes, no modification without written permission (ansperformance.eu disclaimer); not an open licence"},
              "cap_s": CAP, "mapping_leave_one_month_out": {}, "development": {}, "ranking_implied": {}}
    for a in AIRPORTS:
        g = table[table["airport"] == a]
        errors = []
        for m in g["month"]:
            f, _, _ = mapping(table, a, m)
            row = g[g["month"] == m].iloc[0]
            errors.append(row["P"] + f(row["u"]) - row["label_mean"])
        _, slope, _ = mapping(table, a, None)
        report["mapping_leave_one_month_out"][a] = {"uses_slope_in_u": slope, "sd_s": float(np.std(errors)), "max_abs_s": float(np.max(np.abs(errors))), "u_2025_max": float(g["u"].max())}
    print({a: (round(v["sd_s"], 1), round(v["max_abs_s"], 1)) for a, v in report["mapping_leave_one_month_out"].items()})

    rows = pd.read_parquet(ROOT / ROWS)
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = offsets(), offsets_all()
    seg = segments(MONTHS)
    k = read_json(ROOT / "docs/disagreement/tail_stretch_v2_results.json")["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    share, gate = share_gate.candidate_spec(read_json(ROOT / "configs/thinned_regime_round_v1.json"), "n85_nd1200")
    for month in MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        mix, _ = share_gate.mix(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), share, gate)
        control = full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)
        y, days, airport, floor = f["label"].to_numpy(dtype=float), f["day"].to_numpy(), f["airport"].to_numpy(), f["floor"].to_numpy(dtype=float)
        inside = (control > 0) & (control <= CAP)
        shift = np.zeros(len(f))
        entry = {"rows": int(len(f)), "control_rmse": rmse(y, control), "airports": {}}
        for a in AIRPORTS:
            m = (airport == a) & inside
            q = pub[(pub["airport"] == a) & (pub["month"] == month)].iloc[0]
            fmap, slope, _ = mapping(table, a, month)
            target = q["P"] + fmap(q["u"])
            level = float(control[m].mean() - target)
            ordinary = m & (y > 0) & (y <= CAP)
            true_level = float(control[ordinary].mean() - y[ordinary].mean())
            shift[m] = level
            entry["airports"][a] = {"rows": int(m.sum()), "u": float(q["u"]), "target_mean": float(target), "estimated_level_error_s": level, "true_level_error_ordinary_rows_s": true_level,
                                    "sd_of_mapping_s": report["mapping_leave_one_month_out"][a]["sd_s"]}
        entry["level_error_estimate_vs_true"] = {"correlation": float(np.corrcoef([v["estimated_level_error_s"] for v in entry["airports"].values()],
                                                                                   [v["true_level_error_ordinary_rows_s"] for v in entry["airports"].values()])[0, 1]),
                                                 "rms_difference_s": float(np.sqrt(np.mean([(v["estimated_level_error_s"] - v["true_level_error_ordinary_rows_s"]) ** 2 for v in entry["airports"].values()])))}
        entry["weights"] = {}
        for w in (0.5, 1.0):
            for name, keep in (("all_airports", AIRPORTS), ("without_ltfm", [a for a in AIRPORTS if a != "LTFM"])):
                values = np.maximum(control - w * np.where(np.isin(airport, keep), shift, 0.0), floor)
                entry["weights"][f"{name}_w{w}"] = {"change_pct": 100 * (rmse(y, values) / rmse(y, control) - 1), "paired_days": paired_days(days, y, values, control, SEED)["rmse_difference_95pct_interval_seconds"]}
        report["development"][month] = entry
        print(month, "control", round(entry["control_rmse"], 3), {k2: (round(v["change_pct"], 4), [round(x, 3) for x in v["paired_days"]]) for k2, v in entry["weights"].items()})
        print("   level error estimated / true:", {a: (round(v["estimated_level_error_s"], 1), round(v["true_level_error_ordinary_rows_s"], 1)) for a, v in entry["airports"].items()}, entry["level_error_estimate_vs_true"])

    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")]
    raw["month"] = raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    v46 = pd.read_parquet(ROOT / "runs/submissions/vigorous-whistle_v46.parquet").merge(raw[[ID, "ADEP_mvt", "month"]], on=ID, validate="one_to_one")
    total = 0.0
    for (a, m), g in v46.groupby(["ADEP_mvt", "month"]):
        q = pub[(pub["airport"] == a) & (pub["month"] == m)].iloc[0]
        fmap, slope, _ = mapping(table, a, None)
        p = g[TARGET].to_numpy(dtype=float)
        p = p[(p > 0) & (p <= CAP)]
        level = float(p.mean() - (q["P"] + fmap(q["u"])))
        sd = report["mapping_leave_one_month_out"][a]["sd_s"]
        inside_range = bool(q["u"] <= report["mapping_leave_one_month_out"][a]["u_2025_max"] + 0.02)
        net = len(p) * max(level ** 2 - sd ** 2, 0.0)
        total += net
        report["ranking_implied"][f"{a} {m}"] = {"rows": int(len(p)), "published_mean_s": float(q["P"]), "left_out_share": float(q["u"]), "left_out_share_inside_2025_range": inside_range,
                                                 "implied_level_error_s": level, "sd_of_mapping_s": sd, "squared_error_at_stake_net_of_mapping_noise": net}
    n, score = 344841, 223.9339
    report["ranking_implied_total"] = {"squared_error_at_stake": total, "rmse_if_removed_exactly": float(np.sqrt((n * score ** 2 - total) / n)),
                                       "note": "an upper reading: it assumes the 2025 mapping holds in 2026 and that a level shift is the right correction"}
    print({k2: (round(v["implied_level_error_s"], 1), v["left_out_share_inside_2025_range"]) for k2, v in report["ranking_implied"].items()})
    print(report["ranking_implied_total"])
    write_json(ROOT / "docs/clean_line_screens/published_means.json", report)


if __name__ == "__main__":
    main()
