"""xplane_stands_probe_v1 (configs/xplane_stands_probe_v1.json): the X-Plane Scenery Gateway apt.dat as an LFPG stand catalogue.

Part A checks catalogue positions against the stand positions the matcher learned from ADS-B gate anchors at eight other airports and
against LFPG's own gate anchors; part B counts LFPG coverage per month (label-free); part C asks whether the first-point distance to the
catalogue stand explains what the residual stage's twelve-month leave-one-month-out prediction leaves on 2025 LFPG dark matches.
Reads the cached Gateway zips (tools/pull_xplane_gateway.py); writes docs/xplane_stands/probe_results.json. No model is fitted.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TARGET, TIME  # noqa: E402
from prc2026_adsb.features import metres  # noqa: E402

CONFIG = ROOT / "configs/xplane_stands_probe_v1.json"
SOURCES = ROOT / "docs/xplane_stands/sources.json"
MATCHES = ROOT / "data/external/adsblol/matches_v11"
OUTPUT = ROOT / "docs/xplane_stands/probe_results.json"
MONTHS_2025 = [f"2025-{m:02}" for m in range(1, 13)]
ALL_MONTHS = MONTHS_2025 + ["2026-01", "2026-07"]
RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
LEAD_WORDS = {"STAND", "GATE", "RAMP", "PARKING", "POSITION", "POS", "APRON"}
CANONICAL = re.compile(r"^([A-Z]*)(\d+)([A-Z]*)$")


def normalise(name) -> str | None:
    """The declared stand-name key: shared by apt.dat names and STAND_mvt."""
    if not isinstance(name, str):
        return None
    key = name.upper().strip()
    parts = key.split(None, 1)
    if len(parts) == 2 and parts[0] in LEAD_WORDS:
        key = parts[1]
    key = re.sub(r"-\([A-Z]\)$", "", key)
    key = re.sub(r"\s*\([^)]*\)$", "", key)
    key = re.sub(r"[\s\-]", "", key)
    match = CANONICAL.match(key)
    if match:
        key = match.group(1) + str(int(match.group(2))) + match.group(3)
    return key or None


def catalogue(icao: str, role: str) -> tuple[pd.DataFrame, dict]:
    """Resolved (key -> lat, lon) positions of one scenery version, and its counts."""
    airport = next(a for a in read_json(SOURCES)["airports"] if a["icao"] == icao)
    version = airport["versions"][role]
    path = ROOT / version["zip"]
    if sha256_file(path) != version["zip_sha256"]:
        raise ValueError(f"Cached Gateway zip changed: {path}")
    text = zipfile.ZipFile(path).read(f"{icao}.dat").decode("utf-8", "replace").splitlines()
    rows = []
    for line in text:
        if line.startswith("1300 "):
            parts = line.split(None, 6)
            rows.append({"name": parts[6].strip() if len(parts) > 6 else "", "lat": float(parts[1]), "lon": float(parts[2])})
    raw = pd.DataFrame(rows)
    raw["key"] = raw["name"].map(normalise)
    raw = raw[raw["key"].notna()]
    out, ambiguous = [], 0
    for key, group in raw.groupby("key"):
        lat, lon = group["lat"].mean(), group["lon"].mean()
        if metres(lat, lon, group["lat"].to_numpy(), group["lon"].to_numpy()).max() <= 25.0:
            out.append({"key": key, "lat": lat, "lon": lon, "entries": len(group)})
        else:
            ambiguous += 1
    table = pd.DataFrame(out)
    return table, {"scenery_id": version["scenery_id"], "date_approved": version["date_approved"], "startup_locations": len(rows),
                   "named_keys": int(raw["key"].nunique()), "resolved_keys": len(table), "ambiguous_keys": ambiguous}


def distance_summary(d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "median_m": float(np.median(d)), "p90_m": float(np.quantile(d, 0.9)),
            "within_25m": float(np.mean(d <= 25)), "within_40m": float(np.mean(d <= 40)), "within_100m": float(np.mean(d <= 100))}


def attach(frame: pd.DataFrame, cat: pd.DataFrame, stand_col: str) -> pd.DataFrame:
    """Add the catalogue position (cat_lat, cat_lon) of each row's normalised stand; NaN where unresolved."""
    keyed = frame.assign(key=frame[stand_col].map(normalise))
    return keyed.merge(cat.rename(columns={"lat": "cat_lat", "lon": "cat_lon"})[["key", "cat_lat", "cat_lon"]], on="key", how="left")


def part_a(config: dict) -> dict:
    spec = config["part_a_controls"]
    learned = pd.read_parquet(MATCHES / "stands.parquet")
    result = {"airports": {}, "sensitivity_recommended": {}}
    passes = 0
    for icao in spec["control_airports"]:
        for role, bucket in [("pre_data", result["airports"]), ("recommended", result["sensitivity_recommended"])]:
            cat, counts = catalogue(icao, role)
            rows = attach(learned[learned["airport"].eq(icao)], cat, "stand")
            d = metres(rows["lat"].to_numpy(), rows["lon"].to_numpy(), rows["cat_lat"].to_numpy(), rows["cat_lon"].to_numpy())
            summary = distance_summary(d)
            entry = {**counts, "learned_stands": int(len(rows)), "resolved_learned_stands": summary["n"], "distance": summary}
            if role == "pre_data":
                entry["passes"] = bool(summary["n"] >= 30 and summary["median_m"] <= 25 and summary["within_40m"] >= 0.80)
                passes += entry["passes"]
            bucket[icao] = entry
    result["airports_passing"] = passes
    result["catalogue_accepted"] = bool(passes >= 5)

    cat, counts = catalogue("LFPG", "pre_data")
    rec, rec_counts = catalogue("LFPG", "recommended")
    gates = pd.concat([pd.read_parquet(MATCHES / f"matches2_{m}.parquet", columns=["airport", "stand", "gate", "anchor_lat", "anchor_lon"]).assign(month=m)
                       for m in ALL_MONTHS], ignore_index=True)
    gates = attach(gates[gates["airport"].eq("LFPG") & gates["gate"]], cat, "stand")
    anchor = distance_summary(metres(gates["anchor_lat"].to_numpy(), gates["anchor_lon"].to_numpy(), gates["cat_lat"].to_numpy(), gates["cat_lon"].to_numpy()))
    own = attach(learned[learned["airport"].eq("LFPG")], cat, "stand")
    own_d = metres(own["lat"].to_numpy(), own["lon"].to_numpy(), own["cat_lat"].to_numpy(), own["cat_lon"].to_numpy())
    both = cat.merge(rec, on="key", suffixes=("_pre", "_rec"))
    shift = metres(both["lat_pre"].to_numpy(), both["lon_pre"].to_numpy(), both["lat_rec"].to_numpy(), both["lon_rec"].to_numpy())
    lfpg_ok = result["catalogue_accepted"] and (anchor["n"] < 30 or anchor["median_m"] <= 25)
    result["lfpg"] = {"pre_data": counts, "recommended": rec_counts, "gate_matches": int(len(gates)), "gate_anchor_distance": anchor,
                      "learned_stands": {s: (None if not np.isfinite(v) else float(v)) for s, v in zip(own["stand"], own_d)},
                      "version_shift": {"keys_in_both": int(len(both)), "moved_over_10m": int((shift > 10).sum()), "max_m": float(shift.max()) if len(shift) else None,
                                        "only_pre_data": int(len(set(cat["key"]) - set(rec["key"]))), "only_recommended": int(len(set(rec["key"]) - set(cat["key"])))},
                      "accepted": bool(lfpg_ok)}
    return result


def raw_departures(month: str) -> pd.DataFrame:
    if month.startswith("2026"):
        raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, PHASE, TIME, "ADEP_mvt", "STAND_mvt"])
        raw = raw[raw[TIME].dt.strftime("%Y-%m").eq(month)]
    else:
        y, m = int(month[:4]), int(month[5:7])
        name = f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"
        raw = pd.read_parquet(ROOT / "data/raw" / name, columns=[ID, PHASE, TIME, "ADEP_mvt", "STAND_mvt"])
    return raw[raw[PHASE].eq("DEP") & raw["ADEP_mvt"].eq("LFPG")]


def lfpg_matches(month: str, cat: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_parquet(MATCHES / f"matches2_{month}.parquet")
    m = attach(m[m["airport"].eq("LFPG")].reset_index(drop=True), cat, "stand")
    m["d"] = metres(m["first_lat"].to_numpy(), m["first_lon"].to_numpy(), m["cat_lat"].to_numpy(), m["cat_lon"].to_numpy())
    return m


def part_b(cat: pd.DataFrame) -> dict:
    out = {}
    for month in ALL_MONTHS:
        dep = attach(raw_departures(month), cat, "STAND_mvt")
        m = lfpg_matches(month, cat)
        dark = m[~m["gate"]]
        resolved_dark = dark[dark["cat_lat"].notna()]
        out[month] = {"departures": int(len(dep)), "stand_resolved": int(dep["cat_lat"].notna().sum()),
                      "matches": int(len(m)), "gate": int(m["gate"].sum()), "dark": int(len(dark)), "dark_resolved": int(len(resolved_dark)),
                      "dark_first_point": distance_summary(resolved_dark["d"].to_numpy()),
                      "dark_within_40m": int((resolved_dark["d"] <= 40).sum())}
    return out


def part_c(config: dict, cat: pd.DataFrame) -> dict:
    v25 = read_json(ROOT / "configs/adsb_v25_v1.json")
    frames, whole, excluded = [], {}, {}
    for month in MONTHS_2025:
        m = lfpg_matches(month, cat)
        m = m[~m["gate"] & m["cat_lat"].notna()][[ID, "d"]]
        base = pd.read_parquet(ROOT / v25["cross_fit"]["scratch"] / f"{month}.parquet", columns=[ID, TARGET, "v9", "airport"])
        preds = pd.read_parquet(ROOT / f"runs/adsb_v25/dev/predictions_{month}.parquet")
        table = base.merge(preds, on=ID, how="left", validate="one_to_one")
        if month in RECIPE_MONTHS:
            label, recipe = table[TARGET].to_numpy(float), table["mean_adj2_cat"].to_numpy(float)
            whole[month] = {"rows": int(len(table)), "sse": float(np.sum((label - recipe) ** 2))}
        # Deviation (recorded in the results): the residual-stage tables omit some departures (2,502 in June 2025), which have no
        # out-of-month prediction and so no target; they are excluded and counted.
        rows = m.merge(table, on=ID, how="inner", validate="one_to_one")
        excluded[month] = int(len(m) - len(rows))
        if not rows["airport"].eq("LFPG").all():
            raise ValueError(f"LFPG dark matches of {month} do not align with the residual-stage table.")
        rows["month"] = month
        rows["r"] = rows[TARGET] - (rows["v9"] + 0.5 * rows["residual:adj2"] + 0.5 * rows["residual:cat"])
        rows["r_recipe"] = rows[TARGET] - rows["mean_adj2_cat"] if "mean_adj2_cat" in rows else np.nan
        frames.append(rows[[ID, "month", "d", TARGET, "r", "r_recipe"]])
    data = pd.concat(frames, ignore_index=True)
    data["f0"], data["f1"] = np.nan, np.nan
    for month in MONTHS_2025:
        held = data["month"].eq(month).to_numpy()
        fit = ~held & (data[TARGET].to_numpy() <= 7200)
        if not held.any():
            continue
        r, d = data.loc[fit, "r"].to_numpy(), data.loc[fit, "d"].to_numpy()
        f0 = float(r.mean())
        edges = np.quantile(d, np.linspace(0.1, 0.9, 9))
        bins = np.digitize(d, edges)
        shrunk = np.array([(r[bins == b].sum() + 100 * f0) / ((bins == b).sum() + 100) for b in range(10)])
        data.loc[held, "f0"] = f0
        data.loc[held, "f1"] = shrunk[np.digitize(data.loc[held, "d"].to_numpy(), edges)]
    e0, e1 = data["r"] - data["f0"], data["r"] - data["f1"]
    per_month = {}
    for month, g in data.groupby("month"):
        s0, s1 = float(np.sum(e0[g.index] ** 2)), float(np.sum(e1[g.index] ** 2))
        entry = {"rows": int(len(g)), "rmse_f0": float(np.sqrt(s0 / len(g))), "rmse_f1": float(np.sqrt(s1 / len(g))),
                 "residual_path_rmse": float(np.sqrt(np.mean(g["r"] ** 2))), "corr_d_r": float(np.corrcoef(g["d"], g["r"])[0, 1]) if len(g) > 2 else None}
        if month in whole:
            entry["recipe_rmse_on_rows"] = float(np.sqrt(np.mean(g["r_recipe"] ** 2)))
            old = whole[month]["sse"]
            entry["recipe_month_bound_pct"] = float(100 * (np.sqrt((old - (s0 - s1)) / old) - 1))
        per_month[month] = entry
    eligible = [m for m, e in per_month.items() if e["rows"] >= 50]
    better = [m for m in eligible if per_month[m]["rmse_f1"] < per_month[m]["rmse_f0"]]
    bounds = [per_month[m]["recipe_month_bound_pct"] for m in RECIPE_MONTHS if m in per_month]
    ordinary = data[TARGET].to_numpy() <= 7200
    return {"rows": int(len(data)), "excluded_not_in_residual_table": excluded, "d": distance_summary(data["d"].to_numpy()),
            "pooled": {"rmse_f0": float(np.sqrt(np.mean(e0 ** 2))), "rmse_f1": float(np.sqrt(np.mean(e1 ** 2))),
                       "ratio": float(np.sqrt(np.mean(e1 ** 2)) / np.sqrt(np.mean(e0 ** 2))),
                       "ordinary_rows": int(ordinary.sum()), "ordinary_ratio": float(np.sqrt(np.mean(e1[ordinary] ** 2)) / np.sqrt(np.mean(e0[ordinary] ** 2)))},
            "months_eligible": eligible, "months_f1_better": better,
            "mean_recipe_month_bound_pct": float(np.mean(bounds)) if bounds else None, "per_month": per_month, "whole_recipe_months": whole}


def main():
    config = read_json(CONFIG)
    started = utc_now()
    a = part_a(config)
    cat, _ = catalogue("LFPG", "pre_data")
    b = part_b(cat)
    c = part_c(config, cat)
    months_ok = len(c["months_eligible"]) > 0 and len(c["months_f1_better"]) >= 0.75 * len(c["months_eligible"])
    bound = c["mean_recipe_month_bound_pct"]
    go = bool(a["lfpg"]["accepted"] and c["pooled"]["ratio"] <= 0.97 and months_ok and bound is not None and bound <= -0.05)
    write_json(OUTPUT, {"experiment_id": config["experiment_id"], "config_sha256": sha256_file(CONFIG), "code_sha256": sha256_file(Path(__file__)),
                        "sources_sha256": sha256_file(SOURCES), "stands_sha256": sha256_file(MATCHES / "stands.parquet"),
                        "started_utc": started, "finished_utc": utc_now(), "part_a": a, "part_b": b, "part_c": c,
                        "decision": {"lfpg_accepted": a["lfpg"]["accepted"], "pooled_ratio": c["pooled"]["ratio"], "months_rule_met": bool(months_ok),
                                     "mean_recipe_month_bound_pct": bound, "go_for_screen": go}})
    print(f"part A: {a['airports_passing']}/8 control airports pass; catalogue accepted {a['catalogue_accepted']}; LFPG accepted {a['lfpg']['accepted']}")
    print(f"part C: {c['rows']} rows, pooled ratio {c['pooled']['ratio']:.4f}, months better {len(c['months_f1_better'])}/{len(c['months_eligible'])}, bound {bound}")
    print(f"go_for_screen: {go}")


if __name__ == "__main__":
    main()
