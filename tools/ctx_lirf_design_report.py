"""Round B design readings (configs/adsb_ctx_lirf_design_v1.json): the LIRF-inclusive context network, October only.

Rows: the recipe proxy's October table (tools/nn_design_report.RecipeProxy); LIRF non-population rows split by ADS-B match.
L1  LIRF non-population RMSE (all / matched / unmatched) per run and per variant seed mean, against the nnl mean of the same seed count
    (nnl = the 16 adsb_nn2 development seeds' seasonal-fold October predictions, in seed order);
L2  (1 - b) x v9 + b x network, b in {0.25, 0.3, 0.5}, variant mean against nnl mean at equal seed count; paired-day 95% interval of the
    difference at b = 0.3 on the LIRF non-population rows;
L3  descriptive: non-LIRF October RMSE of ctx_lirf seed means against the round-1 ctx design seeds at equal counts, and the round-1
    recipe-proxy reading R2 at b = 1 (table = 0.5 committee + 0.5 network) for both.
Decision (declared): GO if at 4 seeds the ctx_lirf L2 blend at b = 0.3 is at least 1% below nnl's with the interval below zero; the
development variant is ctx_lirf_noadsb instead if, at 2 seeds each, it is within 0.5% of ctx_lirf on all LIRF non-population rows and
better on the unmatched ones. No scoring month is read.
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
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402

DECLARATION = ROOT / "configs/adsb_ctx_lirf_design_v1.json"
B_VALUES = [0.25, 0.3, 0.5]


def read(path: Path, ids: pd.Series) -> np.ndarray:
    return pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx/lirf_design"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx_lirf/design_report_1.json"))
    args = parser.parse_args()
    decl = read_json(DECLARATION)
    proxy = RecipeProxy(pd.read_parquet(ROOT / args.committee))
    table = proxy.table
    ids, label = table[ID], table[TARGET].to_numpy(dtype=float)
    v9 = table["v9"].to_numpy(dtype=float)
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    lirf = table["airport"].eq("LIRF").to_numpy() & ~table["population"].astype(bool).to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    non_lirf = proxy.non_lirf
    parts = {"all": lirf, "matched": lirf & matched, "unmatched": lirf & ~matched}

    runs = {}
    for variant, seeds in decl["runs"].items():
        for seed in seeds:
            path = ROOT / args.runs / f"{variant}_s{seed}" / "design"
            record = read_json(path / "run.json")
            if record["config_sha256"] != sha256_file(DECLARATION) or record["variant"] != variant or record["seed"] != seed:
                raise ValueError(f"{path} was not fitted under the round-B design declaration.")
            runs.setdefault(variant, {})[seed] = read(path / "stop_predictions.parquet", ids)
    nnl = [read(ROOT / f"runs/adsb_nn2/nnl_s{s}/seasonal/stop_predictions.parquet", ids) for s in range(201, 217)]
    ctx1 = [read(ROOT / f"runs/adsb_ctx/design/ctx_s{s}/design/stop_predictions.parquet", ids) for s in range(1, 7)]
    for values in [*nnl, *[v for r in runs.values() for v in r.values()]]:
        if not np.isfinite(values[lirf]).all():
            raise ValueError("A LIRF network does not cover every LIRF non-population October row.")

    def mean(values: list[np.ndarray], n: int) -> np.ndarray:
        return np.mean(values[:n], axis=0)

    def l1(net: np.ndarray) -> dict:
        return {p: rmse(label[k], net[k]) for p, k in parts.items()}

    def l2(net: np.ndarray) -> dict:
        return {str(b): {p: rmse(label[k], (1 - b) * v9[k] + b * net[k]) for p, k in parts.items()} for b in B_VALUES}

    report = {"created_at_utc": utc_now(), "declaration_sha256": sha256_file(DECLARATION), "rows": {p: int(k.sum()) for p, k in parts.items()},
              "v9": {p: rmse(label[k], v9[k]) for p, k in parts.items()}, "runs": {}, "variants": {}, "audit_used": False, "note": "October only; no scoring month read."}
    for variant, seeds in runs.items():
        report["runs"][variant] = {str(s): l1(v) for s, v in seeds.items()}
        values = [seeds[s] for s in sorted(seeds)]
        entry = {}
        for n in range(1, len(values) + 1):
            mine, theirs = mean(values, n), mean(nnl, n)
            blend_mine, blend_theirs = (0.7 * v9 + 0.3 * mine), (0.7 * v9 + 0.3 * theirs)
            entry[str(n)] = {"l1": l1(mine), "nnl_l1": l1(theirs), "l2": l2(mine), "nnl_l2": l2(theirs),
                             "l2_b030_change_pct": {p: 100 * (rmse(label[k], blend_mine[k]) / rmse(label[k], blend_theirs[k]) - 1) for p, k in parts.items()},
                             "l2_b030_interval_s": day_interval(days[lirf], label[lirf], blend_mine[lirf], blend_theirs[lirf])}
            if variant == "ctx_lirf" and n <= len(ctx1):
                entry[str(n)]["l3"] = {"non_lirf_rmse": rmse(label[non_lirf], mine[non_lirf]), "round1_ctx_non_lirf_rmse": rmse(label[non_lirf], mean(ctx1, n)[non_lirf]),
                                       "r2_b1_pct": 100 * (rmse(label, proxy.values(0.5 * proxy.committee + 0.5 * mine)) / rmse(label, proxy.control) - 1),
                                       "round1_ctx_r2_b1_pct": 100 * (rmse(label, proxy.values(0.5 * proxy.committee + 0.5 * mean(ctx1, n))) / rmse(label, proxy.control) - 1)}
        report["variants"][variant] = entry
    go_entry = report["variants"].get("ctx_lirf", {}).get("4")
    go = bool(go_entry and go_entry["l2_b030_change_pct"]["all"] <= -1.0 and go_entry["l2_b030_interval_s"][1] < 0)
    variant = "ctx_lirf"
    a, b = report["variants"].get("ctx_lirf", {}).get("2"), report["variants"].get("ctx_lirf_noadsb", {}).get("2")
    if a and b:
        overall = 100 * (b["l1"]["all"] / a["l1"]["all"] - 1)
        report["variant_choice"] = {"noadsb_vs_ctx_lirf_all_pct": overall, "noadsb_unmatched": b["l1"]["unmatched"], "ctx_lirf_unmatched": a["l1"]["unmatched"]}
        if overall <= 0.5 and b["l1"]["unmatched"] < a["l1"]["unmatched"]:
            variant = "ctx_lirf_noadsb"
    report["decision"] = {"go": go, "development_variant": variant if go else None, "rule": decl["protocol"]["decision"]}
    write_json(ROOT / args.report, report)
    print("rows", report["rows"], "v9", {p: round(x, 2) for p, x in report["v9"].items()})
    for variant, entry in report["variants"].items():
        for n, e in entry.items():
            print(f"{variant:16s} n={n} L1 {{{', '.join(f'{p} {e['l1'][p]:.2f}/{e['nnl_l1'][p]:.2f}' for p in parts)}}} "
                  f"L2 b0.3 change {{{', '.join(f'{p} {e['l2_b030_change_pct'][p]:+.2f}%' for p in parts)}}} interval {[round(x, 2) for x in e['l2_b030_interval_s']]}"
                  + (f" | non-LIRF {e['l3']['non_lirf_rmse']:.2f} vs ctx {e['l3']['round1_ctx_non_lirf_rmse']:.2f}, R2 {e['l3']['r2_b1_pct']:+.3f}% vs {e['l3']['round1_ctx_r2_b1_pct']:+.3f}%" if "l3" in e else ""))
    print("decision", report["decision"], report.get("variant_choice"))


if __name__ == "__main__":
    main()
