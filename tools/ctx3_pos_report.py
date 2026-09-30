"""Readings and decision of the positional batch (configs/adsb_ctx3_pos_v1.json).

R1 / R2 on October exactly as tools/ctx3_combo_report.py (pool P, committee, share 0.75, b in {0.25, 0.5, 1.0}; share 0.85 beside it),
with each variant's four-seed mean as the item. R3 (transfer): the four-seed means of tools/ctx3_pos_predict.py's July / December
predictions, natural and thinned, on non-LIRF departures; thinned rows = EDDF December, EDDF and LSZH July. Before any July or December
label is read, every run's scorer check (October re-prediction against its own stop_predictions, RMS <= 0.05 s) must pass.
GO (declared): (a) pooled thinned-row RMSE on thinned inputs below ref32's, paired-day 95% interval of the difference below zero; (b) whole
non-LIRF RMSE on thinned inputs below ref32's in July and in December; (c) October R2 at b = 0.5 at most 0.02 points above ref32's.
Of several GO items the lowest pooled thinned-row RMSE is carried forward.
Usage: tools/ctx3_pos_report.py [--allow-missing]
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
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402
from ctx_design_report import october  # noqa: E402
from run_adsb_learner import source_file  # noqa: E402

DECLARATION = "configs/adsb_ctx3_pos_v1.json"
B_VALUES = [0.25, 0.5, 1.0]
SHARE, OTHER_SHARE = 0.75, 0.85
COST_POINTS = 0.02
ITEMS = ["nopos", "nocount", "nopos_nocount"]
THINNED = {"2025-07": ["EDDF", "LSZH"], "2025-12": ["EDDF"]}
DECEMBER_BUILD = "data/december/d198dea10625b0a75977"


def month_truth(month: str, catalog: pd.DataFrame, parent: Path) -> pd.DataFrame:
    """ID, airport, day, label of every departure of the month."""
    name = source_file(month)
    if month == "2025-12":
        rows = pd.read_parquet(ROOT / DECEMBER_BUILD / "features" / name, columns=[ID, "airport"])
        times = pd.read_parquet(ROOT / "data/raw" / name, columns=[ID, TIME])
        rows = rows.merge(times, on=ID, how="left", validate="one_to_one")
        labels = pd.read_parquet(ROOT / DECEMBER_BUILD / "labels" / name)
    else:
        rows = catalog.loc[catalog["source_file"].eq(name), [ID, "airport", TIME]]
        labels = pd.read_parquet(parent / "labels" / name)
    out = rows.merge(labels, on=ID, how="left", validate="one_to_one")
    if out[TARGET].isna().any() or out[TIME].isna().any():
        raise ValueError(f"{month}: labels or times missing.")
    out["day"] = pd.to_datetime(out[TIME]).dt.strftime("%Y-%m-%d")
    return out[[ID, "airport", "day", TARGET]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx3_pos/design"))
    parser.add_argument("--transfer", type=Path, default=Path("runs/adsb_ctx3_pos/transfer"))
    parser.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx3_pos/design_report.json"))
    parser.add_argument("--allow-missing", action="store_true", help="interim reading on the seeds every variant has (never the decision)")
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    decl_sha = sha256_file(ROOT / DECLARATION)

    # runs and the scorer check (before any July / December label)
    runs = {}
    for path in sorted((ROOT / args.runs).glob("*/design/run.json")):
        record = read_json(path)
        if record["config_sha256"] != decl_sha or record.get("smoke_limit_rows") is not None or record["inputs"]["device"] != "cpu":
            raise ValueError(f"{path} was not fitted under this declaration.")
        if record["variant"] not in decl["runs"] or record["seed"] not in decl["runs"][record["variant"]]:
            raise ValueError(f"{path}: undeclared run.")
        runs.setdefault(record["variant"], {})[record["seed"]] = {"dir": path.parent, "fit_seconds": record["fit_seconds"],
                                                                   "curve_last": [round(e["monitor_rmse"], 3) for e in record["history"][-3:]],
                                                                   "model_sha256": record["model_sha256"]}
    missing = {v: sorted(set(s) - set(runs.get(v, {}))) for v, s in decl["runs"].items() if set(s) - set(runs.get(v, {}))}
    if missing and not args.allow_missing:
        raise ValueError(f"Declared runs missing: {missing}")
    seeds = sorted(set.intersection(*[set(runs.get(v, {})) for v in decl["runs"]]))
    if not seeds:
        raise ValueError("No seed is complete over the four variants.")
    transfer = read_json(ROOT / args.transfer / "predictions.json")
    if transfer["declaration_sha256"] != decl_sha:
        raise ValueError("Transfer predictions were made under another declaration.")
    checks = {}
    for v in decl["runs"]:
        for s in seeds:
            info = transfer["runs"].get(f"{v}_s{s}")
            if info is None or info["model_sha256"] != runs[v][s]["model_sha256"]:
                raise ValueError(f"{v}_s{s}: transfer predictions missing or of another model.")
            for key, sha in info["files"].items():
                if sha256_file(ROOT / args.transfer / f"{v}_s{s}" / f"{key}.parquet") != sha:
                    raise ValueError(f"{v}_s{s} {key}: prediction file changed.")
            checks[f"{v}_s{s}"] = info["scorer_check"]
    failed = [k for k, c in checks.items() if not c["passes"]]
    if failed:
        raise ValueError(f"Scorer check failed (no July / December label read): {failed}")

    # R1 / R2 on October (as the combination batch)
    proxy = RecipeProxy(pd.read_parquet(ROOT / args.committee))
    ids, label, non_lirf, table = proxy.ids, proxy.label, proxy.non_lirf, proxy.table
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    airport = table["airport"].to_numpy()
    committee = proxy.committee
    ctx = [ROOT / args.pool_runs / f"ctx_s{s}/seasonal/stop_predictions.parquet" for s in range(101, 117)]
    e24 = [ROOT / args.pool_runs / f"ctx_e24_s{s}/seasonal/stop_predictions.parquet" for s in range(117, 133)]
    pool = 0.5 * np.mean([october(p, ids) for p in ctx], axis=0) + 0.5 * np.mean([october(p, ids) for p in e24], axis=0)

    def net(values, rows=None):
        k = non_lirf if rows is None else (non_lirf & rows)
        return rmse(label[k], values[k])

    def proxied(x, b, share=SHARE):
        return proxy.values((1 - share) * committee + share * ((1 - b) * pool + b * x))

    control = {share: rmse(label, proxy.values((1 - share) * committee + share * pool)) for share in (SHARE, OTHER_SHARE)}

    def r2(x, b, share=SHARE):
        return 100 * (rmse(label, proxied(x, b, share)) / control[share] - 1)

    oct_values = {v: {s: october(runs[v][s]["dir"] / "stop_predictions.parquet", ids) for s in seeds} for v in decl["runs"]}
    mean = {v: np.mean([oct_values[v][s] for s in seeds], axis=0) for v in decl["runs"]}
    ref = mean["ref32"]
    ref_proxy, ref_r2 = proxied(ref, 0.5), r2(ref, 0.5)
    items = {}
    for name, x in mean.items():
        entry = {"october": {"rmse_non_lirf": net(x), "rmse_matched": net(x, matched), "rmse_unmatched": net(x, ~matched),
                             "rmse_by_airport": {a: net(x, airport == a) for a in sorted(set(airport[non_lirf]))},
                             "per_seed_rmse_non_lirf": {str(s): net(oct_values[name][s]) for s in seeds},
                             "r2_pct": {str(b): r2(x, b) for b in B_VALUES}, "r2_pct_share_085": {str(b): r2(x, b, OTHER_SHARE) for b in B_VALUES}}}
        if name != "ref32":
            o = entry["october"]
            o.update({"r2_b0.5_minus_ref32_points": o["r2_pct"]["0.5"] - ref_r2, "standalone_vs_ref32_pct": 100 * (o["rmse_non_lirf"] / net(ref) - 1),
                      "interval_vs_ref32_b0.5_s": day_interval(days, label, proxied(x, 0.5), ref_proxy)})
        items[name] = entry

    # R3: July and December, natural and thinned (labels read only now)
    manifest, parent, original, catalog = load_bundle(ROOT, ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"])
    pooled = {}
    for month, thinned_airports in THINNED.items():
        truth = month_truth(month, catalog, parent)
        mids = truth[ID]
        y = truth[TARGET].to_numpy(dtype=float)
        ap = truth["airport"].to_numpy()
        mdays = truth["day"].to_numpy()
        rows_all = ap != "LIRF"
        rows_thin = np.isin(ap, thinned_airports)

        def seed_mean(variant, reading):
            return np.mean([pd.read_parquet(ROOT / args.transfer / f"{variant}_s{s}" / f"{month}_{reading}.parquet").set_index(ID)["adsb_full"]
                            .reindex(mids).to_numpy(dtype=float) for s in seeds], axis=0)

        values = {v: {r: seed_mean(v, r) for r in ("natural", "natural_cov", "thinned")} for v in decl["runs"]}
        for v, by in values.items():
            if not all(np.isfinite(a[rows_all]).all() for a in by.values()):
                raise ValueError(f"{v} {month}: predictions missing.")
            m = {}
            for r, a in by.items():
                m[r] = {"rmse_non_lirf": rmse(y[rows_all], a[rows_all]), "rmse_thinned_rows": rmse(y[rows_thin], a[rows_thin]),
                        "rmse_by_thinned_airport": {t: rmse(y[ap == t], a[ap == t]) for t in thinned_airports}}
            m["penalty_thinned_rows_pct"] = 100 * (m["thinned"]["rmse_thinned_rows"] / m["natural"]["rmse_thinned_rows"] - 1)
            m["penalty_non_lirf_pct"] = 100 * (m["thinned"]["rmse_non_lirf"] / m["natural"]["rmse_non_lirf"] - 1)
            m["thinned_rows"] = int(rows_thin.sum())
            m["natural_vs_natural_cov_rms_s"] = float(np.sqrt(np.mean((by["natural"][rows_all] - by["natural_cov"][rows_all]) ** 2)))
            if v != "ref32":
                ref_t = values["ref32"]["thinned"]
                m["thinned_non_lirf_vs_ref32_pct"] = 100 * (m["thinned"]["rmse_non_lirf"] / rmse(y[rows_all], ref_t[rows_all]) - 1)
                m["thinned_rows_vs_ref32_pct"] = 100 * (m["thinned"]["rmse_thinned_rows"] / rmse(y[rows_thin], ref_t[rows_thin]) - 1)
                m["natural_non_lirf_vs_ref32_pct"] = 100 * (m["natural"]["rmse_non_lirf"] / rmse(y[rows_all], values["ref32"]["natural"][rows_all]) - 1)
            items[v].setdefault("transfer", {})[month] = m
            pooled.setdefault(v, {"label": [], "thinned": [], "natural": [], "day": []})
            pooled[v]["label"].append(y[rows_thin]); pooled[v]["thinned"].append(by["thinned"][rows_thin])
            pooled[v]["natural"].append(by["natural"][rows_thin]); pooled[v]["day"].append(mdays[rows_thin])
    pl = {v: {k: np.concatenate(p[k]) for k in p} for v, p in pooled.items()}
    ref_pool = pl["ref32"]
    for v in decl["runs"]:
        p = pl[v]
        entry = {"rows": int(len(p["label"])), "rmse_thinned": rmse(p["label"], p["thinned"]), "rmse_natural": rmse(p["label"], p["natural"])}
        entry["penalty_pct"] = 100 * (entry["rmse_thinned"] / entry["rmse_natural"] - 1)
        if v != "ref32":
            entry["thinned_vs_ref32_pct"] = 100 * (entry["rmse_thinned"] / rmse(ref_pool["label"], ref_pool["thinned"]) - 1)
            entry["interval_thinned_vs_ref32_s"] = day_interval(p["day"], p["label"], p["thinned"], ref_pool["thinned"])
            entry["natural_vs_ref32_pct"] = 100 * (entry["rmse_natural"] / rmse(ref_pool["label"], ref_pool["natural"]) - 1)
        items[v]["transfer_pooled_thinned_rows"] = entry

    for v in ITEMS:
        e = items[v]
        a = e["transfer_pooled_thinned_rows"]["rmse_thinned"] < items["ref32"]["transfer_pooled_thinned_rows"]["rmse_thinned"] and \
            e["transfer_pooled_thinned_rows"]["interval_thinned_vs_ref32_s"][1] < 0
        b = all(e["transfer"][m]["thinned_non_lirf_vs_ref32_pct"] < 0 for m in THINNED)
        c = e["october"]["r2_b0.5_minus_ref32_points"] <= COST_POINTS
        e["decision"] = {"a_thinned_rows_better_interval_below_zero": bool(a), "b_both_months_better_on_thinned_inputs": bool(b),
                         "c_october_cost_within_0.02_points": bool(c), "go": bool(a and b and c and not missing)}
    go = [v for v in ITEMS if items[v]["decision"]["go"]]
    carried = min(go, key=lambda v: items[v]["transfer_pooled_thinned_rows"]["rmse_thinned"]) if go else None
    report = {"created_at_utc": utc_now(), "declaration_sha256": decl_sha, "seeds_read": seeds, "missing": missing, "interim": bool(missing),
              "scorer_checks": checks, "control_proxy_rmse": control[SHARE], "control_proxy_rmse_share_085": control[OTHER_SHARE],
              "pool_rmse_non_lirf": net(pool), "ref32_r2_b0.5": ref_r2, "items": items, "go": go, "carried_forward": carried,
              "runs": {v: {str(s): {k: r[k] for k in ("fit_seconds", "curve_last", "model_sha256")} for s, r in by.items()} for v, by in runs.items()},
              "transfer_receipt_sha256": sha256_file(ROOT / args.transfer / "predictions.json"), "decision": decl["protocol"]["decision"],
              "tool_sha256": sha256_file(ROOT / "tools/ctx3_pos_report.py"), "audit_used": False,
              "labels_read": "October, July and December 2025 (as declared); no ranking row or official score"}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print(f"seeds {seeds}; control proxy {control[SHARE]:.3f}; ref32 R2(b=0.5) {ref_r2:+.4f}%; scorer checks max RMS "
          f"{max(c['rms_s'] for c in checks.values()):.4f} s")
    for v, e in items.items():
        o, t = e["october"], e["transfer_pooled_thinned_rows"]
        extra = "" if v == "ref32" else (f" | Oct standalone {o['standalone_vs_ref32_pct']:+.3f}%, R2 cost {o['r2_b0.5_minus_ref32_points']:+.4f} pts | thinned rows "
                                          f"{t['thinned_vs_ref32_pct']:+.3f}% [{t['interval_thinned_vs_ref32_s'][0]:+.2f}, {t['interval_thinned_vs_ref32_s'][1]:+.2f}] s | "
                                          f"months {[round(e['transfer'][m]['thinned_non_lirf_vs_ref32_pct'], 3) for m in THINNED]} | {e['decision']}")
        print(f"{v:14s} Oct {o['rmse_non_lirf']:.3f} R2 {o['r2_pct']['0.5']:+.4f}% | thinned rows nat {t['rmse_natural']:.2f} thin {t['rmse_thinned']:.2f} "
              f"(penalty {t['penalty_pct']:+.2f}%){extra}")
    print("GO:", go, "carried forward:", carried)


if __name__ == "__main__":
    main()
