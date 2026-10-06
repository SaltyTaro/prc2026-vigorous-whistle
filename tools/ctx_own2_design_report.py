"""Design readings of lever 5 (configs/ctx_own2_design_v1.json): the own-points context-network member with the reference recipe, October only.

As tools/ctx_ownpoints_report.py (tunnel 21: pool P, share 0.75, R2 grid, paired-day interval), with v66's member weight b = 1.0 as the decision:
  control        dl32own on rows with adsb_matched = 1, m25dl32own on the other rows (own_all as shipped; refitted in this batch on the GPU)
  ref_own        r32own on matched rows, m25r32own on the others (decision item)
  mix_own        0.5 control + 0.5 ref_own (decision item)
  ref_matched    r32own on matched rows, m25dl32own on the others (reported)
  ref_unmatched  dl32own on matched rows, m25r32own on the others (reported)
GO (declared) per decision item: R2 at b = 1.0 at least 0.05 points below the control's, 8-seed standalone October non-LIRF RMSE below the
control's, paired-day 95% interval of the proxy difference at b = 1.0 below zero. Also reports the GPU control fits against tunnel 21's CPU
fits of seeds 9-12 (same settings; not exact across devices). No scoring month, ranking row or official score is read.
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
from prc2026.features import ID, TIME  # noqa: E402
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402
from ctx_design_report import october  # noqa: E402

DECLARATION = "configs/ctx_own2_design_v1.json"
B_VALUES = [0.4, 0.5, 1.0]
B_DECISION = 1.0
SHARE, OTHER_SHARE = 0.75, 0.85
BAR_POINTS = 0.05
PAIRS = {"control": ("dl32own", "m25dl32own"), "ref_own": ("r32own", "m25r32own"), "ref_matched": ("r32own", "m25dl32own"),
         "ref_unmatched": ("dl32own", "m25r32own")}
DECISION_ITEMS = ("ref_own", "mix_own")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/ctx_own2/design"))
    parser.add_argument("--reference-runs", type=Path, default=Path("runs/ctx_ownpoints/design"))
    parser.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--own", type=Path, default=Path("data/external/adsblol/features_v19own/features_v19_2025-10.parquet"))
    parser.add_argument("--report", type=Path, default=Path("docs/ctx_own2/design_report.json"))
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    decl_sha = sha256_file(ROOT / DECLARATION)
    proxy = RecipeProxy(pd.read_parquet(ROOT / args.committee))
    ids, label, non_lirf, table = proxy.ids, proxy.label, proxy.non_lirf, proxy.table
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    airport = table["airport"].to_numpy()
    own = pd.read_parquet(ROOT / args.own, columns=[ID, "p_has_inb", "p_first_mv"]).set_index(ID).reindex(ids)
    has_inb = own["p_has_inb"].eq(1).to_numpy()
    has_motion = own["p_first_mv"].notna().to_numpy()
    committee = proxy.committee
    ctx = [ROOT / args.pool_runs / f"ctx_s{s}/seasonal/stop_predictions.parquet" for s in range(101, 117)]
    e24 = [ROOT / args.pool_runs / f"ctx_e24_s{s}/seasonal/stop_predictions.parquet" for s in range(117, 133)]
    pool = 0.5 * np.mean([october(p, ids) for p in ctx], axis=0) + 0.5 * np.mean([october(p, ids) for p in e24], axis=0)

    def net(values, rows=None):
        k = non_lirf if rows is None else (non_lirf & rows)
        return rmse(label[k], values[k])

    def proxied(x, b, share=SHARE):
        return proxy.values((1 - share) * committee + share * ((1 - b) * pool + b * x))

    base = {share: rmse(label, proxy.values((1 - share) * committee + share * pool)) for share in (SHARE, OTHER_SHARE)}

    def r2(x, b, share=SHARE):
        return 100 * (rmse(label, proxied(x, b, share)) / base[share] - 1)

    runs = {}
    for path in sorted((ROOT / args.runs).glob("*/design/run.json")):
        record = read_json(path)
        if record["config_sha256"] != decl_sha or record.get("smoke_limit_rows") is not None or record["inputs"]["device"] != "cuda":
            raise ValueError(f"{path} was not fitted under this declaration on the GPU.")
        if record["variant"] not in decl["runs"] or record["seed"] not in decl["runs"][record["variant"]]:
            raise ValueError(f"{path}: undeclared run.")
        runs.setdefault(record["variant"], {})[record["seed"]] = {"values": october(path.parent / "stop_predictions.parquet", ids), "fit_seconds": record["fit_seconds"],
                                                                   "curve_last": [round(e["monitor_rmse"], 3) for e in record["history"][-3:]],
                                                                   "epochs": len(record["history"]),
                                                                   "extra_features_used": len(record["inputs"]["extra_features_used"]),
                                                                   "predictions_sha256": sha256_file(path.parent / "stop_predictions.parquet")}
    missing = {v: sorted(set(s) - set(runs.get(v, {}))) for v, s in decl["runs"].items() if set(s) - set(runs.get(v, {}))}
    if missing:
        raise ValueError(f"Declared runs missing: {missing}")
    seeds = sorted(decl["runs"]["dl32own"])
    mean = {v: np.mean([runs[v][s]["values"] for s in seeds], axis=0) for v in decl["runs"]}
    reproduction = {}
    for v in ("dl32own", "m25dl32own"):
        for s in seeds:
            ref = ROOT / args.reference_runs / f"{v}_s{s}/design/stop_predictions.parquet"
            if ref.exists():
                old = october(ref, ids)
                reproduction[f"{v}_s{s}"] = {"rmse_cpu": net(old), "rmse_gpu": net(runs[v][s]["values"]),
                                             "rms_difference_s": float(np.sqrt(np.nanmean((old[non_lirf] - runs[v][s]["values"][non_lirf]) ** 2)))}
    tables = {name: np.where(matched, mean[a], mean[b]) for name, (a, b) in PAIRS.items()}
    tables["mix_own"] = 0.5 * tables["control"] + 0.5 * tables["ref_own"]
    ctrl = tables["control"]
    ctrl_proxy = proxied(ctrl, B_DECISION)
    items = {}
    for name, x in tables.items():
        entry = {"rmse_non_lirf": net(x), "rmse_matched": net(x, matched), "rmse_unmatched": net(x, ~matched),
                 "rmse_inbound_identified": net(x, has_inb), "rmse_own_motion_seen": net(x, has_motion),
                 "rmse_by_airport": {a: net(x, airport == a) for a in sorted(set(airport[non_lirf]))},
                 "r2_pct": {str(b): r2(x, b) for b in B_VALUES}, "r2_pct_share_085": {str(b): r2(x, b, OTHER_SHARE) for b in B_VALUES}}
        if name != "control":
            v = proxied(x, B_DECISION)
            entry.update({"gain_points_vs_control": {str(b): r2(ctrl, b) - entry["r2_pct"][str(b)] for b in B_VALUES},
                          "gain_points_vs_control_share_085": {str(b): r2(ctrl, b, OTHER_SHARE) - entry["r2_pct_share_085"][str(b)] for b in B_VALUES},
                          "standalone_vs_control_pct": 100 * (entry["rmse_non_lirf"] / net(ctrl) - 1),
                          "standalone_below_control": bool(entry["rmse_non_lirf"] < net(ctrl)),
                          "interval_vs_control_b1.0_s": day_interval(days, label, v, ctrl_proxy),
                          "proxy_change_vs_control_b1.0_pct": 100 * (rmse(label, v) / rmse(label, ctrl_proxy) - 1),
                          "by_airport_vs_control_pct": {a: 100 * (net(x, airport == a) / net(ctrl, airport == a) - 1) for a in sorted(set(airport[non_lirf]))},
                          "inbound_identified_vs_control_pct": 100 * (net(x, has_inb) / net(ctrl, has_inb) - 1),
                          "own_motion_seen_vs_control_pct": 100 * (net(x, has_motion) / net(ctrl, has_motion) - 1)})
            if name in DECISION_ITEMS:
                entry["go"] = bool(entry["gain_points_vs_control"][str(B_DECISION)] >= BAR_POINTS and entry["standalone_below_control"]
                                   and entry["interval_vs_control_b1.0_s"][1] < 0)
        items[name] = entry
    networks = {}
    for v in decl["runs"]:
        networks[v] = {"per_seed_rmse_non_lirf": {str(s): net(runs[v][s]["values"]) for s in seeds}, "mean_rmse_non_lirf": net(mean[v]),
                       "mean_rmse_matched": net(mean[v], matched), "mean_rmse_unmatched": net(mean[v], ~matched)}
    corr = {"ref_own_vs_control_error_correlation_non_lirf":
            float(np.corrcoef((tables["ref_own"] - label)[non_lirf], (ctrl - label)[non_lirf])[0, 1])}
    go_items = [n for n in DECISION_ITEMS if items[n]["go"]]
    report = {"created_at_utc": utc_now(), "declaration_sha256": decl_sha, "seeds": seeds, "base_proxy_rmse": base[SHARE], "pool_rmse_non_lirf": net(pool),
              "share": SHARE, "b_decision": B_DECISION, "bar_points": BAR_POINTS, "items": items, "networks": networks, "go_items": go_items, "go": bool(go_items),
              "gpu_controls_vs_tunnel21_cpu_fits": reproduction, "correlation": corr,
              "runs": {v: {str(s): {k: r[k] for k in ("fit_seconds", "epochs", "curve_last", "extra_features_used", "predictions_sha256")} for s, r in by.items()}
                       for v, by in runs.items()},
              "decision": decl["protocol"]["decision"], "tool_sha256": sha256_file(ROOT / "tools/ctx_own2_design_report.py"), "audit_used": False,
              "october_rows": int(len(ids)), "october_inbound_identified_share": float(has_inb.mean()), "october_own_motion_share": float(has_motion.mean()),
              "note": "October only; no scoring month, ranking row or official score read."}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print(f"pool {report['pool_rmse_non_lirf']:.3f}; base proxy {base[SHARE]:.3f}; GO items {go_items}")
    for v, n in networks.items():
        print(f"   {v:11s} mean {n['mean_rmse_non_lirf']:.3f} (m {n['mean_rmse_matched']:.2f} / u {n['mean_rmse_unmatched']:.2f}) seeds {[round(x, 2) for x in n['per_seed_rmse_non_lirf'].values()]}")
    for name, e in items.items():
        extra = ""
        if name != "control":
            extra = (f" standalone {e['standalone_vs_control_pct']:+.3f}%, gain b1.0 {e['gain_points_vs_control']['1.0']:+.4f} pts (b0.5 {e['gain_points_vs_control']['0.5']:+.4f}), "
                     f"interval [{e['interval_vs_control_b1.0_s'][0]:+.2f}, {e['interval_vs_control_b1.0_s'][1]:+.2f}] s" + (f", GO {e['go']}" if "go" in e else ""))
        print(f"{name:13s} R1 {e['rmse_non_lirf']:.3f} (m {e['rmse_matched']:.2f} / u {e['rmse_unmatched']:.2f}) R2 b1.0 {e['r2_pct']['1.0']:+.4f}%{extra}")
    print("GPU vs CPU controls:", {k: (round(v["rmse_cpu"], 2), round(v["rmse_gpu"], 2)) for k, v in reproduction.items()}, corr)


if __name__ == "__main__":
    main()
