"""adsb_rctx design readings (configs/adsb_rctx_design_v1.json protocol): October 2025 only.

Every residual context-network design run (runs/adsb_rctx/design/rctx_s<seed>/2025-10) and the seed means (first k seeds, sorted) are
read against the incumbent's residual stage on October's in-scope rows (matched_no_lirf). The control residual is v26-v28's:
0.05 adj2 + 0.05 cat + 0.9 rnet, adj2/cat the adsb_v25 leave-one-month-out columns for October and rnet the mean of the 16 declared rnb
seeds' October leave-one-month-out predictions (runs/adsb_rnn/rnb_s501..516/2025-10). Candidates replace rnet inside the stage by
(1 - w) x rnet + w x rctx, w = 0.1 .. 1.0:

  scope      RMSE of v9 + r for the network alone, rnet alone, the control and every candidate; the correlation of rctx's and rnet's
             residual errors
  recipe     v28's non-LIRF recipe rebuilt on October from October-out-of-sample parts: max(0.3 x (v9 + r) + 0.7 x (0.25 x committee +
             0.75 x (0.5 ens_ctx + 0.5 ens_ctx_e24)), floor), with the committee rebuilt on October (runs/adsb_nn/design/committee_october
             .parquet) and ens_ctx / ens_ctx_e24 the means of their 16 seeds' seasonal-fold October predictions; LIRF rows stay at v9 for
             every candidate. Ratio to the control on all rows (primary) and on non-LIRF rows, with the paired-day interval at the best w.
No recipe or ranking month is read. Output: a JSON report and a printed table.
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
from nn_design_report import day_interval  # noqa: E402
from run_adsb_v11 import candidate_values, scope_mask  # noqa: E402
from run_adsb_v24 import adsb_features, load_frame, month_table  # noqa: E402

OCTOBER = "2025-10"
WEIGHTS = [round(0.1 * i, 1) for i in range(1, 11)]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def october_column(path: Path, ids: pd.Series, column: str) -> np.ndarray:
    return pd.read_parquet(path).set_index(ID)[column].reindex(ids).to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=Path("configs/adsb_rctx_design_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_rctx/design"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_rctx/design_report.json"))
    args = parser.parse_args()
    design = read_json(ROOT / args.design)
    if sha256_file(ROOT / design["network_config"]["path"]) != design["network_config"]["sha256"]:
        raise ValueError("The network configuration changed since the design declaration.")
    stage = read_json(ROOT / design["network_config"]["path"])["residual_stage"]
    inputs = design["inputs"]
    table = month_table(load_frame(stage, OCTOBER, "cross_fit"), adsb_features(OCTOBER, ROOT / stage["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), OCTOBER)
    ids = table[ID]
    label, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
    floor = float(table["floor"].iloc[0])
    inside = scope_mask(table, "matched_no_lirf")
    non_lirf = table["airport"].ne("LIRF").to_numpy()
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    columns = pd.read_parquet(ROOT / inputs["residual_columns"]).set_index(ID).reindex(ids)
    adj2, cat = (columns[f"residual:{m}"].to_numpy(dtype=float) for m in ("adj2", "cat"))
    rnet_runs = []
    for leaf in inputs["rnet_learners"]:
        frame = pd.read_parquet(ROOT / leaf / OCTOBER / "predictions.parquet").set_index(ID).reindex(ids)
        if not np.array_equal(frame["in_scope"].to_numpy(dtype=bool), inside):
            raise ValueError(f"{leaf} does not cover October's scope.")
        rnet_runs.append(frame["residual"].to_numpy(dtype=float))
    rnet = np.mean(rnet_runs, axis=0)
    committee = october_column(ROOT / inputs["committee_october"], ids, "committee")
    ens = {name: np.mean([october_column(ROOT / leaf / "seasonal" / "stop_predictions.parquet", ids, "adsb_full") for leaf in leaves], axis=0)
           for name, leaves in inputs["context_learners"].items()}
    network = 0.5 * ens["ens_ctx"] + 0.5 * ens["ens_ctx_e24"]
    both = np.isfinite(committee) & np.isfinite(network)
    table["_mix"] = np.where(both, 0.25 * committee + 0.75 * network, committee)
    if not np.isfinite(table.loc[non_lirf, "_mix"]).all() or not (np.isfinite(adj2) & np.isfinite(cat) & np.isfinite(rnet)).all():
        raise ValueError("October parts do not cover every row.")

    def stage_r(member: np.ndarray) -> np.ndarray:
        return 0.05 * adj2 + 0.05 * cat + 0.9 * member

    def recipe(residual: np.ndarray) -> np.ndarray:
        spec = {"weight": 1.0, "scope": "matched_no_lirf", "blend_with": {"name": "_mix", "scope": "all_no_lirf", "weight": 0.7}, "extra_blends": []}
        values, _ = candidate_values(table, v9, residual, spec, floor)
        return values

    control_r = stage_r(rnet)
    control = recipe(control_r)
    base_all, base_nl = rmse(label, control), rmse(label[non_lirf], control[non_lirf])
    y_s, v9_s = label[inside], v9[inside]

    def reading(net: np.ndarray) -> dict:
        out = {"scope": {"v9": rmse(y_s, v9_s), "rnet": rmse(y_s, v9_s + rnet[inside]), "control": rmse(y_s, v9_s + control_r[inside]),
                         "network": rmse(y_s, v9_s + net[inside]),
                         "blend": {str(w): rmse(y_s, v9_s + stage_r((1 - w) * rnet + w * net)[inside]) for w in WEIGHTS},
                         "error_correlation_with_rnet": float(np.corrcoef(y_s - v9_s - rnet[inside], y_s - v9_s - net[inside])[0, 1])},
               "recipe": {"blend": {}}}
        for w in WEIGHTS:
            v = recipe(stage_r((1 - w) * rnet + w * net))
            out["recipe"]["blend"][str(w)] = {"ratio_all": rmse(label, v) / base_all, "ratio_non_lirf": rmse(label[non_lirf], v[non_lirf]) / base_nl}
        best = min(WEIGHTS, key=lambda w: out["recipe"]["blend"][str(w)]["ratio_all"])
        out["recipe"]["best_w"] = best
        out["recipe"]["best_pct_all"] = 100 * (out["recipe"]["blend"][str(best)]["ratio_all"] - 1)
        out["recipe"]["best_pct_non_lirf"] = 100 * (out["recipe"]["blend"][str(best)]["ratio_non_lirf"] - 1)
        out["recipe"]["best_day_interval_seconds"] = day_interval(days, label, recipe(stage_r((1 - best) * rnet + best * net)), control)
        return out

    runs = {}
    for path in sorted((ROOT / args.runs).glob(f"*/{OCTOBER}/run.json")):
        record = read_json(path)
        if record["stage"] != "design" or record["smoke"] or record["config_sha256"] != design["network_config"]["sha256"] or record["variant"] != design["variant"]:
            raise ValueError(f"{path} is not a design run of the declared network configuration.")
        frame = pd.read_parquet(path.parent / "predictions.parquet").set_index(ID).reindex(ids)
        if frame["residual"].isna().any() or not np.array_equal(frame["in_scope"].to_numpy(dtype=bool), inside):
            raise ValueError(f"{path.parent} does not cover October's scope.")
        runs[record["seed"]] = {"values": frame["residual"].to_numpy(dtype=float), "curve": [round(e["monitor_rmse"], 3) for e in record["history"]],
                                "fit_seconds": record["fit_seconds"], "best_monitor_epoch": record["best_monitor_epoch"], "sha256": sha256_file(path.parent / "predictions.parquet")}
    entries = {f"rctx_s{s}": {**reading(r["values"]), **{k: r[k] for k in ["curve", "fit_seconds", "best_monitor_epoch", "sha256"]}} for s, r in sorted(runs.items())}
    ordered = [runs[s]["values"] for s in sorted(runs)]
    for k in range(2, len(ordered) + 1):
        entries[f"rctx_mean{k}"] = {**reading(np.mean(ordered[:k], axis=0)), "seeds": k}
    rnet_curve = {k: rmse(y_s, v9_s + np.mean(rnet_runs[:k], axis=0)[inside]) for k in (1, 2, 4, 8, 16)}
    decision = {"rule": design["protocol"]["decision"], "seeds_read": len(ordered)}
    four = entries.get("rctx_mean4")
    decision["mean4_best_pct_all"] = four["recipe"]["best_pct_all"] if four else None
    decision["mean4_best_w"] = four["recipe"]["best_w"] if four else None
    decision["mean4_interval"] = four["recipe"]["best_day_interval_seconds"] if four else None
    decision["go"] = bool(four and four["recipe"]["best_pct_all"] <= -100 * design["protocol"]["go_threshold_fraction"]
                          and four["recipe"]["best_day_interval_seconds"][1] < 0)
    report = {"created_at_utc": utc_now(), "design": args.design.as_posix(), "design_sha256": sha256_file(ROOT / args.design),
              "rows": int(len(table)), "rows_in_scope": int(inside.sum()), "rows_non_lirf": int(non_lirf.sum()),
              "control": {"recipe_rmse_all": base_all, "recipe_rmse_non_lirf": base_nl, "scope_rmse": rmse(y_s, v9_s + control_r[inside]),
                          "rnet_scope_rmse_by_seeds": rnet_curve},
              "inputs": {"residual_columns": sha256_file(ROOT / inputs["residual_columns"]), "committee_october": sha256_file(ROOT / inputs["committee_october"]),
                         "scratch": sha256_file(ROOT / stage["cross_fit"]["scratch"] / f"{OCTOBER}.parquet")},
              "readings": entries, "decision": decision, "audit_used": False, "note": "October only; no recipe or ranking month read."}
    write_json(ROOT / args.report, report)
    print(f"control: scope {report['control']['scope_rmse']:.3f} (rnet alone by seeds {({k: round(v, 3) for k, v in rnet_curve.items()})})  recipe all {base_all:.3f}  non-LIRF {base_nl:.3f}")
    print(f"{'run':12s} {'net':>8s} {'errcorr':>7s} {'best w':>6s} | {'rec best%':>9s} {'nonLIRF%':>8s} interval")
    for name, e in entries.items():
        s, r = e["scope"], e["recipe"]
        print(f"{name:12s} {s['network']:8.3f} {s['error_correlation_with_rnet']:7.4f} {r['best_w']:6.1f} | {r['best_pct_all']:9.4f} {r['best_pct_non_lirf']:8.4f} "
              f"[{r['best_day_interval_seconds'][0]:.3f}, {r['best_day_interval_seconds'][1]:.3f}]  {e.get('curve', [])[-3:]}")
    print("decision", decision)


if __name__ == "__main__":
    main()
