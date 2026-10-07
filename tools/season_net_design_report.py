"""Design readings of the seasonal-weight lever (fork S; configs/season_net_design_v1.json): the own-points context network own_all refitted with
a seasonal loss weight (tools/run_season_net_ctx3.py), mixed into the shipped member own_all on every row, October only.

As tools/lfpg_net_design_report.py (tunnel 21's pool P, share 0.75, R2 grid, paired-day interval; v66 / v69 member weight b = 1.0 as the decision):
  control   C = dl32own on rows with adsb_matched = 1, m25dl32own on the other rows: lever 5's GPU design fits (runs/ctx_own2/design, seeds
            9-16), reused; each run's variant_sha256 must equal this declaration's dl32own / m25dl32own and its device must be cuda
  S_V       variant V's matched-row network seed mean on matched rows, its unmatched-row network seed mean on the others
  items     <V>_c50 / <V>_c100: X = (1 - c) C + c S_V on every row
GO (declared): R2 at b = 1.0 at least 0.05 points below the control's, 8-seed October non-LIRF RMSE below the control's, paired-day 95%
interval of the proxy difference at b = 1.0 below zero. Several GO items: the lowest R2 at b = 1.0.
Every season run must carry its season.json (weights by month) and the declared variant settings. No scoring month, ranking row or official
score is read.
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
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TIME  # noqa: E402
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402
from ctx_design_report import october  # noqa: E402
from run_adsb_ctx3 import resolve_variant  # noqa: E402

DECLARATION = "configs/season_net_design_v1.json"
B_VALUES = [0.4, 0.5, 1.0]
B_DECISION = 1.0
SHARE, OTHER_SHARE = 0.75, 0.85
BAR_POINTS = 0.05
C_VALUES = {"c50": 0.5, "c100": 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/season_net/design"))
    parser.add_argument("--control-runs", type=Path, default=Path("runs/ctx_own2/design"))
    parser.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, default=Path("docs/season_net/design_report.json"))
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    decl_sha = sha256_file(ROOT / DECLARATION)
    proxy = RecipeProxy(pd.read_parquet(ROOT / args.committee))
    ids, label, non_lirf, table = proxy.ids, proxy.label, proxy.non_lirf, proxy.table
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    airport = table["airport"].to_numpy()
    lfpg = airport == "LFPG"
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
        if record["variant_sha256"] != object_hash(resolve_variant(decl, record["variant"])):
            raise ValueError(f"{path}: variant settings differ from the declaration.")
        runs.setdefault(record["variant"], {})[record["seed"]] = record | {"_dir": path.parent}
    for v in ("dl32own", "m25dl32own"):
        want = object_hash(resolve_variant(decl, v))
        for s in decl["control_runs"][v]:
            path = ROOT / args.control_runs / f"{v}_s{s}/design/run.json"
            record = read_json(path)
            if record["variant_sha256"] != want or record["inputs"]["device"] != "cuda" or record.get("smoke_limit_rows") is not None:
                raise ValueError(f"{path}: the reused control differs from this declaration's {v} or did not run on the GPU.")
            runs.setdefault(v, {})[s] = record | {"_dir": path.parent}
    needed = {**decl["runs"], "dl32own": decl["control_runs"]["dl32own"], "m25dl32own": decl["control_runs"]["m25dl32own"]}
    missing = {v: sorted(set(s) - set(runs.get(v, {}))) for v, s in needed.items() if set(s) - set(runs.get(v, {}))}
    if missing:
        raise ValueError(f"Declared runs missing: {missing}")
    seeds = sorted(decl["control_runs"]["dl32own"])
    values = {v: {s: october(runs[v][s]["_dir"] / "stop_predictions.parquet", ids) for s in seeds} for v in needed}
    mean = {v: np.mean([values[v][s] for s in seeds], axis=0) for v in needed}
    pairs = {name: np.where(matched, mean[a], mean[b]) for name, (a, b) in decl["pairs"].items()}
    ctrl = pairs["control"]
    ctrl_proxy = proxied(ctrl, B_DECISION)
    groups = {"matched": matched, "unmatched": ~matched, **{f"airport_{a}": airport == a for a in sorted(set(airport)) if a != "LIRF"}}
    tables = {"control": ctrl}
    for v in [n for n in decl["pairs"] if n != "control"]:
        for tag, c in C_VALUES.items():
            tables[f"{v}_{tag}"] = (1 - c) * ctrl + c * pairs[v]
    items = {}
    for name, x in tables.items():
        entry = {"rmse_non_lirf": net(x), "rmse_by_group": {g: net(x, rows) for g, rows in groups.items()},
                 "r2_pct": {str(b): r2(x, b) for b in B_VALUES}, "r2_pct_share_085": {str(b): r2(x, b, OTHER_SHARE) for b in B_VALUES}}
        if name != "control":
            v = proxied(x, B_DECISION)
            entry.update({"gain_points_vs_control": {str(b): r2(ctrl, b) - entry["r2_pct"][str(b)] for b in B_VALUES},
                          "gain_points_vs_control_share_085": {str(b): r2(ctrl, b, OTHER_SHARE) - entry["r2_pct_share_085"][str(b)] for b in B_VALUES},
                          "non_lirf_vs_control_pct": 100 * (net(x) / net(ctrl) - 1),
                          "non_lirf_below_control": bool(net(x) < net(ctrl)),
                          "by_group_vs_control_pct": {g: 100 * (net(x, rows) / net(ctrl, rows) - 1) for g, rows in groups.items()},
                          "interval_vs_control_b1.0_s": day_interval(days, label, v, ctrl_proxy),
                          "proxy_change_vs_control_b1.0_pct": 100 * (rmse(label, v) / rmse(label, ctrl_proxy) - 1)})
            entry["go"] = bool(entry["gain_points_vs_control"][str(B_DECISION)] >= BAR_POINTS and entry["non_lirf_below_control"]
                               and entry["interval_vs_control_b1.0_s"][1] < 0)
        items[name] = entry
    networks = {v: {"per_seed_rmse_non_lirf": {str(s): net(values[v][s]) for s in seeds}, "mean_rmse_non_lirf": net(mean[v]),
                    "mean_rmse_matched": net(mean[v], matched), "mean_rmse_unmatched": net(mean[v], ~matched)} for v in needed}
    specialists = {name: {"rmse_non_lirf": net(s), "rmse_matched": net(s, matched), "rmse_unmatched": net(s, ~matched),
                          "error_correlation_with_control": float(np.corrcoef((s - label)[non_lirf], (ctrl - label)[non_lirf])[0, 1])}
                   for name, s in pairs.items()}
    season_records = {v: {str(s): read_json(runs[v][s]["_dir"] / "season.json") for s in seeds} for v in decl["runs"]}
    for v, by in season_records.items():
        for s, rec in by.items():
            if rec["season"] != resolve_variant(decl, v)["season"] or rec["variant"] != v:
                raise ValueError(f"{v} seed {s}: season.json does not match the declaration.")
    go_items = sorted([n for n, e in items.items() if e.get("go")], key=lambda n: items[n]["r2_pct"][str(B_DECISION)])
    report = {"created_at_utc": utc_now(), "declaration_sha256": decl_sha, "seeds": seeds, "base_proxy_rmse": base[SHARE], "pool_rmse_non_lirf": net(pool),
              "share": SHARE, "b_decision": B_DECISION, "bar_points": BAR_POINTS, "items": items, "specialists": specialists, "networks": networks,
              "go_items": go_items, "selected": go_items[0] if go_items else None, "go": bool(go_items),
              "october_rows": int(len(ids)), "october_non_lirf_rows": int(non_lirf.sum()),
              "season_weights_by_month": {v: by[str(seeds[0])]["by_month"] for v, by in season_records.items()},
              "runs": {v: {str(s): {"fit_seconds": r["fit_seconds"], "fitted_rows": r["fitted_rows"], "epochs": len(r["history"]),
                                    "predictions_sha256": sha256_file(r["_dir"] / "stop_predictions.parquet"), "dir": r["_dir"].relative_to(ROOT).as_posix()}
                           for s, r in by.items()} for v, by in runs.items()},
              "decision": decl["protocol"]["decision"], "tool_sha256": sha256_file(Path(__file__)), "audit_used": False,
              "note": "October only; no scoring month, ranking row or official score read."}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print(f"base proxy {base[SHARE]:.3f}; October non-LIRF rows {report['october_non_lirf_rows']}; GO items {go_items}")
    for name, s in specialists.items():
        print(f"   {name:10s} non-LIRF {s['rmse_non_lirf']:.3f} (m {s['rmse_matched']:.2f} / u {s['rmse_unmatched']:.2f}) corr {s['error_correlation_with_control']:.4f}")
    for name, e in items.items():
        extra = ""
        if name != "control":
            extra = (f" non-LIRF {e['non_lirf_vs_control_pct']:+.3f}%, gain b1.0 {e['gain_points_vs_control']['1.0']:+.4f} pts, "
                     f"interval [{e['interval_vs_control_b1.0_s'][0]:+.3f}, {e['interval_vs_control_b1.0_s'][1]:+.3f}] s, GO {e['go']}")
        print(f"{name:16s} R2 b1.0 {e['r2_pct']['1.0']:+.4f}%{extra}")


if __name__ == "__main__":
    main()
