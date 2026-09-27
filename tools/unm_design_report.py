"""Design readings of the unmatched-row specialist (configs/adsb_unm_design_v1.json): October only.

C   the committee (ens_v20_bag_q_all) on October, from runs/adsb_nn/design/committee_october.parquet (hash-bound); its member columns are
    the seed-matched all-row controls M of every specialist run.
P   v30's network table on October: 0.25 ens_ctx_e24 + 0.25 ens_ctx_e24b + 0.5 ens_ctx_e32 from the seeds' seasonal stop predictions.
R1  standalone October RMSE (non-LIRF, matched, unmatched, unmatched by airport) of every run, item seed mean and bag, next to its M.
R2  recipe proxy (tools/nn_design_report.RecipeProxy) of T = 0.25 C' + 0.75 P, C' = C with (1 - c) C + c X on non-LIRF unmatched rows,
    c in {0.5, 1.0}; % against T = 0.25 C + 0.75 P. Descriptive: T' = (1 - c) T + c X on the same rows, c in {0.1, 0.25}.
Unmatched rows for routing and readings are the proxy's own flag (the recipe's scopes); agreement with the learners' pass-2 flag is reported.
GO (declared, bags only): R2 at c = 0.5 at least 0.05 points below the control, paired-day 95% interval below zero, and standalone unmatched
RMSE below the seed-matched member bag M. The reproduction run must equal lgb_x19_s24's October column first. No scoring month is read.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TIME  # noqa: E402

DECLARATION = "configs/adsb_unm_design_v1.json"
C_VALUES = [0.5, 1.0]
TABLE_C = [0.1, 0.25]
DECISION_C = 0.5
BAR_POINTS = 0.05
AIRPORTS = ("LTFM", "LFPG", "EGLL", "LEMD", "EDDM", "EDDF", "LEBL", "LSZH", "EHAM")


def expand(pattern: str) -> list[str]:
    """runs/x_s{117..132}/y -> the sixteen paths."""
    match = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if not match:
        return [pattern]
    return [pattern[:match.start()] + str(i) + pattern[match.end():] for i in range(int(match.group(1)), int(match.group(2)) + 1)]


def route(base: np.ndarray, x: np.ndarray, rows: np.ndarray, c: float) -> np.ndarray:
    """(1 - c) base + c x on the selected rows; every other row keeps base exactly."""
    out = np.array(base, dtype=float, copy=True)
    out[rows] = (1 - c) * base[rows] + c * x[rows]
    return out


def bag(values: dict[str, np.ndarray], spec: dict) -> np.ndarray:
    return sum(float(w) * values[spec[kind]] for kind, w in spec["weights"].items())


def main():
    from ctx_design_report import october
    from nn_design_report import RecipeProxy, day_interval, rmse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runs", type=Path, help="Default: the declaration's output_root")
    args = parser.parse_args()
    decl_path = ROOT / DECLARATION
    decl, decl_sha = read_json(decl_path), sha256_file(decl_path)
    if not decl.get("declared_at_utc"):
        raise SystemExit("The declaration is still a draft.")
    reference = decl["parents"]["october_reference"]
    if sha256_file(ROOT / reference["path"]) != reference["sha256"]:
        raise ValueError("The October committee reference changed.")
    frame = pd.read_parquet(ROOT / reference["path"])
    proxy = RecipeProxy(frame)
    ids, label, non_lirf, table = proxy.ids, proxy.label, proxy.non_lirf, proxy.table
    members = frame.set_index(ID).reindex(ids)
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    unmatched = non_lirf & table["adsb_matched"].ne(1).to_numpy()
    matched = non_lirf & ~unmatched
    pass2_unmatched = non_lirf & members["adsb_matched"].ne(1).to_numpy()
    airport = table["airport"].to_numpy()
    committee = proxy.committee
    net = decl["parents"]["network_table"]
    pool = sum(float(w) * np.mean([october(ROOT / p, ids) for p in expand(net["paths"][name])], axis=0) for name, w in net["weights"].items())
    share = float(net["network_share"])
    control = proxy.values((1 - share) * committee + share * pool)
    control_rmse = rmse(label, control)

    def r2(values):
        return 100 * (rmse(label, values) / control_rmse - 1)

    def slot(x, c):
        return proxy.values((1 - share) * route(committee, x, unmatched, c) + share * pool)

    def table_level(x, c):
        return proxy.values(route((1 - share) * committee + share * pool, x, unmatched, c))

    def standalone(x):
        return {"non_lirf": rmse(label[non_lirf], x[non_lirf]), "matched": rmse(label[matched], x[matched]), "unmatched": rmse(label[unmatched], x[unmatched]),
                "unmatched_by_airport": {a: rmse(label[unmatched & (airport == a)], x[unmatched & (airport == a)]) for a in AIRPORTS if (unmatched & (airport == a)).any()}}

    root = ROOT / (args.runs or decl["output_root"])
    values, records = {}, {}
    for name in decl["runs"]:
        path = root / name / "design" / "run.json"
        if not path.exists():
            raise ValueError(f"Declared run missing: {name}")
        record = read_json(path)
        if record["config_sha256"] != decl_sha or record.get("smoke_limit_rows") is not None or record["run"] != name:
            raise ValueError(f"{path} was not fitted under this declaration (or is a smoke run).")
        values[name] = october(path.parent / "stop_predictions.parquet", ids)
        records[name] = record
    check = decl["reproduction_check"]
    repro_diff = float(np.max(np.abs(values[check["run"]] - members[check["reference_column"]].to_numpy(dtype=float))))
    if repro_diff > float(check["tolerance_seconds"]):
        raise ValueError(f"The reproduction run differs from {check['reference_column']} by {repro_diff} s: the runner's path is not the members'.")

    control_member = {name: members[decl["runs"][name]["control_member"]].to_numpy(dtype=float) for name in decl["runs"]}
    item_values = {item: np.mean([values[r] for r in runs], axis=0) for item, runs in decl["items"].items()}
    item_members = {item: np.mean([control_member[r] for r in runs], axis=0) for item, runs in decl["items"].items()}
    bag_values = {name: bag(item_values, spec) for name, spec in decl["bags"].items()}
    bag_members = {name: bag(item_members, spec) for name, spec in decl["bags"].items()}

    def describe(x, m):
        entry = {"standalone": standalone(x), "member_control_standalone": standalone(m),
                 "r2_pct": {str(c): r2(slot(x, c)) for c in C_VALUES}, "member_control_r2_pct": {str(c): r2(slot(m, c)) for c in C_VALUES},
                 "table_level_r2_pct": {str(c): r2(table_level(x, c)) for c in TABLE_C}}
        entry["gain_points"] = -entry["r2_pct"][str(DECISION_C)]
        entry["specialisation_points"] = entry["member_control_r2_pct"][str(DECISION_C)] - entry["r2_pct"][str(DECISION_C)]
        entry["interval_vs_control_s"] = day_interval(days, label, slot(x, DECISION_C), control)
        entry["unmatched_below_member_control"] = bool(entry["standalone"]["unmatched"] < entry["member_control_standalone"]["unmatched"])
        return entry

    items = {name: describe(x, item_members[name]) for name, x in item_values.items()}
    bags = {}
    for name, x in bag_values.items():
        entry = describe(x, bag_members[name])
        entry["decision_item"] = name in decl["protocol"]["decision_items"]
        entry["go"] = bool(entry["decision_item"] and entry["gain_points"] >= BAR_POINTS and entry["interval_vs_control_s"][1] < 0 and entry["unmatched_below_member_control"])
        bags[name] = entry
    go = sorted((n for n, e in bags.items() if e["go"]), key=lambda n: bags[n]["r2_pct"][str(DECISION_C)])
    per_run = {name: {"standalone_unmatched": rmse(label[unmatched], values[name][unmatched]), "member_unmatched": rmse(label[unmatched], control_member[name][unmatched]),
                      "best_iteration": records[name]["best_iteration"], "fitted_rows": records[name]["fitted_rows"], "seconds": records[name]["seconds"],
                      "predictions_sha256": sha256_file(root / name / "design" / "stop_predictions.parquet")} for name in decl["runs"]}
    report = {"created_at_utc": utc_now(), "declaration_sha256": decl_sha, "control_proxy_rmse": control_rmse, "committee_standalone": standalone(committee),
              "network_pool_standalone": standalone(pool), "network_share": share, "bar_points": BAR_POINTS, "decision_c": DECISION_C,
              "reproduction_max_abs_difference_s": repro_diff, "unmatched_rows": int(unmatched.sum()), "non_lirf_rows": int(non_lirf.sum()),
              "unmatched_flag_agreement_with_pass2": float(np.mean(unmatched[non_lirf] == pass2_unmatched[non_lirf])),
              "items": items, "bags": bags, "go": go, "selected": go[0] if go else None, "runs": per_run,
              "decision": decl["protocol"]["decision"], "audit_used": False, "note": "October only; no scoring month read."}
    write_json(ROOT / args.report, report)
    print(f"control proxy {control_rmse:.3f}; committee October unmatched {report['committee_standalone']['unmatched']:.3f}; reproduction {repro_diff:.2e} s; "
          f"flag agreement {report['unmatched_flag_agreement_with_pass2']:.4f}")
    for name, e in {**items, **bags}.items():
        flag = ("GO" if e.get("go") else "no") if e.get("decision_item") else "  "
        print(f"{flag:3s} {name:12s} unm {e['standalone']['unmatched']:.3f} (M {e['member_control_standalone']['unmatched']:.3f}) "
              f"R2 c0.5 {e['r2_pct']['0.5']:+.4f}% c1 {e['r2_pct']['1.0']:+.4f}% (M c0.5 {e['member_control_r2_pct']['0.5']:+.4f}%) "
              f"table c0.25 {e['table_level_r2_pct']['0.25']:+.4f}% interval [{e['interval_vs_control_s'][0]:+.2f}, {e['interval_vs_control_s'][1]:+.2f}] s")
    print("GO:", go)


if __name__ == "__main__":
    main()
