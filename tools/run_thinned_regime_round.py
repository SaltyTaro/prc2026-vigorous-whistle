"""Declared round on the ranking-regime (coverage-thinned) development sets: configs/thinned_regime_round_v1.json. No model is fitted.

The coverage campaign's thinned months (configs/coverage_regime_v1.json: July 2025 thinned to July 2026's adsb.lol coverage, December 2025
thinned to January 2026's) carry saved re-predictions of every component on the thinned inputs: the committee, the LIRF trees and the
residual members (runs/coverage_regime/measure/<month>/predictions.parquet), the context-network ensembles
(runs/adsb_ctx/regime_check4/<month>.parquet) and the LIRF context network (runs/adsb_ctx/regime_check_lirf/<month>.parquet). This tool
assembles v34's recipe per row from those tables, natural and thinned, in the layout of runs/lomo_dev/rows.parquet, and scores the
declared candidates (network share inside the committee mix, by scope; the networks-far-below-the-committee gate) with
tools/run_fair_weights.build.

  --check             assemble both kinds; the natural assembly must reproduce the shipped-provenance development rows' v33
                      (runs/fair_dev/rows_shipped.parquet) within the regime checks' tolerance: month RMSE within 0.01 s (the saved
                      context-network re-predictions differ from the stored ones on a few rows, docs/adsb_ctx/regime_check_*.json).
                      Reads no candidate score.
  --score --select    the governing run -> runs/thinned_regime/selection.json and docs/thinned_regime/results_v1.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_fair_weights import build, candidate_weights  # noqa: E402

DECLARATION = "configs/thinned_regime_round_v1.json"
SHIPPED = "runs/fair_dev/rows_shipped.parquet"
LOMO = "runs/lomo_dev/rows.parquet"
MEASURE = "runs/coverage_regime/measure/{month}/predictions.parquet"
NETS = "runs/adsb_ctx/regime_check4/{month}.parquet"
LIRF_NET = "runs/adsb_ctx/regime_check_lirf/{month}.parquet"
OUT = ROOT / "runs/thinned_regime"
REPORT = ROOT / "docs/thinned_regime/results_v1.json"
NETWORK = {"ens_ctx_e24": 0.25, "ens_ctx_e24b": 0.25, "ens_ctx_e32": 0.5}  # v30's network table (configs/adsb_ctx_v5.json c_e24x32_e32h)
RESIDUAL = {"adj2": 0.05, "cat": 0.05, "rnet": 0.9}                        # v26's residual stage (configs/adsb_rnn_v1.json rnet_w90)
V33_LIRF = {"lirf_nonpop_matched": {"v9": 0.15, "net": 0.40, "tree": 0.45}, "lirf_nonpop_unmatched": {"v9": 0.40, "net": 0.35, "tree": 0.25}}
V33_OUTER = {"ltfm": 1.0, "matched_other": 0.6, "unmatched_other": 0.7}
SEED = 20260929
CODE = ["tools/run_thinned_regime_round.py", "tools/run_fair_weights.py", "tools/run_adsb_nn_dev.py", "tools/run_adsb_v11.py"]


def aligned(path: Path, ids: np.ndarray) -> pd.DataFrame:
    table = pd.read_parquet(path).set_index(ID)
    if table.index.duplicated().any() or len(table) != len(ids) or not table.index.isin(ids).all():
        raise ValueError(f"{path} does not hold exactly the month's rows.")
    return table.reindex(ids)


def assemble(month: str, kind: str, shipped: pd.DataFrame) -> pd.DataFrame:
    """v34-layout rows of one month from the saved natural or thinned component tables (shipped-provenance v9 side)."""
    s = shipped[shipped["month"] == month].reset_index(drop=True)
    ids = s[ID].to_numpy()
    measure = aligned(ROOT / MEASURE.format(month=month), ids)
    nets = aligned(ROOT / NETS.format(month=month), ids)
    lirf_net = aligned(ROOT / LIRF_NET.format(month=month), ids)
    if np.max(np.abs(measure["TAXITIME_SEC_mvt"].to_numpy(dtype=float) - s["label"].to_numpy(dtype=float))) > 1e-6 or not (measure["airport"].to_numpy() == s["airport"].to_numpy()).all():
        raise ValueError(f"Labels or airports differ between the measure table and the development rows on {month}.")
    airport = s["airport"].to_numpy()
    matched = measure[f"class_{kind}"].ne("unmatched").to_numpy()
    nonpop = (s["lirf_m"] | s["lirf_u"]).to_numpy()
    v9 = s["v9"].to_numpy(dtype=float)
    residual = np.sum([w * measure[f"{kind}:residual:{m}"].to_numpy(dtype=float) for m, w in RESIDUAL.items()], axis=0)
    base = v9 + np.where(matched & (airport != "LIRF"), np.nan_to_num(residual, nan=0.0), 0.0)
    committee = measure[f"{kind}:ens_v20_bag_q_all"].to_numpy(dtype=float)
    network = np.sum([w * nets[f"{name}:{kind}"].to_numpy(dtype=float) for name, w in NETWORK.items()], axis=0)
    frame = pd.DataFrame({ID: ids, "month": month, "kind": kind, "day": s["day"].to_numpy(), "airport": airport, "label": s["label"].to_numpy(dtype=float),
                          "floor": s["floor"].to_numpy(dtype=float), "matched": matched, "class": measure[f"class_{kind}"].to_numpy(),
                          "lirf_m": nonpop & matched & (airport == "LIRF"), "lirf_u": nonpop & ~matched & (airport == "LIRF"),
                          "v9": v9, "residual": residual, "base": base, "committee": committee, "nets": network,
                          "mix": mixed(committee, network, 0.75), "lirf_tree": measure[f"{kind}:ens_v22_lirf_q"].to_numpy(dtype=float),
                          "lirf_net": lirf_net[f"ens_ctx_lirf:{kind}"].to_numpy(dtype=float), "v33": s["v33"].to_numpy(dtype=float)})
    return frame


def gate_values(f: pd.DataFrame, values: np.ndarray, spec: dict, outer: dict) -> np.ndarray:
    """Networks far below the committee (configs/snow_higher_v1.json net_down): on non-LIRF rows with committee - nets > T the mix takes `share_inside`."""
    airport = f["airport"].to_numpy()
    committee, nets = f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float)
    inside = (airport != "LIRF") & np.isfinite(nets) & (committee - nets > spec["threshold_s"])
    mix = mixed(committee, nets, spec["share_inside"])
    matched = f["matched"].to_numpy()
    w = np.where(airport == "LTFM", outer["ltfm"], np.where(matched, outer["matched_other"], outer["unmatched_other"]))
    gated = np.maximum((1 - w) * f["base"].to_numpy(dtype=float) + w * mix, f["floor"].to_numpy(dtype=float))
    return np.where(inside, gated, values)


def values_for(f: pd.DataFrame, control: dict, spec: dict) -> np.ndarray:
    outer, lirf = candidate_weights(control, spec)
    share = control["network_share"]
    committee, nets = f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float)
    mix = mixed(committee, nets, share)
    if "network_share" in spec:
        scope = spec.get("scope", "all")
        matched = f["matched"].to_numpy()
        inside = {"all": np.ones(len(f), dtype=bool), "unmatched": ~matched, "matched": matched}[scope]
        mix = np.where(inside, mixed(committee, nets, spec["network_share"]), mix)
    values = build(f.assign(mix=mix), outer, lirf)
    if "gate" in spec:
        values = gate_values(f, values, spec["gate"], outer)
    return values


def check(shipped: pd.DataFrame, months: list[str]) -> dict:
    out = {}
    for month in months:
        natural, thinned = assemble(month, "natural", shipped), assemble(month, "thinned", shipped)
        s = shipped[shipped["month"] == month].reset_index(drop=True)
        rebuilt = build(natural, V33_OUTER, V33_LIRF)
        diff = np.abs(rebuilt - s["v33"].to_numpy(dtype=float))
        y = s["label"].to_numpy(dtype=float)
        entry = {"rows": int(len(s)), "natural_matched_equals_development": bool((natural["matched"].to_numpy() == s["matched"].to_numpy()).all()),
                 "v33_development_rmse": rmse(y, s["v33"].to_numpy(dtype=float)), "v33_natural_rebuild_rmse": rmse(y, rebuilt), "tolerance_s": 0.01,
                 "natural_v33_max_abs": float(np.max(diff)), "natural_v33_rms": float(np.sqrt(np.mean(diff ** 2))), "natural_v33_rows_over_1s": int((diff > 1.0).sum()),
                 "matched_share": {k: {a: float(g["matched"].mean()) for a, g in t.groupby("airport")} for k, t in (("natural", natural), ("thinned", thinned))},
                 "finite": {k: bool(np.isfinite(build(t, V33_OUTER, V33_LIRF)).all()) for k, t in (("natural", natural), ("thinned", thinned))}}
        entry["ok"] = bool(abs(entry["v33_natural_rebuild_rmse"] - entry["v33_development_rmse"]) <= 0.01 and entry["natural_v33_rows_over_1s"] <= 50
                           and entry["natural_matched_equals_development"] and all(entry["finite"].values()))
        out[month] = entry
    return out


def main() -> None:
    shipped = pd.read_parquet(ROOT / SHIPPED)
    decl = read_json(ROOT / DECLARATION) if (ROOT / DECLARATION).exists() else None
    thinned_months = list(decl["thinned_months"]) if decl else ["2025-07", "2025-12"]
    checks = check(shipped, thinned_months)
    if "--check" in sys.argv:
        for month, entry in checks.items():
            print(month, {k: v for k, v in entry.items() if k != "matched_share"})
            print("   matched share natural -> thinned:", {a: (round(entry["matched_share"]["natural"][a], 3), round(entry["matched_share"]["thinned"][a], 3)) for a in entry["matched_share"]["natural"]})
        return
    if "--score" not in sys.argv:
        raise SystemExit("Use --check or --score --select.")
    if decl is None or not decl.get("declared_at_utc"):
        raise ValueError("The round is not declared.")
    if decl.get("audit_used"):
        raise ValueError("Configuration does not respect its own contract.")
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    if not all(e["ok"] for e in checks.values()):
        raise ValueError(f"The natural assembly does not reproduce the development rows: {checks}")
    control = decl["control"]
    sets = {f"thinned:{m}": assemble(m, "thinned", shipped) for m in thinned_months}
    lomo = pd.read_parquet(ROOT / LOMO)
    for month in decl["natural_months"]:
        sets[f"natural:{month}"] = lomo[lomo["month"] == month].reset_index(drop=True)
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "audit_used": False, "assembly_check": checks,
              "inputs": {"rows_shipped_sha256": sha256_file(ROOT / SHIPPED), "lomo_rows_sha256": sha256_file(ROOT / LOMO),
                         **{f"{name}:{m}": sha256_file(ROOT / p.format(month=m)) for m in thinned_months for name, p in (("measure", MEASURE), ("nets", NETS), ("lirf_net", LIRF_NET))}},
              "sets": {}, "candidates": {}}
    per_set = {}
    for key, f in sets.items():
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        base_values = values_for(f, control, {})
        if not np.isfinite(base_values).all():
            raise ValueError(f"The control has non-finite values on {key}.")
        report["sets"][key] = {"rows": int(len(f)), "control_rmse": rmse(y, base_values), "matched_share_non_lirf": float(f.loc[f["airport"].ne("LIRF"), "matched"].mean())}
        per_set[key] = {}
        for name, spec in decl["candidates"].items():
            values = values_for(f, control, spec)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {key}.")
            changed = np.abs(values - base_values) > 1e-9
            per_set[key][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, base_values), "rows_differing_from_control": int(changed.sum()),
                                  "paired_days_vs_control": paired_days(days, y, values, base_values, SEED)}
        print(key, {"control": round(report["sets"][key]["control_rmse"], 4)}, {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in per_set[key].items()}, flush=True)
    thinned_keys = [f"thinned:{m}" for m in thinned_months]
    natural_keys = [f"natural:{m}" for m in decl["natural_months"]]
    for name in decl["candidates"]:
        entry = {**decl["candidates"][name], "sets": {k: per_set[k][name] for k in sets}}
        ratios = [entry["sets"][k]["ratio_to_control"] for k in thinned_keys]
        entry["mean_thinned_ratio"] = float(np.mean(ratios))
        below = [k for k in thinned_keys if entry["sets"][k]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["thinned_intervals_below_zero"] = below
        entry["natural_worse"] = [k for k in natural_keys if entry["sets"][k]["ratio_to_control"] > 1.0]
        entry["eligible"] = bool(all(r < 1.0 for r in ratios) and entry["mean_thinned_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"]
                                 and len(below) == len(thinned_keys) and not entry["natural_worse"])
        report["candidates"][name] = entry
    eligible = [n for n, e in report["candidates"].items() if e["eligible"]]
    report["eligible"] = eligible
    report["selected_candidate"] = min(eligible, key=lambda n: (report["candidates"][n]["mean_thinned_ratio"], n)) if eligible else None
    report["model_code"] = {p: sha256_file(ROOT / p) for p in CODE}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": report["selected_candidate"], "eligible": eligible})
    for n, e in report["candidates"].items():
        line = " ".join(f"{k}:{100 * (e['sets'][k]['ratio_to_control'] - 1):+.4f}% [{e['sets'][k]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},"
                        f"{e['sets'][k]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for k in sets)
        print(f"{n:10s} thinned mean {100 * (e['mean_thinned_ratio'] - 1):+.4f}% | {line} | eligible={e['eligible']}")


if __name__ == "__main__":
    main()
