"""Leave-one-month-out evaluation of configs/pop_signals_v1.json: new classifier inputs for the v9 LIRF fill mixture.

For every declared candidate (a feature set on top of v9's classifier features, bagged over the declared seeds) and the control
(v9's features, the same bag), each 2025 month is held out in turn: the classifier and the normal component are fitted on the other
eleven months' population rows, the held-out month's population rows with a schedule interval at or below the mixture threshold get
the mixture value p_fill * interval + (1 - p_fill) * normal, and the squared error against the held-out labels is recorded.
Selection follows the declared rule only. Writes docs/pop_signals/results_v1.json (or --report). Nothing is exported or uploaded.

    python tools/run_pop_signals.py [--declaration configs/pop_signals_v1.json] [--report PATH] [--rows PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_final import pop_signals as ps  # noqa: E402

MONTH_FILES = {f"2025-{m:02d}": ROOT / f"data/raw/training_2025-{m:02d}-01_{2025 + (m == 12)}-{(m % 12) + 1:02d}-01.parquet" for m in range(1, 13)}
CODE = ["src/prc2026_final/pop_signals.py", "src/prc2026_lirf_mixture/features.py", "src/prc2026_lirf_mixture/model.py", "tools/run_pop_signals.py"]


def load_table(config: dict, mixture_config: dict) -> pd.DataFrame:
    months = list(config["months"])
    movements = pd.concat([pd.read_parquet(MONTH_FILES[m], columns=ps.RAW_COLUMNS) for m in months], ignore_index=True)
    table = ps.population_table(movements, mixture_config)
    table = table[table["lomo_month"].isin(months)].reset_index(drop=True)
    table["mode"] = ps.modes_of(table, mixture_config)
    return table


def fold(table: pd.DataFrame, held: str, candidate: dict, config: dict, mixture_config: dict) -> pd.DataFrame:
    train = table[table["lomo_month"].ne(held)].copy()
    hold = table[table["lomo_month"].eq(held)].copy()
    names = ps.candidate_names(candidate)
    if "cs_rate" in names:
        fill = train["mode"].eq(0).to_numpy(dtype=float)
        smoothing = float(config["callsign_rate"]["smoothing"])
        train["cs_rate"] = ps.callsign_rates(train["flight"], train["lomo_month"], fill, train["flight"], train["lomo_month"], smoothing)
        hold["cs_rate"] = ps.callsign_rates(train["flight"], train["lomo_month"], fill, hold["flight"], None, smoothing)
    classifier = {**mixture_config["classifier"], "params": {**mixture_config["classifier"]["params"], "num_threads": int(config.get("threads", 4))}}
    component = ps.normal_component(train["label"].to_numpy(dtype=float), train["schedule_interval_s"].to_numpy(dtype=float), train["mode"].to_numpy(),
                                    mixture_config)
    weight = ps.stake_weights(train, component, candidate["stake_weight"]) if candidate.get("stake_weight") else None
    bag = ps.fit_bag(train, train["mode"].to_numpy(), train["movement_date_utc"].to_numpy(), names, classifier, config["seeds"], weight)
    mixture, p_fill = ps.mixture_for(bag, component, hold)
    return pd.DataFrame({ID: hold[ID].to_numpy(), "month": held, "interval_s": hold["schedule_interval_s"].to_numpy(dtype=float),
                         "label": hold["label"].to_numpy(dtype=float), "mode": hold["mode"].to_numpy(), "p_fill": p_fill, "mixture": mixture,
                         "rounds": float(np.mean([m["rounds"] for m in bag["members"]]))})


def summarise(rows: pd.DataFrame, control: pd.DataFrame | None, threshold: float, months: list[str], rng: np.random.Generator, draws: int,
              ranking_rows: dict | None = None) -> dict:
    scope = rows[rows["interval_s"] <= threshold]
    err = (scope["mixture"] - scope["label"]) ** 2
    per_month = err.groupby(scope["month"]).sum().reindex(months).fillna(0.0)
    out = {"rows_in_scope": int(len(scope)), "sse": float(err.sum()), "sse_by_month": {m: float(v) for m, v in per_month.items()},
           "auc_fill": float(roc_auc_score(scope["mode"].eq(0), scope["p_fill"])), "mean_rounds": float(rows["rounds"].mean())}
    if control is not None:
        cscope = control[control["interval_s"] <= threshold]
        cerr = ((cscope["mixture"] - cscope["label"]) ** 2).groupby(cscope["month"]).sum().reindex(months).fillna(0.0)
        a, b = per_month.to_numpy(), cerr.to_numpy()
        rel = (a.sum() - b.sum()) / b.sum()
        draws_rel = np.array([(a[i].sum() - b[i].sum()) / b[i].sum() for i in (rng.integers(0, len(months), len(months)) for _ in range(draws))])
        out.update({"relative_sse_change": float(rel), "months_better": int((a < b).sum()), "months": len(months),
                    "relative_change_by_month": {m: float((a[k] - b[k]) / b[k]) for k, m in enumerate(months)},
                    "bootstrap_95": [float(np.quantile(draws_rel, 0.025)), float(np.quantile(draws_rel, 0.975))]})
        ranking_rows = {m: int(n) for m, n in (ranking_rows or {}).items() if m in months}  # month keys only (the declaration also documents itself)
        if ranking_rows:  # informational: the ranking months' analogs, per-row SSE scaled to the ranking's in-scope population counts
            counts = {m: int((cscope["month"] == m).sum()) for m in ranking_rows}
            ctrl = sum(b[months.index(m)] / counts[m] * n for m, n in ranking_rows.items())
            cand = sum(a[months.index(m)] / counts[m] * n for m, n in ranking_rows.items())
            out["ranking_composition_read"] = {"relative_sse_change": float((cand - ctrl) / ctrl), "sse_change": float(cand - ctrl), "ranking_rows": ranking_rows}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", default="configs/pop_signals_v1.json")
    parser.add_argument("--report")
    parser.add_argument("--rows", help="optional parquet with every held-out row of every candidate")
    args = parser.parse_args()
    config = read_json(ROOT / args.declaration)
    if not config.get("declared_at_utc"):
        raise ValueError("The declaration is not recorded.")
    mixture_config = read_json(ROOT / config["mixture_declaration"])
    threshold = float(mixture_config["mixture"]["schedule_interval_threshold_seconds"])
    months = list(config["months"])
    table = load_table(config, mixture_config)
    rng = np.random.default_rng(int(config["rule"]["bootstrap_seed"]))
    results, frames = {}, []
    control_rows = None
    order = ["control", *config["candidates"]]
    for name in order:
        candidate = config["control"] if name == "control" else config["candidates"][name]
        rows = pd.concat([fold(table, m, candidate, config, mixture_config) for m in months], ignore_index=True)
        rows["candidate"] = name
        frames.append(rows)
        if name == "control":
            control_rows = rows
        results[name] = {"feature_set": candidate["feature_set"], "stake_weight": candidate.get("stake_weight"),
                         **summarise(rows, None if name == "control" else control_rows, threshold, months, rng, int(config["rule"]["bootstrap_draws"]),
                                     config.get("ranking_composition_read"))}
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[name].items() if k not in ("sse_by_month", "relative_change_by_month")}, flush=True)
    rule = config["rule"]
    eligible = [n for n in config["candidates"] if results[n]["relative_sse_change"] <= -float(rule["min_relative_sse_gain"])
                and results[n]["months_better"] >= int(rule["min_months_better"]) and results[n]["bootstrap_95"][1] < 0]
    selected = min(eligible, key=lambda n: results[n]["sse"]) if eligible else None
    report = {"created_at_utc": utc_now(), "declaration": {args.declaration: {"sha256": sha256_file(ROOT / args.declaration), "object_hash": object_hash(config)}},
              "mixture_declaration": {config["mixture_declaration"]: {"sha256": sha256_file(ROOT / config["mixture_declaration"])}},
              "code": {p: sha256_file(ROOT / p) for p in CODE}, "inputs": {str(p.relative_to(ROOT)): sha256_file(p) for p in MONTH_FILES.values()},
              "population_rows": int(len(table)), "rows_in_scope": int((table["schedule_interval_s"] <= threshold).sum()),
              "mode_counts": {str(k): int((table["mode"] == k).sum()) for k in range(3)}, "threshold_s": threshold, "seeds": config["seeds"],
              "results": results, "eligible": eligible, "selected_candidate": selected, "rule": rule}
    report_path = ROOT / (args.report or config["report"])
    write_json(report_path, report)
    if args.rows:
        pd.concat(frames, ignore_index=True).to_parquet(ROOT / args.rows, index=False)
    print("selected:", selected, "eligible:", eligible, "->", report_path)


if __name__ == "__main__":
    main()
