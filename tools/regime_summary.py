"""Print the coverage_regime_v1 result tables (regime match, penalty, avoidability) as Markdown from the saved JSON records."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/coverage_regime"
DOCS = ROOT / "docs/coverage_regime"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def regime_table() -> None:
    config = load(ROOT / "configs/coverage_regime_v1.json")
    print("| regime | airport | matched % (natural 2025 / thinned / 2026 target) | gate % | points per segment | dark first point s |")
    print("|---|---|---|---|---|---|")
    for key, spec in config["regimes"].items():
        month = key.split(" ")[0]
        rec = load(RUNS / spec["attempt"] / f"attempt_{month}.json")
        for a in spec["airports_thinned"]:
            s = {k: rec["stats"][k][a] for k in ("natural_source", "thinned_source", "natural_target")}
            f = lambda name, d=1: " / ".join("-" if s[k][name] is None else f"{s[k][name]:.{d}f}" for k in s)  # noqa: E731
            print(f"| {key} | {a} | {f('match_pct')} | {f('gate_pct')} | {f('points_median', 0)} | {f('dark_first_point_s_median', 0)} |")


def penalty_tables() -> None:
    for month in ("2025-12", "2025-07"):
        rep = load(DOCS / f"penalty_{month}.json")
        if rep is None:
            continue
        w, n = rep["whole_month"], rep["non_lirf"]
        print(f"\n{month}: whole month {w['natural_rmse']:.3f} -> {w['thinned_rmse']:.3f} s ({w['change_pct']:+.3f}%), non-LIRF {n['natural_rmse']:.3f} -> {n['thinned_rmse']:.3f} ({n['change_pct']:+.3f}%); "
              f"SSE change {w['sse_change_total']:.4g}, of which blend tables alone {w['sse_change_blend_tables_only']:.4g}")
        print(f"reproduction: {rep['reproduction']}; residual check: {rep['residual_check']}")
        print("\n| airport | rows | natural | thinned | change | rows changed |\n|---|---:|---:|---:|---:|---:|")
        for a, e in rep["airports"].items():
            print(f"| {a} | {e['rows']:,} | {e['natural_rmse']:.2f} | {e['thinned_rmse']:.2f} | {e['change_pct']:+.3f}% | {e['rows_changed']:,} |")
        print("\n| airport: natural -> thinned class | rows | natural | thinned | SSE change share |\n|---|---:|---:|---:|---:|")
        for k, e in rep["transitions"].items():
            share = "-" if e["share_of_airport_sse_change"] is None else f"{100 * e['share_of_airport_sse_change']:.1f}%"
            print(f"| {k} | {e['rows']:,} | {e['natural_rmse']:.1f} | {e['thinned_rmse']:.1f} | {share} |")


def avoid_tables() -> None:
    for month in ("2025-12", "2025-07"):
        rep = load(DOCS / f"avoidability_{month}.json")
        if rep is None:
            continue
        print(f"\n{month} avoidability (adj2; thinned month, training thinned at {rep['airports_thinned_in_training']}):")
        print("| trained on | scope RMSE | shifted-airport scope RMSE | pure-adj2 recipe RMSE | natural month scope RMSE |\n|---|---:|---:|---:|---:|")
        for name in ("natural", "mix", "thinned"):
            t, n = rep["scores"][f"thinned:{name}_trained"], rep["scores"][f"natural:{name}_trained"]
            print(f"| {name} | {t['scope_rmse']:.3f} | {t['shifted_scope_rmse']:.3f} | {t['pure_adj2_recipe_rmse']:.3f} | {n['scope_rmse']:.3f} |")
        print(json.dumps(rep["avoidable"], indent=1))


if __name__ == "__main__":
    what = sys.argv[1:] or ["regime", "penalty", "avoid"]
    if "regime" in what:
        regime_table()
    if "penalty" in what:
        penalty_tables()
    if "avoid" in what:
        avoid_tables()
