"""Design reading of configs/adsb_ctx3_lirf_design_v1.json (with its amendment): LIRF non-population rows only.

Readings (tools/ctx3_lirf_predict.py transfer files of the declared runs and, with --baseline, of the round-1 ctx_lirf design seeds):
  october            labels, cross-fitted v9, population flag and match flag from the recipe proxy's October table (as round B)
  2025-07_natural    labels, v9 and the LIRF non-population scope (lirf_m | lirf_u) from runs/clean_line_screens/hour/dev_stack.parquet;
  2025-07_thinned    match flag from the reading's own ADS-B table (natural: the dev stack; thinned: coverage-regime attempt a13)
  2025-12_natural / 2025-12_thinned  (a13 thins EDDF only; LIRF reads as natural)
Items: B = baseline four-seed mean; refresh = lref32 mean; refresh_mask = lref32 on matched rows, lm25_32 on unmatched rows; lm25_32
alone is reported. L1 = standalone RMSE (all / matched / unmatched); L2 = (1 - b) v9 + b X against the same with B, b in 0.35, 0.5, with
the paired-day 95% interval at b = 0.35; masking = refresh_mask against refresh on unmatched rows.
GO (declared): L2 at b = 0.35 at least 0.3% below B with the interval below zero on october AND on 2025-07_thinned, and not above B on
2025-07_natural and 2025-12_natural. If both items are GO, refresh_mask is carried only if it beats refresh on thinned July's unmatched rows.
The 16-seed ens_ctx_lirf July reading (runs/adsb_ctx/regime_check_lirf/2025-07.parquet) is reported as a reference, not in the rule.
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

DECL = "configs/adsb_ctx3_lirf_design_v1.json"
READINGS = ["october", "2025-07_natural", "2025-07_thinned", "2025-12_natural", "2025-12_thinned"]
THINNED_TABLE = {"2025-07": "runs/coverage_regime/a13_jul/features/thinned/committee/adsb_training_2025-07-01_2025-08-01.parquet",
                 "2025-12": "runs/coverage_regime/a13_dec/features/thinned/committee/adsb_training_2025-12-01_2026-01-01.parquet"}
B_VALUES, BAR = [0.35, 0.5], 0.3


def frame_for(reading: str, proxy: RecipeProxy | None, stack: pd.DataFrame) -> pd.DataFrame:
    """ID, label, v9, day, matched for the reading's LIRF non-population rows."""
    if reading == "october":
        t = proxy.table
        keep = t["airport"].eq("LIRF").to_numpy() & ~t["population"].astype(bool).to_numpy()
        return pd.DataFrame({ID: t[ID].to_numpy()[keep], "label": t[TARGET].to_numpy(dtype=float)[keep], "v9": t["v9"].to_numpy(dtype=float)[keep],
                             "day": t[TIME].dt.strftime("%Y-%m-%d").to_numpy()[keep], "matched": t["adsb_matched"].eq(1).to_numpy()[keep]})
    month, regime = reading.split("_")
    s = stack[stack["month"].eq(month) & (stack["lirf_m"] | stack["lirf_u"])]
    out = pd.DataFrame({ID: s[ID].to_numpy(), "label": s["label"].to_numpy(dtype=float), "v9": s["v9"].to_numpy(dtype=float),
                        "day": s["day"].astype(str).to_numpy(), "matched": s["lirf_m"].to_numpy(dtype=bool)})
    if regime == "thinned":
        thin = pd.read_parquet(ROOT / THINNED_TABLE[month], columns=[ID, "adsb_matched"]).set_index(ID)["adsb_matched"]
        flag = thin.reindex(out[ID])
        if flag.isna().any():
            raise ValueError(f"{reading}: thinned table misses rows.")
        out["matched"] = flag.eq(1).to_numpy()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transfer", type=Path, default=Path("runs/adsb_ctx3_lirf/transfer"))
    p.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    p.add_argument("--stack", type=Path, default=Path("runs/clean_line_screens/hour/dev_stack.parquet"))
    p.add_argument("--reference", type=Path, default=Path("runs/adsb_ctx/regime_check_lirf/2025-07.parquet"))
    p.add_argument("--report", type=Path, default=Path("docs/adsb_ctx3_lirf/design_report.json"))
    a = p.parse_args()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc") or not decl.get("amendments"):
        raise ValueError("Declaration or amendment missing.")
    proxy = RecipeProxy(pd.read_parquet(ROOT / a.committee))
    stack = pd.read_parquet(ROOT / a.stack, columns=[ID, "month", "day", "airport", "label", "v9", "lirf_m", "lirf_u"])
    tr = ROOT / a.transfer
    seeds = {v: decl["runs"][v] for v in ("lref32", "lm25_32")}
    files = {}

    def mean_of(names: list[str], reading: str, ids: pd.Series, root: Path) -> np.ndarray:
        vals = []
        for name in names:
            path = root / name / f"{reading}.parquet"
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
            v = pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
            if not np.isfinite(v).all():
                raise ValueError(f"{path} does not cover the reading's rows.")
            vals.append(v)
        return np.mean(vals, axis=0)

    out, go_parts = {}, {"refresh": {}, "refresh_mask": {}}
    for reading in READINGS:
        f = frame_for(reading, proxy, stack)
        ids, y, v9, days, m = f[ID], f["label"].to_numpy(), f["v9"].to_numpy(), f["day"].to_numpy(), f["matched"].to_numpy()
        base = mean_of([f"ctx_lirf_s{s}" for s in (1, 2, 3, 4)], reading, ids, tr / "baseline")
        ref = mean_of([f"lref32_s{s}" for s in seeds["lref32"]], reading, ids, tr)
        msk = mean_of([f"lm25_32_s{s}" for s in seeds["lm25_32"]], reading, ids, tr)
        items = {"B": base, "refresh": ref, "refresh_mask": np.where(m, ref, msk), "lm25_32_alone": msk}
        parts = {"all": np.ones(len(y), dtype=bool), "matched": m, "unmatched": ~m}
        e = {"rows": {k: int(v.sum()) for k, v in parts.items()}, "v9_rmse": rmse(y, v9),
             "L1": {name: {k: (rmse(y[v], x[v]) if v.any() else None) for k, v in parts.items()} for name, x in items.items()}, "L2": {}}
        for name in ("refresh", "refresh_mask", "lm25_32_alone"):
            x = items[name]
            e["L2"][name] = {}
            for b in B_VALUES:
                cand, ctrl = (1 - b) * v9 + b * x, (1 - b) * v9 + b * base
                e["L2"][name][str(b)] = {"rmse": rmse(y, cand), "B_rmse": rmse(y, ctrl), "change_pct": 100 * (rmse(y, cand) / rmse(y, ctrl) - 1),
                                         "unmatched_change_pct": (100 * (rmse(y[~m], cand[~m]) / rmse(y[~m], ctrl[~m]) - 1)) if (~m).any() else None}
            cand, ctrl = 0.65 * v9 + 0.35 * x, 0.65 * v9 + 0.35 * base
            e["L2"][name]["interval_b0.35_s"] = day_interval(days, y, cand, ctrl)
        if (~m).any():
            e["masking_unmatched"] = {"refresh": rmse(y[~m], ref[~m]), "refresh_mask": rmse(y[~m], msk[~m]),
                                      "change_pct": 100 * (rmse(y[~m], msk[~m]) / rmse(y[~m], ref[~m]) - 1),
                                      "interval_s": day_interval(days[~m], y[~m], msk[~m], ref[~m])}
        out[reading] = e
        for name in ("refresh", "refresh_mask"):
            go_parts[name][reading] = (e["L2"][name]["0.35"]["change_pct"], e["L2"][name]["interval_b0.35_s"])
    # the 16-seed reference on July
    refj = pd.read_parquet(ROOT / a.reference)
    files[a.reference.as_posix()] = sha256_file(ROOT / a.reference)
    ref_out = {}
    for regime in ("natural", "thinned"):
        f = frame_for(f"2025-07_{regime}", proxy, stack)
        x = refj.set_index(ID)[f"ens_ctx_lirf:{regime}"].reindex(f[ID]).to_numpy(dtype=float)
        y, v9 = f["label"].to_numpy(), f["v9"].to_numpy()
        ref_out[regime] = {"ens_ctx_lirf16_L1_all": rmse(y, x), "ens_ctx_lirf16_L2_b0.35": rmse(y, 0.65 * v9 + 0.35 * x),
                           "refresh_mask_L2_b0.35": out[f"2025-07_{regime}"]["L2"]["refresh_mask"]["0.35"]["rmse"],
                           "refresh_L2_b0.35": out[f"2025-07_{regime}"]["L2"]["refresh"]["0.35"]["rmse"]}
    go = {}
    for name, g in go_parts.items():
        strong = all(g[r][0] <= -BAR and g[r][1][1] < 0 for r in ("october", "2025-07_thinned"))
        safe = all(g[r][0] <= 0 for r in ("2025-07_natural", "2025-12_natural"))
        go[name] = {"strong_readings": strong, "not_worse_natural": safe, "go": bool(strong and safe)}
    carried = None
    if go["refresh_mask"]["go"] and go["refresh"]["go"]:
        carried = "refresh_mask" if out["2025-07_thinned"]["masking_unmatched"]["change_pct"] < 0 else "refresh"
    elif go["refresh_mask"]["go"] or go["refresh"]["go"]:
        carried = "refresh_mask" if go["refresh_mask"]["go"] else "refresh"
    rep = {"created_at_utc": utc_now(), "declaration_sha256": sha256_file(ROOT / DECL), "readings": out, "reference_16_seed_july": ref_out,
           "go": go, "carried": carried, "inputs_sha256": files, "audit_used": False,
           "note": "Design-fold models (fit 2025-02..06, 08, 09; October monitored); July / December are transfer readings as in configs/adsb_ctx3_pos_v1.json. No official score, ranking row or published aggregate read."}
    (ROOT / a.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / a.report, rep)
    for reading, e in out.items():
        l1 = e["L1"]
        print(f"{reading:16s} rows {e['rows']['all']:6d} (unm {e['rows']['unmatched']:5d})  L1 all B {l1['B']['all']:.2f} R {l1['refresh']['all']:.2f} RM {l1['refresh_mask']['all']:.2f}"
              f" | unm B {l1['B']['unmatched'] or float('nan'):.2f} R {l1['refresh']['unmatched'] or float('nan'):.2f} RM {l1['refresh_mask']['unmatched'] or float('nan'):.2f}")
        for name in ("refresh", "refresh_mask"):
            q = e["L2"][name]
            print(f"    L2 {name:13s} b.35 {q['0.35']['change_pct']:+.3f}% ({q['interval_b0.35_s'][0]:+.2f}..{q['interval_b0.35_s'][1]:+.2f})  b.5 {q['0.5']['change_pct']:+.3f}%")
        if "masking_unmatched" in e:
            mk = e["masking_unmatched"]
            print(f"    masking on unmatched {mk['change_pct']:+.3f}% ({mk['interval_s'][0]:+.2f}..{mk['interval_s'][1]:+.2f})")
    print("reference 16-seed July:", ref_out)
    print("GO:", go, "carried:", carried)


if __name__ == "__main__":
    main()
