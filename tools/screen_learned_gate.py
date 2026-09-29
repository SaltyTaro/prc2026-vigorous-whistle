"""Screen (descriptive): can a small gradient-boosted model over the stack's components predict the v46 stack's residual out of month?

Every layer shipped since v35 is a hand-made disagreement gate. This screen asks whether a learned gate finds more: a LightGBM fitted on
three development months' residuals (label - v46 stack) from the components and their differences, read on the fourth month
(month-blocked). Rows: runs/lomo_dev/rows.parquet, non-LIRF. The control is the full v46 stack on those rows
(tools/run_share_gate_reread.full_stack with the n85_nd1200 mix). Two feature sets: `components` (the stack's own columns) and
`context` (components plus label-free clock and schedule intervals from the raw movement files). Nothing is declared, selected or
exported; anything it suggests needs its own declaration. Report: docs/clean_line_screens/learned_gate_screen.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_final import share_gate  # noqa: E402
from run_adsb_v11 import rmse  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402
from run_share_gate_reread import full_stack  # noqa: E402

ROWS = "runs/lomo_dev/rows.parquet"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
RAW = {"2025-01": "training_2025-01-01_2025-02-01", "2025-07": "training_2025-07-01_2025-08-01", "2025-11": "training_2025-11-01_2025-12-01",
       "2025-12": "training_2025-12-01_2026-01-01"}
CAP = 20000.0
SEED = 20260929
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_child_samples=400, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
              reg_lambda=50.0, n_estimators=300, verbose=-1, n_jobs=6, random_state=SEED)
COMPONENTS = ["stack", "v9", "base", "committee", "nets", "mix", "d_mix_base", "d_nets_committee", "d_v9_mix", "matched", "airport_code"]
CONTEXT = ["hour", "proxy_s", "clock_missing", "sched_interval_s", "eobt_interval_s", "no_nm", "weekday"]


def context(month: str, ids: np.ndarray) -> pd.DataFrame:
    raw = pd.read_parquet(ROOT / f"data/raw/{RAW[month]}.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "ADEP_flt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(ids)
    t = raw["MVT_TIME_UTC_mvt"]
    clock = raw["AOBT_3_flt"].fillna(raw["LOBT_flt"])
    return pd.DataFrame({"hour": (t.dt.hour + t.dt.minute / 60.0).to_numpy(), "weekday": t.dt.weekday.to_numpy(dtype=float),
                         "proxy_s": (t - clock).dt.total_seconds().to_numpy(), "clock_missing": clock.isna().to_numpy(dtype=float),
                         "sched_interval_s": (t - raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy(),
                         "eobt_interval_s": (t - raw["EOBT_1_flt"]).dt.total_seconds().to_numpy(), "no_nm": raw["ADEP_flt"].isna().to_numpy(dtype=float)})


def main() -> None:
    rows = pd.read_parquet(ROOT / ROWS)
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = offsets(), offsets_all()
    seg = segments(MONTHS)
    k = read_json(ROOT / "docs/disagreement/tail_stretch_v2_results.json")["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    share, gate = share_gate.candidate_spec(read_json(ROOT / "configs/thinned_regime_round_v1.json"), "n85_nd1200")
    parts = []
    for month in MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        mix, _ = share_gate.mix(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), share, gate)
        f = f.assign(mix=mix)
        f["stack"] = full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)
        q = seg.reindex(f[ID])
        f["anchored"] = cs.reindex(f[ID])["motion_start"].notna().to_numpy() | (q["in_scope"].fillna(False).to_numpy(dtype=bool) & q["source"].eq("regular").to_numpy())
        parts.append(pd.concat([f, context(month, f[ID].to_numpy())], axis=1))
    t = pd.concat(parts, ignore_index=True)
    t["d_mix_base"], t["d_nets_committee"], t["d_v9_mix"] = t["mix"] - t["base"], t["nets"] - t["committee"], t["v9"] - t["mix"]
    t["matched"] = t["matched"].astype(float)
    t["airport_code"] = t["airport"].astype("category").cat.codes
    t["res"] = t["label"] - t["stack"]
    scope = (t["airport"].ne("LIRF") & ~t["anchored"]).to_numpy()
    rng = np.random.default_rng(SEED)
    report = {"created_at_utc": utc_now(), "rows_sha256": sha256_file(ROOT / ROWS), "params": PARAMS, "cap_s": CAP, "control": "the full v46 stack on the development rows",
              "scope": "non-LIRF rows outside the callsign and queue layers", "feature_sets": {"components": COMPONENTS, "context": COMPONENTS + CONTEXT}, "months": {}}
    for held in MONTHS:
        fit = scope & t["month"].ne(held).to_numpy() & (t["label"] < CAP).to_numpy()
        test = t["month"].eq(held).to_numpy()
        inside = scope[test]
        f = t.loc[test]
        y, control, days = f["label"].to_numpy(dtype=float), f["stack"].to_numpy(dtype=float), f["day"].to_numpy()
        entry = {"rows": int(test.sum()), "scope_rows": int(inside.sum()), "control_rmse": rmse(y, control), "provenance": sorted(set(f["provenance"]))}
        for name, cols in (("components", COMPONENTS), ("context", COMPONENTS + CONTEXT)):
            model = lgb.LGBMRegressor(**PARAMS).fit(t.loc[fit, cols], t.loc[fit, "res"].clip(-3600, 3600))
            pred = np.where(inside, model.predict(f[cols]), 0.0)
            gain = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
            entry[name] = {"top_features": {c: round(float(g / gain.sum()), 4) for c, g in gain.head(6).items()}, "weights": {}}
            for w in (0.25, 0.5, 1.0):
                values = np.maximum(control + w * pred, f["floor"].to_numpy(dtype=float))
                uniq, inv = np.unique(days, return_inverse=True)
                sa, sb, n = np.bincount(inv, (values - y) ** 2), np.bincount(inv, (control - y) ** 2), np.bincount(inv).astype(float)
                draws = [np.sqrt(sa[p].sum() / n[p].sum()) - np.sqrt(sb[p].sum() / n[p].sum()) for p in (rng.integers(0, len(uniq), len(uniq)) for _ in range(2000))]
                ordinary = y < CAP
                entry[name]["weights"][str(w)] = {"change_pct": 100 * (rmse(y, values) / rmse(y, control) - 1),
                                                   "change_pct_labels_below_cap": 100 * (rmse(y[ordinary], values[ordinary]) / rmse(y[ordinary], control[ordinary]) - 1),
                                                   "interval_s": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                                                   "mean_abs_move_s": float(np.abs(values - control)[inside].mean()),
                                                   "alignment": float(-np.sum((values - control) * (control - y)) / max(np.sum((values - control) ** 2), 1e-9))}
            airports = {}
            values = np.maximum(control + 0.5 * pred, f["floor"].to_numpy(dtype=float))
            for a in sorted(f["airport"].unique()):
                m = (f["airport"].to_numpy() == a) & inside
                if m.any():
                    airports[a] = round(100 * (rmse(y[m], values[m]) / rmse(y[m], control[m]) - 1), 3)
            entry[name]["airports_pct_at_0.5"] = airports
        report["months"][held] = entry
        print(held, entry["provenance"], "control", round(entry["control_rmse"], 3))
        for name in ("components", "context"):
            print("  ", name, {w: (round(e["change_pct"], 4), [round(v, 3) for v in e["interval_s"]], "align", round(e["alignment"], 2)) for w, e in entry[name]["weights"].items()})
            print("      airports at 0.5:", entry[name]["airports_pct_at_0.5"], "top:", entry[name]["top_features"], flush=True)
    write_json(ROOT / "docs/clean_line_screens/learned_gate_screen.json", report)


if __name__ == "__main__":
    main()
