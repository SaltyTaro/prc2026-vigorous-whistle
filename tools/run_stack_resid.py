"""Stack residual learner (configs/stack_resid_v1.json): a month-blocked LightGBM on the shipped chain's residual from the stack's
own components and the published day discrepancy, added at a fixed weight after v57's variance re-allocation.

  --check                      label-free: the ranking components (final-file runner stages) and their reproduction checks, the
                               transfer table (development vs ranking feature spreads by airport; prediction RMS / mean by airport
                               of the all-four-month bag on the ranking against development) -> docs/stack_resid/transfer_check.json
  --score                      the governing run on the four development months (bag fitted without the scored month) and the
                               declared rule -> docs/stack_resid/results_v1.json + README.md
  --dry-run --version N [--base FILE --base-sha256 SHA]
                               the bag fitted on all four development months, applied to the base file (default the uploaded v57)
                               -> runs/stack_resid/dry_v<N>/ (never runs/submissions)

Development rows: runs/scout/refit/ctl_v55m.parquet (`new` = the shipped chain on the v55m model part) with the components of
runs/lomo_dev/rows.parquet and runs/dev_model_part/rows.parquet; published totals per (airport, off-block day predicted by v55m)
from runs/scout/refit/rows_v55m.parquet. Ranking rows: the final-file runner's stage outputs (runs/final_dryrun/run; the LFPG
July EOBT_1 repair as the shipped 50/50 blend from runs/final_lfpg_eobt/run on the edited rows), the v55m model-part file for the
off-block day, the base file as the chain's value. No departure label or block time of the ranking enters; no official score is used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402

DECL = "configs/stack_resid_v1.json"
OUT_DOCS = ROOT / "docs/stack_resid"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
GUARD = (213142471.0, 213142469.0)
SEEDS = [20261003, 11, 12]
WEIGHT = 0.5
FLOOR = -12.0
CLIP = 3600.0
BOOT_SEED = 20261003
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=400, feature_fraction=0.5, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=4, max_bin=127, feature_pre_filter=False, deterministic=True,
              force_col_wise=True)
ROUNDS, PATIENCE = 800, 60
RUNNER = ROOT / "runs/final_dryrun/run"
REPAIRED = ROOT / "runs/final_lfpg_eobt/run"
MODEL_PART = ("runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet", None)  # sha recorded at run time
V57 = ("runs/submissions/vigorous-whistle_v57.parquet", "31d909ff2e5d1937e65404c396893b9e5524d082c07422be9390c75aad4daed8")
COMPONENTS = ["v9", "residual", "base", "mix", "lirf_tree", "lirf_net", "committee", "nets", "v30", "v33", "v46", "v51m", "v52m", "v53m", "v54m", "v55m"]
PAIRS = [("nets", "committee"), ("mix", "base"), ("v9", "mix"), ("lirf_net", "lirf_tree"), ("v55m", "v46"), ("v51m", "v46"), ("v52m", "v51m"),
         ("v53m", "v52m"), ("v54m", "v53m"), ("v55m", "v54m"), ("stack", "v9"), ("committee", "v9"), ("nets", "v9")]
SPREAD = ["v9", "base", "mix", "committee", "nets", "lirf_tree", "lirf_net"]
TOP = ["d_comp_std", "d_lirf_net_lirf_tree", "d_v55m_v46", "d_nets_committee", "d_stack_v9", "p_disc_new_pr", "p_own_share"]
CODE = ["tools/run_stack_resid.py"]


# ---------------------------------------------------------------- features

def features(t: pd.DataFrame) -> pd.DataFrame:
    """46 label-free columns from components (`stack` = v55m), a, matched, and the published day total against the chain (`new`)."""
    f = pd.DataFrame(index=t.index)
    f["c_stack"] = t["stack"].to_numpy(dtype=float)
    for c in COMPONENTS:
        f[f"c_{c}"] = t[c].to_numpy(dtype=float)
    for x, y in PAIRS:
        f[f"d_{x}_{y}"] = (t[x] - t[y]).to_numpy(dtype=float)
    s = t[SPREAD].to_numpy(dtype=float)
    f["d_comp_std"] = np.nanstd(s, axis=1)
    f["d_comp_range"] = np.nanmax(s, axis=1) - np.nanmin(s, axis=1)
    f["c_a"] = t["a"].to_numpy(dtype=float)
    f["c_matched"] = t["matched"].astype(float).to_numpy()
    pos_new = np.maximum(t["a"].to_numpy(dtype=float) - t["new"].to_numpy(dtype=float), 0.0)
    pos_m = np.maximum(t["a"].to_numpy(dtype=float) - t["v55m_day"].to_numpy(dtype=float), 0.0)
    g = pd.DataFrame({"airport": t["airport"].to_numpy(), "bday": t["bday"].to_numpy(), "pub": t["pub"].to_numpy(dtype=float),
                      "n_pub": t["n_pub"].to_numpy(dtype=float), "pos_new": pos_new, "pos_m": pos_m, "one": 1.0})
    day = g.groupby(["airport", "bday"]).agg(pub=("pub", "first"), n_pub=("n_pub", "first"), s_new=("pos_new", "sum"), s_m=("pos_m", "sum"), n=("one", "sum"))
    day["disc_new"], day["disc_m"] = day["pub"] - day["s_new"], day["pub"] - day["s_m"]
    day["disc_new_pr"], day["disc_m_pr"] = day["disc_new"] / day["n"], day["disc_m"] / day["n"]
    day["pub_pr"], day["n_ratio"] = day["pub"] / day["n_pub"], day["n"] / day["n_pub"]
    j = g[["airport", "bday"]].join(day, on=["airport", "bday"])
    for c in ["pub", "n_pub", "disc_new", "disc_m", "disc_new_pr", "disc_m_pr", "pub_pr", "n_ratio", "n"]:
        f[f"p_{c}"] = j[c].to_numpy(dtype=float)
    f["p_complete"] = t["complete"].astype(float).to_numpy()
    f["p_own_share"] = pos_new / j["s_new"].replace(0, np.nan).to_numpy(dtype=float)
    f["p_pos_new"] = pos_new
    return f.astype(np.float32)


def fixed_rows(t: pd.DataFrame) -> np.ndarray:
    return t["cls"].eq("LIRF_pop").to_numpy() | t[ID].isin(GUARD).to_numpy()


# ---------------------------------------------------------------- tables

def takeoff(ids: np.ndarray, paths: list[Path]) -> pd.Series:
    raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]) for p in paths], ignore_index=True)
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(ids)
    if raw["MVT_TIME_UTC_mvt"].isna().any():
        raise ValueError("Take-off times do not cover the rows.")
    return raw


def dev_table() -> pd.DataFrame:
    ctl = pd.read_parquet(ROOT / "runs/scout/refit/ctl_v55m.parquet", columns=[ID, "month", "airport", "bday", "label", "stack", "new"])
    rows = pd.read_parquet(ROOT / "runs/scout/refit/rows_v55m.parquet", columns=[ID, "bday", "a", "matched", "cls", "pub", "n_pub", "complete"])
    lomo = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet", columns=[ID, "v9", "residual", "base", "mix", "lirf_tree", "lirf_net", "committee", "nets", "v30", "v33"])
    dmp = pd.read_parquet(ROOT / "runs/dev_model_part/rows.parquet", columns=[ID, "v46", "v51m", "v52m", "v53m", "v54m", "v55m"])
    t = ctl.merge(rows, on=ID, suffixes=("", "_rows"), validate="one_to_one").merge(lomo, on=ID, validate="one_to_one").merge(dmp, on=ID, validate="one_to_one")
    if not (t["bday"] == t["bday_rows"]).all() or np.abs(t["stack"] - t["v55m"]).max() > 1e-9:
        raise ValueError("The development tables disagree on the off-block day or on v55m.")
    t["v55m_day"] = t["v55m"]
    raw = takeoff(t[ID].to_numpy(), sorted((ROOT / "data/raw").glob("training_2025-*.parquet")))
    tk = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    t["takeoff_day"] = tk.dt.strftime("%Y-%m-%d").to_numpy()
    t["dom"] = tk.dt.day.to_numpy()
    return t.drop(columns=["bday_rows"])


def ranking_components(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from run_adsb_nn2_dev import weighted
    from run_adsb_nn_dev import mixed
    from run_lomo_rows import configs
    config, net, lirf_net, _ = configs()
    rec = pd.read_parquet(root / "recipe/detail.parquet")
    ids = rec[ID].to_numpy()
    nets = pd.read_parquet(root / "networks/ensembles.parquet").set_index(ID).reindex(ids).reset_index(drop=True)
    leaves = pd.read_parquet(root / "leaves/tables.parquet").set_index(ID).reindex(ids).reset_index(drop=True)
    tab = pd.concat([nets, leaves], axis=1)
    g = rec[[ID, "airport", "month", "scope", "v9", "residual"]].copy()
    g["base"] = g["v9"] + g["residual"]
    g["committee"], g["nets"] = tab[config["committee_table"]].to_numpy(dtype=float), weighted(tab, net["network"])
    g["mix"] = mixed(g["committee"].to_numpy(), g["nets"].to_numpy(), float(net["network_share"]))
    g["lirf_tree"], g["lirf_net"] = tab[config["lirf_table"]].to_numpy(dtype=float), weighted(tab, lirf_net)
    return g, rec


def ranking_table(base: tuple[str, str], checks: dict | None = None) -> pd.DataFrame:
    from export_daily_totals import SCOPES
    from run_fair_weights import build
    from run_lomo_rows import V30, configs
    from screen_daily_totals import published
    _, _, _, v33 = configs()
    g, rec = ranking_components(RUNNER)
    g_rep, _ = ranking_components(REPAIRED)
    if not np.array_equal(g[ID].to_numpy(), g_rep[ID].to_numpy()):
        raise ValueError("The repaired run's rows differ.")
    st = pd.read_parquet(RUNNER / "stack/detail.parquet").set_index(ID).reindex(g[ID])
    g["matched"] = g["scope"].map(SCOPES).astype(bool)
    g["lirf_m"], g["lirf_u"] = g["scope"].eq("lirf_nonpop_matched"), g["scope"].eq("lirf_nonpop_unmatched")
    g["floor"] = FLOOR
    g["v33"] = st["v33"].to_numpy(dtype=float)  # read by build only on the LIRF population (fixed rows)
    v30_unrepaired = build(g, *V30)
    if checks is not None:
        checks["v30_rebuilt_vs_runner_max_abs_s"] = float(np.abs(v30_unrepaired - st["v30"].to_numpy(dtype=float)).max())
        checks["committee_mix_vs_recipe_max_abs_s"] = float(np.abs(g["mix"].to_numpy() - rec["committee_mix"].to_numpy(dtype=float)).max())
    edited = g[ID].isin(set(pd.read_parquet(RUNNER / "repair/edited_ids.parquet")[ID])).to_numpy()
    for c in ["v9", "residual", "base", "committee", "nets", "mix", "lirf_tree", "lirf_net"]:
        g.loc[edited, c] = 0.5 * g.loc[edited, c].to_numpy(dtype=float) + 0.5 * g_rep.loc[edited, c].to_numpy(dtype=float)
    g["v30"] = build(g, *V30)
    g["v33"] = build(g, v33["outer"], v33["lirf"])
    for c in ["v46", "v51m", "v52m", "v53m", "v54m", "v55m"]:
        g[c] = st[c].to_numpy(dtype=float)
    g["stack"] = g["v55m"]
    raw = takeoff(g[ID].to_numpy(), [ROOT / "data/raw/ranking.parquet"])
    tk = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).reset_index(drop=True)
    g["a"] = (tk - pd.to_datetime(raw["SCHED_TIME_UTC_mvt"], utc=True).reset_index(drop=True)).dt.total_seconds().to_numpy()
    part = pd.read_parquet(ROOT / MODEL_PART[0]).set_index(ID)[TARGET].reindex(g[ID]).to_numpy(dtype=float)
    g["v55m_day"] = part
    g["bday"] = (tk - pd.to_timedelta(part, unit="s")).dt.strftime("%Y-%m-%d").to_numpy()
    chain = pd.read_parquet(ROOT / base[0]).set_index(ID)[TARGET].reindex(g[ID])
    if chain.isna().any():
        raise ValueError("The base file does not cover the ranking rows.")
    g["new"] = chain.to_numpy(dtype=float)
    g["cls"] = np.where(g["scope"].eq("fixed").to_numpy(), "LIRF_pop", np.where(g["matched"], g["airport"] + "_m", g["airport"] + "_u"))
    g = g.join(published(), on=["airport", "bday"])
    g["complete"] = (g["pub"].notna() & (g.groupby(["airport", "bday"])["a"].transform("size") >= 0.9 * g["n_pub"])).to_numpy()
    g["takeoff_day"] = tk.dt.strftime("%Y-%m-%d").to_numpy()
    if checks is not None:
        checks["model_part_file"] = {"path": MODEL_PART[0], "sha256": sha256_file(ROOT / MODEL_PART[0])}
        diff = np.rint(g["v55m"].to_numpy()) - part
        checks["runner_v55m_rounded_vs_model_part_file"] = {"rows_differ": int((diff != 0).sum()), "max_abs_s": float(np.abs(diff).max())}
        va = pd.read_parquet(ROOT / "runs/scout/var_alloc_export_v57/rows.parquet").set_index(ID).reindex(g[ID])
        checks["off_block_day_equals_v57_chain"] = bool((va["bday"].to_numpy() == g["bday"].to_numpy()).all())
        checks["repair_edited_rows_blended_50_50"] = int(edited.sum())
    return g


# ---------------------------------------------------------------- fitting

def fit_bag(X: np.ndarray, res: np.ndarray, fit: np.ndarray, dom: np.ndarray) -> list:
    val = fit & (dom % 5 == 0)
    trn = fit & ~val
    models = []
    for seed in SEEDS:
        p = {**PARAMS, "seed": seed}
        dtr = lgb.Dataset(X[trn], res[trn], params=p, free_raw_data=True)
        dva = lgb.Dataset(X[val], res[val], reference=dtr, params=p, free_raw_data=True)
        models.append(lgb.train(p, dtr, num_boost_round=ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)]))
    return models


def predict_bag(models: list, X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)


def apply(base: np.ndarray, pred: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    return np.where(fixed, base, np.maximum(base + WEIGHT * pred, FLOOR))


def paired(days, y, cand, ref, seed=BOOT_SEED):
    f = pd.DataFrame({"d": days, "c": (cand - y) ** 2, "r": (ref - y) ** 2}).groupby("d").agg(c=("c", "sum"), r=("r", "sum"), n=("c", "size"))
    idx = np.random.default_rng(seed).integers(0, len(f), size=(2000, len(f)))
    n = f["n"].to_numpy()[idx].sum(1)
    rel = np.sqrt(f["c"].to_numpy()[idx].sum(1) / n) / np.sqrt(f["r"].to_numpy()[idx].sum(1) / n) - 1
    delta = np.sqrt(f["c"].to_numpy()[idx].sum(1) / n) - np.sqrt(f["r"].to_numpy()[idx].sum(1) / n)
    return {"interval_s": [float(v) for v in np.quantile(delta, [0.025, 0.975])], "interval_pct": [float(100 * v) for v in np.quantile(rel, [0.025, 0.975])],
            "days": int(len(f)), "days_better": int((f["c"] < f["r"]).sum())}


def dev_matrix(t: pd.DataFrame):
    X = features(t)
    res = np.clip(t["label"].to_numpy(dtype=float) - t["new"].to_numpy(dtype=float), -CLIP, CLIP)
    return X, res


def by_airport_rms(t: pd.DataFrame, pred: np.ndarray, fixed: np.ndarray) -> dict:
    d = pd.DataFrame({"airport": t["airport"].to_numpy(), "p": pred})[~fixed]
    return {a: {"rms_s": float(np.sqrt((g["p"] ** 2).mean())), "mean_s": float(g["p"].mean()), "rows": int(len(g))} for a, g in d.groupby("airport")}


# ---------------------------------------------------------------- commands

def check() -> None:
    checks: dict = {}
    dev = dev_table()
    rk = ranking_table(V57, checks)
    Xd, res = dev_matrix(dev)
    Xr = features(rk)
    if list(Xd.columns) != list(Xr.columns):
        raise ValueError("Feature columns differ.")
    fx_d, fx_r = fixed_rows(dev), fixed_rows(rk)
    models = fit_bag(Xd.to_numpy(), res, ~fx_d, dev["dom"].to_numpy())
    pd_in = predict_bag(models, Xd.to_numpy())
    pr = predict_bag(models, Xr.to_numpy())
    pr[fx_r] = 0.0
    pd_in[fx_d] = 0.0
    spread = {}
    for c in TOP:
        spread[c] = {}
        for a in sorted(dev["airport"].unique()):
            dv = Xd.loc[(dev["airport"].eq(a) & ~fx_d).to_numpy(), c].dropna()
            rv = Xr.loc[(rk["airport"].eq(a) & ~fx_r).to_numpy(), c].dropna()
            spread[c][a] = {"dev_q05_q50_q95": [float(v) for v in dv.quantile([0.05, 0.5, 0.95])], "rank_q05_q50_q95": [float(v) for v in rv.quantile([0.05, 0.5, 0.95])],
                            "dev_sd": float(dv.std()), "rank_sd": float(rv.std()), "sd_ratio": float(rv.std() / dv.std()) if dv.std() > 0 else None}
    out = {"created_at_utc": utc_now(), "checks": checks, "trees": [int(m.best_iteration) for m in models],
           "prediction_by_airport": {"dev_in_sample": by_airport_rms(dev, pd_in, fx_d), "ranking": by_airport_rms(rk, pr, fx_r)}, "spreads": spread,
           "note": "label-free on the ranking; the bag is fitted on all four development months (in-sample on development)"}
    rat = {a: out["prediction_by_airport"]["ranking"][a]["rms_s"] / out["prediction_by_airport"]["dev_in_sample"][a]["rms_s"] for a in out["prediction_by_airport"]["ranking"]}
    out["ranking_over_dev_in_sample_rms"] = rat
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DOCS / "transfer_check.json", out)
    print("checks", checks)
    print("trees", out["trees"])
    for a in sorted(rat):
        r, d = out["prediction_by_airport"]["ranking"][a], out["prediction_by_airport"]["dev_in_sample"][a]
        print(f"{a} dev rms {d['rms_s']:.1f} mean {d['mean_s']:+.1f} | ranking rms {r['rms_s']:.1f} mean {r['mean_s']:+.1f} | ratio {rat[a]:.2f}")
    for c in TOP:
        print(c, {a: (round(v["sd_ratio"], 2) if v["sd_ratio"] else None) for a, v in spread[c].items()})


def score() -> None:
    decl = read_json(ROOT / DECL)
    dev = dev_table()
    X, res = dev_matrix(dev)
    Xa = X.to_numpy()
    fx = fixed_rows(dev)
    month, y, new = dev["month"].to_numpy(), dev["label"].to_numpy(dtype=float), dev["new"].to_numpy(dtype=float)
    pred = np.zeros(len(dev))
    out = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": object_hash(decl), "code": {c: sha256_file(ROOT / c) for c in CODE},
           "control": "runs/scout/refit/ctl_v55m.parquet `new`", "months": {}, "trees": {}}
    for m in MONTHS:
        models = fit_bag(Xa, res, (month != m) & ~fx, dev["dom"].to_numpy())
        ev = month == m
        p = predict_bag(models, Xa[ev])
        p[fx[ev]] = 0.0
        pred[ev] = p
        cand = apply(new[ev], p, fx[ev])
        ctl = float(np.sqrt(np.mean((y[ev] - new[ev]) ** 2)))
        r = float(np.sqrt(np.mean((y[ev] - cand) ** 2)))
        out["trees"][m] = [int(b.best_iteration) for b in models]
        out["months"][m] = {"control_rmse": ctl, "rmse": r, "change_pct": 100 * (r / ctl - 1), **paired(dev["takeoff_day"].to_numpy()[ev], y[ev], cand, new[ev]),
                            "rows_moved": int((np.abs(cand - new[ev]) > 1e-9).sum()), "rms_move_s": float(np.sqrt(np.mean((cand - new[ev]) ** 2))),
                            "max_abs_move_s": float(np.abs(cand - new[ev]).max())}
        print(m, {k: out["months"][m][k] for k in ["control_rmse", "rmse", "change_pct", "interval_pct"]}, flush=True)
    eq = float(np.mean([out["months"][m]["change_pct"] for m in EQUAL]))
    out["equal_footing_mean_pct"] = eq
    out["dev_oof_prediction_by_airport"] = by_airport_rms(dev, pred, fx)
    cand_all = apply(new, pred, fx)
    by = pd.DataFrame({"airport": dev["airport"], "month": month, "c": (new - y) ** 2, "n": (cand_all - y) ** 2}).groupby(["airport", "month"]).sum()
    out["airport_month_change_pct"] = {f"{a} {m}": float(100 * (np.sqrt(r["n"]) / np.sqrt(r["c"]) - 1)) for (a, m), r in by.iterrows()}
    # rule (4) needs the ranking side: the bag on all four months applied to the ranking (label-free)
    rk = ranking_table(V57)
    models = fit_bag(Xa, res, ~fx, dev["dom"].to_numpy())
    pr = predict_bag(models, features(rk).to_numpy())
    fxr = fixed_rows(rk)
    pr[fxr] = 0.0
    out["ranking_prediction_by_airport"] = by_airport_rms(rk, pr, fxr)
    ratios = {a: out["ranking_prediction_by_airport"][a]["rms_s"] / out["dev_oof_prediction_by_airport"][a]["rms_s"] for a in out["ranking_prediction_by_airport"]}
    out["ranking_over_dev_rms"] = ratios
    # sensitivity, reported and never selectable: ranking rows outside the development support of the top features keep the chain
    Xr = features(rk)
    outside = np.zeros(len(rk), dtype=bool)
    for a in sorted(dev["airport"].unique()):
        dm, rm = (dev["airport"].eq(a).to_numpy() & ~fx), rk["airport"].eq(a).to_numpy()
        for c in TOP:
            lo, hi = X.loc[dm, c].quantile([0.001, 0.999])
            v = Xr.loc[rm, c].to_numpy()
            outside[np.flatnonzero(rm)] |= np.isfinite(v) & ((v < lo) | (v > hi))
    ps = np.where(outside, 0.0, pr)
    sup = by_airport_rms(rk, ps, fxr)
    out["sensitivity_support_guard"] = {"rows_outside_support": int((outside & ~fxr).sum()),
                                        "rows_outside_by_airport": {a: int(n) for a, n in pd.Series(rk["airport"].to_numpy()[outside & ~fxr]).value_counts().items()},
                                        "ranking_over_dev_rms": {a: sup[a]["rms_s"] / out["dev_oof_prediction_by_airport"][a]["rms_s"] for a in sup},
                                        "selectable": False}
    rule = {"no_month_worse": all(out["months"][m]["change_pct"] <= 0 for m in MONTHS),
            "equal_footing_mean_at_most_-0.05pct": eq <= -0.05,
            "equal_footing_intervals_below_zero": all(out["months"][m]["interval_s"][1] < 0 for m in EQUAL),
            "transfer_guard_0.5x_2x": all(0.5 <= v <= 2.0 for v in ratios.values())}
    out["rule"] = rule
    out["accepted"] = all(rule.values())
    out["twelve_month_check"] = "none exists: the component tables cover only the four development months"
    out["official_scores_used"] = "none"
    write_json(OUT_DOCS / "results_v1.json", out)
    write_readme(out)
    print("equal-footing mean", eq, "rule", rule, "accepted", out["accepted"])


def write_readme(out: dict) -> None:
    lines = ["# Stack residual learner (`configs/stack_resid_v1.json`) — governing run", "",
             f"Accepted by the declared rule: **{out['accepted']}**. Results: [`results_v1.json`](results_v1.json); transfer check: "
             "[`transfer_check.json`](transfer_check.json). Exploration: [`docs/dig/residual_ore.md`](../dig/residual_ore.md).", "",
             "| Month | Control RMSE | Candidate | Change | Paired-day 95% (s) | Rows moved | RMS move (s) | Max move (s) |", "|---|---|---|---|---|---|---|---|"]
    for m in MONTHS:
        r = out["months"][m]
        lines.append(f"| {m} | {r['control_rmse']:.4f} | {r['rmse']:.4f} | {r['change_pct']:+.4f}% | [{r['interval_s'][0]:+.3f}, {r['interval_s'][1]:+.3f}] | "
                     f"{r['rows_moved']:,} | {r['rms_move_s']:.1f} | {r['max_abs_move_s']:.0f} |")
    lines += ["", f"Equal-footing mean (January, July, December): **{out['equal_footing_mean_pct']:+.4f}%**.", "",
              "Rule: " + "; ".join(f"{k} = {v}" for k, v in out["rule"].items()) + ".", "",
              "Transfer guard (ranking prediction RMS over development out-of-month RMS, by airport): " +
              ", ".join(f"{a} {v:.2f}" for a, v in sorted(out["ranking_over_dev_rms"].items())) + ".", "",
              "Sensitivity (reported, not selectable): ranking rows outside the development support of the top features left at the chain "
              f"({out['sensitivity_support_guard']['rows_outside_support']:,} rows): ratios " +
              ", ".join(f"{a} {v:.2f}" for a, v in sorted(out["sensitivity_support_guard"]["ranking_over_dev_rms"].items())) + ".", "",
              "No twelve-month check exists (the component tables cover only the four development months). No official score is used."]
    (OUT_DOCS / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(args) -> None:
    out_dir = ROOT / "runs/stack_resid" / f"dry_v{args.version}"
    if (ROOT / "runs/submissions").resolve() in [out_dir.resolve(), *out_dir.resolve().parents]:
        raise ValueError("A dry run never writes under runs/submissions.")
    output = out_dir / f"vigorous-whistle_v{args.version}.parquet"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    base = (args.base, args.base_sha256)
    if sha256_file(ROOT / base[0]) != base[1]:
        raise ValueError("The base file does not have the expected SHA-256.")
    dev = dev_table()
    X, res = dev_matrix(dev)
    fx = fixed_rows(dev)
    models = fit_bag(X.to_numpy(), res, ~fx, dev["dom"].to_numpy())
    checks: dict = {}
    rk = ranking_table(base, checks)
    pr = predict_bag(models, features(rk).to_numpy())
    fxr = fixed_rows(rk)
    pr[fxr] = 0.0
    b = rk["new"].to_numpy(dtype=float)
    values = apply(b, pr, fxr)
    provenance = {"experiment_id": "stack_resid_v1", "dry_run": True, "base": {"path": base[0], "sha256": base[1]}, "declaration": {DECL: sha256_file(ROOT / DECL)},
                  "code": {c: sha256_file(ROOT / c) for c in CODE}, "official_scores_used": "none", "audit_used": False,
                  "recipe": "base + 0.5 x (3-seed LightGBM bag on the chain's development residual from the stack's components and the published day discrepancy); LIRF population and the two guard rows unchanged; floor -12 s"}
    receipt = export_submission(pd.DataFrame({ID: rk[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(rk[ID]).to_numpy().astype(np.int64)
    change = exported - b.astype(np.int64)
    if (change[fxr] != 0).any():
        raise ValueError("A fixed row changed.")
    by = pd.DataFrame({"airport": rk["airport"], "month": rk["month"], "ch": change.astype(float)})
    rep = {"created_at_utc": utc_now(), "version": args.version, "output": output.relative_to(ROOT).as_posix(), "sha256": sha256_file(output), "receipt": receipt,
           "checks": checks, "trees": [int(m.best_iteration) for m in models], "rows_changed": int((change != 0).sum()),
           "sum_squared_change": float((change.astype(float) ** 2).sum()), "rms_change_s": float(np.sqrt((change.astype(float) ** 2).mean())),
           "max_abs_change_s": float(np.abs(change).max()), "mean_change_s": float(change.mean()),
           "by_airport_month": {f"{a} {m}": {"rows_changed": int((g["ch"] != 0).sum()), "sum_squared_change": float((g["ch"] ** 2).sum()),
                                             "rms_change_s": float(np.sqrt((g["ch"] ** 2).mean())), "mean_change_s": float(g["ch"].mean())}
                                for (a, m), g in by.groupby(["airport", "month"])},
           "ranking_prediction_by_airport": by_airport_rms(rk, pr, fxr)}
    write_json(out_dir / f"export_v{args.version}.json", rep)
    pd.DataFrame({ID: rk[ID].to_numpy(), "airport": rk["airport"].to_numpy(), "base": b, "pred": pr, "value": exported}).to_parquet(out_dir / "rows.parquet", index=False)
    print({k: rep[k] for k in ["sha256", "rows_changed", "sum_squared_change", "rms_change_s", "max_abs_change_s", "mean_change_s", "trees"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", type=int)
    parser.add_argument("--base", default=V57[0])
    parser.add_argument("--base-sha256", default=V57[1])
    args = parser.parse_args()
    if args.check:
        check()
    if args.score:
        score()
    if args.dry_run:
        if args.version is None:
            raise ValueError("--dry-run needs --version.")
        dry_run(args)


if __name__ == "__main__":
    main()
