"""Governing run of configs/stack_second_v1.json: a second guarded residual pass on top of v71 at fixed airport sets.

  --build   (workstation, light)  development and ranking tables -> runs/stack_second/{dev,rank}_table.parquet (+ hashes); the control
                                   (v71's development values) is rebuilt and checked against docs/nbr_state/dark_results_v1.json
  --fit     (Azure VM)             five-seed LightGBM bags per candidate: month-blocked development predictions and ranking predictions
                                   -> runs/stack_second/preds_{dev,rank}.parquet
  --score   (workstation)          clauses 1-4, selection, dry run of the selected candidate -> docs/stack_second/results_v1.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DECL = ROOT / "configs/stack_second_v1.json"
OUT = ROOT / "runs/stack_second"
DOCS = ROOT / "docs/stack_second"
V70 = (ROOT / "runs/submissions/vigorous-whistle_v70.parquet", "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f")
V71 = (ROOT / "runs/submissions/vigorous-whistle_v71.parquet", "2dfb3b9f53284c7ae3a02882d19cfd61b19116b1b00504176551bcae8b82e6ba")
DARK = ROOT / "docs/nbr_state/dark_results_v1.json"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
MONTHS, EQUAL = ["2025-01", "2025-07", "2025-11", "2025-12"], ["2025-01", "2025-07", "2025-12"]
GUARD = (213142471.0, 213142469.0)
NB = ["nx", "nxk", "nn", "x_nmerr"]
QU = ["gp", "gn", "gp2", "gp4", "gn4", "bpos", "blen"]
CL = ["c_sched", "c_aobt", "c_lobt", "c_eobt", "c_iobt", "c_arvt", "c_hour", "c_dow", "c_layers"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=800, lambda_l2=50.0, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=16, max_bin=127, deterministic=True, force_col_wise=True)
SEEDS, ROUNDS, W, CAP, FLOOR, CLIP_Y = [7, 11, 12, 13, 14], 400, 0.5, 120.0, -12.0, 1800.0


def sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stacker_names(cols) -> list[str]:
    out: list[str] = []
    for i, c in enumerate(cols):
        n = f"s_{c}"
        out.append(n if n not in out else f"{n}_{i}")
    return out


def clocks(t: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    r = raw.set_index(ID).reindex(t[ID].to_numpy())
    tk = t["MVT_TIME_UTC_mvt"].to_numpy()
    tk = pd.Series(pd.to_datetime(t["MVT_TIME_UTC_mvt"].to_numpy(), utc=True))
    out = pd.DataFrame(index=range(len(t)))
    for nm, col in [("c_sched", "SCHED_TIME_UTC_mvt"), ("c_aobt", "AOBT_3_flt"), ("c_lobt", "LOBT_flt"), ("c_eobt", "EOBT_1_flt"), ("c_iobt", "IOBT_flt")]:
        out[nm] = (tk - pd.Series(r[col].to_numpy())).dt.total_seconds().to_numpy()
    out["c_arvt"] = (pd.Series(r["ARVT_3_flt"].to_numpy()) - tk).dt.total_seconds().to_numpy()
    out["c_hour"] = (tk.dt.hour + tk.dt.minute / 60).to_numpy()
    out["c_dow"] = tk.dt.dayofweek.to_numpy()
    return out


def nm_error_state(t: pd.DataFrame, raw: pd.DataFrame) -> np.ndarray:
    import run_nbr_state as N
    r = raw.set_index(ID).reindex(t[ID].to_numpy())
    proxy = (pd.Series(pd.to_datetime(t["MVT_TIME_UTC_mvt"].to_numpy(), utc=True)) - pd.Series(r["AOBT_3_flt"].to_numpy())).dt.total_seconds().to_numpy()
    ok = np.isfinite(proxy) & (proxy >= -3600) & (proxy <= 7200)
    s = t[["airport", "ts", "date"]].reset_index(drop=True).copy()
    s["matched"], s["value"], s["v9"] = t["matched"].to_numpy() & ok, t["value"].to_numpy(dtype=float), np.nan_to_num(proxy)
    return N.state(s)


def gate_flags(ids: np.ndarray, months: list[str]) -> np.ndarray:
    g = pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/matches/matches2_{m}.parquet", columns=[ID, "gate"]) for m in months])
    return g.drop_duplicates(ID).set_index(ID)["gate"].reindex(ids).fillna(False).astype(bool).to_numpy()


def control_dev(d: pd.DataFrame, gate: np.ndarray) -> np.ndarray:
    """v71's development values: nbr_state_v1 w75 (per-airport month-blocked slopes) then nbr_dark_v1 (pooled month-blocked slope)."""
    import run_nbr_state as N
    x = N.state(d)
    y = (d["label"] - d["value"]).to_numpy()
    new = d["value"].to_numpy(dtype=float).copy()
    unm = ~d["matched"].to_numpy() & ~d["fixed"].to_numpy() & ~np.isnan(x)
    for ap in d["airport"].unique():
        a = unm & (d["airport"] == ap).to_numpy()
        for m in MONTHS:
            f, t = a & (d["month"] != m).to_numpy(), a & (d["month"] == m).to_numpy()
            if f.sum() >= N.MIN_FIT and t.any():
                b = N.slope(x[f], y[f])
                new[t] = np.maximum(new[t] + np.clip(0.75 * b * x[t], -N.CAP, N.CAP), N.FLOOR)
    dk = d["matched"].to_numpy() & ~gate & ~d["fixed"].to_numpy() & ~np.isnan(x)
    y2 = np.clip(d["label"].to_numpy() - new, -N.CLIP_Y, N.CLIP_Y)
    base = new.copy()
    for m in MONTHS:
        f, t = dk & (d["month"] != m).to_numpy(), dk & (d["month"] == m).to_numpy()
        b = float((x[f] * y2[f]).sum() / (x[f] ** 2).sum())
        new[t] = np.maximum(base[t] + np.clip(0.75 * b * x[t], -N.CAP, N.CAP), N.FLOOR)
    return new


def build() -> None:
    import nbr_columns as C
    import run_nbr_state as N
    import run_stack_queue as Q
    import run_stack_weight as SW
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- development
    d = N.dev_table()
    gate = gate_flags(d[ID].to_numpy(), MONTHS)
    ctl = control_dev(d, gate)
    dark = json.loads(DARK.read_text(encoding="utf-8"))
    for m in MONTHS:
        s = (d["month"] == m).to_numpy()
        rm = float(np.sqrt(((d["label"].to_numpy()[s] - ctl[s]) ** 2).mean()))
        if abs(rm - dark["dev"][m]["rmse"]) > 1e-9:
            raise ValueError(f"control not reproduced in {m}: {rm} vs {dark['dev'][m]['rmse']}")
    dev, X46, X65, X79, Pn, Pg, Po, res = SW.dev_inputs()
    X79 = X79.reset_index(drop=True)
    X79.columns = stacker_names(X79.columns)
    X79[ID] = dev[ID].to_numpy()
    q = Q.queue_dev(dev.reset_index(drop=True))
    q[ID] = dev[ID].to_numpy()
    raw = pd.concat([pd.read_parquet(ROOT / f"data/raw/{N.RAW[m]}.parquet", columns=[ID, "PHASE_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt",
                                                                                     "EOBT_1_flt", "IOBT_flt", "ARVT_3_flt"]) for m in MONTHS])
    raw = raw[raw["PHASE_mvt"] == "DEP"].drop(columns="PHASE_mvt")
    T = d[[ID, "month", "airport", "day", "label"]].copy()
    T["v70"], T["v71"] = d["value"].to_numpy(dtype=float), ctl
    T["fixed"] = d["fixed"].to_numpy()
    T["kind"] = np.where(T["fixed"], 3, np.where(~d["matched"].to_numpy(), 0, np.where(gate, 2, 1)))
    T["ap"] = d["airport"].map({a: i for i, a in enumerate(AIRPORTS)}).to_numpy()
    c = C.columns(d)
    for col in ["nx", "nxk", "nn"]:
        T[col] = c[col].to_numpy()
    T["x_nmerr"] = nm_error_state(d, raw)
    T = pd.concat([T.reset_index(drop=True), clocks(d, raw)], axis=1)
    T["c_layers"] = T["v71"] - T["v70"]
    T = T.merge(q, on=ID, how="left", validate="one_to_one").merge(X79, on=ID, how="left", validate="one_to_one")
    if T[X79.columns[0]].isna().any() or T["bpos"].isna().any():
        raise ValueError("development join incomplete")
    T.to_parquet(OUT / "dev_table.parquet", index=False)
    # ---- ranking
    if sha(V70[0]) != V70[1] or sha(V71[0]) != V71[1]:
        raise ValueError("v70 / v71 files changed")
    k = N.ranking_table()
    gk = gate_flags(k[ID].to_numpy(), ["2026-01", "2026-07"])
    _, _, _, mats, rk, fxr, h2, h3, ho, Por = SW.rank_inputs()
    X79r = [m for m in mats if m[0] == "po"][0][2].reset_index(drop=True)
    X79r.columns = X79.columns[:-1]
    X79r[ID] = rk[ID].to_numpy()
    qr = Q.queue_rank(rk)
    qr[ID] = rk[ID].to_numpy()
    rraw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt",
                                                                     "IOBT_flt", "ARVT_3_flt"])
    rraw = rraw[rraw["PHASE_mvt"] == "DEP"].drop(columns="PHASE_mvt")
    v71 = pd.read_parquet(V71[0]).set_index(ID)[TARGET]
    K = k[[ID, "month", "airport"]].copy()
    K["v70"], K["v71"] = k["value"].to_numpy(dtype=float), v71.reindex(k[ID]).to_numpy(dtype=float)
    K["fixed"] = k["fixed"].to_numpy()
    K["kind"] = np.where(K["fixed"], 3, np.where(~k["matched"].to_numpy(), 0, np.where(gk, 2, 1)))
    K["ap"] = k["airport"].map({a: i for i, a in enumerate(AIRPORTS)}).to_numpy()
    ck = C.columns(k)
    for col in ["nx", "nxk", "nn"]:
        K[col] = ck[col].to_numpy()
    K["x_nmerr"] = nm_error_state(k, rraw)
    K = pd.concat([K.reset_index(drop=True), clocks(k, rraw)], axis=1)
    K["c_layers"] = K["v71"] - K["v70"]
    K = K.merge(qr, on=ID, how="left", validate="one_to_one").merge(X79r, on=ID, how="left", validate="one_to_one")
    if K[X79.columns[0]].isna().any() or K["bpos"].isna().any() or K["v71"].isna().any():
        raise ValueError("ranking join incomplete")
    K.to_parquet(OUT / "rank_table.parquet", index=False)
    rec = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "dev_rows": len(T), "rank_rows": len(K),
           "features": features(T), "control_dev_rmse": {m: dark["dev"][m]["rmse"] for m in MONTHS},
           "sha256": {"dev_table": sha(OUT / "dev_table.parquet"), "rank_table": sha(OUT / "rank_table.parquet"), "declaration": sha(DECL),
                      "tool": sha(Path(__file__))}}
    (OUT / "tables.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec["sha256"], indent=1), len(rec["features"]), "features")


def features(T: pd.DataFrame) -> list[str]:
    return [c for c in T.columns if c.startswith("s_")] + NB + QU + CL + ["kind", "ap"]


def fit(table_dir: Path) -> None:
    import lightgbm as lgb
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    T, K = pd.read_parquet(table_dir / "dev_table.parquet"), pd.read_parquet(table_dir / "rank_table.parquet")
    F = features(T)
    y = np.clip((T["label"] - T["v71"]).to_numpy(), -CLIP_Y, CLIP_Y)
    pd_dev, pd_rank = pd.DataFrame({ID: T[ID]}), pd.DataFrame({ID: K[ID]})
    XT, XK = T[F].astype(np.float32), K[F].astype(np.float32)
    for c in decl["candidates"]:
        inset = T["airport"].isin(c["airports"]).to_numpy() & (T["kind"] != 3).to_numpy()
        p = np.full(len(T), np.nan)
        for m in MONTHS:
            f, t = inset & (T["month"] != m).to_numpy(), inset & (T["month"] == m).to_numpy()
            acc = np.zeros(int(t.sum()))
            for s in SEEDS:
                mdl = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(XT[f], y[f], categorical_feature=["kind", "ap"]), ROUNDS)
                acc += mdl.predict(XT[t]) / len(SEEDS)
            p[t] = acc
            print(c["name"], m, "done", flush=True)
        pd_dev[c["name"]] = p
        kin = K["airport"].isin(c["airports"]).to_numpy() & (K["kind"] != 3).to_numpy()
        acc = np.zeros(int(kin.sum()))
        for s in SEEDS:
            mdl = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(XT[inset], y[inset], categorical_feature=["kind", "ap"]), ROUNDS)
            acc += mdl.predict(XK[kin]) / len(SEEDS)
        pr = np.full(len(K), np.nan)
        pr[kin] = acc
        pd_rank[c["name"]] = pr
        print(c["name"], "ranking done", flush=True)
    pd_dev.to_parquet(table_dir / "preds_dev.parquet", index=False)
    pd_rank.to_parquet(table_dir / "preds_rank.parquet", index=False)


def score() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import run_intruder_bound as R
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    tabs = json.loads((OUT / "tables.json").read_text(encoding="utf-8"))
    for nm in ["dev_table", "rank_table"]:
        if sha(OUT / f"{nm}.parquet") != tabs["sha256"][nm]:
            raise ValueError(f"{nm} changed")
    T, K = pd.read_parquet(OUT / "dev_table.parquet"), pd.read_parquet(OUT / "rank_table.parquet")
    PD = pd.read_parquet(OUT / "preds_dev.parquet").set_index(ID).reindex(T[ID]).reset_index()
    PR = pd.read_parquet(OUT / "preds_rank.parquet").set_index(ID).reindex(K[ID]).reset_index()
    out = {"experiment_id": decl["experiment_id"], "declaration_sha256": sha(DECL), "audit_used": False, "official_scores_used": "none",
           "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tables": tabs["sha256"],
           "preds_sha256": {"dev": sha(OUT / "preds_dev.parquet"), "rank": sha(OUT / "preds_rank.parquet")}, "candidates": {}}
    for c in decl["candidates"]:
        name, aps = c["name"], c["airports"]
        p = PD[name].to_numpy()
        mv = ~np.isnan(p)
        new = T["v71"].to_numpy(dtype=float).copy()
        new[mv] = np.maximum(new[mv] + np.clip(c["weight"] * p[mv], -CAP, CAP), FLOOR)
        rec = {"dev": {}}
        rng = np.random.default_rng(20261004)
        for m in MONTHS:
            s = (T["month"] == m).to_numpy()
            sub = T[s].assign(value=T["v71"][s])
            pct, lo, hi = R.interval(sub, "value", new[s], rng)
            rec["dev"][m] = {"pct": pct, "lo": lo, "hi": hi, "rmse": float(np.sqrt(((T["label"].to_numpy()[s] - new[s]) ** 2).mean())),
                             "control_rmse": float(np.sqrt(((T["label"].to_numpy()[s] - T["v71"].to_numpy()[s]) ** 2).mean())),
                             "squared_change": float(((new[s] - T["v71"].to_numpy()[s]) ** 2).sum())}
        eq = float(np.mean([rec["dev"][m]["pct"] for m in EQUAL]))
        rec.update({"eq_mean_pct": eq, "clause_1": all(rec["dev"][m]["pct"] <= 0.005 for m in MONTHS), "clause_2": eq <= -0.02,
                    "clause_3": all(rec["dev"][m]["hi"] < 0 for m in EQUAL)})
        rec["airport_month_dsse"] = (((T["label"] - new) ** 2 - (T["label"] - T["v71"]) ** 2).groupby([T["airport"], T["month"]]).sum()
                                      .unstack().round(0).to_dict(orient="index"))
        pk = PR[name].to_numpy()
        mk = ~np.isnan(pk)
        newk = K["v71"].to_numpy(dtype=float).copy()
        newk[mk] = np.maximum(newk[mk] + np.clip(c["weight"] * pk[mk], -CAP, CAP), FLOOR)
        cd, ck = new - T["v71"].to_numpy(), newk - K["v71"].to_numpy()
        guard = {}
        for ap in aps:
            sd, sk = mv & (T["airport"] == ap).to_numpy(), mk & (K["airport"] == ap).to_numpy()
            if not sd.any() or not sk.any():
                continue
            rd, rk_ = float(np.sqrt((cd[sd] ** 2).mean())), float(np.sqrt((ck[sk] ** 2).mean()))
            guard[ap] = {"rms_dev": rd, "rms_rank": rk_, "ratio": rk_ / rd, "ok": bool(0.5 <= rk_ / rd <= 2.0), "rows_rank": int(sk.sum())}
        excl = sorted(a for a, g in guard.items() if not g["ok"])
        rec.update({"transfer": guard, "excluded_airports": excl, "clause_4": len(excl) <= 2})
        rec["accepted"] = bool(rec["clause_1"] and rec["clause_2"] and rec["clause_3"] and rec["clause_4"])
        keep = K["airport"].isin(excl).to_numpy()
        rec["_newk"] = np.where(keep, K["v71"].to_numpy(dtype=float), newk)
        out["candidates"][name] = rec
        print(name, {m: (round(v["pct"], 4), round(v["lo"], 4), round(v["hi"], 4)) for m, v in rec["dev"].items()}, "eq", round(eq, 4),
              "transfer", {a: round(g["ratio"], 2) for a, g in guard.items()}, "accepted", rec["accepted"], flush=True)
    passing = [n for n, r in out["candidates"].items() if r["accepted"]]
    sel = min(passing, key=lambda n: out["candidates"][n]["eq_mean_pct"]) if passing else None
    out["selected"] = sel
    if sel:
        v71 = pd.read_parquet(V71[0])
        if sha(V71[0]) != V71[1]:
            raise ValueError("v71 changed")
        newk = pd.Series(np.rint(out["candidates"][sel]["_newk"]), index=K[ID].to_numpy()).reindex(v71[ID].to_numpy()).to_numpy()
        res = v71.copy()
        res[TARGET] = newk.astype(v71[TARGET].dtype)
        dry = OUT / "rank/dry_v72/vigorous-whistle_v72.parquet"
        dry.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(dry, index=False)
        back = pd.read_parquet(dry)
        fixed_ids = set(K.loc[K["fixed"], ID]) | set(GUARD)
        fx = back[ID].isin(fixed_ids).to_numpy()
        out["dry_run"] = {"file": dry.relative_to(ROOT).as_posix(), "sha256": sha(dry),
                          "rows_changed_vs_v71": int((back[TARGET].to_numpy() != v71[TARGET].to_numpy()).sum()),
                          "squared_change_vs_v71": float(((back[TARGET].to_numpy(float) - v71[TARGET].to_numpy(float)) ** 2).sum()),
                          "fixed_rows_unchanged": bool((back[TARGET].to_numpy()[fx] == v71[TARGET].to_numpy()[fx]).all()),
                          "lirf_rows_changed": int(((back[TARGET].to_numpy() != v71[TARGET].to_numpy()) & back[ID].isin(K.loc[K["airport"] == "LIRF", ID]).to_numpy()).sum()),
                          "min": float(back[TARGET].min()), "template_order_same": bool((back[ID].to_numpy() == v71[ID].to_numpy()).all()),
                          "dtype": str(back[TARGET].dtype)}
        print("dry run", out["dry_run"])
    for r in out["candidates"].values():
        r.pop("_newk", None)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "results_v1.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("selected", sel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fit", type=Path)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.fit:
        fit(a.fit)
    if a.score:
        score()


if __name__ == "__main__":
    main()
