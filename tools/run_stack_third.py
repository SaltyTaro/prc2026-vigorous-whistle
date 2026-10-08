"""Governing run of configs/stack_third_v1.json: a third guarded residual pass on v72 with raw ADS-B match inputs at EDDF / EHAM / LSZH.

  --build   (workstation, light)  runs/stack_third/{dev,rank}_t3.parquet from runs/stack_second's tables and predictions, the raw tables of
                                   docs/audit_v72/raw_build.py (runs/audit_v72/raw_{dev,rank}.parquet) and the uploaded v72
  --fit DIR (Azure VM)             five-seed bags per candidate -> DIR/preds_{dev,rank}.parquet
  --score   (workstation)          clauses 1-4, selection, dry run -> docs/stack_third/results_v1.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DECL = ROOT / "configs/stack_third_v1.json"
SS, OUT, DOCS = ROOT / "runs/stack_second", ROOT / "runs/stack_third", ROOT / "docs/stack_third"
V72 = (ROOT / "runs/submissions/vigorous-whistle_v72.parquet", "4280c399aa148847c4be4588dfc3d98d4adfd1a5e64981035b4e0fa3ee802f80")
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
MONTHS, EQUAL = ["2025-01", "2025-07", "2025-11", "2025-12"], ["2025-01", "2025-07", "2025-12"]
CL = ["c_sched", "c_aobt", "c_lobt", "c_eobt", "c_iobt", "c_arvt", "c_hour", "c_dow", "c_layers"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=800, lambda_l2=50.0, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=16, max_bin=127, deterministic=True, force_col_wise=True)
SEEDS, ROUNDS, CAP, FLOOR, CLIP_Y = [7, 11, 12, 13, 14], 400, 120.0, -12.0, 1800.0
# Final-file chain options (docs/final_file/lane_L.md, tools/final_late_chain.py); the defaults reproduce the governing run.
RAW_DIR = ROOT / "runs/audit_v72"   # raw_{dev,rank}.parquet of docs/audit_v72/raw_build.py
DEV_PREDS_FROM = None    # an existing preds_dev.parquet: fit() copies it and fits only the ranking bags


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def second(v71: np.ndarray, p: np.ndarray) -> np.ndarray:
    v = v71.copy()
    m = ~np.isnan(p)
    v[m] = np.maximum(v[m] + np.clip(0.5 * p[m], -120, 120), -12)
    return v


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = json.loads((SS / "tables.json").read_text(encoding="utf-8"))
    for nm in ["dev_table", "rank_table"]:
        if sha(SS / f"{nm}.parquet") != tabs["sha256"][nm]:
            raise ValueError(f"{nm} changed")
    if sha(V72[0]) != V72[1]:
        raise ValueError("v72 changed")
    cols = [ID, "month", "airport", "v71", "kind", "fixed", "ap"] + CL
    T = pd.read_parquet(SS / "dev_table.parquet", columns=cols + ["day", "label"])
    K = pd.read_parquet(SS / "rank_table.parquet", columns=cols)
    T["p2"] = pd.read_parquet(SS / "preds_dev.parquet").set_index(ID)["sp_hicov7"].reindex(T[ID]).to_numpy()
    K["p2"] = pd.read_parquet(SS / "preds_rank.parquet").set_index(ID)["sp_hicov7"].reindex(K[ID]).to_numpy()
    T["ship"] = second(T["v71"].to_numpy(dtype=float), T["p2"].to_numpy())
    K["ship"] = pd.read_parquet(V72[0]).set_index(ID)[TARGET].reindex(K[ID]).to_numpy(dtype=float)
    if np.abs(np.rint(second(K["v71"].to_numpy(dtype=float), K["p2"].to_numpy())) - K["ship"].to_numpy()).max() > 0:
        raise ValueError("ranking v72 not reproduced from v71 + sp_hicov7")
    for X in (T, K):
        X["d2"] = X["ship"] - X["v71"]
    for X, nm in ((T, "dev"), (K, "rank")):
        raw = pd.read_parquet(RAW_DIR / f"raw_{nm}.parquet")
        if not (raw[ID].to_numpy() == X[ID].to_numpy()).all():
            raise ValueError(f"raw_{nm} not aligned")
        out = pd.concat([X.reset_index(drop=True), raw.drop(columns=[ID]).reset_index(drop=True)], axis=1)
        out.to_parquet(OUT / f"{nm}_t3.parquet", index=False)
    rec = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "features": features(pd.read_parquet(OUT / "dev_t3.parquet")),
           "sha256": {"dev_t3": sha(OUT / "dev_t3.parquet"), "rank_t3": sha(OUT / "rank_t3.parquet"), "declaration": sha(DECL), "tool": sha(Path(__file__)),
                      "raw_dev": sha(RAW_DIR / "raw_dev.parquet"), "raw_rank": sha(RAW_DIR / "raw_rank.parquet")}}
    (OUT / "tables.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec["sha256"], indent=1), len(rec["features"]), "features")


def features(T: pd.DataFrame) -> list[str]:
    raw = [c for c in T.columns if c.startswith(("adsb_", "sfc_", "p_", "m_"))]
    return raw + ["p2", "d2", "ship"] + CL + ["kind", "ap"]


def fit(table_dir: Path) -> None:
    import lightgbm as lgb
    decl = json.loads(DECL.read_text(encoding="utf-8")) if DECL.exists() else json.loads((table_dir / "stack_third_v1.json").read_text(encoding="utf-8"))
    T, K = pd.read_parquet(table_dir / "dev_t3.parquet"), pd.read_parquet(table_dir / "rank_t3.parquet")
    F = features(T)
    y = np.clip((T["label"] - T["ship"]).to_numpy(), -CLIP_Y, CLIP_Y)
    pd_dev, pd_rank = pd.DataFrame({ID: T[ID]}), pd.DataFrame({ID: K[ID]})
    XT, XK = T[F].astype(np.float32), K[F].astype(np.float32)
    nf = (T["kind"] != 3).to_numpy()
    stored = pd.read_parquet(DEV_PREDS_FROM) if DEV_PREDS_FROM is not None else None
    for c in decl["candidates"]:
        tr = T["airport"].isin(c["fit_airports"]).to_numpy() & nf
        ap = T["airport"].isin(c["apply_airports"]).to_numpy() & nf
        p = np.full(len(T), np.nan)
        for m in (MONTHS if stored is None else []):
            f, t = tr & (T["month"] != m).to_numpy(), ap & (T["month"] == m).to_numpy()
            acc = np.zeros(int(t.sum()))
            for s in SEEDS:
                mdl = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(XT[f], y[f], categorical_feature=["kind", "ap"]), ROUNDS)
                acc += mdl.predict(XT[t]) / len(SEEDS)
            p[t] = acc
            print(c["name"], m, "done", flush=True)
        pd_dev[c["name"]] = p if stored is None else stored.set_index(ID)[c["name"]].reindex(T[ID]).to_numpy()
        kin = K["airport"].isin(c["apply_airports"]).to_numpy() & (K["kind"] != 3).to_numpy()
        acc = np.zeros(int(kin.sum()))
        for s in (SEEDS if kin.any() else []):
            mdl = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(XT[tr], y[tr], categorical_feature=["kind", "ap"]), ROUNDS)
            acc += mdl.predict(XK[kin]) / len(SEEDS)
        pr = np.full(len(K), np.nan)
        pr[kin] = acc
        pd_rank[c["name"]] = pr
        print(c["name"], "ranking done", flush=True)
    pd_dev.to_parquet(table_dir / "preds_dev.parquet", index=False)
    pd_rank.to_parquet(table_dir / "preds_rank.parquet", index=False)


def score() -> None:
    import run_intruder_bound as R
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    tabs = json.loads((OUT / "tables.json").read_text(encoding="utf-8"))
    for nm in ["dev_t3", "rank_t3"]:
        if sha(OUT / f"{nm}.parquet") != tabs["sha256"][nm]:
            raise ValueError(f"{nm} changed")
    T = pd.read_parquet(OUT / "dev_t3.parquet", columns=[ID, "month", "airport", "day", "label", "ship", "kind"])
    K = pd.read_parquet(OUT / "rank_t3.parquet", columns=[ID, "month", "airport", "ship", "kind"])
    PD = pd.read_parquet(OUT / "preds_dev.parquet").set_index(ID).reindex(T[ID]).reset_index()
    PR = pd.read_parquet(OUT / "preds_rank.parquet").set_index(ID).reindex(K[ID]).reset_index()
    out = {"experiment_id": decl["experiment_id"], "declaration_sha256": sha(DECL), "audit_used": False, "official_scores_used": "none",
           "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tables": tabs["sha256"],
           "preds_sha256": {"dev": sha(OUT / "preds_dev.parquet"), "rank": sha(OUT / "preds_rank.parquet")}, "candidates": {}}
    base, y = T["ship"].to_numpy(dtype=float), T["label"].to_numpy(dtype=float)
    for c in decl["candidates"]:
        name, aps = c["name"], c["apply_airports"]
        p = PD[name].to_numpy()
        mv = ~np.isnan(p)
        new = base.copy()
        new[mv] = np.maximum(base[mv] + np.clip(c["weight"] * p[mv], -CAP, CAP), FLOOR)
        rec = {"dev": {}}
        rng = np.random.default_rng(20261004)
        for m in MONTHS:
            s = (T["month"] == m).to_numpy()
            sub = T[s].assign(value=base[s])
            pct, lo, hi = R.interval(sub, "value", new[s], rng)
            rec["dev"][m] = {"pct": pct, "lo": lo, "hi": hi, "rmse": float(np.sqrt(((y[s] - new[s]) ** 2).mean())),
                             "control_rmse": float(np.sqrt(((y[s] - base[s]) ** 2).mean())), "squared_change": float(((new[s] - base[s]) ** 2).sum()),
                             "gain": float(((y[s] - base[s]) ** 2).sum() - ((y[s] - new[s]) ** 2).sum())}
        eq = float(np.mean([rec["dev"][m]["pct"] for m in EQUAL]))
        rec.update({"eq_mean_pct": eq, "clause_1": all(rec["dev"][m]["pct"] <= 0.005 for m in MONTHS), "clause_2": eq <= -0.02,
                    "clause_3": all(rec["dev"][m]["hi"] < 0 for m in EQUAL)})
        rec["airport_month_dsse"] = (((T["label"] - new) ** 2 - (T["label"] - base) ** 2).groupby([T["airport"], T["month"]]).sum()
                                      .unstack().round(0).to_dict(orient="index"))
        pk = PR[name].to_numpy()
        mk = ~np.isnan(pk)
        basek = K["ship"].to_numpy(dtype=float)
        newk = basek.copy()
        newk[mk] = np.maximum(basek[mk] + np.clip(c["weight"] * pk[mk], -CAP, CAP), FLOOR)
        cd, ck = new - base, newk - basek
        guard = {}
        for ap in aps:
            sd, sk = mv & (T["airport"] == ap).to_numpy(), mk & (K["airport"] == ap).to_numpy()
            rd, rk_ = float(np.sqrt((cd[sd] ** 2).mean())), float(np.sqrt((ck[sk] ** 2).mean()))
            guard[ap] = {"rms_dev": rd, "rms_rank": rk_, "ratio": rk_ / rd, "ok": bool(0.5 <= rk_ / rd <= 2.0), "rows_rank": int(sk.sum())}
        excl = sorted(a for a, g in guard.items() if not g["ok"])
        rec.update({"transfer": guard, "excluded_airports": excl, "clause_4": len(excl) <= 1})
        rec["accepted"] = bool(rec["clause_1"] and rec["clause_2"] and rec["clause_3"] and rec["clause_4"])
        rec["_newk"] = np.where(K["airport"].isin(excl).to_numpy(), basek, newk)
        out["candidates"][name] = rec
        print(name, {m: (round(v["pct"], 4), round(v["lo"], 4), round(v["hi"], 4)) for m, v in rec["dev"].items()}, "eq", round(eq, 4),
              "g/S", {m: round(v["gain"] / v["squared_change"], 2) for m, v in rec["dev"].items() if v["squared_change"] > 0},
              "transfer", {a: round(g["ratio"], 2) for a, g in guard.items()}, "accepted", rec["accepted"], flush=True)
    passing = [n for n, r in out["candidates"].items() if r["accepted"]]
    sel = min(passing, key=lambda n: out["candidates"][n]["eq_mean_pct"]) if passing else None
    out["selected"] = sel
    if sel:
        v72 = pd.read_parquet(V72[0])
        if sha(V72[0]) != V72[1]:
            raise ValueError("v72 changed")
        newk = pd.Series(np.rint(out["candidates"][sel]["_newk"]), index=K[ID].to_numpy()).reindex(v72[ID].to_numpy()).to_numpy()
        res = v72.copy()
        res[TARGET] = newk.astype(v72[TARGET].dtype)
        dry = OUT / "rank/dry_v73/vigorous-whistle_v73.parquet"
        dry.parent.mkdir(parents=True, exist_ok=True)
        if dry.exists():
            raise ValueError("dry run exists; never overwrite")
        res.to_parquet(dry, index=False)
        back = pd.read_parquet(dry)
        fixed_ids = set(K.loc[K["kind"] == 3, ID])
        fx = back[ID].isin(fixed_ids).to_numpy()
        ch = back[TARGET].to_numpy() != v72[TARGET].to_numpy()
        apk = pd.Series(K["airport"].to_numpy(), index=K[ID].to_numpy()).reindex(back[ID].to_numpy()).to_numpy()
        out["dry_run"] = {"file": dry.relative_to(ROOT).as_posix(), "sha256": sha(dry), "rows_changed_vs_v72": int(ch.sum()),
                          "rows_changed_by_airport": pd.Series(apk[ch]).value_counts().to_dict(),
                          "squared_change_vs_v72": float(((back[TARGET].to_numpy(float) - v72[TARGET].to_numpy(float)) ** 2).sum()),
                          "fixed_rows_unchanged": bool((~ch[fx]).all()), "min": float(back[TARGET].min()),
                          "template_order_same": bool((back[ID].to_numpy() == v72[ID].to_numpy()).all()), "dtype": str(back[TARGET].dtype)}
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
