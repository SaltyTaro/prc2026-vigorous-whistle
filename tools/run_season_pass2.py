"""Governing run of configs/season_pass_v2.json: a second W25 (winter-season) late pass, on v74, with the stacker / neighbour / queue
inputs added to the raw ADS-B inputs; one LightGBM bag per airport fitted on November / December 2025 and applied to January 2026 only.
Copy of tools/run_season_pass.py (v74's tool, frozen) with v74 as the base.

  --build   (workstation, light)  runs/season_pass/{dev,rank}_sp.parquet from runs/stack_third's tables and t3_ro_3's predictions; the
                                   ranking v73 must be reproduced exactly from v72 + t3_ro_3
  --fit DIR (Azure VM)             per airport: Dec -> Nov, Nov -> Dec, Dec -> Jan 2025 (descriptive) and Nov+Dec -> Jan 2026 bags
                                   -> DIR/preds_{dev,rank}.parquet
  --score   (workstation)          clauses 1-3, selection, dry run -> docs/season_pass/results_v1.json
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
DECL = ROOT / "configs/season_pass_v2.json"
SP, SS, OUT, DOCS = ROOT / "runs/season_pass", ROOT / "runs/stack_second", ROOT / "runs/season_pass2", ROOT / "docs/season_pass2"
V74 = (ROOT / "runs/submissions/vigorous-whistle_v74.parquet", "85e57d05322009d886d7423ba818ec6435f023bb8fd054463dd8adca6983f761")
NBQ = ["nx", "nxk", "nn", "x_nmerr", "gp", "gn", "gp2", "gp4", "gn4", "bpos", "blen"]
CL = ["c_sched", "c_aobt", "c_lobt", "c_eobt", "c_iobt", "c_arvt", "c_hour", "c_dow", "c_layers"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=800, lambda_l2=50.0, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=16, max_bin=127, deterministic=True, force_col_wise=True)
SEEDS, ROUNDS, CAP, FLOOR, CLIP_Y = [7, 11, 12, 13, 14], 400, 120.0, -12.0, 1800.0
NOV, DEC, JAN25, JAN26 = "2025-11", "2025-12", "2025-01", "2026-01"
DEV_PREDS_FROM = None    # final-file chain: an existing preds_dev.parquet copied by fit(), which then fits only the ranking bags


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def third(v72: np.ndarray, p: np.ndarray) -> np.ndarray:  # one guarded 0.5-weight pass (used for v73 -> v74 here)
    v = v72.copy()
    m = ~np.isnan(p)
    v[m] = np.maximum(v[m] + np.clip(0.5 * p[m], -120, 120), -12)
    return v


def features(T: pd.DataFrame) -> list[str]:
    raw = [c for c in T.columns if c.startswith(("adsb_", "sfc_", "p_", "m_"))]
    stk = [c for c in T.columns if c.startswith("s_")]
    return stk + NBQ + raw + CL + ["p2", "d2", "p3", "d3", "p4", "d4", "ship"] + ["kind", "ap"]


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tabs, ss_tabs = json.loads((SP / "tables.json").read_text(encoding="utf-8")), json.loads((SS / "tables.json").read_text(encoding="utf-8"))
    for nm in ["dev_sp", "rank_sp"]:
        if sha(SP / f"{nm}.parquet") != tabs["sha256"][nm]:
            raise ValueError(f"{nm} changed")
    for nm in ["dev_table", "rank_table"]:
        if sha(SS / f"{nm}.parquet") != ss_tabs["sha256"][nm]:
            raise ValueError(f"{nm} changed")
    if sha(V74[0]) != V74[1]:
        raise ValueError("v74 changed")
    for nm, ssn in [("dev", "dev_table"), ("rank", "rank_table")]:
        X = pd.read_parquet(SP / f"{nm}_sp.parquet")
        S = pd.read_parquet(SS / f"{ssn}.parquet")
        if not (S[ID].to_numpy() == X[ID].to_numpy()).all():
            raise ValueError(f"{ssn} not aligned")
        keep = [c for c in S.columns if c.startswith("s_")] + NBQ
        X = pd.concat([X.reset_index(drop=True), S[keep].reset_index(drop=True)], axis=1)
        P = pd.read_parquet(SP / f"preds_{nm}.parquet").set_index(ID).reindex(X[ID])
        p4 = np.full(len(X), np.nan)
        for a_ in ["EGLL", "LIRF", "LEBL"]:
            v = P[f"sp_{a_}"].to_numpy()
            p4[~np.isnan(v)] = v[~np.isnan(v)]
        v73 = X["ship"].to_numpy(dtype=float)
        X["p4"] = p4
        if nm == "dev":
            X["ship"] = third(v73, p4)
        else:
            X["ship"] = pd.read_parquet(V74[0]).set_index(ID)[TARGET].reindex(X[ID]).to_numpy(dtype=float)
            if np.abs(np.rint(third(v73, p4)) - X["ship"].to_numpy()).max() > 0:
                raise ValueError("ranking v74 not reproduced from v73 + w25_e3")
        X["d4"] = X["ship"] - v73
        X.to_parquet(OUT / f"{nm}_sp2.parquet", index=False)
    T = pd.read_parquet(OUT / "dev_sp2.parquet")
    rec = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "features": features(T),
           "sha256": {"dev_sp2": sha(OUT / "dev_sp2.parquet"), "rank_sp2": sha(OUT / "rank_sp2.parquet"), "declaration": sha(DECL), "tool": sha(Path(__file__))}}
    (OUT / "tables.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec["sha256"], indent=1), len(rec["features"]), "features")


def fit(table_dir: Path) -> None:
    import lightgbm as lgb
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    T, K = pd.read_parquet(table_dir / "dev_sp2.parquet"), pd.read_parquet(table_dir / "rank_sp2.parquet")
    F = features(T)
    y = np.clip((T["label"] - T["ship"]).to_numpy(), -CLIP_Y, CLIP_Y)
    XT, XK = T[F].astype(np.float32), K[F].astype(np.float32)
    nf, nfk = (T["kind"] != 3).to_numpy(), (K["kind"] != 3).to_numpy()
    mon, monk, apt, aptk = T["month"].to_numpy(), K["month"].to_numpy(), T["airport"].to_numpy(), K["airport"].to_numpy()
    airports = sorted({a for c in decl["candidates"] for a in c["airports"]})
    pdv, prk = pd.DataFrame({ID: T[ID]}), pd.DataFrame({ID: K[ID]})

    def bag(f: np.ndarray, X: pd.DataFrame, t: np.ndarray) -> np.ndarray:
        acc = np.zeros(int(t.sum()))
        for s in SEEDS:
            mdl = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(XT[f], y[f], categorical_feature=["kind", "ap"]), ROUNDS)
            acc += mdl.predict(X[t]) / len(SEEDS)
        return acc

    stored = pd.read_parquet(DEV_PREDS_FROM).set_index(ID).reindex(T[ID]) if DEV_PREDS_FROM is not None else None
    for a in airports:
        base = nf & (apt == a)
        col, desc = np.full(len(T), np.nan), np.full(len(T), np.nan)
        if stored is None:
            for tr, te in [(DEC, NOV), (NOV, DEC)]:
                t = base & (mon == te)
                col[t] = bag(base & (mon == tr), XT, t)
            t = base & (mon == JAN25)
            desc[t] = bag(base & (mon == DEC), XT, t)
        else:
            col, desc = stored[f"sp_{a}"].to_numpy(), stored[f"desc_{a}"].to_numpy()
        pdv[f"sp_{a}"], pdv[f"desc_{a}"] = col, desc
        tk = nfk & (aptk == a) & (monk == JAN26)
        pr = np.full(len(K), np.nan)
        if tk.any():   # the declared rule acts on January 2026 rows only; a file without them gets no ranking bag
            pr[tk] = bag(base & np.isin(mon, [NOV, DEC]), XK, tk)
        prk[f"sp_{a}"] = pr
        print(a, "done", flush=True)
    pdv.to_parquet(table_dir / "preds_dev.parquet", index=False)
    prk.to_parquet(table_dir / "preds_rank.parquet", index=False)


def score() -> None:
    import run_intruder_bound as R
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    tabs = json.loads((OUT / "tables.json").read_text(encoding="utf-8"))
    for nm in ["dev_sp2", "rank_sp2"]:
        if sha(OUT / f"{nm}.parquet") != tabs["sha256"][nm]:
            raise ValueError(f"{nm} changed")
    T = pd.read_parquet(OUT / "dev_sp2.parquet", columns=[ID, "month", "airport", "day", "label", "ship", "kind"])
    K = pd.read_parquet(OUT / "rank_sp2.parquet", columns=[ID, "month", "airport", "ship", "kind"])
    PD = pd.read_parquet(OUT / "preds_dev.parquet").set_index(ID).reindex(T[ID]).reset_index()
    PR = pd.read_parquet(OUT / "preds_rank.parquet").set_index(ID).reindex(K[ID]).reset_index()
    out = {"experiment_id": decl["experiment_id"], "declaration_sha256": sha(DECL), "audit_used": False, "official_scores_used": "none",
           "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tables": tabs["sha256"],
           "preds_sha256": {"dev": sha(OUT / "preds_dev.parquet"), "rank": sha(OUT / "preds_rank.parquet")}, "candidates": {}}
    base, y = T["ship"].to_numpy(dtype=float), T["label"].to_numpy(dtype=float)
    basek = K["ship"].to_numpy(dtype=float)

    def apply(b: np.ndarray, p: np.ndarray, w: float) -> np.ndarray:
        n = b.copy()
        m = ~np.isnan(p)
        n[m] = np.maximum(b[m] + np.clip(w * p[m], -CAP, CAP), FLOOR)
        return n

    for c in decl["candidates"]:
        name, aps, w = c["name"], c["airports"], c["weight"]
        new, desc, newk = base.copy(), base.copy(), basek.copy()
        for a in aps:
            new = apply(new, PD[f"sp_{a}"].to_numpy(), w)
            desc = apply(desc, PD[f"desc_{a}"].to_numpy(), w)
            newk = apply(newk, PR[f"sp_{a}"].to_numpy(), w)
        rec = {"dev": {}, "descriptive_cross_season": {}}
        rng = np.random.default_rng(20261004)
        for m in ["2025-11", "2025-12"]:
            s = (T["month"] == m).to_numpy()
            pct, lo, hi = R.interval(T[s].assign(value=base[s]), "value", new[s], rng)
            rec["dev"][m] = {"pct": pct, "lo": lo, "hi": hi, "squared_change": float(((new[s] - base[s]) ** 2).sum()),
                             "gain": float(((y[s] - base[s]) ** 2).sum() - ((y[s] - new[s]) ** 2).sum())}
        s = (T["month"] == JAN25).to_numpy()
        pct, lo, hi = R.interval(T[s].assign(value=base[s]), "value", desc[s], np.random.default_rng(20261004))
        rec["descriptive_cross_season"]["2025-12 -> 2025-01"] = {"pct": pct, "lo": lo, "hi": hi}
        mean2 = float(np.mean([rec["dev"][m]["pct"] for m in rec["dev"]]))
        rec.update({"mean_pct": mean2, "clause_1": all(v["pct"] < 0 and v["hi"] < 0 for v in rec["dev"].values()), "clause_2": mean2 <= -0.02})
        rec["airport_month_dsse"] = (((T["label"] - new) ** 2 - (T["label"] - base) ** 2).groupby([T["airport"], T["month"]]).sum()
                                      .unstack().round(0).to_dict(orient="index"))
        cd, ck = new - base, newk - basek
        guard = {}
        for a in aps:
            sd = (cd != 0) & (T["airport"] == a).to_numpy()
            sk = (ck != 0) & (K["airport"] == a).to_numpy()
            rd, rk_ = float(np.sqrt((cd[sd] ** 2).mean())), float(np.sqrt((ck[sk] ** 2).mean()))
            guard[a] = {"rms_dev": rd, "rms_rank": rk_, "ratio": rk_ / rd, "ok": bool(0.5 <= rk_ / rd <= 2.0), "rows_rank": int(sk.sum())}
        excl = sorted(a for a, g in guard.items() if not g["ok"])
        rec.update({"transfer": guard, "excluded_airports": excl, "clause_3": len(excl) <= 1})
        rec["accepted"] = bool(rec["clause_1"] and rec["clause_2"] and rec["clause_3"])
        rec["_newk"] = np.where(K["airport"].isin(excl).to_numpy(), basek, newk)
        out["candidates"][name] = rec
        print(name, {m: (round(v["pct"], 4), round(v["lo"], 4), round(v["hi"], 4)) for m, v in rec["dev"].items()}, "mean", round(mean2, 4),
              "g/S", {m: round(v["gain"] / v["squared_change"], 2) for m, v in rec["dev"].items() if v["squared_change"] > 0},
              "cross-season Jan 2025", round(rec["descriptive_cross_season"]["2025-12 -> 2025-01"]["pct"], 4),
              "transfer", {a: round(g["ratio"], 2) for a, g in guard.items()}, "accepted", rec["accepted"], flush=True)
    passing = [n for n, r in out["candidates"].items() if r["accepted"]]
    sel = min(passing, key=lambda n: out["candidates"][n]["mean_pct"]) if passing else None
    out["selected"] = sel
    if sel:
        v73 = pd.read_parquet(V74[0])
        if sha(V74[0]) != V74[1]:
            raise ValueError("v74 changed")
        newk = pd.Series(np.rint(out["candidates"][sel]["_newk"]), index=K[ID].to_numpy()).reindex(v73[ID].to_numpy()).to_numpy()
        res = v73.copy()
        res[TARGET] = newk.astype(v73[TARGET].dtype)
        dry = OUT / "rank/dry_v75/vigorous-whistle_v75.parquet"
        dry.parent.mkdir(parents=True, exist_ok=True)
        if dry.exists():
            raise ValueError("dry run exists; never overwrite")
        res.to_parquet(dry, index=False)
        back = pd.read_parquet(dry)
        ch = back[TARGET].to_numpy() != v73[TARGET].to_numpy()
        info = K.set_index(ID).reindex(back[ID].to_numpy())
        out["dry_run"] = {"file": dry.relative_to(ROOT).as_posix(), "sha256": sha(dry), "rows_changed_vs_v74": int(ch.sum()),
                          "rows_changed_by_airport_month": pd.Series(list(zip(info["airport"].to_numpy()[ch], info["month"].to_numpy()[ch]))).astype(str).value_counts().to_dict(),
                          "squared_change_vs_v74": float(((back[TARGET].to_numpy(float) - v73[TARGET].to_numpy(float)) ** 2).sum()),
                          "fixed_rows_unchanged": bool((~ch[(info["kind"] == 3).to_numpy()]).all()), "min": float(back[TARGET].min()),
                          "template_order_same": bool((back[ID].to_numpy() == v73[ID].to_numpy()).all()), "dtype": str(back[TARGET].dtype)}
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
