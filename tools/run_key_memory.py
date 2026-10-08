"""Governing run of configs/key_memory_v1.json: same-season (W25) flight-number memory. The residual of airport + FLIGHT_mvt in one
W25 month corrects the adjacent month; ranking: November + December 2025 pooled -> January 2026 only. Light pandas, workstation.

The base is the chain current at the run: v74, or v74 + season_pass_v2's selected candidate (docs/season_pass2/results_v1.json).
Writes docs/key_memory/results_v1.json and, if a candidate is accepted, a dry run under runs/key_memory/rank/ (never runs/submissions)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_intruder_bound as R  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DECL = ROOT / "configs/key_memory_v1.json"
SP2, OUT, DOCS = ROOT / "runs/season_pass2", ROOT / "runs/key_memory", ROOT / "docs/key_memory"
V74 = (ROOT / "runs/submissions/vigorous-whistle_v74.parquet", "85e57d05322009d886d7423ba818ec6435f023bb8fd054463dd8adca6983f761")
SP2_DECL, SP2_RES = ROOT / "configs/season_pass_v2.json", ROOT / "docs/season_pass2/results_v1.json"
FILES = {"2025-01": "training_2025-01-01_2025-02-01", "2025-11": "training_2025-11-01_2025-12-01", "2025-12": "training_2025-12-01_2026-01-01"}
BETA, LAM, CAP, FLOOR, CLIP_R, ORD = 0.25, 30.0, 120.0, -12.0, 1800.0, 20000.0
NOV, DEC, JAN25, JAN26 = "2025-11", "2025-12", "2025-01", "2026-01"
RANKING = ROOT / "data/raw/ranking.parquet"   # the movement file (final-file chain: a new file)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def apply_pass(b: np.ndarray, p: np.ndarray, w: float) -> np.ndarray:  # tools/run_season_pass2.py score()'s apply
    n = b.copy()
    m = ~np.isnan(p)
    n[m] = np.maximum(b[m] + np.clip(w * p[m], -120.0, 120.0), -12.0)
    return n


def flight_numbers(paths: list[Path], ids: pd.Series) -> pd.Series:
    raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", "FLIGHT_mvt"]) for p in paths])
    raw = raw[raw["PHASE_mvt"].eq("DEP")]
    fn = raw.set_index(ID)["FLIGHT_mvt"].astype("string").str.strip().str.upper()
    fn = fn.where(fn.str.len() > 0)
    return fn.reindex(ids.to_numpy())


def base_chain() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tabs = json.loads((SP2 / "tables.json").read_text(encoding="utf-8"))
    for nm in ["dev_sp2", "rank_sp2"]:
        if sha(SP2 / f"{nm}.parquet") != tabs["sha256"][nm]:
            raise ValueError(f"{nm} changed")
    T = pd.read_parquet(SP2 / "dev_sp2.parquet", columns=[ID, "month", "airport", "day", "label", "ship", "kind"])
    K = pd.read_parquet(SP2 / "rank_sp2.parquet", columns=[ID, "month", "airport", "kind"])
    T["base"] = T["ship"].to_numpy(dtype=float).copy()
    res = json.loads(SP2_RES.read_text(encoding="utf-8"))
    sel = res.get("selected")
    info = {"season_pass_v2_selected": sel, "season_pass_v2_results_sha256": sha(SP2_RES)}
    if sel:
        cand = next(c for c in json.loads(SP2_DECL.read_text(encoding="utf-8"))["candidates"] if c["name"] == sel)
        if sha(SP2 / "preds_dev.parquet") != res["preds_sha256"]["dev"]:
            raise ValueError("season_pass_v2 preds changed")
        PD = pd.read_parquet(SP2 / "preds_dev.parquet").set_index(ID).reindex(T[ID])
        b = T["base"].to_numpy()
        for a in cand["airports"]:
            if a in res["candidates"][sel]["excluded_airports"]:
                continue  # season_pass_v2 left this airport at v74 in its ranking file; its development values stay too
            b = apply_pass(b, PD[f"sp_{a}"].to_numpy(), cand["weight"])
        T["base"] = b
        bf = ROOT / res["dry_run"]["file"]
        if sha(bf) != res["dry_run"]["sha256"]:
            raise ValueError("season_pass_v2 dry run changed")
        info.update({"base_file": res["dry_run"]["file"], "base_sha256": res["dry_run"]["sha256"], "base_version": 75})
    else:
        bf = V74[0]
        if sha(bf) != V74[1]:
            raise ValueError("v74 changed")
        info.update({"base_file": bf.relative_to(ROOT).as_posix(), "base_sha256": V74[1], "base_version": 74})
    B = pd.read_parquet(bf)
    return T, K, B, info


def key_means(T: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    s = T[rows]
    return s.groupby(["airport", "fn"], observed=True, dropna=True)["r"].agg(["sum", "count"]).reset_index()


def corr(target: pd.DataFrame, g: pd.DataFrame) -> np.ndarray:
    x = target[["airport", "fn"]].merge(g, on=["airport", "fn"], how="left")
    n = x["count"].fillna(0).to_numpy()
    mean = np.where(n > 0, x["sum"].fillna(0).to_numpy() / np.maximum(n, 1), 0.0)
    return np.clip(BETA * n / (n + LAM) * mean, -CAP, CAP)


def run() -> dict:
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    T, K, B, info = base_chain()
    T["fn"] = flight_numbers([ROOT / f"data/raw/{f}.parquet" for f in FILES.values()], T[ID]).to_numpy()
    K["fn"] = flight_numbers([RANKING], K[ID]).to_numpy()
    y, base = T["label"].to_numpy(dtype=float), T["base"].to_numpy()
    T["r"] = np.clip(y - base, -CLIP_R, CLIP_R)
    fixed, fixk = (T["kind"] == 3).to_numpy(), (K["kind"] == 3).to_numpy()
    fit_ok = ~fixed & (y < ORD) & T["fn"].notna().to_numpy()
    mon, monk = T["month"].to_numpy(), K["month"].to_numpy()
    bk = pd.Series(B[TARGET].to_numpy(dtype=float), index=B[ID].to_numpy()).reindex(K[ID].to_numpy()).to_numpy()
    out = {"experiment_id": decl["experiment_id"], "declaration_sha256": sha(DECL), "audit_used": False, "official_scores_used": "none",
           "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "base": info, "candidates": {}}
    c_dev = np.zeros(len(T))
    for src, tgt in [(NOV, DEC), (DEC, NOV), (DEC, JAN25)]:
        tg = mon == tgt
        c = corr(T[tg], key_means(T, fit_ok & (mon == src)))
        c[fixed[tg]] = 0.0
        if tgt == JAN25:
            c_desc = np.zeros(len(T))
            c_desc[tg] = c
        else:
            c_dev[tg] = c
    tgk = (monk == JAN26) & ~fixk
    c_rank = np.zeros(len(K))
    c_rank[tgk] = corr(K[tgk], key_means(T, fit_ok & np.isin(mon, [NOV, DEC])))
    air, airk = T["airport"].to_numpy(), K["airport"].to_numpy()
    for cand in decl["candidates"]:
        name, aps = cand["name"], cand["airports"]
        on, onk = np.isin(air, aps), np.isin(airk, aps)
        new = np.where(on, np.maximum(base + c_dev, FLOOR), base)
        desc = np.where(on, np.maximum(base + c_desc, FLOOR), base)
        newk = np.where(onk & tgk, np.maximum(bk + c_rank, FLOOR), bk)
        rec = {"dev": {}, "descriptive_cross_season": {}}
        rng = np.random.default_rng(20261004)
        for m in [NOV, DEC]:
            s = mon == m
            pct, lo, hi = R.interval(T[s].assign(value=base[s]), "value", new[s], rng)
            rec["dev"][m] = {"pct": pct, "lo": lo, "hi": hi, "squared_change": float(((new[s] - base[s]) ** 2).sum()),
                             "gain": float(((y[s] - base[s]) ** 2).sum() - ((y[s] - new[s]) ** 2).sum())}
        s = mon == JAN25
        pct, lo, hi = R.interval(T[s].assign(value=base[s]), "value", desc[s], np.random.default_rng(20261004))
        rec["descriptive_cross_season"]["2025-12 -> 2025-01"] = {"pct": pct, "lo": lo, "hi": hi}
        mean2 = float(np.mean([v["pct"] for v in rec["dev"].values()]))
        rec.update({"mean_pct": mean2, "clause_1": all(v["pct"] < 0 and v["hi"] < 0 for v in rec["dev"].values()), "clause_2": mean2 <= -0.02})
        dsse = (y - new) ** 2 - (y - base) ** 2
        rec["airport_month_dsse"] = pd.Series(dsse).groupby([air, mon]).sum().unstack().round(0).to_dict(orient="index")
        cd, ck = new - base, newk - bk
        guard = {}
        for a in aps:
            sd, sk = (cd != 0) & (air == a), (ck != 0) & (airk == a)
            rd = float(np.sqrt((cd[sd] ** 2).mean())) if sd.any() else float("nan")
            rk_ = float(np.sqrt((ck[sk] ** 2).mean())) if sk.any() else float("nan")
            ratio = rk_ / rd if (sd.any() and sk.any()) else float("nan")
            guard[a] = {"rms_dev": rd, "rms_rank": rk_, "ratio": ratio, "ok": bool(0.5 <= ratio <= 2.0), "rows_dev": int(sd.sum()), "rows_rank": int(sk.sum())}
        excl = sorted(a for a, g in guard.items() if not g["ok"])
        rec.update({"transfer": guard, "excluded_airports": excl, "clause_3": len(excl) <= 1})
        rec["accepted"] = bool(rec["clause_1"] and rec["clause_2"] and rec["clause_3"])
        rec["_newk"] = np.where(np.isin(airk, excl), bk, newk)
        out["candidates"][name] = rec
        print(name, {m: (round(v["pct"], 4), round(v["lo"], 4), round(v["hi"], 4)) for m, v in rec["dev"].items()}, "mean", round(mean2, 4),
              "g/S", {m: round(v["gain"] / v["squared_change"], 2) for m, v in rec["dev"].items() if v["squared_change"] > 0},
              "cross-season Jan 2025", round(rec["descriptive_cross_season"]["2025-12 -> 2025-01"]["pct"], 4),
              "transfer", {a: round(g["ratio"], 2) for a, g in guard.items()}, "accepted", rec["accepted"], flush=True)
    passing = [n for n, r in out["candidates"].items() if r["accepted"]]
    sel = min(passing, key=lambda n: out["candidates"][n]["mean_pct"]) if passing else None
    out["selected"] = sel
    if sel:
        v = info["base_version"] + 1
        newk = pd.Series(np.rint(out["candidates"][sel]["_newk"]), index=K[ID].to_numpy()).reindex(B[ID].to_numpy()).to_numpy()
        res = B.copy()
        res[TARGET] = newk.astype(B[TARGET].dtype)
        dry = OUT / f"rank/dry_v{v}/vigorous-whistle_v{v}.parquet"
        dry.parent.mkdir(parents=True, exist_ok=True)
        if dry.exists():
            raise ValueError("dry run exists; never overwrite")
        res.to_parquet(dry, index=False)
        back = pd.read_parquet(dry)
        ch = back[TARGET].to_numpy() != B[TARGET].to_numpy()
        kk = K.set_index(ID).reindex(back[ID].to_numpy())
        out["dry_run"] = {"file": dry.relative_to(ROOT).as_posix(), "sha256": sha(dry), "version": v, f"rows_changed_vs_v{v - 1}": int(ch.sum()),
                          "rows_changed_by_airport_month": pd.Series(list(zip(kk["airport"].to_numpy()[ch], kk["month"].to_numpy()[ch]))).astype(str).value_counts().to_dict(),
                          "squared_change": float(((back[TARGET].to_numpy(float) - B[TARGET].to_numpy(float)) ** 2).sum()),
                          "fixed_rows_unchanged": bool((~ch[(kk["kind"] == 3).to_numpy()]).all()), "min": float(back[TARGET].min()),
                          "template_order_same": bool((back[ID].to_numpy() == B[ID].to_numpy()).all()), "dtype": str(back[TARGET].dtype)}
        print("dry run", out["dry_run"])
    for r in out["candidates"].values():
        r.pop("_newk", None)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "results_v1.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("selected", sel)
    return out


if __name__ == "__main__":
    run()
