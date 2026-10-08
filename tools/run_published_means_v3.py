"""Governing run of configs/published_means_v3.json: v47's published-means procedure (configs/published_means_v2.json) on v72's chain.
Development values = runs/stack_second v71 + sp_hicov7; ranking = the uploaded v72. Fixed rows (LIRF population, guard rows) never move;
candidate pm_np also leaves them out of LIRF's means. Writes docs/published_means/results_v3.json and, if a candidate is accepted by the
primary rule, a dry run under runs/published_means_v3/ (never under runs/submissions). Light pandas; runs on the workstation."""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from run_published_means import fit_mapping, shift_for  # noqa: E402
from screen_published_means import AIRPORTS, CAP, WORKBOOK, published  # noqa: E402
import run_intruder_bound as R  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DECL = ROOT / "configs/published_means_v3.json"
SS = ROOT / "runs/stack_second"
V72 = (ROOT / "runs/submissions/vigorous-whistle_v72.parquet", "4280c399aa148847c4be4588dfc3d98d4adfd1a5e64981035b4e0fa3ee802f80")
OUT = ROOT / "runs/published_means_v3"
REPORT = ROOT / "docs/published_means/results_v3.json"
MONTHS, EQUAL = ["2025-01", "2025-07", "2025-11", "2025-12"], ["2025-01", "2025-07", "2025-12"]
FLOOR, MAX_SIGMA = -12.0, 5.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def label_table(nonpop_lirf: bool) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))):
        d = pd.read_parquet(path, columns=["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", TARGET, "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
        d = d[d["PHASE_mvt"].eq("DEP")]
        if nonpop_lirf:
            pop = d["ADEP_mvt"].eq("LIRF") & d["ADEP_flt"].isna() & d["AIRCRAFT_TYPE_mvt"].isna()
            d = d[~pop]
        month = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
        for (a, m), g in d.groupby([d["ADEP_mvt"], month]):
            y = g[TARGET].to_numpy(dtype=float)
            if len(y) > 100 and m.startswith("2025"):
                rows.append({"airport": a, "month": m, "label_mean": float(y[(y > 0) & (y <= CAP)].mean())})
    return pd.DataFrame(rows).merge(published(), on=["airport", "month"], how="inner", validate="one_to_one")


def dev_values() -> pd.DataFrame:
    T = pd.read_parquet(SS / "dev_table.parquet", columns=[ID, "month", "airport", "day", "label", "v71", "kind"])
    p = pd.read_parquet(SS / "preds_dev.parquet").set_index(ID)["sp_hicov7"].reindex(T[ID]).to_numpy()
    v = T["v71"].to_numpy(dtype=float).copy()
    m = ~np.isnan(p)
    v[m] = np.maximum(v[m] + np.clip(0.5 * p[m], -120, 120), -12)
    T["v72"], T["fixed"] = v, (T["kind"] == 3).to_numpy()
    return T


def selected(airport: np.ndarray, fixed: np.ndarray, a: str, cand: str) -> np.ndarray:
    m = airport == a
    return m & ~fixed if (cand == "pm_np" and a == "LIRF") else m


def run() -> dict:
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    if sha(ROOT / WORKBOOK) != decl["external_data"]["sha256"]:
        raise ValueError("workbook differs from the declared file")
    if sha(V72[0]) != V72[1]:
        raise ValueError("v72 changed")
    T = dev_values()
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")]
    v72 = pd.read_parquet(V72[0])
    K = pd.read_parquet(SS / "rank_table.parquet", columns=[ID, "kind"]).set_index(ID)
    rk = raw.set_index(ID).reindex(v72[ID])
    k_air, k_mon = rk["ADEP_mvt"].to_numpy(), rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()
    k_fix = (K["kind"].reindex(v72[ID]).to_numpy() == 3)
    k_val = v72[TARGET].to_numpy(dtype=float)
    out = {"experiment_id": decl["experiment_id"], "declaration_sha256": sha(DECL), "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "audit_used": False, "official_scores_used": "none", "workbook_sha256": sha(ROOT / WORKBOOK), "v72_sha256": V72[1], "candidates": {}}
    for c in decl["candidates"]:
        cand = c["name"]
        table = label_table(cand == "pm_np")
        pub = published()
        y, air, fx = T["label"].to_numpy(dtype=float), T["airport"].to_numpy(), T["fixed"].to_numpy()
        ctl = T["v72"].to_numpy(dtype=float)
        levels = {}
        for m in MONTHS:
            s = (T["month"] == m).to_numpy()
            levels[m] = {}
            for a in AIRPORTS:
                sel = s & selected(air, fx, a, cand) & (ctl > 0) & (ctl <= CAP)
                q = pub[(pub["airport"] == a) & (pub["month"] == m)].iloc[0]
                mp = fit_mapping(table, a, m)
                ordy = sel & (y > 0) & (y <= CAP)
                levels[m][a] = {"estimate": float(ctl[sel].mean() - (q["P"] + mp["function"](q["u"]))), "true": float(ctl[ordy].mean() - y[ordy].mean()),
                                "sd_map": mp["sd_map"], "u": float(q["u"]), "u_max": mp["u_max"]}

        def method(a, exclude):
            d = np.array([levels[m][a]["estimate"] - levels[m][a]["true"] for m in MONTHS if m != exclude])
            return float(d.mean()), float(np.sqrt(np.mean((d - d.mean()) ** 2)))

        new = ctl.copy()
        rec = {"dev": {}}
        rng = np.random.default_rng(20261004)
        for m in MONTHS:
            s = (T["month"] == m).to_numpy()
            cells = {}
            for a in AIRPORTS:
                e = levels[m][a]
                bias, noise = method(a, m)
                shift, r = shift_for(e["estimate"], bias, max(e["sd_map"], noise), e["u"], e["u_max"], MAX_SIGMA)
                mv = s & (air == a) & ~fx & (ctl > 0) & (ctl <= CAP)
                new[mv] = np.maximum(ctl[mv] - shift, FLOOR)
                cells[a] = {**r, "true_level_error_s": e["true"]}
            sub = T[s].assign(value=ctl[s])
            pct, lo, hi = R.interval(sub, "value", new[s], rng)
            rec["dev"][m] = {"pct": pct, "lo": lo, "hi": hi, "eligible": {a: [round(cl["shift_s"], 2), round(cl["true_level_error_s"], 2)] for a, cl in cells.items() if cl["eligible"]},
                             "cells": cells}
        eq = float(np.mean([rec["dev"][m]["pct"] for m in EQUAL]))
        rec["eq_mean_pct"] = eq
        rec["accepted_primary"] = bool(all(rec["dev"][m]["pct"] <= 0 for m in EQUAL) and eq < 0)
        rec["v57_clauses"] = {"clause_1": all(rec["dev"][m]["pct"] <= 0.005 for m in MONTHS), "clause_2": eq <= -0.02,
                              "clause_3": all(rec["dev"][m]["hi"] < 0 for m in EQUAL)}
        # ranking cells
        newk, cellk = k_val.copy(), {}
        for a in AIRPORTS:
            mp = fit_mapping(table, a, None)
            bias, noise = method(a, None)
            for mo in ["2026-01", "2026-07"]:
                q = pub[(pub["airport"] == a) & (pub["month"] == mo)].iloc[0]
                selk = (k_air == a) & (k_mon == mo) & (k_val > 0) & (k_val <= CAP)
                if cand == "pm_np" and a == "LIRF":
                    selk &= ~k_fix
                lev = float(k_val[selk].mean() - (q["P"] + mp["function"](q["u"])))
                shift, r = shift_for(lev, bias, max(mp["sd_map"], noise), float(q["u"]), mp["u_max"], MAX_SIGMA)
                mvk = (k_air == a) & (k_mon == mo) & ~k_fix & (k_val > 0) & (k_val <= CAP)
                if r["eligible"]:
                    newk[mvk] = np.maximum(k_val[mvk] - shift, FLOOR)
                cellk[f"{a} {mo}"] = {**r, "rows_moved": int(mvk.sum()) if r["eligible"] else 0}
        ch = np.rint(newk) - k_val
        rec["ranking"] = {"cells": cellk, "eligible": {k: round(v["shift_s"], 2) for k, v in cellk.items() if v["eligible"]},
                          "rows_changed": int((ch != 0).sum()), "squared_change": float((ch ** 2).sum()),
                          "expected_gain_if_estimates_hold": float(sum(v["rows_moved"] * (2 * v["shift_s"] * v["effective_level_s"] - v["shift_s"] ** 2) for v in cellk.values() if v["eligible"]))}
        rec["_newk"] = newk
        out["candidates"][cand] = rec
        print(cand, {m: (round(v["pct"], 4), round(v["lo"], 3), round(v["hi"], 3), v["eligible"]) for m, v in rec["dev"].items()}, "eq", round(eq, 4),
              "primary", rec["accepted_primary"], "v57", rec["v57_clauses"], "| ranking", rec["ranking"]["eligible"], "S", round(rec["ranking"]["squared_change"] / 1e6, 2), "e6", flush=True)
    passing = [n for n, r in out["candidates"].items() if r["accepted_primary"]]
    sel = min(passing, key=lambda n: out["candidates"][n]["eq_mean_pct"]) if passing else None
    out["selected_primary"] = sel
    if sel:
        res = v72.copy()
        res[TARGET] = np.rint(out["candidates"][sel]["_newk"]).astype(v72[TARGET].dtype)
        dry = OUT / "dry_v73/vigorous-whistle_v73.parquet"
        dry.parent.mkdir(parents=True, exist_ok=True)
        if dry.exists():
            raise ValueError("dry run exists; never overwrite")
        res.to_parquet(dry, index=False)
        back = pd.read_parquet(dry)
        out["dry_run"] = {"file": dry.relative_to(ROOT).as_posix(), "sha256": sha(dry), "rows_changed_vs_v72": int((back[TARGET].to_numpy() != v72[TARGET].to_numpy()).sum()),
                          "squared_change_vs_v72": float(((back[TARGET].to_numpy(float) - k_val) ** 2).sum()),
                          "fixed_rows_unchanged": bool((back[TARGET].to_numpy()[k_fix] == v72[TARGET].to_numpy()[k_fix]).all()),
                          "template_order_same": bool((back[ID].to_numpy() == v72[ID].to_numpy()).all()), "dtype": str(back[TARGET].dtype), "min": float(back[TARGET].min())}
        print("dry run", out["dry_run"])
    for r in out["candidates"].values():
        r.pop("_newk", None)
    REPORT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("selected (primary rule)", sel)
    return out


if __name__ == "__main__":
    run()
