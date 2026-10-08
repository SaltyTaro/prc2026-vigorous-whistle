"""Governing run of configs/nbr_dark_v1.json: nbr_state_v1 extended to dark rows (matched, pushback not traced), one pooled slope.

Control = nbr_state_v1 w75 (rebuilt here with tools/run_nbr_state.py's functions and checked against docs/nbr_state/results_v1.json).
Development (month-blocked pooled slope), fresh August-October 2025 (rows12, development slope), ranking (on the v71 dry run).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import run_nbr_state as N  # noqa: E402

ID, TARGET = N.ID, N.TARGET
DECL = ROOT / "configs/nbr_dark_v1.json"
V71 = (ROOT / "runs/nbr_state/rank/dry_v71/vigorous-whistle_v71.parquet", "7a0b1431eef1146d7afd40c728bea7af6b90a411e257ca277564866e410b84ed")
STATE_RESULTS = ROOT / "docs/nbr_state/results_v1.json"
W_STATE = 0.75
DRY = ROOT / "runs/nbr_state/rank/dry_v72/vigorous-whistle_v72.parquet"
RANK_MONTHS = ["2026-01", "2026-07"]   # months of the movement file (final-file chain: the new file's months)
MATCHES = ROOT / "data/external/adsblol/matches"   # folder of matches2_<month>.parquet
DARK_RESULTS = ROOT / "docs/nbr_state/dark_results_v1.json"


def gates(months: list[str]) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(MATCHES / f"matches2_{m}.parquet", columns=[ID, "gate"]) for m in months])
    return g.drop_duplicates(ID)


def with_gate(t: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    t = t.merge(gates(months), on=ID, how="left")
    t["gate"] = t["gate"].fillna(False).astype(bool)
    return t


def main() -> None:
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    w = float(decl["candidates"][0]["weight"])
    state_res = json.loads(STATE_RESULTS.read_text(encoding="utf-8"))
    d, f, k = N.dev_table(), N.fresh_table(), N.ranking_table()
    d, f = with_gate(d, N.DEV_MONTHS), with_gate(f, N.FRESH)
    k = with_gate(k, RANK_MONTHS)
    for t in (d, f, k):
        t["x"] = N.state(t)
        t["unm"] = ~t["matched"].to_numpy() & ~t["fixed"].to_numpy() & ~np.isnan(t["x"].to_numpy())
        t["dark"] = t["matched"].to_numpy() & ~t["gate"].to_numpy() & ~t["fixed"].to_numpy() & ~np.isnan(t["x"].to_numpy())
    airports = sorted(d["airport"].unique())
    y0 = (d["label"] - d["value"]).to_numpy()
    # control: nbr_state_v1 w75
    b_dev, final = np.zeros(len(d)), {}
    for ap in airports:
        a = d["unm"].to_numpy() & (d["airport"] == ap).to_numpy()
        if a.sum() >= N.MIN_FIT:
            final[ap] = N.slope(d["x"].to_numpy()[a], y0[a])
        for m in N.DEV_MONTHS:
            fit, tgt = a & (d["month"] != m).to_numpy(), a & (d["month"] == m).to_numpy()
            if fit.sum() >= N.MIN_FIT and tgt.any():
                b_dev[tgt] = N.slope(d["x"].to_numpy()[fit], y0[fit])
    if any(abs(final[a] - state_res["slopes_final"][a]) > 1e-12 for a in final):
        raise ValueError("nbr_state slopes not reproduced")
    d["ctl"] = N.apply(d["value"].to_numpy(float), d["x"].to_numpy(), b_dev, W_STATE, d["unm"].to_numpy())
    f["ctl"] = N.apply(f["value"].to_numpy(float), f["x"].to_numpy(), f["airport"].map(final).fillna(0.0).to_numpy(float), W_STATE, f["unm"].to_numpy())
    for t, months, part in ((d, N.DEV_MONTHS, "dev"), (f, N.FRESH, "fresh")):
        for m in months:
            s = (t["month"] == m).to_numpy()
            rm = float(np.sqrt(((t["label"].to_numpy()[s] - t["ctl"].to_numpy()[s]) ** 2).mean()))
            if abs(rm - state_res["candidates"]["w75"][part][m]["rmse"]) > 1e-9:
                raise ValueError(f"control not reproduced: {part} {m}")
    v71 = pd.read_parquet(V71[0])
    if N.sha(V71[0]) != V71[1]:
        raise ValueError("v71 dry run changed")
    k["ctl"] = v71.set_index(ID)[TARGET].reindex(k[ID]).to_numpy(dtype=float)
    # dark-row layer
    y = np.clip((d["label"] - d["ctl"]).to_numpy(), -N.CLIP_Y, N.CLIP_Y)
    x, dk = d["x"].to_numpy(), d["dark"].to_numpy()
    B = float((x[dk] * y[dk]).sum() / (x[dk] ** 2).sum())
    bd = np.zeros(len(d))
    blocked = {}
    for m in N.DEV_MONTHS:
        fit, tgt = dk & (d["month"] != m).to_numpy(), dk & (d["month"] == m).to_numpy()
        blocked[m] = float((x[fit] * y[fit]).sum() / (x[fit] ** 2).sum())
        bd[tgt] = blocked[m]
    new_d = d["ctl"].to_numpy(float).copy()
    new_d[dk] = np.maximum(new_d[dk] + np.clip(w * bd[dk] * x[dk], -N.CAP, N.CAP), N.FLOOR)
    fdk = f["dark"].to_numpy()
    new_f = f["ctl"].to_numpy(float).copy()
    new_f[fdk] = np.maximum(new_f[fdk] + np.clip(w * B * f["x"].to_numpy()[fdk], -N.CAP, N.CAP), N.FLOOR)
    kdk = k["dark"].to_numpy()
    new_k = k["ctl"].to_numpy(float).copy()
    new_k[kdk] = np.maximum(new_k[kdk] + np.clip(w * B * k["x"].to_numpy()[kdk], -N.CAP, N.CAP), N.FLOOR)
    out = {"experiment_id": decl["experiment_id"], "declaration_sha256": N.sha(DECL), "audit_used": False, "official_scores_used": "none",
           "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "control": "nbr_state_v1 w75 (reproduced exactly)",
           "slope_final": B, "slopes_blocked": blocked,
           "dark_rows": {"dev": int(dk.sum()), "fresh": int(fdk.sum()), "ranking": int(kdk.sum())}, "dev": {}, "fresh": {}}
    for t, new, months, part in ((d, new_d, N.DEV_MONTHS, "dev"), (f, new_f, N.FRESH, "fresh")):
        rng = np.random.default_rng(N.SEED)
        for m in months:
            s = (t["month"] == m).to_numpy()
            out[part][m] = N.month_row(t[s].assign(value=t["ctl"][s]), new[s], rng)
    guard = {}
    cd = new_d - d["ctl"].to_numpy()
    for ap in airports:
        sd, sk = dk & (d["airport"] == ap).to_numpy(), kdk & (k["airport"] == ap).to_numpy()
        if not sd.any() or not sk.any():
            continue
        rd, rk = float(np.sqrt((cd[sd] ** 2).mean())), float(np.sqrt(((new_k - k["ctl"].to_numpy())[sk] ** 2).mean()))
        guard[ap] = {"rms_dev": rd, "rms_rank": rk, "ratio": rk / rd, "rows_rank": int(sk.sum()), "ok": bool(0.5 <= rk / rd <= 2.0)}
    excluded = sorted(a for a, g in guard.items() if not g["ok"])
    eq = float(np.mean([out["dev"][m]["pct"] for m in N.EQUAL]))
    c1 = all(out["dev"][m]["pct"] <= 0.005 for m in N.DEV_MONTHS)
    c3 = all(out["dev"][m]["hi"] < 0 for m in N.EQUAL)
    c4 = all(out["fresh"][m]["pct"] < 0 for m in N.FRESH) and sum(out["fresh"][m]["hi"] < 0 for m in N.FRESH) >= 2
    out.update({"transfer": guard, "excluded_airports": excluded, "eq_mean_pct": eq, "clause_1": c1, "clause_2": eq <= -0.02,
                "clause_3": c3, "clause_4": c4, "accepted": bool(c1 and eq <= -0.02 and c3 and c4)})
    print({m: (round(v["pct"], 4), round(v["lo"], 4), round(v["hi"], 4)) for m, v in out["dev"].items()}, "eq", round(eq, 4))
    print("fresh", {m: (round(v["pct"], 4), round(v["lo"], 4), round(v["hi"], 4)) for m, v in out["fresh"].items()})
    print("transfer", {a: round(g["ratio"], 2) for a, g in guard.items()}, "accepted", out["accepted"], flush=True)
    if out["accepted"]:
        keep = k["airport"].isin(excluded).to_numpy()
        final_k = np.where(keep, k["ctl"].to_numpy(float), new_k)
        res = v71.copy()
        res[TARGET] = pd.Series(np.rint(final_k), index=k[ID].to_numpy()).reindex(v71[ID].to_numpy()).to_numpy().astype(v71[TARGET].dtype)
        DRY.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(DRY, index=False)
        back = pd.read_parquet(DRY)
        v70 = pd.read_parquet(N.V70[0])
        out["dry_run"] = {"file": DRY.relative_to(ROOT).as_posix(), "sha256": N.sha(DRY),
                          "rows_changed_vs_v71": int((back[TARGET].to_numpy() != v71[TARGET].to_numpy()).sum()),
                          "squared_change_vs_v71": float(((back[TARGET].to_numpy(float) - v71[TARGET].to_numpy(float)) ** 2).sum()),
                          "rows_changed_vs_v70": int((back[TARGET].to_numpy() != v70[TARGET].to_numpy()).sum()),
                          "squared_change_vs_v70": float(((back[TARGET].to_numpy(float) - v70[TARGET].to_numpy(float)) ** 2).sum()),
                          "template_order_same": bool((back[ID].to_numpy() == v71[ID].to_numpy()).all()), "dtype": str(back[TARGET].dtype)}
        print("dry run", out["dry_run"])
    DARK_RESULTS.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
