"""Tier A readings of configs/febjun_check_v1.json: nbr_state_v1, nbr_dark_v1 and oooi_pin_v1 on February / June 2025, and the
key_memory W24 analogue screen (screen only).

Each layer is read with its own declared fresh-check code (tools/run_nbr_state.py, tools/run_nbr_dark.py, tools/run_oooi_pin.py),
the month list being the only parameter. Before the new months, the same path must reproduce the stored August-October readings
exactly. Light (pandas); runs on the workstation. The June 2025 data-link messages are the hash-checked mirror of the arcw1
extraction (runs/febjun_check/oooi/vm/); the archive has no February 2025 assets (it starts 14 June 2025).

Usage: .venv\\Scripts\\python.exe -X utf8 -B tools\\run_febjun_check.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import oooi_pin_join as J  # noqa: E402
import run_intruder_bound as R  # noqa: E402
import run_nbr_dark as K  # noqa: E402
import run_nbr_state as N  # noqa: E402
import run_oooi_pin as O  # noqa: E402
import run_stack_resid as S  # noqa: E402

ID = "MVT_ID_mvt"
DECL = ROOT / "configs/febjun_check_v1.json"
OUT = ROOT / "docs/final_file/febjun_results_v1.json"
RUNS = ROOT / "runs/febjun_check"
MONTHS = ["2025-02", "2025-06"]
FRESH_STORED = ["2025-08", "2025-09", "2025-10"]
RAW_NEW = {"2025-02": "training_2025-02-01_2025-03-01", "2025-06": "training_2025-06-01_2025-07-01"}
OOOI_VM = RUNS / "oooi/vm"
OOOI_MESSAGES_SHA = "c8cf467d0f76379eda35bab3a76cc3d8e03abd30068bcb24218c526f388df4c2"
OOOI_JOIN = RUNS / "oooi/join"
SEED = 20261004


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same(a: dict, b: dict, keys=("pct", "lo", "hi", "rmse")) -> bool:
    return all(a[k] == b[k] for k in keys)


_TAKEOFFS: pd.DataFrame | None = None


def all_takeoffs() -> pd.DataFrame:
    """DEP take-off times from all twelve 2025 raw files. The declared tools look a month's rows up in that month's raw file only;
    rows12 holds one February 'extra' LTFM row (MVT 185722540, take-off 2025-02-28 23:59:57) that sits in the March file. Reading
    every file adds lookups only (IDs are unique), so the stored August-October readings still reproduce exactly."""
    global _TAKEOFFS
    if _TAKEOFFS is None:
        parts = [pd.read_parquet(p, columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"]) for p in sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))]
        d = pd.concat(parts, ignore_index=True)
        d = d[d["PHASE_mvt"] == "DEP"][[ID, "MVT_TIME_UTC_mvt"]].drop_duplicates(ID)
        _TAKEOFFS = d.reset_index(drop=True)
    return _TAKEOFFS


def takeoffs_any(months: list[str]) -> pd.DataFrame:   # replaces run_nbr_state.takeoffs inside nbr()
    return all_takeoffs()


def takeoff_days_any(months: list[str]) -> pd.DataFrame:   # replaces run_oooi_pin.takeoff_days for rows12 rows
    d = all_takeoffs().copy()
    d["tday"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m-%d")
    return d[[ID, "tday"]]


# ---------------------------------------------------------------- neighbour layers

def nbr(months: list[str]) -> dict:
    """nbr_state_v1 w75 and nbr_dark_v1 on rows12 'stack' for the given months, as their declared fresh checks."""
    state_res = json.loads((ROOT / "docs/nbr_state/results_v1.json").read_text(encoding="utf-8"))
    dark_res = json.loads((ROOT / "docs/nbr_state/dark_results_v1.json").read_text(encoding="utf-8"))
    w_state = {c["name"]: float(c["weight"]) for c in json.loads(N.DECL.read_text(encoding="utf-8"))["candidates"]}["w75"]
    w_dark = float(json.loads(K.DECL.read_text(encoding="utf-8"))["candidates"][0]["weight"])
    final, B = state_res["slopes_final"], float(dark_res["slope_final"])
    saved = (N.FRESH, dict(N.RAW), N.takeoffs)
    N.FRESH = list(months)
    N.RAW.update(RAW_NEW)
    N.takeoffs = takeoffs_any
    try:
        f = N.fresh_table()
    finally:
        N.FRESH, N.RAW, N.takeoffs = saved
    f = K.with_gate(f, months)
    f["x"] = N.state(f)
    nan_x = np.isnan(f["x"].to_numpy())
    unm = ~f["matched"].to_numpy() & ~f["fixed"].to_numpy() & ~nan_x
    dark = f["matched"].to_numpy() & ~f["gate"].to_numpy() & ~f["fixed"].to_numpy() & ~nan_x
    b_f = f["airport"].map(final).fillna(0.0).to_numpy(dtype=float)
    value = f["value"].to_numpy(dtype=float)
    new_state = N.apply(value, f["x"].to_numpy(), b_f, w_state, unm)
    new_dark = new_state.copy()
    new_dark[dark] = np.maximum(new_dark[dark] + np.clip(w_dark * B * f["x"].to_numpy()[dark], -N.CAP, N.CAP), N.FLOOR)
    out = {"state": {}, "dark": {}, "rows": {"unmatched_scope": int(unm.sum()), "dark_scope": int(dark.sum())}}
    rng = np.random.default_rng(N.SEED)
    for m in months:
        s = (f["month"] == m).to_numpy()
        out["state"][m] = N.month_row(f[s], new_state[s], rng)
    rng = np.random.default_rng(N.SEED)
    for m in months:
        s = (f["month"] == m).to_numpy()
        out["dark"][m] = N.month_row(f[s].assign(value=new_state[s]), new_dark[s], rng)
    out["rows"]["by_month"] = {m: {"rows": int((f["month"] == m).sum()), "unmatched_scope": int(unm[(f["month"] == m).to_numpy()].sum()),
                                   "dark_scope": int(dark[(f["month"] == m).to_numpy()].sum())} for m in months}
    return out


def nbr_control() -> dict:
    got = nbr(FRESH_STORED)
    state_res = json.loads((ROOT / "docs/nbr_state/results_v1.json").read_text(encoding="utf-8"))
    dark_res = json.loads((ROOT / "docs/nbr_state/dark_results_v1.json").read_text(encoding="utf-8"))
    ok_state = all(same(got["state"][m], state_res["candidates"]["w75"]["fresh"][m]) for m in FRESH_STORED)
    ok_dark = all(same(got["dark"][m], dark_res["fresh"][m]) for m in FRESH_STORED)
    if not (ok_state and ok_dark):
        raise ValueError(f"stored August-October neighbour readings not reproduced (state {ok_state}, dark {ok_dark})")
    return {"state_reproduced": ok_state, "dark_reproduced": ok_dark}


# ---------------------------------------------------------------- OUT pin

def pin_offsets() -> tuple[pd.Series, dict]:
    stored = json.loads(O.RESULTS.read_text(encoding="utf-8"))
    if sha(O.PROXY) != O.PROXY_SHA:
        raise ValueError("v69 development proxy changed")
    if sha(O.RUN / "joined_dev.parquet") != stored["joined_sha256"]["dev"] or sha(O.FRESH / "joined_dev.parquet") != stored["joined_sha256"]["fresh"]:
        raise ValueError("stored joined tables changed")
    pin_dev = O.scope(pd.read_parquet(O.RUN / "joined_dev.parquet"))
    dev = pd.read_parquet(O.PROXY, columns=[ID, "month", "airport", "label", "cls", "day", "value"])
    dev = dev.merge(O.takeoff_days(O.DEV_MONTHS), on=ID, how="left", validate="one_to_one")
    dev["day"] = dev["tday"]
    dev = dev.merge(pin_dev[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    dev["in_scope"] = dev["raw"].notna().to_numpy() & ~S.fixed_rows(dev) & dev["airport"].eq("LFPG").to_numpy()
    off = O.offsets(dev[dev["in_scope"] & dev["month"].isin(O.PIN_DEV_MONTHS)])
    if {k: float(v) for k, v in off.items()} != stored["offsets_jul_nov_dec"]:
        raise ValueError("July / November / December offsets not reproduced")
    return off, stored


def pin_reading(months: list[str], joined: Path, off: pd.Series, w: float) -> dict:
    """oooi_pin_v1's clause-4 reading (rows12 'stack', LFPG non-fixed rows with a guarded pin) for the given months."""
    pin = O.scope(pd.read_parquet(joined))
    fr = pd.read_parquet(O.ROWS12, columns=[ID, "month", "airport", "label", "stack", "cls", "population"])
    fr = fr[fr["month"].isin(months)].copy()
    fr = fr.merge(takeoff_days_any(months), on=ID, how="left", validate="one_to_one")
    if fr["tday"].isna().any():
        raise ValueError("take-off days do not cover the rows")
    fr["day"] = fr["tday"]
    fr = fr.merge(pin[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    fixed = S.fixed_rows(fr) | fr["population"].fillna(False).to_numpy(dtype=bool)
    fr["in_scope"] = fr["raw"].notna().to_numpy() & ~fixed & fr["airport"].eq("LFPG").to_numpy()
    fr["pin"] = np.where(fr["in_scope"], fr["raw"] + fr["source"].map(off), np.nan)
    f = fr[[ID, "month", "label", "day"]].copy()
    f["ctl"] = fr["stack"].to_numpy(dtype=float)
    f["cand"] = f["ctl"]
    fs = fr["in_scope"].to_numpy()
    f.loc[fs, "cand"] = O.cand_value(w, fr.loc[fs, "pin"].to_numpy(), f.loc[fs, "ctl"].to_numpy())
    rng = np.random.default_rng(O.SEED)
    out = {}
    for m in months:
        out[m] = O.month_stats(f[f["month"] == m], rng)
        out[m]["pinned_rows"] = int(fs[(f["month"] == m).to_numpy()].sum())
    out["pins_missing_from_rows12"] = int((~pin[ID].isin(fr[ID])).sum())
    out["scope_by_source"] = {k: int(v) for k, v in pin["source"].value_counts().items()}
    return out


# ---------------------------------------------------------------- key_memory analogue (screen only)

def key_memory_screen() -> dict:
    months = ["2025-01", "2025-02", "2025-03"]
    t = pd.read_parquet(O.ROWS12, columns=[ID, "month", "airport", "label", "stack"])
    t = t[t["month"].isin(months)].reset_index(drop=True)
    paths = [next((ROOT / "data/raw").glob(f"training_{m}-01_*.parquet")) for m in months]
    raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
                     for p in paths])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(t[ID].to_numpy())
    if raw["MVT_TIME_UTC_mvt"].isna().any():
        raise ValueError("raw rows do not cover rows12")
    fn = raw["FLIGHT_mvt"].astype("string").str.strip().str.upper()
    t["fn"] = fn.where(fn.str.len() > 0).to_numpy()
    t["fixed"] = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    t["day"] = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    t["r"] = np.clip(t["label"] - t["stack"], -1800.0, 1800.0)
    airports = sorted(t["airport"].unique())
    sets = {"km_10": airports, "km_9": [a for a in airports if a != "EGLL"]}
    out = {}
    for name_pair, (mem, tgt) in {"analogue_jan_to_mar": ("2025-01", "2025-03"), "adjacent_feb_to_mar": ("2025-02", "2025-03")}.items():
        fit = (t["month"] == mem).to_numpy() & ~t["fixed"].to_numpy() & (t["label"] < 20000).to_numpy() & t["fn"].notna().to_numpy()
        g = t[fit].groupby(["airport", "fn"], observed=True, dropna=True)["r"].agg(["sum", "count"]).reset_index()
        s = t[(t["month"] == tgt).to_numpy()].reset_index(drop=True)
        x = s[["airport", "fn"]].merge(g, on=["airport", "fn"], how="left")
        n = x["count"].fillna(0).to_numpy()
        mean = np.where(n > 0, x["sum"].fillna(0).to_numpy() / np.maximum(n, 1), 0.0)
        c = np.clip(0.25 * n / (n + 30.0) * mean, -120.0, 120.0)
        for set_name, aps in sets.items():
            m = ~s["fixed"].to_numpy() & s["fn"].notna().to_numpy() & s["airport"].isin(aps).to_numpy() & (n > 0)
            base = s["stack"].to_numpy(dtype=float)
            new = base.copy()
            new[m] = np.maximum(base[m] + c[m], -12.0)
            rng = np.random.default_rng(SEED)
            pct, lo, hi = R.interval(s, "stack", new, rng)
            ch = new - base
            lab = s["label"].to_numpy(dtype=float)
            gain = float(((lab - base) ** 2).sum() - ((lab - new) ** 2).sum())
            sq = float((ch ** 2).sum())
            out[f"{name_pair}/{set_name}"] = {"memory_month": mem, "target_month": tgt, "pct": pct, "lo": lo, "hi": hi,
                                              "rows_changed": int((np.abs(ch) > 0).sum()), "squared_change": sq,
                                              "gain_over_squared_change": gain / sq if sq else None}
    return out


# ---------------------------------------------------------------- run

def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    res = {"experiment_id": decl["experiment_id"], "declaration_sha256": sha(DECL), "audit_used": False, "official_scores_used": "none",
           "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "code_sha256": {p: sha(ROOT / p) for p in ["tools/run_febjun_check.py", "tools/run_nbr_state.py", "tools/run_nbr_dark.py",
                                                       "tools/run_oooi_pin.py", "tools/oooi_pin_join.py", "tools/run_intruder_bound.py"]},
           "inputs_sha256": {"rows12": sha(O.ROWS12), "oooi_june_2025_messages_unique": OOOI_MESSAGES_SHA}}
    # neighbour layers
    res["nbr_control"] = nbr_control()
    print("nbr control", res["nbr_control"], flush=True)
    res["nbr"] = nbr(MONTHS)
    for part in ("state", "dark"):
        print(part, {m: (round(v["pct"], 5), round(v["lo"], 4), round(v["hi"], 4)) for m, v in res["nbr"][part].items()}, flush=True)
    # OUT pin
    off, stored = pin_offsets()
    w = {c["name"]: float(c["weight"]) for c in json.loads(O.CONFIG.read_text(encoding="utf-8"))["candidates"]}["w50"]
    ctl = pin_reading(FRESH_STORED, O.FRESH / "joined_dev.parquet", off, w)
    ok = all(same(ctl[m], stored["candidates"]["w50"]["fresh"][m]) for m in FRESH_STORED)
    if not ok:
        raise ValueError("stored August-October OUT-pin readings not reproduced")
    res["pin_control"] = {"offsets_reproduced": True, "fresh_reproduced": True, "offsets": {k: float(v) for k, v in off.items()}}
    print("pin control ok", res["pin_control"]["offsets"], flush=True)
    if sha(OOOI_VM / "messages_unique.parquet") != OOOI_MESSAGES_SHA:
        raise ValueError("June 2025 messages changed")
    OOOI_JOIN.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OOOI_VM / "messages_unique.parquet", OOOI_JOIN / "messages_unique.parquet")
    J.main(OOOI_JOIN, ["2025-06"])
    res["pin"] = {"2025-02": "unreadable: the adsb.lol Aircraft Data Links archive has no asset before 14 June 2025 (listing 10 October, runs/febjun_check/oooi/assets_all.json)",
                  **pin_reading(["2025-06"], OOOI_JOIN / "joined_dev.parquet", off, w)}
    res["pin"]["joined_sha256"] = sha(OOOI_JOIN / "joined_dev.parquet")
    print("pin June", {k: v for k, v in res["pin"]["2025-06"].items() if k in ("pct", "lo", "hi", "pinned_rows")}, flush=True)
    # screen
    res["key_memory_screen"] = key_memory_screen()
    for k, v in res["key_memory_screen"].items():
        print("screen", k, round(v["pct"], 5), round(v["lo"], 4), round(v["hi"], 4), v["rows_changed"], flush=True)
    # rule
    readings = {"nbr_state_v1 w75": res["nbr"]["state"], "nbr_dark_v1": res["nbr"]["dark"], "oooi_pin_v1 w50": {"2025-06": res["pin"]["2025-06"]}}
    known = decl["governed_layers_and_readings"]["already_known_at_declaration"]["readings_pct_and_interval_s"]
    for layer, v in known.items():
        readings[layer] = {m: {"pct": v[m][0], "lo": v[m][1], "hi": v[m][2]} for m in MONTHS}
    decision = {}
    for layer, months in readings.items():
        decision[layer] = {}
        for m in MONTHS:
            r = months.get(m)
            if r is None:
                decision[layer][m] = "stays on (unreadable)"
            else:
                off_ = r["pct"] > 0 and r["lo"] > 0
                decision[layer][m] = "switch off" if off_ else "stays on"
    res["rule_decision"] = decision
    res["switched_off"] = sorted(f"{l} {m}" for l, d in decision.items() for m, v in d.items() if v == "switch off")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("switched off:", res["switched_off"] or "none", flush=True)


if __name__ == "__main__":
    main()
