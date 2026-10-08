"""Lever 7 of the final-week campaign (docs/opdi_stack/README.md): OPDI flight-event surface times as extra inputs of the guarded
stacking learner (stack_resid_v3's place in v66's chain), read in v66's pinned development chain.

  --features KIND   KIND = dev | rank: OPDI columns per departure -> runs/opdi_stack/features_<KIND>.parquet (+ receipt)
  --coverage        coverage of the OPDI columns by airport and month (dev vs rank) -> docs/opdi_stack/coverage.json
  --score NAME      NAME = identity (no OPDI column; must reproduce the pinned v66 control runs/pubrefit/chain/ship/proxy.parquet exactly)
                    or opdi (stack_resid_v3's learner refitted with the OPDI columns added, guarded like v3's own-point columns; LIRF keeps
                    v2 as in v3) -> runs/opdi_stack/chain/<NAME>/proxy.parquet; the chain after the stack is v66's (intruder bound, identity
                    extension, own-motion v1, own_motion_v2), exactly tools/run_ownpoints_weight.score_chain
  --read            the screen against the pinned control -> docs/opdi_stack/screen.json (GO bar of the dig campaign)

Matching (label-free): an OPDI flight is a departure from airport A if it has a take-off event inside A's box or entry-runway events
tagged A; it is matched to a departure of A by callsign (CALLSIGN_flt, else the rematch key map) with its take-off reference (take-off
event, else the last runway entry) within [take-off - 900 s, take-off + 300 s] of the supplied take-off (nearest). Only OPDI events
before the OPDI take-off reference enter. Departure labels and block times are never read.
No official score enters. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402

EVENTS = ROOT / "data/external/opdi/stack_v1"
OUT = ROOT / "runs/opdi_stack"
DOCS = ROOT / "docs/opdi_stack"
SHIP = ROOT / "runs/pubrefit/chain/ship"
KEYMAP = ROOT / "runs/rematch_nonm/key_map.parquet"
RANKING = "data/raw/ranking.parquet"   # tools/final_early_chain.py: the movement file of a final file
AIRPORTS = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
            "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
            "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
BOX = (0.10, 0.15)
LO, HI = -900.0, 300.0
CORE = ["o_ref_rel", "o_stand_exit", "o_apron_exit", "o_twy_first", "o_first_seen", "o_rwy_entry", "o_n_twy", "o_deice", "o_deice_dur", "o_has"]
REL = [("q_stand_vs_pred", "o_stand_exit"), ("q_apron_vs_pred", "o_apron_exit"), ("q_twy_vs_pred", "o_twy_first"), ("q_seen_vs_pred", "o_first_seen")]
NEW_COLS = CORE + [q for q, _ in REL]
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
# screened variants (fixed before any OPDI score): the OPDI learner's value on OPDI-matched rows at every airport except LIRF, or only at
# the three airports where the 18 September probe found OPDI surface times that adsb.lol lacks (LSZH, EDDF, LEMD)
VARIANTS = {"opdi": ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM"], "opdi3": ["LSZH", "EDDF", "LEMD"]}
BAR = {"eq_mean_pct": -0.05, "max_month_pct": 0.01, "min_intervals_below_zero": 2}


# ---------------------------------------------------------------- features

def departures(kind: str) -> pd.DataFrame:
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
    if kind == "dev":
        files = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
        mv = pd.concat([pd.read_parquet(files[i], columns=cols) for i in (0, 6, 10, 11)], ignore_index=True)
    else:
        mv = pd.read_parquet(ROOT / RANKING, columns=cols)
    dep = mv[mv["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).rename(columns={"ADEP_mvt": "airport"}).copy()
    km = pd.read_parquet(KEYMAP)[["ADEP_mvt", "FLIGHT_mvt", "key"]].rename(columns={"ADEP_mvt": "airport"})
    dep = dep.merge(km, on=["airport", "FLIGHT_mvt"], how="left")
    dep["cs"] = dep["CALLSIGN_flt"].fillna(dep["key"]).astype("string").str.upper().str.strip()
    dep["t"] = (dep["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return dep[[ID, "airport", "cs", "t"]]


def box_airport(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    out = np.full(len(lat), None, dtype=object)
    for a, (la, lo) in AIRPORTS.items():
        out[(np.abs(lat - la) <= BOX[0]) & (np.abs(lon - lo) <= BOX[1])] = a
    return out


def opdi_flights(files: list[Path]) -> pd.DataFrame:
    """One row per (OPDI flight, airport) departure with its take-off reference and surface times (epoch seconds)."""
    ev = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ev = ev.drop_duplicates(["flight_id", "type", "event_time", "osm_ref"])
    ev["ts"] = (ev["event_time"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    ev["box"] = box_airport(ev["latitude"].to_numpy(dtype=float), ev["longitude"].to_numpy(dtype=float))
    ev["ap"] = ev["osm_airport"].where(ev["osm_airport"].isin(list(AIRPORTS)), ev["box"])
    ev = ev[ev["ap"].notna()]
    cs = ev[ev["callsign"].notna()].groupby("flight_id")["callsign"].agg(lambda s: s.mode().iat[0]).str.upper().str.strip()
    tko = ev[ev["type"].eq("take-off")].groupby(["flight_id", "ap"])["ts"].min()
    rwy = ev[ev["type"].eq("entry-runway")].groupby(["flight_id", "ap"])["ts"].max()
    ref = pd.concat([tko.rename("tko"), rwy.rename("rwy")], axis=1)
    ref["ref"] = ref["tko"].fillna(ref["rwy"])
    ref = ref.reset_index()
    ev = ev.merge(ref[["flight_id", "ap", "ref"]], on=["flight_id", "ap"], how="inner")
    ev = ev[ev["ts"] <= ev["ref"]]
    g = ev.groupby(["flight_id", "ap"])

    def last(tp):
        return ev[ev["type"].eq(tp)].groupby(["flight_id", "ap"])["ts"].max()

    def first(tp):
        return ev[ev["type"].eq(tp)].groupby(["flight_id", "ap"])["ts"].min()

    f = pd.DataFrame({"ref": g["ref"].first()})
    f["stand_exit"] = last("exit-parking_position")
    f["apron_exit"] = last("exit-apron")
    f["twy_first"] = first("entry-taxiway")
    f["first_seen"] = first("first_seen")
    rw = ev[ev["type"].eq("entry-runway")].groupby(["flight_id", "ap"])["ts"].max()
    f["rwy_entry"] = rw
    f["n_twy"] = ev[ev["type"].eq("entry-taxiway")].groupby(["flight_id", "ap"]).size()
    de_in, de_out = first("entry-deicing_pad"), last("exit-deicing_pad")
    f["deice"] = de_in.notna().reindex(f.index).fillna(False).astype(float)
    f["deice_dur"] = (de_out - de_in).reindex(f.index)
    f = f.reset_index().rename(columns={"ap": "airport"})
    f["cs"] = f["flight_id"].map(cs)
    return f[f["cs"].notna()]


def build(kind: str) -> None:
    dest = OUT / f"features_{kind}.parquet"
    dep = departures(kind)
    files = sorted(EVENTS.glob("events_*.parquet"))
    if kind == "dev":
        files = [f for f in files if f.name[7:11] == "2024" or f.name[7:11] == "2025"]
    else:
        files = [f for f in files if f.name[7:11] == "2026" or f.name.startswith("events_20251231")]
    of = opdi_flights(files)
    m = dep.merge(of, on=["airport", "cs"], how="inner")
    m["dt"] = m["ref"] - m["t"]
    m = m[(m["dt"] >= LO) & (m["dt"] <= HI)].copy()
    m["adt"] = m["dt"].abs()
    m = m.sort_values([ID, "adt"]).drop_duplicates(ID)
    t = m["t"]
    out = pd.DataFrame({ID: m[ID].to_numpy()})
    out["o_ref_rel"] = (m["dt"]).to_numpy()
    for c, src in [("o_stand_exit", "stand_exit"), ("o_apron_exit", "apron_exit"), ("o_twy_first", "twy_first"), ("o_first_seen", "first_seen"),
                   ("o_rwy_entry", "rwy_entry")]:
        out[c] = (t - m[src]).to_numpy()
    out["o_n_twy"] = m["n_twy"].fillna(0).to_numpy()
    out["o_deice"] = m["deice"].fillna(0).to_numpy()
    out["o_deice_dur"] = m["deice_dur"].to_numpy()
    out["o_has"] = 1.0
    out.to_parquet(dest, index=False)
    write_json(dest.with_suffix(".receipt.json"), {"written_at_utc": utc_now(), "kind": kind, "rows": int(len(out)), "departures": int(len(dep)),
                                                    "sha256": sha256_file(dest), "event_files": {f.name: sha256_file(f) for f in files},
                                                    "code_sha256": sha256_file(Path(__file__)), "keymap_sha256": sha256_file(KEYMAP)})
    a = dep.merge(out, on=ID, how="left")
    print(kind, "departures", len(dep), "matched", len(out), flush=True)
    print(a.groupby("airport")[["o_has", "o_stand_exit", "o_twy_first"]].agg(lambda s: round(float(s.notna().mean()), 3)), flush=True)


def opdi_columns(kind: str, ids: np.ndarray, new: np.ndarray) -> pd.DataFrame:
    f = pd.read_parquet(OUT / f"features_{kind}.parquet").set_index(ID).reindex(ids)
    f["o_has"] = f["o_has"].fillna(0.0)
    P = f[CORE].astype(np.float64).reset_index(drop=True)
    for q, c in REL:
        P[q] = P[c].to_numpy(dtype=np.float64) - new
    return P[NEW_COLS].astype(np.float32)


def coverage() -> None:
    res = {}
    for kind in ["dev", "rank"]:
        dep = departures(kind)
        f = pd.read_parquet(OUT / f"features_{kind}.parquet")
        raw = pd.read_parquet(ROOT / ("data/raw/ranking.parquet" if kind == "rank" else "data/raw/training_2025-01-01_2025-02-01.parquet"), columns=[ID]).head(0)
        del raw
        mo = dep[[ID, "airport", "t"]].copy()
        mo["month"] = pd.to_datetime(mo["t"], unit="s", utc=True).dt.strftime("%Y-%m")
        a = mo.merge(f, on=ID, how="left")
        for (ap, m), g in a.groupby(["airport", "month"]):
            res.setdefault(ap, {})[m] = {"rows": int(len(g)), "matched": round(float(g["o_has"].notna().mean()), 4),
                                         "stand_exit": round(float(g["o_stand_exit"].notna().mean()), 4),
                                         "twy_first": round(float(g["o_twy_first"].notna().mean()), 4),
                                         "deice": round(float((g["o_deice"] > 0).mean()), 4)}
    write_json(DOCS / "coverage.json", {"created_at_utc": utc_now(), "by_airport_month": res})
    for ap, d in res.items():
        print(ap, {m: (v["matched"], v["stand_exit"]) for m, v in d.items()}, flush=True)


# ---------------------------------------------------------------- the pinned chain

def score(name: str) -> None:
    """tools/run_ownpoints_weight.score_chain on the pinned v66 chain files (runs/pubrefit/chain/ship), with the OPDI columns appended to
    stack_resid_v3's 65 inputs (and to its new-column guard) when name == 'opdi'."""
    import run_ctx_ownpoints_chain as CH
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    if sha256_file(ROOT / CH.V63[0]) != CH.V63[1]:
        raise ValueError("The v63 development proxy differs from the declared file.")
    files = (SHIP / "ctl.parquet", SHIP / "rows_all.parquet", SHIP / "dmp.parquet", SHIP / "cand.parquet")
    dev = CH.dev_table_from(*files[:3])
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    core = V3.own_points("dev", dev[ID].to_numpy())
    Pn = V3.new_columns(core, dev["new"].to_numpy(dtype=float))
    X65 = pd.concat([X46, Pn], axis=1)
    pr = V3.dev_predictions(dev, X46, X65, Pn, res)
    new, air = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy()
    use = np.zeros(len(dev), bool)
    if name in VARIANTS:
        # stack_resid_v3's learner refitted with the OPDI columns appended (and guarded like v3's own-point columns); it replaces v3's
        # stacking value only on rows with an OPDI match that its guard does not hold (and, per variant, at the listed airports);
        # every other row keeps the identity value (v3), LIRF keeps v2 as in v3
        Po = opdi_columns("dev", dev[ID].to_numpy(), dev["new"].to_numpy(dtype=float))
        Pg = pd.concat([Pn, Po], axis=1)
        po = V3.dev_predictions(dev, X46, pd.concat([X46, Pg], axis=1), Pg, res)
        use = (Po["o_has"].to_numpy() > 0) & ~po["h3"] & np.isin(air, VARIANTS[name])
        pr = {**pr, "base3": np.where(use, po["base3"], pr["base3"]), "h3": po["h3"], "trees": {**pr["trees"], **{f"opdi {k}": v for k, v in po["trees"].items()}}}
    elif name != "identity":
        raise ValueError(name)
    s1 = pd.read_parquet(files[3]).set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    keys, _ = O2.O.dev_table()
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
    t["s2"] = s1 + (np.where(air == "LIRF", pr["base2"], pr["base3"]) - new)
    _, cand63 = V3.Chain().run(t, "s2")
    dev12, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    t["c63"] = cand63
    c66 = CH.own_motion_v2(t, "c63", t12, trig)
    d = OUT / "chain" / name
    d.mkdir(parents=True, exist_ok=True)
    out = t[[ID, "month", "airport", "label", "cls", "day"]].copy()
    out["value"] = c66
    out["held3"] = pr["h3"]
    out["opdi_used"] = use
    out.to_parquet(d / "proxy.parquet", index=False)
    rm = {m: float(np.sqrt(np.mean((g["label"] - g["value"]) ** 2))) for m, g in out.groupby("month")}
    ship = pd.read_parquet(SHIP / "proxy.parquet").set_index(ID)["value"].reindex(out[ID]).to_numpy(dtype=float)
    diff = float(np.abs(c66 - ship).max())
    write_json(d / "score.json", {"created_at_utc": utc_now(), "name": name, "month_rmse": rm, "max_abs_diff_vs_pinned_control": diff,
                                  "inputs": {p.name: sha256_file(p) for p in files}, "stacker_trees": pr["trees"],
                                  "proxy_sha256": sha256_file(d / "proxy.parquet"), "code": sha256_file(Path(__file__)),
                                  "opdi_features": read_json(OUT / "features_dev.receipt.json") if name in VARIANTS else None, "opdi_rows_used": int(use.sum())})
    print(name, "month RMSE", {m: round(v, 4) for m, v in rm.items()}, "max |diff| vs pinned control", diff, flush=True)
    if name == "identity" and diff != 0.0:
        raise ValueError("identity: the copied chain does not reproduce the pinned v66 control.")


def read() -> None:
    import run_intruder_bound as R
    ctl = pd.read_parquet(SHIP / "proxy.parquet")
    res = {"created_at_utc": utc_now(), "control": "pinned v66 development proxy (runs/pubrefit/chain/ship/proxy.parquet)", "bar": BAR,
           "variants": {}, "official_scores_used": "none"}
    for name in VARIANTS:
        p = OUT / "chain" / name / "proxy.parquet"
        if not p.exists():
            continue
        c = pd.read_parquet(p).set_index(ID).reindex(ctl[ID])
        t = ctl.rename(columns={"value": "ctl"}).assign(cand=c["value"].to_numpy(dtype=float), used=c["opdi_used"].to_numpy(dtype=bool))
        rng = np.random.default_rng(20261004)
        months = {}
        for m in MONTHS:
            s = t[t["month"] == m]
            pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
            ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
            months[m] = {"pct": pct, "lo": lo, "hi": hi, "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))),
                         "control_rmse": float(np.sqrt(np.mean((s["label"] - s["ctl"]) ** 2))), "opdi_rows_used": int(s["used"].sum()),
                         "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum())}
        eq = float(np.mean([months[m]["pct"] for m in EQUAL]))
        n_below = sum(months[m]["hi"] < 0 for m in EQUAL)
        go = eq <= BAR["eq_mean_pct"] and all(months[m]["pct"] <= BAR["max_month_pct"] for m in MONTHS) and n_below >= BAR["min_intervals_below_zero"]
        by = {}
        for (a, m), g in t.groupby(["airport", "month"]):
            tot = float(((t.loc[t["month"] == m, "label"] - t.loc[t["month"] == m, "ctl"]) ** 2).sum())
            c0, c1 = float(((g["label"] - g["ctl"]) ** 2).sum()), float(((g["label"] - g["cand"]) ** 2).sum())
            by[f"{a} {m}"] = {"sse_change_pct_of_month": 100 * (c1 - c0) / tot, "rows_changed": int((np.abs(g["cand"] - g["ctl"]) > 1e-9).sum()),
                              "opdi_rows_used": int(g["used"].sum())}
        res["variants"][name] = {"airports": VARIANTS[name], "months": months, "equal_footing_mean_pct": eq, "equal_intervals_below_zero": int(n_below),
                                 "go": bool(go), "airport_month": by}
        print(name, {m: (round(v["pct"], 4), round(v["lo"], 3), round(v["hi"], 3)) for m, v in months.items()}, "eq", round(eq, 4), "GO", go, flush=True)
        for k, v in by.items():
            if k.startswith(("LSZH", "EDDF", "LEMD")):
                print("  ", k, round(v["sse_change_pct_of_month"], 4), v["rows_changed"], v["opdi_rows_used"], flush=True)
    write_json(DOCS / "screen.json", res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--score")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args()
    if a.features:
        build(a.features)
    if a.coverage:
        coverage()
    if a.score:
        score(a.score)
    if a.read:
        read()


if __name__ == "__main__":
    main()
