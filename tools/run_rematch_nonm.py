"""Governing run of configs/rematch_nonm_v1.json: no-NM departures at EDDF / EHAM / LSZH rematched to adsb.lol surface tracks through a
learned flight-number -> callsign key, blended into the model part (w 0.5) before the published-data chain.

  --matches         key map and rematch of every no-NM departure at the three airports (2025 and the ranking; label-free)
                    -> runs/rematch_nonm/key_map.parquet, matches_2025.parquet, matches_rank.parquet
  --dev             the development chain (as tools/scout_refit_run.py builds runs/scout/refit/ctl_v55m.parquet) for the control
                    (unchanged v55m, must reproduce ctl_v55m `new` exactly) and the candidate -> runs/rematch_nonm/dev/{ctl,cand}.parquet
  --score           the declared rule (development + twelve-month check) -> docs/rematch_nonm/results_v1.json
  --ranking-base    the ranking model part with the layer: runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet + the layer
                    -> runs/rematch_nonm/base/vigorous-whistle_v55m_rematch.parquet (never under runs/submissions)
  --ranking-check   label-free: the dry-run chain output against v57 and against the published daily totals
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_adsb.features import load_surface, segment  # noqa: E402
from adsb_rematch_nonm import IATA  # noqa: E402

DECLARATION = "configs/rematch_nonm_v1.json"
OUT = ROOT / "runs/rematch_nonm"
DEFAULT_OUT = OUT   # the 2025 matches and offsets always come from here; --out moves only the new file's matches and base
AIRPORTS = ["EDDF", "EHAM", "LSZH"]
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
SEED = 20260929
W = 0.5
FLOOR = -12.0
GUARD = (213142471.0, 213142469.0)
BASE = ("runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet", "266b8c936f2625578e4980f1b11c3a26cc63594a13916510fa3fe86925ec800f")
COLS = [ID, "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "STAND_mvt"]
TRAINING = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
RANKING = ROOT / "data/raw/ranking.parquet"


# ---------------------------------------------------------------- key map and rematch (label-free)

def departures() -> pd.DataFrame:
    frames = [pd.read_parquet(p, columns=COLS) for p in [*TRAINING, RANKING]]
    d = pd.concat(frames, ignore_index=True)
    return d[d["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).reset_index(drop=True)


def key_map(d: pd.DataFrame) -> pd.DataFrame:
    w = d[d["CALLSIGN_flt"].notna() & d["FLIGHT_mvt"].notna()][["ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt"]]
    w = w.assign(FLIGHT_mvt=w["FLIGHT_mvt"].str.strip().str.upper(), CALLSIGN_flt=w["CALLSIGN_flt"].str.strip().str.upper())
    cnt = w.value_counts().rename("k").reset_index()
    cnt = cnt.sort_values(["ADEP_mvt", "FLIGHT_mvt", "k", "CALLSIGN_flt"], ascending=[True, True, False, True], kind="mergesort")
    return cnt.drop_duplicates(["ADEP_mvt", "FLIGHT_mvt"]).rename(columns={"CALLSIGN_flt": "key"}).reset_index(drop=True)


def iata_key(flight: str) -> str | None:
    return IATA[flight[:2]] + flight[2:] if flight[:2] in IATA and flight[2:3].isdigit() else None


def rematch(rows: pd.DataFrame) -> pd.DataFrame:
    """Per row and key (existing FLIGHT_mvt, v35's IATA mapping, the learned key): the segment nearest take-off (tools/adsb_rematch_nonm.py)."""
    rows = rows.copy()
    rows["takeoff"] = (rows["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    rows["flight"] = rows["FLIGHT_mvt"].fillna("").astype(str).str.strip().str.upper()
    stands = pd.read_parquet(ROOT / "data/external/adsblol/matches_v11/stands.parquet")
    lookup = {(r.airport, r.stand): (r.lat, r.lon) for r in stands.itertuples(index=False)}
    rows["dayk"] = rows["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m%d")
    out = []
    for dayk, group in rows.groupby("dayk", sort=True):
        days = [(pd.Timestamp(dayk) - pd.Timedelta(days=1)).strftime("%Y%m%d"), dayk]
        surface = load_surface(ROOT / "data/external/adsblol/surface", days)
        for airport, g in group.groupby("ADEP_mvt", sort=True):
            keys = {}
            for row in g.itertuples(index=False):
                keys[row[0]] = {"key": row.flight or None, "iata": iata_key(row.flight), "learned": row.key if isinstance(row.key, str) else None}
            s = surface[surface["airport"].eq(airport)] if not surface.empty else surface
            wanted = {k for v in keys.values() for k in v.values() if k}
            hit = s[s["callsign"].isin(wanted)] if len(s) else s
            by_callsign = {c: f["icao"].unique() for c, f in hit.groupby("callsign")} if len(hit) else {}
            s = s[s["icao"].isin(hit["icao"].unique())] if len(hit) else s.iloc[0:0]
            by_icao = {i: f for i, f in s.groupby("icao")} if len(s) else {}
            for row in g.itertuples(index=False):
                stand = lookup.get((airport, row.STAND_mvt)) if isinstance(row.STAND_mvt, str) else None
                rec = {ID: row[0], "airport": airport, "takeoff": row.takeoff}
                for method, key in keys[row[0]].items():
                    best = None
                    for icao in (by_callsign.get(key, []) if key else []):
                        track = by_icao[icao]
                        before = track[(track["time"] <= row.takeoff + 30) & (track["time"] >= row.takeoff - 4 * 3600) & track["ground"]]
                        after = track[(track["time"] >= row.takeoff - 30) & (track["time"] <= row.takeoff + 900) & ~track["ground"]]
                        if len(before) >= 3 and len(after) >= 1:
                            info = segment(track, row.takeoff, stand=stand)
                            if info is not None and (best is None or abs(info["last_ground"] - row.takeoff) < abs(best["last_ground"] - row.takeoff)):
                                best = {**info, "icao": icao}
                    rec[f"{method}_found"] = best is not None
                    if best is not None:
                        for k in ["icao", "gate", "at_known_stand", "motion_start", "points", "taxi_stops", "taxi_stopped_s"]:
                            rec[f"{method}_{k}"] = best[k]
                out.append(rec)
        print(dayk, len(out), flush=True)
    t = pd.DataFrame(out)
    for method in ["key", "iata", "learned"]:
        for k in ["gate", "at_known_stand"]:
            c = f"{method}_{k}"
            t[c] = t[c].fillna(False).astype(bool) if c in t else False
    t["observed_taxi"] = t["takeoff"] - t.get("learned_motion_start", np.nan)
    t["learned_ok"] = t["learned_gate"] & t["learned_at_known_stand"]            # usable for the offsets
    t["qualifying"] = t["learned_ok"] & ~t["key_gate"] & ~t["iata_gate"]       # rows the layer changes
    return t


def step_matches() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = departures()
    km = key_map(d)
    km.to_parquet(OUT / "key_map.parquet", index=False)
    rows = d[d["ADEP_flt"].isna() & d["ADEP_mvt"].isin(AIRPORTS)].copy()
    rows["FLIGHT_mvt"] = rows["FLIGHT_mvt"].fillna("").astype(str).str.strip().str.upper()
    rows = rows.merge(km[["ADEP_mvt", "FLIGHT_mvt", "key"]], on=["ADEP_mvt", "FLIGHT_mvt"], how="left")
    ranking_ids = set(pd.read_parquet(RANKING, columns=[ID])[ID])
    rank = rows[rows[ID].isin(ranking_ids)]
    train = rows[~rows[ID].isin(ranking_ids)]
    for name, part in [("rank", rank), ("2025", train)]:
        if name == "2025" and OUT != DEFAULT_OUT:   # final file: the shipped 2025 matches stay in runs/rematch_nonm
            continue
        if (OUT / f"matches_{name}.parquet").exists():
            print("keep", name)
            continue
        t = rematch(part[[ID, "ADEP_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_mvt", "STAND_mvt", "key"]])
        t["month"] = pd.to_datetime(t["takeoff"], unit="s", utc=True).dt.strftime("%Y-%m")
        t.to_parquet(OUT / f"matches_{name}.parquet", index=False)
        print(name, len(t), "qualifying", int(t["qualifying"].sum()), flush=True)


# ---------------------------------------------------------------- offsets and the layer

def labels_2025() -> pd.Series:
    raw = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", TARGET]) for p in TRAINING], ignore_index=True)
    return raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID)[TARGET].astype(float)


def offsets(m25: pd.DataFrame, lab: pd.Series, exclude_month: str | None) -> dict:
    f = m25[m25["learned_ok"] & (m25["month"] != exclude_month if exclude_month else True)]
    e = lab.reindex(f[ID]).to_numpy() - f["observed_taxi"].to_numpy()
    return {a: float(np.median(e[(f["airport"] == a).to_numpy()])) for a in AIRPORTS}


def apply(values: np.ndarray, ids: np.ndarray, matches: pd.DataFrame, off: dict) -> tuple[np.ndarray, np.ndarray]:
    q = matches[matches["qualifying"]].set_index(ID)
    pos = pd.Index(ids).get_indexer(q.index)
    keep = pos >= 0
    q, pos = q[keep], pos[keep]
    if np.isin(q.index.to_numpy(), GUARD).any():
        raise ValueError("A guard row qualified.")
    target = q["observed_taxi"].to_numpy() + q["airport"].map(off).to_numpy(dtype=float)
    new = values.astype(float).copy()
    new[pos] = np.maximum(new[pos] + W * (target - new[pos]), FLOOR)
    changed = np.zeros(len(ids), dtype=bool)
    changed[pos] = True
    return new, changed


# ---------------------------------------------------------------- development chain (as tools/scout_refit_run.py)

def dev_tables(kind: str, col: np.ndarray, base: pd.DataFrame, tk, pub) -> None:
    from scout_refit_prep import with_stack
    new = with_stack(base, col, tk, pub)
    for m in MONTHS:
        mix = pd.concat([base[base["month"] != m], new[new["month"] == m]], ignore_index=True)
        mix.to_parquet(OUT / f"dev/rows_{kind}_{m}.parquet", index=False)


def step_dev() -> None:
    import run_daily_totals as rd
    import scout_extreme_shift as se
    import scout_var_alloc as sva
    from scout_refit_prep import takeoff
    from screen_daily_totals import published
    (OUT / "dev").mkdir(parents=True, exist_ok=True)
    m25 = pd.read_parquet(OUT / "matches_2025.parquet")
    lab = labels_2025()
    base = pd.read_parquet(ROOT / "runs/daily_totals/rows.parquet")
    dm = pd.read_parquet(ROOT / "runs/dev_model_part/rows.parquet").drop_duplicates(ID).set_index(ID)
    ids = base[ID].to_numpy()
    v = dm["v55m"].reindex(ids).to_numpy(dtype=float)
    v = np.where(np.isnan(v), base["stack"].to_numpy(dtype=float), v)
    cand, changed = v.copy(), np.zeros(len(v), dtype=bool)
    offs = {}
    for m in MONTHS:
        idx = np.flatnonzero(base["month"].eq(m).to_numpy())
        offs[m] = offsets(m25, lab, m)
        c, ch = apply(v[idx], ids[idx], m25, offs[m])
        cand[idx], changed[idx] = c, ch
    pd.DataFrame({ID: ids, "month": base["month"], "v55m": v, "cand": cand, "changed": changed}).to_parquet(OUT / "dev/model_part.parquet", index=False)
    write_json(OUT / "dev/offsets.json", offs)
    tk, pub = takeoff(ids), published()
    settings, tau_scale, prior = rd.settings_from(read_json(ROOT / "configs/daily_totals_v1.json"))
    for kind, col in [("ctl", v), ("cand", cand)]:
        dev_tables(kind, col, base, tk, pub)
        parts = []
        for m in MONTHS:
            tag = f"rematch_nonm_{kind}"
            if not (ROOT / f"runs/daily_totals/{tag}/{m}.parquet").exists():
                rd.run_month(m, tag, tau_scale, settings, rows=f"runs/rematch_nonm/dev/rows_{kind}_{m}.parquet", prior=prior)
            d = OUT / f"dev/v48_{kind}_{m}"
            d.mkdir(parents=True, exist_ok=True)
            for k in MONTHS:
                src = ROOT / (f"runs/daily_totals/{tag}/{k}.parquet" if k == m else f"runs/daily_totals/daily_totals_v1/{k}.parquet")
                shutil.copyfile(src, d / f"{k}.parquet")
            se.TABLES[f"{kind}_{m}"] = (OUT / f"dev/rows_{kind}_{m}.parquet", d)
            t, ks = sva.layered(f"{kind}_{m}")
            parts.append(t.loc[t["month"] == m, [ID, "month", "airport", "bday", "label", "stack", "v48", "ext", "new"]].assign(k=ks[m]))
            print(kind, m, "k", ks[m], flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(OUT / f"dev/{kind}.parquet", index=False)
        if kind == "ctl":
            ref = pd.read_parquet(ROOT / "runs/scout/refit/ctl_v55m.parquet").set_index(ID)
            got = pd.read_parquet(OUT / "dev/ctl.parquet").set_index(ID).reindex(ref.index)
            diff = float(np.max(np.abs(got["new"].to_numpy() - ref["new"].to_numpy())))
            print("control against runs/scout/refit/ctl_v55m.parquet: max |new difference| =", diff, flush=True)
            if diff != 0.0:
                raise ValueError("The control does not reproduce ctl_v55m exactly.")


# ---------------------------------------------------------------- the rule

def twelve(m25: pd.DataFrame, lab: pd.Series) -> dict:
    from run_adsb_v11 import paired_days
    r12 = pd.read_parquet(ROOT / "runs/daily_totals/rows12.parquet", columns=[ID, "month", "label", "stack", "bday"])
    out = {}
    for m in sorted(r12["month"].unique()):
        g = r12[r12["month"] == m].reset_index(drop=True)
        y, s = g["label"].to_numpy(dtype=float), g["stack"].to_numpy(dtype=float)
        c, ch = apply(s, g[ID].to_numpy(), m25, offsets(m25, lab, m))
        r = lambda v: float(np.sqrt(np.mean((y - v) ** 2)))
        out[m] = {"rows_changed": int(ch.sum()), "control_rmse": r(s), "rmse": r(c), "change_pct": 100 * (r(c) / r(s) - 1),
                  "sum_squared_change": float(((c - s) ** 2).sum()), "sse_change": float(((y - c) ** 2).sum() - ((y - s) ** 2).sum()),
                  "paired_days": paired_days(g["bday"].to_numpy(), y, c, s, SEED)["rmse_difference_95pct_interval_seconds"]}
    return out


def step_score() -> None:
    from run_adsb_v11 import paired_days
    decl = read_json(ROOT / DECLARATION)
    rule = decl["rule"]
    c = pd.read_parquet(OUT / "dev/ctl.parquet").set_index(ID)
    n = pd.read_parquet(OUT / "dev/cand.parquet").set_index(ID).reindex(c.index)
    ref = pd.read_parquet(ROOT / "runs/scout/refit/ctl_v55m.parquet").set_index(ID).reindex(c.index)
    if float(np.max(np.abs(c["new"].to_numpy() - ref["new"].to_numpy()))) != 0.0:
        raise ValueError("The control does not reproduce ctl_v55m exactly.")
    mp = pd.read_parquet(OUT / "dev/model_part.parquet").set_index(ID).reindex(c.index)
    dev = {}
    for m in MONTHS:
        k = (c["month"] == m).to_numpy()
        y, a, b = c["label"].to_numpy()[k], c["new"].to_numpy()[k], n["new"].to_numpy()[k]
        r = lambda v: float(np.sqrt(np.mean((y - v) ** 2)))
        iv = paired_days(c["bday"].to_numpy()[k], y, b, a, SEED)
        d = np.sort((y - b) ** 2 - (y - a) ** 2)
        dev[m] = {"model_part_rmse": r(mp["v55m"].to_numpy()[k]), "control_rmse": r(a), "candidate_rmse": r(b), "change_pct": 100 * (r(b) / r(a) - 1),
                  "interval_seconds": iv["rmse_difference_95pct_interval_seconds"], "days_better": iv["days_with_lower_squared_error"], "days": iv["days"],
                  "layer_rows": int(mp["changed"].to_numpy()[k].sum()), "rows_changed_after_chain": int((np.abs(b - a) > 1e-9).sum()),
                  "model_part_squared_change": float(((mp["cand"] - mp["v55m"]).to_numpy()[k] ** 2).sum()),
                  "squared_change_after_chain": float(((b - a) ** 2).sum()), "sse_change": float(((y - b) ** 2).sum() - ((y - a) ** 2).sum()),
                  "best_ten_rows_share_of_gain_pct": float(100 * d[:10].sum() / d.sum()) if d.sum() < 0 else None,
                  "k_control": float(c["k"].to_numpy()[k][0]), "k_candidate": float(n["k"].to_numpy()[k][0])}
        print(m, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in dev[m].items()}, flush=True)
    m25 = pd.read_parquet(OUT / "matches_2025.parquet")
    tw = twelve(m25, labels_2025())
    for m, v in tw.items():
        print("twelve", m, round(v["change_pct"], 4), v["rows_changed"], [round(x, 3) for x in v["paired_days"]], flush=True)
    mean = float(np.mean([dev[m]["change_pct"] for m in EQUAL]))
    checks = {"1_no_month_worse": all(dev[m]["change_pct"] <= rule["max_month_worsening_pct"] for m in MONTHS),
              "2_equal_footing_mean": mean <= -rule["min_mean_improvement_pct"],
              "3_intervals_below_zero": all(dev[m]["interval_seconds"][1] < 0 for m in EQUAL),
              "4_twelve": all(v["change_pct"] <= rule["max_month_worsening_pct"] for v in tw.values())
                          and np.mean([v["change_pct"] < 0 for v in tw.values()]) >= rule["twelve_min_better_share"]}
    checks = {k: bool(v) for k, v in checks.items()}
    accepted = all(checks.values())
    report = {"experiment_id": decl["experiment_id"], "created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": object_hash(decl),
              "declaration_file_sha256": sha256_file(ROOT / DECLARATION), "code": {"tools/run_rematch_nonm.py": sha256_file(ROOT / "tools/run_rematch_nonm.py")},
              "inputs": {p: sha256_file(ROOT / p) for p in ["runs/dev_model_part/rows.parquet", "runs/daily_totals/rows.parquet", "runs/daily_totals/rows12.parquet",
                                                           "runs/scout/refit/ctl_v55m.parquet", "data/external/adsblol/matches_v11/stands.parquet"]},
              "matches": {p: sha256_file(OUT / p) for p in ["key_map.parquet", "matches_2025.parquet", "matches_rank.parquet"] if (OUT / p).exists()},
              "offsets_by_scored_month": read_json(OUT / "dev/offsets.json"), "control_reproduces_ctl_v55m": True,
              "development": dev, "equal_footing_mean_change_pct": mean, "twelve_months": tw,
              "twelve_better": int(sum(v["change_pct"] < 0 for v in tw.values())), "twelve_mean_change_pct": float(np.mean([v["change_pct"] for v in tw.values()])),
              "checks": checks, "accepted": accepted, "official_scores_used": "none"}
    path = ROOT / decl["report"]
    if path.exists():
        raise ValueError(f"{path} exists; never overwrite.")
    write_json(path, report)
    print("equal-footing mean %:", round(mean, 4), "checks", checks, "ACCEPTED" if accepted else "NOT ACCEPTED")


# ---------------------------------------------------------------- ranking

def step_ranking_base() -> None:
    if sha256_file(ROOT / BASE[0]) != BASE[1]:
        raise ValueError("The base model part changed.")
    b = pd.read_parquet(ROOT / BASE[0])
    ids = b[ID].to_numpy()
    mr = pd.read_parquet(OUT / "matches_rank.parquet")
    off = offsets(pd.read_parquet(DEFAULT_OUT / "matches_2025.parquet"), labels_2025(), None)
    new, changed = apply(b[TARGET].to_numpy(dtype=float), ids, mr, off)
    out = b.copy()
    out[TARGET] = np.where(changed, np.rint(new), b[TARGET].to_numpy()).astype(b[TARGET].dtype)
    d = OUT / "base"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "vigorous-whistle_v55m_rematch.parquet"
    if path.exists():
        raise ValueError(f"{path} exists; never overwrite.")
    out.to_parquet(path, index=False)
    s = ((out[TARGET].to_numpy(dtype=float) - b[TARGET].to_numpy(dtype=float)) ** 2)
    rec = {"created_at_utc": utc_now(), "base": {"path": BASE[0], "sha256": BASE[1]}, "offsets": off, "rows_changed": int((s > 0).sum()), "layer_rows": int(changed.sum()),
           "sum_squared_change": float(s.sum()), "output": {"path": str(path.relative_to(ROOT).as_posix()), "sha256": sha256_file(path)}}
    write_json(d / "base_receipt.json", rec)
    print(rec)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ["matches", "dev", "score", "ranking-base"]:
        p.add_argument(f"--{f}", action="store_true")
    p.add_argument("--out", default=None, help="final file: directory for the new file's key map, matches and rematched base (default runs/rematch_nonm)")
    p.add_argument("--base", default=None, help="final file: the runner's v55m model-part file (default the shipped v55 model part)")
    p.add_argument("--base-sha256", default=None)
    import final_chain_overrides
    final_chain_overrides.add_arguments(p)
    a = p.parse_args()
    final_chain_overrides.apply(a)
    global OUT, BASE
    if a.out:
        OUT = ROOT / final_chain_overrides._rel(a.out)
    if a.base:
        if not a.base_sha256:
            raise ValueError("--base needs --base-sha256.")
        BASE = (final_chain_overrides._rel(a.base), a.base_sha256)
    if (a.dev or a.score) and OUT != DEFAULT_OUT:
        raise ValueError("--dev / --score are the governing run on 2025; they never take --out.")
    if a.matches:
        step_matches()
    if a.dev:
        step_dev()
    if a.score:
        step_score()
    if a.ranking_base:
        step_ranking_base()


if __name__ == "__main__":
    main()
