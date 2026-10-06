"""Lever 6 of the final-week campaign: the published-data layers refitted on v66's development chain (docs/pubrefit/README.md).

v66's development chain (runs/ctx_ownpoints_weight/chain/own_all_w100, tools/run_ctx_ownpoints_chain.step_chain) applies the extreme-day
layer and the variance re-allocation month by month on a mixed table: the scored month carries v66's model part, the fitting months the
v46 development stack, so both layers keep the parameters fitted on the older model part. This tool rebuilds that chain with the variance
re-allocation's LightGBM pinned (deterministic, column-wise; tools/lirf_own_pinned.py) and reads three refits against it:

  ship   control: tools/scout_var_alloc.layered on the shipped mixed tables (pinned) -> the pinned v66 development proxy
  va     variance re-allocation models and k refitted month-blocked on v66's chain rows; extreme-day values as in ship
  ext    extreme-day calibration refitted month-blocked on v66's chain days; ship's variance re-allocation correction on top
  both   tools/scout_var_alloc.layered on the all-v66 table (both layers refitted month-blocked, the layers' own governing procedure)

Every variant then goes through tools/run_ownpoints_weight.score_chain (stack_resid_v3 refitted on the rebuilt components, intruder
bound, identity extension, own-motion v1, own_motion_v2) exactly as v66's stage 2.

  --tables          all-v66 rows and v48 tables per kind (ctl: without the no-NM rematch, cand: with it)
  --layers V        the layer values of variant V (ship first; va / ext need ship, ext needs both)
  --check-va        the va arithmetic fed with both's extreme-day values must reproduce both exactly (code check)
  --score V         score_chain on variant V's files -> runs/pubrefit/chain/<V>/proxy.parquet
  --read            every scored variant against ship: v57's clauses 1-3 -> runs/pubrefit/screen.json
No official score enters. Nothing is uploaded.
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
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_scout import var_alloc as VA  # noqa: E402

VA.PARAMS = {**VA.PARAMS, "deterministic": True, "force_col_wise": True}

import scout_extreme_shift as se  # noqa: E402
import scout_var_alloc as sva  # noqa: E402

SRC = ROOT / "runs/ctx_ownpoints_weight/chain/own_all_w100"
STORED = SRC / "dev_v66_proxy.parquet"
PINNED_B = ROOT / "runs/lirf_own/chain/identity_pinned_v66"
OUT = ROOT / "runs/pubrefit"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
KINDS = ["ctl", "cand"]
KEEP = [ID, "month", "airport", "bday", "label", "stack", "v48", "ext", "new", "I", "w", "base", "elig", "k"]
RULE = {"max_month_worsening_pct": 0.005, "min_mean_improvement_pct": 0.02}


def tables() -> None:
    d = OUT / "tables"
    d.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        parts = [pd.read_parquet(SRC / f"rows_{kind}_{m}.parquet") for m in MONTHS]
        new = pd.concat([p[p["month"] == m] for p, m in zip(parts, MONTHS)], ignore_index=True)
        if len(new) != len(parts[0]) or new[ID].duplicated().any():
            raise ValueError("The all-v66 table does not cover the development rows once.")
        new.to_parquet(d / f"rows_{kind}_new.parquet", index=False)
        vd = d / f"v48_{kind}_new"
        vd.mkdir(exist_ok=True)
        for m in MONTHS:
            shutil.copyfile(SRC / f"v48_{kind}_{m}" / f"{m}.parquet", vd / f"{m}.parquet")
        if kind == "ctl":
            ra = pd.read_parquet(SRC / "rows_all.parquet").set_index(ID).sort_index()
            mine = new.set_index(ID).sort_index()[ra.columns]
            pd.testing.assert_frame_equal(mine, ra, check_exact=True)
            print("ctl all-v66 table equals rows_all.parquet exactly", flush=True)
    write_json(d / "tables.json", {"created_at_utc": utc_now(), "files": {p.name: sha256_file(p) for p in sorted(d.glob("*.parquet"))}})


def register(name: str, rows: Path, v48: Path) -> str:
    se.TABLES[name] = (rows, v48)
    return name


def keep(t: pd.DataFrame, ks: dict, months: list[str]) -> pd.DataFrame:
    t = t.assign(elig=VA.eligible(t), k=t["month"].map(ks))
    return t.loc[t["month"].isin(months), KEEP].reset_index(drop=True)


def layers_ship(kind: str) -> pd.DataFrame:
    parts = []
    for m in MONTHS:
        t, ks = sva.layered(register(f"pr_ship_{kind}_{m}", SRC / f"rows_{kind}_{m}.parquet", SRC / f"v48_{kind}_{m}"))
        parts.append(keep(t, ks, [m]))
        print("ship", kind, m, "k", ks[m], flush=True)
    return pd.concat(parts, ignore_index=True)


def layers_both(kind: str) -> pd.DataFrame:
    d = OUT / "tables"
    t, ks = sva.layered(register(f"pr_new_{kind}", d / f"rows_{kind}_new.parquet", d / f"v48_{kind}_new"))
    print("both", kind, "k", ks, flush=True)
    return keep(t, ks, MONTHS)


def va_refit(kind: str, given: pd.DataFrame) -> pd.DataFrame:
    """tools/scout_var_alloc.layered's variance re-allocation part on the all-v66 table, with the extreme-day values and I_d taken from
    `given` (one row per development row: ext, I)."""
    d = OUT / "tables"
    t = se.dev_rows(register(f"pr_new_{kind}", d / f"rows_{kind}_new.parquet", d / f"v48_{kind}_new"))
    g = given.set_index(ID)
    t["ext"] = g["ext"].reindex(t[ID]).to_numpy(dtype=float)
    t["I"] = g["I"].reindex(t[ID]).to_numpy(dtype=float)
    if np.isnan(t["ext"]).any() or np.isnan(t["I"]).any():
        raise ValueError("Missing extreme-day values.")
    t["r"] = t["label"] - t["ext"]
    elig = VA.eligible(t)
    f = VA.features(t, sva.departures(sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))), se.KEYS)
    a, label, r = t["a"].to_numpy(dtype=float), t["label"].to_numpy(dtype=float), t["r"].to_numpy(dtype=float)
    month, extra = t["month"].to_numpy(), t["extra"].to_numpy(dtype=bool)
    w = np.full(len(t), np.nan)
    for m in MONTHS:
        models = VA.fit_models(f, a, label, r, elig & (month != m))
        ev = month == m
        w[ev] = VA.weights(models, f.loc[ev], elig[ev])
        print("va", kind, m, "models fitted without the month", flush=True)
    base = VA.basis(t, w, t["I"].to_numpy(dtype=float), se.KEYS)
    ext = t["ext"].to_numpy(dtype=float)
    new, ks = ext.copy(), {}
    for m in MONTHS:
        ks[m] = VA.slope(r, base, elig & (month != m) & ~extra)
        ev = month == m
        new[ev] = VA.update(ext[ev], ks[m], base[ev], elig[ev])
    t["w"], t["base"], t["new"] = w, base, new
    print("va", kind, "k", ks, flush=True)
    return keep(t, ks, MONTHS)


def ext_refit(ship: pd.DataFrame, both: pd.DataFrame) -> pd.DataFrame:
    """both's extreme-day values with ship's variance re-allocation correction (k x base, capped, floor) on top."""
    s = ship.set_index(ID)
    b = both.set_index(ID).reindex(s.index)
    out = s.copy()
    elig = s["elig"].to_numpy(dtype=bool)
    ext_b = b["ext"].to_numpy(dtype=float)
    corr = np.clip(s["k"].to_numpy(dtype=float) * s["base"].to_numpy(dtype=float), -VA.CAP_S, VA.CAP_S)
    out["ext"] = ext_b
    out["new"] = np.where(elig, np.maximum(ext_b + corr, VA.FLOOR_S), ext_b)
    out["v48"] = b["v48"].to_numpy(dtype=float)
    return out.reset_index()[KEEP]


def layers(variant: str) -> None:
    d = OUT / "layers"
    d.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        dest = d / f"{variant}_{kind}.parquet"
        if dest.exists():
            print("exists", dest, flush=True)
            continue
        if variant == "ship":
            v = layers_ship(kind)
        elif variant == "both":
            v = layers_both(kind)
        elif variant == "va":
            v = va_refit(kind, pd.read_parquet(d / f"ship_{kind}.parquet"))
        elif variant == "ext":
            v = ext_refit(pd.read_parquet(d / f"ship_{kind}.parquet"), pd.read_parquet(d / f"both_{kind}.parquet"))
        else:
            raise ValueError(variant)
        v.to_parquet(dest, index=False)
        print("wrote", dest, sha256_file(dest), flush=True)
        if variant == "ship":
            stored = pd.read_parquet(SRC / f"{kind}.parquet").set_index(ID)["new"].reindex(v[ID]).to_numpy(dtype=float)
            pinned_b = pd.read_parquet(PINNED_B / f"{kind}.parquet").set_index(ID)["new"].reindex(v[ID]).to_numpy(dtype=float)
            print("ship", kind, "max |diff| vs stored (unpinned)", float(np.abs(v["new"].to_numpy() - stored).max()),
                  "vs fork B's pinned rebuild", float(np.abs(v["new"].to_numpy() - pinned_b).max()), flush=True)


def check_va() -> None:
    d = OUT / "layers"
    both = pd.read_parquet(d / "both_ctl.parquet")
    mine = va_refit("ctl", both).set_index(ID).reindex(both[ID])
    diff = float(np.abs(mine["new"].to_numpy() - both["new"].to_numpy()).max())
    print("check-va: the va arithmetic on both's extreme-day values reproduces both, max |diff| =", diff, flush=True)
    write_json(OUT / "check_va.json", {"created_at_utc": utc_now(), "max_abs_diff_s": diff})
    if diff > 1e-9:
        raise ValueError("The va arithmetic does not reproduce layered.")


def score(variant: str) -> None:
    import run_ownpoints_weight as W
    d = OUT / "chain" / variant
    d.mkdir(parents=True, exist_ok=True)
    cols = [ID, "month", "airport", "bday", "label", "stack", "v48", "ext", "new", "k"]
    for kind in KINDS:
        pd.read_parquet(OUT / "layers" / f"{variant}_{kind}.parquet")[cols].to_parquet(d / f"{kind}.parquet", index=False)
    for f in ["rows_all.parquet", "dmp.parquet"]:
        shutil.copyfile(SRC / f, d / f)
    t, c66, inputs, trees = W.score_chain(d)
    out = t[[ID, "month", "airport", "label", "cls", "day"]].copy()
    out["value"] = c66
    out.to_parquet(d / "proxy.parquet", index=False)
    rm = {m: float(np.sqrt(np.mean((g["label"] - g["value"]) ** 2))) for m, g in out.groupby("month")}
    write_json(d / "score.json", {"created_at_utc": utc_now(), "variant": variant, "month_rmse": rm, "inputs": inputs, "stacker_trees": trees,
                                  "proxy_sha256": sha256_file(d / "proxy.parquet"), "code": sha256_file(ROOT / "tools/run_pubrefit.py")})
    print(variant, "month RMSE", {m: round(v, 4) for m, v in rm.items()}, flush=True)
    if variant == "ship":
        stored = pd.read_parquet(STORED).set_index(ID)["v66"].reindex(out[ID]).to_numpy(dtype=float)
        b = pd.read_parquet(PINNED_B / "identity_rebuilt.parquet")
        bcol = [c for c in b.columns if c not in (ID, "month", "airport", "label", "cls", "day")]
        print("ship vs stored v66 proxy max |diff|", float(np.abs(c66 - stored).max()), "; fork B identity columns", bcol, flush=True)
        for c in bcol:
            bv = b.set_index(ID)[c].reindex(out[ID]).to_numpy(dtype=float)
            print("ship vs fork B", c, "max |diff|", float(np.nanmax(np.abs(c66 - bv))), flush=True)


def read() -> None:
    import run_intruder_bound as R
    ctl = pd.read_parquet(OUT / "chain/ship/proxy.parquet")
    res = {"created_at_utc": utc_now(), "control": "ship (pinned rebuild of v66's development chain)", "rule": RULE, "equal_footing": EQUAL,
           "control_rmse": {m: float(np.sqrt(np.mean((g["label"] - g["value"]) ** 2))) for m, g in ctl.groupby("month")}, "variants": {}}
    for variant in ["va", "ext", "both"]:
        p = OUT / "chain" / variant / "proxy.parquet"
        if not p.exists():
            continue
        c = pd.read_parquet(p).set_index(ID).reindex(ctl[ID])
        t = ctl.rename(columns={"value": "ctl"}).assign(cand=c["value"].to_numpy(dtype=float))
        rng = np.random.default_rng(20261004)
        months = {}
        for m in MONTHS:
            s = t[t["month"] == m]
            pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
            ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
            months[m] = {"pct": pct, "lo": lo, "hi": hi, "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))),
                         "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum())}
        eq = float(np.mean([months[m]["pct"] for m in EQUAL]))
        checks = {"1_no_month_worse": all(months[m]["pct"] <= RULE["max_month_worsening_pct"] for m in MONTHS), "2_equal_footing_mean_pct": eq,
                  "2_mean_ok": eq <= -RULE["min_mean_improvement_pct"], "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in EQUAL)}
        checks["go"] = bool(checks["1_no_month_worse"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"])
        res["variants"][variant] = {"months": months, "checks": checks}
        print(variant, {m: (round(v["pct"], 4), round(v["lo"], 3), round(v["hi"], 3)) for m, v in months.items()}, checks, flush=True)
    write_json(OUT / "screen.json", res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--layers")
    ap.add_argument("--check-va", action="store_true")
    ap.add_argument("--score")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args()
    if a.tables:
        tables()
    if a.layers:
        layers(a.layers)
    if a.check_va:
        check_va()
    if a.score:
        score(a.score)
    if a.read:
        read()


if __name__ == "__main__":
    main()
