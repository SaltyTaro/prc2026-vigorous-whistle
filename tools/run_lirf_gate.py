"""Governing run of configs/lirf_gate_v1.json: the LIRF disagreement gate on the thinned July set and the four natural months.

Evaluation sets: the July 2025 set thinned to July 2026's coverage (LIRF dark; seasonal-fold tree and network re-predictions saved by the
coverage and ctx_lirf regime checks, v9 and days from the LOMO rows) for the unmatched scope, and the natural LOMO/fair rows of January,
July, November and December for both scopes. Population rows (|v33 - v9| <= 1 s) are excluded. Selection follows the declared rule only.
Writes docs/lirf_gate/results_v1.json. Nothing is exported or uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_final import lirf_gate as lg  # noqa: E402

ROWS = "runs/lomo_dev/rows.parquet"
MEASURE = "runs/coverage_regime/measure/2025-07/predictions.parquet"
NETS = "runs/adsb_ctx/regime_check_lirf/2025-07.parquet"
CODE = ["src/prc2026_final/lirf_gate.py", "tools/run_lirf_gate.py"]
CAP = 20000.0


def load_sets() -> dict:
    rows = pd.read_parquet(ROOT / ROWS, columns=[ID, "month", "day", "airport", "label", "matched", "v9", "lirf_net", "lirf_tree", "v33"])
    rows = rows[rows["airport"].eq("LIRF")].copy()
    rows["population"] = (rows["v33"] - rows["v9"]).abs() <= 1.0
    sets = {}
    for month, g in rows.groupby("month"):
        g = g[~g["population"]]
        for scope, mask in (("matched", g["matched"].astype(bool)), ("unmatched", ~g["matched"].astype(bool))):
            d = g[mask]
            if len(d) < 50:  # January has no matched LIRF rows; a scope below 50 rows carries no read
                continue
            sets[f"natural:{month}:{scope}"] = {"v9": d["v9"].to_numpy(float), "net": d["lirf_net"].to_numpy(float), "tree": d["lirf_tree"].to_numpy(float),
                                                "labels": d["label"].to_numpy(float), "days": d["day"].to_numpy(), "scope": scope}
    measure = pd.read_parquet(ROOT / MEASURE, columns=[ID, "TAXITIME_SEC_mvt", "airport", "class_thinned", "thinned:ens_v22_lirf_q"])
    measure = measure[measure["airport"].eq("LIRF")]
    nets = pd.read_parquet(ROOT / NETS, columns=[ID, "ens_ctx_lirf:thinned"])
    july = rows[rows["month"].eq("2025-07") & ~rows["population"]][[ID, "day", "v9", "label"]]
    t = july.merge(measure, on=ID, how="inner", validate="one_to_one").merge(nets, on=ID, how="inner", validate="one_to_one")
    if len(t) != len(july):
        raise ValueError("The thinned July tables do not cover the July LIRF rows.")
    if np.max(np.abs(t["label"] - t["TAXITIME_SEC_mvt"])) > 1e-6:
        raise ValueError("Labels differ between the LOMO rows and the measure table.")
    d = t[t["class_thinned"].eq("unmatched")]
    sets["thinned_july:unmatched"] = {"v9": d["v9"].to_numpy(float), "net": d["ens_ctx_lirf:thinned"].to_numpy(float), "tree": d["thinned:ens_v22_lirf_q"].to_numpy(float),
                                      "labels": d["label"].to_numpy(float), "days": d["day"].to_numpy(), "scope": "unmatched",
                                      "still_matched_rows_excluded": int((~t["class_thinned"].eq("unmatched")).sum())}
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", default="configs/lirf_gate_v1.json")
    parser.add_argument("--report")
    args = parser.parse_args()
    config = read_json(ROOT / args.declaration)
    if not config.get("declared_at_utc"):
        raise ValueError("The declaration is not recorded.")
    sets = load_sets()
    draws, seed = 10000, 20260929
    results = {}
    for name, cand in config["candidates"].items():
        res = {}
        for set_name, s in sets.items():
            spec = cand.get(s["scope"])
            if spec is None:
                continue
            res[set_name] = lg.evaluate(s["v9"], s["net"], s["tree"], s["labels"], s["days"], s["scope"], spec, CAP, draws, seed)
        results[name] = res
    # the rule
    eligible, score = [], {}
    for name, cand in config["candidates"].items():
        ok, total = True, 0.0
        if "unmatched" in cand:
            r = results[name]["thinned_july:unmatched"]["primary"]
            ok &= r["relative_change"] <= -0.01 and r["interval_95"][1] < 0
            for m in ["2025-01", "2025-07", "2025-11", "2025-12"]:
                k = f"natural:{m}:unmatched"
                if k in results[name]:
                    ok &= not (results[name][k]["primary"]["interval_95"][0] > 0)
            total += r["relative_change"]
        if "matched" in cand:
            r = results[name]["natural:2025-07:matched"]["primary"]
            ok &= r["relative_change"] <= -0.005 and r["interval_95"][1] < 0
            for m in ["2025-11", "2025-12"]:
                k = f"natural:{m}:matched"
                if k in results[name]:
                    ok &= not (results[name][k]["primary"]["interval_95"][0] > 0)
            total += r["relative_change"]
        score[name] = total
        if ok:
            eligible.append(name)
        line = {k: (round(v["primary"]["relative_change"] * 100, 2), [round(x * 100, 2) for x in v["primary"]["interval_95"]], v["gated_rows"]) for k, v in results[name].items()}
        print(name, "eligible" if ok else "-", line, flush=True)
    selected = min(eligible, key=lambda n: score[n]) if eligible else None
    report = {"created_at_utc": utc_now(), "declaration": {args.declaration: {"sha256": sha256_file(ROOT / args.declaration), "object_hash": object_hash(config)}},
              "code": {p: sha256_file(ROOT / p) for p in CODE}, "inputs": {p: sha256_file(ROOT / p) for p in [ROWS, MEASURE, NETS]},
              "sets": {k: {"rows": int(len(v["labels"])), "scope": v["scope"], **({"still_matched_rows_excluded": v["still_matched_rows_excluded"]} if "still_matched_rows_excluded" in v else {})} for k, v in sets.items()},
              "cap_s": CAP, "bootstrap": {"draws": draws, "seed": seed}, "results": results, "scores": score, "eligible": eligible, "selected_candidate": selected, "rule": config["rule"]}
    path = ROOT / (args.report or config["report"])
    write_json(path, report)
    print("selected:", selected, "eligible:", eligible, "->", path)


if __name__ == "__main__":
    main()
