"""Scorer of the LIRF masked-member round (configs/adsb_ctx3_lirf_mask_dev_v1.json); written before any member fit finished.

Sets (LIRF non-population rows without an ADS-B match only; every other row is unchanged by construction):
  natural 2025-01 / 2025-07 / 2025-11 / 2025-12  runs/lomo_dev/rows.parquet rows with lirf_u (v9, lirf_net = ens_ctx_lirf, lirf_tree =
                                                ens_v22_lirf_q, label, day); member = the fold's 16-seed mean (lomo_2025-01, lomo_2025-07,
                                                forward, refit's December predictions)
  thinned_july                                  July's non-population LIRF rows whose a13-thinned class is unmatched: the seasonal-fold
                                                LIRF tree (runs/coverage_regime/measure) and network (runs/adsb_ctx/regime_check_lirf) on
                                                thinned inputs, v9 / label / day from the rows; member = the seasonal fold's 16 seeds
                                                re-predicted on the same thinned inputs (tools/ctx3_lirf_predict.py, transfer directory)
Control check (before any candidate is read): each natural month's lirf_net must equal the mean of the shipped network's saved fold
predictions (16 seeds 301-316) within 0.01 s RMS.
Values: v34's unmatched blend with v45's gate u600 (src/prc2026_final/lirf_gate.py) on network' = (1 - w) ens_ctx_lirf + w M.
Metric and rule: the declaration's. --smoke reads whatever member seeds exist (never a result, nothing written to docs/).
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

DECL = "configs/adsb_ctx3_lirf_mask_dev_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
MEASURE = "runs/coverage_regime/measure/2025-07/predictions.parquet"
NETS = "runs/adsb_ctx/regime_check_lirf/2025-07.parquet"
TRANSFER = "runs/adsb_ctx3_lirf_mask/transfer"
GATE = {"T": 600, "lean": {"v9": 0.0, "net": 0.2, "tree": 0.8}}
CAP, DRAWS, SEED = 20000.0, 10000, 20261001
NATURAL = {"2025-01": ("lomo_2025-01", "predictions.parquet"), "2025-07": ("lomo_2025-07", "predictions.parquet"),
           "2025-11": ("forward", "predictions.parquet"), "2025-12": ("refit", "december_predictions.parquet")}
SHIPPED = list(range(301, 317))


def shipped_path(month: str, seed: int) -> Path:
    if month in ("2025-01", "2025-07"):
        return ROOT / f"runs/lomo_dev/nets/ctx_lirf_s{seed}/lomo_{month}/predictions.parquet"
    if month == "2025-11":
        return ROOT / f"runs/adsb_ctx/ctx_lirf_s{seed}/forward/predictions.parquet"
    return ROOT / f"runs/december/learners/runs__adsb_ctx__ctx_lirf_s{seed}.parquet"


def read(path: Path, ids: np.ndarray) -> np.ndarray:
    v = pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
    if not np.isfinite(v).all():
        raise ValueError(f"{path} does not cover every row of the set.")
    return v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report")
    a = p.parse_args()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc"):
        raise ValueError("The declaration is not recorded.")
    fits = decl["fits"]
    nets_root = ROOT / fits["outputs"].split("<variant>")[0]
    variant = fits["variants"][0]
    rows = pd.read_parquet(ROOT / ROWS, columns=[ID, "month", "day", "airport", "label", "lirf_m", "lirf_u", "v9", "lirf_net", "lirf_tree"])
    rows = rows[rows["airport"].eq("LIRF")]
    sets, checks, seeds_used, inputs = {}, {}, {}, {ROWS: sha256_file(ROOT / ROWS)}

    def member(paths: list[Path], ids: np.ndarray, key: str) -> np.ndarray:
        have = [q for q in paths if q.exists()]
        if len(have) != len(paths) and not a.smoke:
            raise ValueError(f"{key}: {len(paths) - len(have)} member files missing.")
        if not have:
            raise ValueError(f"{key}: no member file.")
        seeds_used[key] = len(have)
        return np.mean([read(q, ids) for q in have], axis=0)

    for month, (fold, file) in NATURAL.items():
        d = rows[rows["month"].eq(month) & rows["lirf_u"].astype(bool)]
        ids = d[ID].to_numpy()
        shipped = np.mean([read(shipped_path(month, s), ids) for s in SHIPPED], axis=0)
        rms = float(np.sqrt(np.mean((shipped - d["lirf_net"].to_numpy(float)) ** 2)))
        checks[month] = {"rows": int(len(d)), "lirf_net_vs_shipped_runs_rms_s": rms, "passes": rms <= 0.01}
        if rms > 0.01:
            raise ValueError(f"{month}: the rows' lirf_net is not the shipped network's mean ({rms:.4f} s RMS).")
        m = member([nets_root / f"{variant}_s{s}" / fold / file for s in fits["seeds"]], ids, month)
        sets[f"natural:{month}"] = {"v9": d["v9"].to_numpy(float), "net": d["lirf_net"].to_numpy(float), "tree": d["lirf_tree"].to_numpy(float),
                                    "member": m, "labels": d["label"].to_numpy(float), "days": d["day"].astype(str).to_numpy()}
    measure = pd.read_parquet(ROOT / MEASURE, columns=[ID, "TAXITIME_SEC_mvt", "airport", "class_thinned", "thinned:ens_v22_lirf_q"])
    nets = pd.read_parquet(ROOT / NETS, columns=[ID, "ens_ctx_lirf:thinned"])
    inputs.update({MEASURE: sha256_file(ROOT / MEASURE), NETS: sha256_file(ROOT / NETS)})
    july = rows[rows["month"].eq("2025-07") & (rows["lirf_m"] | rows["lirf_u"]).astype(bool)]
    t = july.merge(measure, on=ID, how="inner", validate="one_to_one").merge(nets, on=ID, how="inner", validate="one_to_one")
    if len(t) != len(july) or np.max(np.abs(t["label"] - t["TAXITIME_SEC_mvt"])) > 1e-6:
        raise ValueError("The thinned July tables do not match the July LIRF rows.")
    t = t[t["class_thinned"].eq("unmatched")]
    ids = t[ID].to_numpy()
    m = member([ROOT / TRANSFER / f"{variant}_s{s}" / "2025-07_thinned.parquet" for s in fits["seeds"]], ids, "thinned_july")
    sets["thinned_july"] = {"v9": t["v9"].to_numpy(float), "net": t["ens_ctx_lirf:thinned"].to_numpy(float), "tree": t["thinned:ens_v22_lirf_q"].to_numpy(float),
                            "member": m, "labels": t["label"].to_numpy(float), "days": t["day"].astype(str).to_numpy()}
    results = {}
    for name, cand in decl["candidates"].items():
        w = float(cand["member_weight"])
        res = {}
        for key, s in sets.items():
            base, _ = lg.gated(s["v9"], s["net"], s["tree"], "unmatched", GATE)
            net2 = (1 - w) * s["net"] + w * s["member"]
            values, fire = lg.gated(s["v9"], net2, s["tree"], "unmatched", GATE)
            y, days = s["labels"], s["days"]
            out = {"rows": int(len(y)), "gate_rows_control": int(lg.gated(s["v9"], s["net"], s["tree"], "unmatched", GATE)[1].sum()), "gate_rows": int(fire.sum())}
            for metric, keep in (("primary", y < CAP), ("unfiltered", np.ones(len(y), dtype=bool))):
                e0, e1 = (base - y) ** 2, (values - y) ** 2
                total = float(e0[keep].sum())
                by_day = np.array([float(e1[keep & (days == d)].sum() - e0[keep & (days == d)].sum()) for d in np.unique(days)])
                lo, hi = lg.paired_day_interval(by_day, total, DRAWS, SEED) if total > 0 else (0.0, 0.0)
                out[metric] = {"rows_kept": int(keep.sum()), "rmse_control": float(np.sqrt(e0[keep].mean())), "rmse": float(np.sqrt(e1[keep].mean())),
                               "relative_change": float((e1[keep].sum() - total) / total), "interval_95": [lo, hi],
                               "days_better": int((by_day < 0).sum()), "days": int(len(by_day))}
            res[key] = out
        results[name] = res
    eligible, score = [], {}
    for name in decl["candidates"]:
        r = results[name]
        tj, jan = r["thinned_july"]["primary"], r["natural:2025-01"]["primary"]
        ok = tj["relative_change"] <= -0.005 and tj["interval_95"][1] < 0 and jan["relative_change"] <= 0
        ok &= all(not (r[k]["primary"]["interval_95"][0] > 0) for k in r if k.startswith("natural:"))
        score[name] = tj["relative_change"] + jan["relative_change"]
        if ok:
            eligible.append(name)
        print(name, "eligible" if ok else "-", {k: (round(100 * v["primary"]["relative_change"], 3), [round(100 * x, 3) for x in v["primary"]["interval_95"]], v["rows"])
                                                  for k, v in r.items()}, flush=True)
    selected = min(eligible, key=lambda n: score[n]) if eligible else None
    report = {"created_at_utc": utc_now(), "smoke": a.smoke, "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
              "code": {q: sha256_file(ROOT / q) for q in ["tools/run_lirf_member.py", "src/prc2026_final/lirf_gate.py"]}, "inputs": inputs,
              "control_checks": checks, "member_seeds_used": seeds_used, "cap_s": CAP, "bootstrap": {"draws": DRAWS, "seed": SEED}, "gate": GATE,
              "results": results, "scores": score, "eligible": eligible, "selected_candidate": None if a.smoke else selected, "rule": decl["rule"]}
    path = ROOT / (a.report or ("runs/adsb_ctx3_lirf_mask/smoke.json" if a.smoke else decl["report"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    print("control checks:", {k: round(v["lirf_net_vs_shipped_runs_rms_s"], 5) for k, v in checks.items()}, "seeds:", seeds_used)
    print("selected:", None if a.smoke else selected, "eligible:", eligible, "->", path)


if __name__ == "__main__":
    main()
