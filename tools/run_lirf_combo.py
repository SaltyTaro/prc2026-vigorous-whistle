"""Scorer of the LIRF matched-row member re-read (configs/lirf_combo_reread_v1.json); written before any candidate was read.

Sets (LIRF non-population rows WITH an ADS-B match only; every other row is unchanged by construction):
  natural 2025-07 / 2025-11 / 2025-12  runs/lomo_dev/rows.parquet rows with lirf_m (v9, lirf_net = ens_ctx_lirf, lirf_tree = ens_v22_lirf_q,
                                       label, day); member = the fold's 16-seed lm25_32 mean (lomo_2025-07, forward, refit's December
                                       predictions), the same files as tools/run_lirf_member.py
Control check (before any candidate is read): each month's lirf_net must equal the mean of the shipped network's saved fold predictions
(16 seeds 301-316) within 0.01 s RMS. --check-control stops after that check and reads no member file.
Values: v34's matched blend with v45's matched gate m600 (src/prc2026_final/lirf_gate.py) on network' = (1 - w) ens_ctx_lirf + w M.
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

DECL = "configs/lirf_combo_reread_v1.json"
MASK_DECL = "configs/adsb_ctx3_lirf_mask_dev_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
GATE = {"T": 600, "lean": {"v9": 0.15, "net": 0.2, "tree": 0.65}}
CAP, DRAWS, SEED = 20000.0, 10000, 20261002
NATURAL = {"2025-07": ("lomo_2025-07", "predictions.parquet"), "2025-11": ("forward", "predictions.parquet"),
           "2025-12": ("refit", "december_predictions.parquet")}
SHIPPED = list(range(301, 317))


def shipped_path(month: str, seed: int) -> Path:
    if month == "2025-07":
        return ROOT / f"runs/lomo_dev/nets/ctx_lirf_s{seed}/lomo_{month}/predictions.parquet"
    if month == "2025-11":
        return ROOT / f"runs/adsb_ctx/ctx_lirf_s{seed}/forward/predictions.parquet"
    return ROOT / f"runs/december/learners/runs__adsb_ctx__ctx_lirf_s{seed}.parquet"


def read(path: Path, ids: np.ndarray) -> np.ndarray:
    v = pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
    if not np.isfinite(v).all():
        raise ValueError(f"{path} does not cover every row of the set.")
    return v


def values(v9: np.ndarray, net: np.ndarray, tree: np.ndarray, member: np.ndarray | None, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Matched-scope value with the m600 gate on network' = (1 - w) net + w member (w 0 or member None: the control)."""
    net2 = net if member is None or w == 0 else (1 - w) * net + w * member
    return lg.gated(v9, net2, tree, "matched", GATE)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-control", action="store_true")
    a = p.parse_args()
    decl = read_json(ROOT / DECL)
    if not a.check_control and not decl.get("declared_at_utc"):
        raise ValueError("The declaration is not recorded.")
    fits = read_json(ROOT / MASK_DECL)["fits"]
    nets_root = ROOT / fits["outputs"].split("<variant>")[0]
    variant = fits["variants"][0]
    rows = pd.read_parquet(ROOT / ROWS, columns=[ID, "month", "day", "airport", "label", "lirf_m", "v9", "lirf_net", "lirf_tree", "v33"])
    month_sse = {m: float(((g["v33"] - g["label"]) ** 2).sum()) for m, g in rows.groupby("month")}
    rows = rows[rows["airport"].eq("LIRF")]
    sets, checks = {}, {}
    for month, (fold, file) in NATURAL.items():
        d = rows[rows["month"].eq(month) & rows["lirf_m"].astype(bool)]
        ids = d[ID].to_numpy()
        shipped = np.mean([read(shipped_path(month, s), ids) for s in SHIPPED], axis=0)
        rms = float(np.sqrt(np.mean((shipped - d["lirf_net"].to_numpy(float)) ** 2)))
        checks[month] = {"rows": int(len(d)), "lirf_net_vs_shipped_runs_rms_s": rms, "passes": rms <= 0.01}
        print(month, checks[month], flush=True)
        if rms > 0.01:
            raise ValueError(f"{month}: the rows' lirf_net is not the shipped network's mean ({rms:.4f} s RMS).")
        if a.check_control:
            continue
        paths = [nets_root / f"{variant}_s{s}" / fold / file for s in fits["seeds"]]
        missing = [q for q in paths if not q.exists()]
        if missing:
            raise ValueError(f"{month}: {len(missing)} member files missing.")
        sets[f"natural:{month}"] = {"v9": d["v9"].to_numpy(float), "net": d["lirf_net"].to_numpy(float), "tree": d["lirf_tree"].to_numpy(float),
                                    "member": np.mean([read(q, ids) for q in paths], axis=0), "labels": d["label"].to_numpy(float),
                                    "days": d["day"].astype(str).to_numpy(), "month_sse_v33": month_sse[month]}
    if a.check_control:
        print("control checks pass; no member file read.")
        return
    results = {}
    for name, cand in decl["candidates"].items():
        w = float(cand["member_weight"])
        res = {}
        for key, s in sets.items():
            base, fire0 = values(s["v9"], s["net"], s["tree"], None, 0.0)
            vals, fire = values(s["v9"], s["net"], s["tree"], s["member"], w)
            y, days = s["labels"], s["days"]
            out = {"rows": int(len(y)), "gate_rows_control": int(fire0.sum()), "gate_rows": int(fire.sum()),
                   "route_sse_change_share_of_month_v33": float((((vals - y) ** 2).sum() - ((base - y) ** 2).sum()) / s["month_sse_v33"])}
            for metric, keep in (("primary", y < CAP), ("unfiltered", np.ones(len(y), dtype=bool))):
                e0, e1 = (base - y) ** 2, (vals - y) ** 2
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
        dec, jul, nov = (r[f"natural:{m}"]["primary"] for m in ("2025-12", "2025-07", "2025-11"))
        ok = dec["relative_change"] <= -0.0025 and dec["interval_95"][1] < 0
        ok &= jul["relative_change"] <= 0 and nov["relative_change"] <= 0
        ok &= all(not (v["primary"]["interval_95"][0] > 0) for v in r.values())
        score[name] = float(np.mean([dec["relative_change"], jul["relative_change"], nov["relative_change"]]))
        if ok:
            eligible.append(name)
        print(name, "eligible" if ok else "-", {k: (round(100 * v["primary"]["relative_change"], 3), [round(100 * x, 3) for x in v["primary"]["interval_95"]],
                                                    v["rows"]) for k, v in r.items()}, flush=True)
    selected = min(eligible, key=lambda n: score[n]) if eligible else None
    standalone = {k: {"control_network_rmse": float(np.sqrt(np.mean((s["net"] - s["labels"])[s["labels"] < CAP] ** 2))),
                      "member_rmse": float(np.sqrt(np.mean((s["member"] - s["labels"])[s["labels"] < CAP] ** 2)))} for k, s in sets.items()}
    report = {"created_at_utc": utc_now(), "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
              "code": {q: sha256_file(ROOT / q) for q in ["tools/run_lirf_combo.py", "src/prc2026_final/lirf_gate.py"]},
              "inputs": {ROWS: sha256_file(ROOT / ROWS)}, "member": {"variant": variant, "seeds": fits["seeds"]},
              "control_checks": checks, "cap_s": CAP, "bootstrap": {"draws": DRAWS, "seed": SEED}, "gate": GATE,
              "standalone_primary_rows": standalone, "results": results, "scores": score, "eligible": eligible, "selected_candidate": selected,
              "rule": decl["rule"]}
    path = ROOT / decl["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    print("standalone:", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in standalone.items()})
    print("selected:", selected, "eligible:", eligible, "->", path)


if __name__ == "__main__":
    main()
