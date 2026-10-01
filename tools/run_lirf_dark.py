"""Scorer of the LIRF dark-regime re-read (configs/lirf_dark_reread_v1.json); written before any candidate was read.

Sets are built exactly as tools/run_lirf_member.py builds them (same tables, same member files, same control check of lirf_net against the
shipped network's fold runs). Dark sets (selectable): thinned_july, natural:2025-01. Covered months (natural:2025-07 / -11 / -12) are read
descriptively only: the candidate never applies there.
Values: network' = (1 - w) ens_ctx_lirf + w M; blend s network' + (1 - s) tree (v9 0), floor -12; gate u600 lean (v9 0 / 0.2 / 0.8).
Control (w 0.55, s 0.35/0.60) must reproduce results_v1.json's w55 route RMSE on both dark sets within 0.01 s before any candidate is read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_final import lirf_gate as lg  # noqa: E402
import run_lirf_member as rlm  # noqa: E402

DECL = "configs/lirf_dark_reread_v1.json"
MASK_DECL = "configs/adsb_ctx3_lirf_mask_dev_v1.json"
MASK_RESULTS = "docs/adsb_ctx3_lirf_mask/results_v1.json"
CONTROL = {"member_weight": 0.55, "network_share": 0.35 / 0.60}
LEAN = {"v9": 0.0, "net": 0.2, "tree": 0.8}
DARK = ("thinned_july", "natural:2025-01")
CAP, DRAWS, SEED = rlm.CAP, rlm.DRAWS, rlm.SEED


def build_sets() -> tuple[dict, dict, dict]:
    """The member round's sets, rebuilt with run_lirf_member's own tables and checks."""
    decl = read_json(ROOT / MASK_DECL)
    fits = decl["fits"]
    nets_root = ROOT / fits["outputs"].split("<variant>")[0]
    variant = fits["variants"][0]
    rows = pd.read_parquet(ROOT / rlm.ROWS, columns=[ID, "month", "day", "airport", "label", "lirf_m", "lirf_u", "v9", "lirf_net", "lirf_tree"])
    rows = rows[rows["airport"].eq("LIRF")]
    sets, checks = {}, {}
    inputs = {rlm.ROWS: sha256_file(ROOT / rlm.ROWS)}

    def member(paths: list[Path], ids: np.ndarray) -> np.ndarray:
        missing = [q for q in paths if not q.exists()]
        if missing:
            raise ValueError(f"{len(missing)} member files missing.")
        return np.mean([rlm.read(q, ids) for q in paths], axis=0)

    for month, (fold, file) in rlm.NATURAL.items():
        d = rows[rows["month"].eq(month) & rows["lirf_u"].astype(bool)]
        ids = d[ID].to_numpy()
        shipped = np.mean([rlm.read(rlm.shipped_path(month, s), ids) for s in rlm.SHIPPED], axis=0)
        rms = float(np.sqrt(np.mean((shipped - d["lirf_net"].to_numpy(float)) ** 2)))
        checks[month] = {"rows": int(len(d)), "lirf_net_vs_shipped_runs_rms_s": rms}
        if rms > 0.01:
            raise ValueError(f"{month}: lirf_net is not the shipped network's mean ({rms:.4f} s RMS).")
        sets[f"natural:{month}"] = {"v9": d["v9"].to_numpy(float), "net": d["lirf_net"].to_numpy(float), "tree": d["lirf_tree"].to_numpy(float),
                                    "member": member([nets_root / f"{variant}_s{s}" / fold / file for s in fits["seeds"]], ids),
                                    "labels": d["label"].to_numpy(float), "days": d["day"].astype(str).to_numpy()}
    measure = pd.read_parquet(ROOT / rlm.MEASURE, columns=[ID, "TAXITIME_SEC_mvt", "airport", "class_thinned", "thinned:ens_v22_lirf_q"])
    nets = pd.read_parquet(ROOT / rlm.NETS, columns=[ID, "ens_ctx_lirf:thinned"])
    inputs.update({rlm.MEASURE: sha256_file(ROOT / rlm.MEASURE), rlm.NETS: sha256_file(ROOT / rlm.NETS)})
    july = rows[rows["month"].eq("2025-07") & (rows["lirf_m"] | rows["lirf_u"]).astype(bool)]
    t = july.merge(measure, on=ID, how="inner", validate="one_to_one").merge(nets, on=ID, how="inner", validate="one_to_one")
    if len(t) != len(july) or np.max(np.abs(t["label"] - t["TAXITIME_SEC_mvt"])) > 1e-6:
        raise ValueError("The thinned July tables do not match the July LIRF rows.")
    t = t[t["class_thinned"].eq("unmatched")]
    ids = t[ID].to_numpy()
    sets["thinned_july"] = {"v9": t["v9"].to_numpy(float), "net": t["ens_ctx_lirf:thinned"].to_numpy(float), "tree": t["thinned:ens_v22_lirf_q"].to_numpy(float),
                            "member": member([ROOT / rlm.TRANSFER / f"{variant}_s{s}" / "2025-07_thinned.parquet" for s in fits["seeds"]], ids),
                            "labels": t["label"].to_numpy(float), "days": t["day"].astype(str).to_numpy()}
    return sets, checks, inputs


def values(s: dict, w: float, share: float, T: float | None) -> tuple[np.ndarray, np.ndarray]:
    net2 = (1 - w) * s["net"] + w * s["member"]
    out = lg.blend(s["v9"], net2, s["tree"], {"v9": 0.0, "net": share, "tree": 1.0 - share})
    fire = np.zeros(len(out), dtype=bool) if T is None else (s["tree"] - net2) > T
    if fire.any():
        out[fire] = lg.blend(s["v9"][fire], net2[fire], s["tree"][fire], LEAN)
    return out, fire


def compare(s: dict, base: np.ndarray, new: np.ndarray, fire: np.ndarray) -> dict:
    y, days = s["labels"], s["days"]
    out = {"rows": int(len(y)), "gate_rows": int(fire.sum())}
    for metric, keep in (("primary", y < CAP), ("unfiltered", np.ones(len(y), dtype=bool))):
        e0, e1 = (base - y) ** 2, (new - y) ** 2
        total = float(e0[keep].sum())
        by_day = np.array([float(e1[keep & (days == d)].sum() - e0[keep & (days == d)].sum()) for d in np.unique(days)])
        lo, hi = lg.paired_day_interval(by_day, total, DRAWS, SEED) if total > 0 else (0.0, 0.0)
        out[metric] = {"rows_kept": int(keep.sum()), "rmse_control": float(np.sqrt(e0[keep].mean())), "rmse": float(np.sqrt(e1[keep].mean())),
                       "relative_change": float((e1[keep].sum() - total) / total), "interval_95": [lo, hi],
                       "days_better": int((by_day < 0).sum()), "days": int(len(by_day))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc") or decl["declared_at_utc"].startswith("<"):
        raise ValueError("The declaration is not recorded.")
    sets, checks, inputs = build_sets()
    prior = read_json(ROOT / MASK_RESULTS)["results"]["w55"]
    controls = {k: values(sets[k], CONTROL["member_weight"], CONTROL["network_share"], 600.0) for k in sets}
    control_check = {}
    for k in sets:
        y = sets[k]["labels"]
        keep = y < CAP
        rmse = float(np.sqrt(((controls[k][0] - y)[keep] ** 2).mean()))
        ref = prior[k]["primary"]["rmse"]
        control_check[k] = {"rmse": rmse, "results_v1_w55_rmse": ref, "diff_s": rmse - ref}
        if abs(rmse - ref) > 0.01:
            raise ValueError(f"{k}: the control does not reproduce v53's w55 reading ({rmse:.4f} vs {ref:.4f}).")
    print("control check:", {k: round(v["diff_s"], 6) for k, v in control_check.items()}, flush=True)

    results = {}
    for name, c in decl["candidates"].items():
        res = {}
        for k in sets:
            new, fire = values(sets[k], float(c["member_weight"]), float(c["network_share"]), 600.0)
            res[k] = compare(sets[k], controls[k][0], new, fire)
            res[k]["role"] = "dark (selectable)" if k in DARK else "covered month (descriptive; the candidate never applies)"
        results[name] = res
    descriptive = {"gate": {}, "least_squares_grid": {}}
    for label, T in (("T1200", 1200.0), ("no_gate", None)):
        descriptive["gate"][label] = {k: compare(sets[k], controls[k][0], *values(sets[k], CONTROL["member_weight"], CONTROL["network_share"], T)) for k in sets}
    for k in DARK:
        s = sets[k]
        keep = s["labels"] < CAP
        best = None
        for w in np.round(np.arange(0.0, 1.0001, 0.05), 2):
            for share in np.round(np.arange(0.30, 1.0001, 0.025), 3):
                v, _ = values(s, float(w), float(share), 600.0)
                sse = float(((v - s["labels"])[keep] ** 2).sum())
                if best is None or sse < best[0]:
                    best = (sse, float(w), float(share))
        e0 = float(((controls[k][0] - s["labels"])[keep] ** 2).sum())
        descriptive["least_squares_grid"][k] = {"member_weight": best[1], "network_share": best[2], "relative_change": (best[0] - e0) / e0,
                                                "grid": "w 0..1 step 0.05, s 0.30..1 step 0.025, gate u600"}

    eligible, score = [], {}
    for name, r in results.items():
        tj, jan = r["thinned_july"]["primary"], r["natural:2025-01"]["primary"]
        ok = tj["relative_change"] <= -0.0025 and tj["interval_95"][1] < 0 and jan["relative_change"] <= 0 and not jan["interval_95"][0] > 0
        score[name] = tj["relative_change"] + jan["relative_change"]
        if ok:
            eligible.append(name)
        print(f"{name:10s}", "eligible" if ok else "-       ", {k: (round(100 * v["primary"]["relative_change"], 3), [round(100 * x, 3) for x in v["primary"]["interval_95"]])
                                                             for k, v in r.items()}, flush=True)
    selected = min(eligible, key=lambda n: score[n]) if eligible else None
    report = {"created_at_utc": utc_now(), "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
              "code": {q: sha256_file(ROOT / q) for q in ["tools/run_lirf_dark.py", "tools/run_lirf_member.py", "src/prc2026_final/lirf_gate.py"]},
              "inputs": inputs, "lirf_net_checks": checks, "control_check": control_check, "cap_s": CAP, "bootstrap": {"draws": DRAWS, "seed": SEED},
              "control": CONTROL, "results": results, "descriptive": descriptive, "scores": score, "eligible": eligible, "selected_candidate": selected,
              "rule": decl["rule"]}
    path = ROOT / decl["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    print("gate (descriptive):", {lab: {k: round(100 * v["primary"]["relative_change"], 3) for k, v in d.items()} for lab, d in descriptive["gate"].items()})
    print("least squares (descriptive):", descriptive["least_squares_grid"])
    print("selected:", selected, "eligible:", eligible, "->", path)


if __name__ == "__main__":
    main()
