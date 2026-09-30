"""Readings and decision of the thinned member-weight check (configs/member_weights_thin_v1.json): December 2025, network-table level.

From tools/member_thin_predict.py's CPU predictions: nets = 0.25 e24 + 0.25 e24b + 0.5 e32 (seed means), combo_all = dl32 on matched rows
and m25dl32 on the others, mask_unmatched = m25_32 on unmatched rows (each regime routed by its own adsb_matched; LIRF rows never take a
member), T(c, m) as tools/run_member_weights.table. Before any candidate is read: every learner's natural-prediction check passes and the
natural nets table reproduces runs/lomo_dev/rows.parquet's December nets (RMS <= 0.25 s). Decision as declared.
Usage: tools/member_thin_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from nn_design_report import day_interval, rmse  # noqa: E402
from run_adsb_learner import source_file  # noqa: E402
from run_member_weights import table  # noqa: E402
from member_thin_predict import DECLARATION, MONTH, families  # noqa: E402
from ctx3_pos_predict import DECEMBER_BUILD, NATURAL_CACHE, REGIME  # noqa: E402

GAIN = 0.05  # percent of the thinned whole-month table RMSE


def main() -> None:
    decl = read_json(ROOT / DECLARATION)
    receipt = read_json(ROOT / "runs/member_thin/predictions.json")
    if receipt["declaration_sha256"] != sha256_file(ROOT / DECLARATION):
        raise ValueError("Predictions were made under another declaration.")
    fams = families(decl)
    checks, missing = {}, []
    for family, learners in fams.items():
        for learner in learners:
            name = Path(learner).name
            info = receipt["learners"].get(name)
            if info is None or len(info.get("files", {})) != 2:
                missing.append(name)
                continue
            for regime, sha in info["files"].items():
                if sha256_file(ROOT / "runs/member_thin" / family / name / f"{regime}.parquet") != sha:
                    raise ValueError(f"{name} {regime}: prediction file changed.")
            checks[name] = info["check"]
    if missing:
        raise ValueError(f"Learners not predicted: {missing}")
    failed = [k for k, c in checks.items() if not c["passes"]]
    if failed:
        raise ValueError(f"Natural-prediction check failed: {failed}")

    name = source_file(MONTH)
    rows = pd.read_parquet(ROOT / DECEMBER_BUILD / "features" / name, columns=[ID, "airport"])
    ids = rows[ID].to_numpy()
    airport = rows["airport"].to_numpy()
    matched = {"natural": pd.read_parquet(ROOT / NATURAL_CACHE / f"adsb_{name.replace('.parquet', '')}.parquet", columns=[ID, "adsb_matched"]),
               "thinned": pd.read_parquet(ROOT / REGIME[MONTH] / "thinned" / "committee" / f"adsb_{name.replace('.parquet', '')}.parquet", columns=[ID, "adsb_matched"])}
    matched = {r: m.set_index(ID)["adsb_matched"].reindex(ids).eq(1).to_numpy() for r, m in matched.items()}

    def seed_mean(family: str, regime: str) -> np.ndarray:
        return np.mean([pd.read_parquet(ROOT / "runs/member_thin" / family / Path(l).name / f"{regime}.parquet").set_index(ID)["adsb_full"]
                        .reindex(ids).to_numpy(dtype=float) for l in fams[family]], axis=0)

    w = decl["models"]["nets"]["weights"]
    lirf = airport == "LIRF"
    parts = {}
    for regime in ("natural", "thinned"):
        nets = sum(w[f] * seed_mean(f, regime) for f in ("e24", "e24b", "e32"))
        dl, m25dl, m25 = seed_mean("dl32", regime), seed_mean("m25dl32", regime), seed_mean("m25_32", regime)
        mk = matched[regime]
        combo = np.where(lirf, np.nan, np.where(mk, dl, m25dl))
        mask = np.where(lirf | mk, np.nan, m25)
        parts[regime] = (nets, combo, mask, mk)
    dev = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet", columns=[ID, "month", "nets"])
    dev = dev[dev["month"].eq(MONTH)].set_index(ID)["nets"].reindex(ids).to_numpy(dtype=float)
    both = np.isfinite(dev) & np.isfinite(parts["natural"][0])
    nets_check = {"rms_s": float(np.sqrt(np.mean((parts["natural"][0][both] - dev[both]) ** 2))), "rows": int(both.sum())}
    nets_check["passes"] = bool(nets_check["rms_s"] <= 0.25)
    if not nets_check["passes"]:
        raise ValueError(f"Natural nets table does not reproduce the development table: {nets_check}")

    # labels only now
    labels = pd.read_parquet(ROOT / DECEMBER_BUILD / "labels" / name).set_index(ID)[TARGET].reindex(ids).to_numpy(dtype=float)
    times = pd.read_parquet(ROOT / "data/raw" / name, columns=[ID, TIME]).set_index(ID)[TIME].reindex(ids)
    days = pd.to_datetime(times).dt.strftime("%Y-%m-%d").to_numpy()
    if np.isnan(labels).any():
        raise ValueError("December labels missing.")
    non_lirf, eddf = ~lirf, airport == "EDDF"

    def reading(values: np.ndarray, regime: str) -> dict:
        mk = parts[regime][3]
        return {"non_lirf": rmse(labels[non_lirf], values[non_lirf]), "eddf": rmse(labels[eddf], values[eddf]),
                "matched": rmse(labels[non_lirf & mk], values[non_lirf & mk]), "unmatched": rmse(labels[non_lirf & ~mk], values[non_lirf & ~mk])}

    tables = {r: {k: table(parts[r][0], parts[r][1], parts[r][2], s["combo_weight"], s["mask_weight"])[0] for k, s in decl["candidates"].items()}
              for r in ("natural", "thinned")}
    control = "c40_m40"
    out = {}
    for k in decl["candidates"]:
        e = {r: reading(tables[r][k], r) for r in ("natural", "thinned")}
        for r in ("natural", "thinned"):
            c = reading(tables[r][control], r)
            e[r]["change_vs_control_pct"] = {s: 100 * (e[r][s] / c[s] - 1) for s in ("non_lirf", "eddf")}
        if k != control:
            e["thinned_interval_vs_control_s"] = {
                "non_lirf": day_interval(days[non_lirf], labels[non_lirf], tables["thinned"][k][non_lirf], tables["thinned"][control][non_lirf]),
                "eddf": day_interval(days[eddf], labels[eddf], tables["thinned"][k][eddf], tables["thinned"][control][eddf])}
            e["shifts"] = bool(e["thinned"]["change_vs_control_pct"]["non_lirf"] <= -GAIN and e["thinned_interval_vs_control_s"]["non_lirf"][1] < 0)
        out[k] = e
    shifting = [k for k in out if out[k].get("shifts")]
    carried = min(shifting, key=lambda k: out[k]["thinned"]["non_lirf"]) if shifting else None
    nets_only = {r: reading(parts[r][0], r) for r in ("natural", "thinned")}
    report = {"created_at_utc": utc_now(), "declaration_sha256": sha256_file(ROOT / DECLARATION), "learner_checks": checks, "nets_check": nets_check,
              "rows": {"december": int(len(ids)), "non_lirf": int(non_lirf.sum()), "eddf": int(eddf.sum()),
                       "matched_non_lirf": {r: int((non_lirf & parts[r][3]).sum()) for r in parts}},
              "nets_only": nets_only, "candidates": out, "shifting": shifting, "carried_forward": carried,
              "decision": decl["decision"], "tool_sha256": sha256_file(ROOT / "tools/member_thin_report.py"),
              "predictions_receipt_sha256": sha256_file(ROOT / "runs/member_thin/predictions.json"), "labels_read": "December 2025 only", "audit_used": False}
    target = ROOT / decl["paths"]["report"]
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, report)
    worst = max(c["rms_s"] for c in checks.values())
    print(f"checks: {len(checks)} learners pass (max RMS {worst:.3f} s); nets table vs development {nets_check['rms_s']:.3f} s")
    print(f"nets only: natural {nets_only['natural']['non_lirf']:.2f} / EDDF {nets_only['natural']['eddf']:.2f}; thinned {nets_only['thinned']['non_lirf']:.2f} / EDDF {nets_only['thinned']['eddf']:.2f}")
    for k, e in sorted(out.items(), key=lambda kv: kv[1]["thinned"]["non_lirf"]):
        iv = e.get("thinned_interval_vs_control_s", {}).get("non_lirf")
        print(f"{k:8s} natural {e['natural']['non_lirf']:.3f} ({e['natural']['change_vs_control_pct']['non_lirf']:+.3f}%) | thinned {e['thinned']['non_lirf']:.3f} "
              f"({e['thinned']['change_vs_control_pct']['non_lirf']:+.3f}%, EDDF {e['thinned']['change_vs_control_pct']['eddf']:+.3f}%)"
              + (f" interval [{iv[0]:+.3f}, {iv[1]:+.3f}] s" if iv else " control") + (" SHIFTS" if e.get("shifts") else ""))
    print("shifting:", shifting, "carried forward:", carried)


if __name__ == "__main__":
    main()
