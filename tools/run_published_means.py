"""Declared round and export of configs/published_means_v1.json: a gated level shift from EUROCONTROL's published airport-month
taxi-out aggregates (the user's decision of 29 September 2026; source, terms and the organiser's words are in the declaration).

  --score                                   development read, scored months' cells, acceptance -> docs/published_means/results_v1.json
  --export --version 47 [--dry-run DIR]     v46 + the accepted procedure's shifts (needs the results file and an accepted procedure)
Nothing is uploaded.
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
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_final import share_gate  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from screen_published_means import AIRPORTS, CAP, MONTHS, WORKBOOK, label_means, published  # noqa: E402

DECLARATION = "configs/published_means_v1.json"
REPORT = "docs/published_means/results_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
V46 = ("runs/submissions/vigorous-whistle_v46.parquet", "8abdd87ad5098d7753927f407ffd7e8cd342aa84a2864392e1b64c0dde015af9", 223.9339)
FLOOR, SEED, SLOPE_RANGE, MAX_SIGMA, GATE, RANGE_MARGIN = -12.0, 20260929, 0.03, 8.0, 3.0, 0.02


def fit_mapping(table: pd.DataFrame, airport: str, exclude: str | None) -> dict:
    """D(u) of one airport from its 2025 months except `exclude`, with its leave-one-month-out sd and the largest u it saw."""
    g = table[(table["airport"] == airport) & (table["month"] != exclude)]
    x, y = g["u"].to_numpy(dtype=float), (g["label_mean"] - g["P"]).to_numpy(dtype=float)
    slope = bool(x.max() - x.min() > SLOPE_RANGE)

    def solve(xx, yy):
        if slope:
            b = np.polyfit(xx, yy, 1)
            return lambda q: float(np.polyval(b, q))
        m = float(yy.mean())
        return lambda q: m
    loo = [solve(np.delete(x, i), np.delete(y, i))(x[i]) - y[i] for i in range(len(x))]
    return {"function": solve(x, y), "slope": slope, "sd_map": float(np.std(loo)), "u_max": float(x.max()), "months": int(len(g))}


def shift_for(level: float, bias: float, sigma: float, u: float, u_max: float) -> tuple[float, dict]:
    """The declared gate and shrinkage for one cell: (shift to subtract from the predictions, record)."""
    effective = level - bias
    inside = bool(u <= u_max + RANGE_MARGIN)
    eligible = bool(inside and sigma <= MAX_SIGMA and abs(effective) >= GATE * sigma)
    shift = effective * max(0.0, 1.0 - sigma ** 2 / effective ** 2) if eligible and effective != 0 else 0.0
    return shift, {"level_estimate_s": level, "method_bias_s": bias, "sigma_s": sigma, "effective_level_s": effective, "u": u, "u_inside_range": inside,
                   "eligible": eligible, "shift_s": shift}


def development_stack(rows: pd.DataFrame) -> dict:
    from run_callsign_fix import offsets, scope_table
    from run_queue_trust import offsets_all, segments
    from run_share_gate_reread import full_stack
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    seg = segments(MONTHS)
    k = read_json(ROOT / "docs/disagreement/tail_stretch_v2_results.json")["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    share, gate = share_gate.candidate_spec(read_json(ROOT / "configs/thinned_regime_round_v1.json"), "n85_nd1200")
    out = {}
    for month in MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        mix, _ = share_gate.mix(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), share, gate)
        out[month] = (f, full_stack(f, mix, cs, offsets(), seg, offsets_all(), k, gate_spec))
    return out


def levels(f: pd.DataFrame, control: np.ndarray, table: pd.DataFrame, pub: pd.DataFrame, month: str) -> dict:
    """Per airport of a development month: estimated and true level error of the stack (mapping without that month)."""
    y, airport = f["label"].to_numpy(dtype=float), f["airport"].to_numpy()
    inside = (control > 0) & (control <= CAP)
    out = {}
    for a in AIRPORTS:
        m = (airport == a) & inside
        q = pub[(pub["airport"] == a) & (pub["month"] == month)].iloc[0]
        mp = fit_mapping(table, a, month)
        ordinary = m & (y > 0) & (y <= CAP)
        out[a] = {"estimate": float(control[m].mean() - (q["P"] + mp["function"](q["u"]))), "true": float(control[ordinary].mean() - y[ordinary].mean()),
                  "sd_map": mp["sd_map"], "u": float(q["u"]), "u_max": mp["u_max"], "rows": int(m.sum())}
    return out


def method(level_by_month: dict, airport: str, exclude: str | None) -> tuple[float, float]:
    d = np.array([level_by_month[m][airport]["estimate"] - level_by_month[m][airport]["true"] for m in MONTHS if m != exclude])
    return float(d.mean()), float(np.sqrt(np.mean((d - d.mean()) ** 2)))


def score() -> dict:
    decl = read_json(ROOT / DECLARATION)
    if sha256_file(ROOT / WORKBOOK) != decl["external_data"]["sha256"]:
        raise ValueError("The workbook differs from the declared file.")
    pub = published()
    table = label_means().merge(pub, on=["airport", "month"], how="inner", validate="one_to_one")
    stacks = development_stack(pd.read_parquet(ROOT / ROWS))
    level_by_month = {m: levels(f, c, table, pub, m) for m, (f, c) in stacks.items()}
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "workbook_sha256": sha256_file(ROOT / WORKBOOK),
              "rows_sha256": sha256_file(ROOT / ROWS), "audit_used": False, "development": {}, "scored_months": {}}
    for month, (f, control) in stacks.items():
        y, days, airport, floor = f["label"].to_numpy(dtype=float), f["day"].to_numpy(), f["airport"].to_numpy(), f["floor"].to_numpy(dtype=float)
        values, cells = control.copy(), {}
        for a in AIRPORTS:
            e = level_by_month[month][a]
            bias, noise = method(level_by_month, a, month)
            shift, record = shift_for(e["estimate"], bias, max(e["sd_map"], noise), e["u"], e["u_max"])
            m = (airport == a) & (control > 0) & (control <= CAP)
            values[m] = np.maximum(control[m] - shift, floor[m])
            cells[a] = {**record, "true_level_error_s": e["true"], "rows": e["rows"]}
        report["development"][month] = {"control_rmse": rmse(y, control), "rmse": rmse(y, values), "change_pct": 100 * (rmse(y, values) / rmse(y, control) - 1),
                                        "paired_days": paired_days(days, y, values, control, SEED)["rmse_difference_95pct_interval_seconds"],
                                        "eligible_cells": [a for a, c in cells.items() if c["eligible"]], "cells": cells}
        print(month, round(report["development"][month]["change_pct"], 4), [round(v, 3) for v in report["development"][month]["paired_days"]],
              {a: (round(c["shift_s"], 1), round(c["true_level_error_s"], 1)) for a, c in cells.items() if c["eligible"]}, flush=True)
    changes = [report["development"][m]["change_pct"] for m in decl["development_read"]["equal_footing"]]
    report["accepted"] = bool(all(c <= 0 for c in changes) and float(np.mean(changes)) < 0)
    report["mean_change_pct_equal_footing"] = float(np.mean(changes))

    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].assign(month=lambda d: d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m"))
    v46 = pd.read_parquet(ROOT / V46[0]).merge(raw[[ID, "ADEP_mvt", "month"]], on=ID, validate="one_to_one")
    stake = 0.0
    for (a, month), g in v46.groupby(["ADEP_mvt", "month"]):
        q = pub[(pub["airport"] == a) & (pub["month"] == month)].iloc[0]
        mp = fit_mapping(table, a, None)
        p = g[TARGET].to_numpy(dtype=float)
        p = p[(p > 0) & (p <= CAP)]
        bias, noise = method(level_by_month, a, None)
        shift, record = shift_for(float(p.mean() - (q["P"] + mp["function"](q["u"]))), bias, max(mp["sd_map"], noise), float(q["u"]), mp["u_max"])
        stake += len(p) * (2 * shift * record["effective_level_s"] - shift ** 2) if record["eligible"] else 0.0
        report["scored_months"][f"{a} {month}"] = {**record, "rows": int(len(p)), "published_mean_s": float(q["P"]), "sd_map_s": mp["sd_map"], "method_noise_s": noise}
    n = len(v46)
    report["scored_months_summary"] = {"eligible_cells": [k for k, c in report["scored_months"].items() if c["eligible"]], "squared_error_expected_if_estimates_hold": stake,
                                       "rmse_expected_if_estimates_hold": float(np.sqrt((n * V46[2] ** 2 - stake) / n))}
    report["code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_published_means.py", "tools/screen_published_means.py", "tools/run_share_gate_reread.py"]}
    write_json(ROOT / REPORT, report)
    print({"accepted": report["accepted"], "mean_change_pct_equal_footing": round(report["mean_change_pct_equal_footing"], 4)})
    print({k: round(c["shift_s"], 1) for k, c in report["scored_months"].items() if c["eligible"]}, report["scored_months_summary"])
    return report


def export(version: int, dry_run: Path | None) -> None:
    decl, report = read_json(ROOT / DECLARATION), read_json(ROOT / REPORT)
    if report["declaration_sha256"] != object_hash(decl) or report["workbook_sha256"] != sha256_file(ROOT / WORKBOOK):
        raise ValueError("The results are not for this declaration and workbook.")
    if not report["accepted"]:
        raise ValueError("The development read did not accept the procedure.")
    if sha256_file(ROOT / V46[0]) != V46[1]:
        raise ValueError("The uploaded v46 changed.")
    out_dir = (ROOT / "runs/submissions") if dry_run is None else dry_run.resolve()
    if dry_run is not None and (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
        raise ValueError("A dry run never writes under runs/submissions.")
    output = out_dir / f"vigorous-whistle_v{version}.parquet"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    base = pd.read_parquet(ROOT / V46[0])
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].assign(month=lambda d: d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")).set_index(ID).reindex(base[ID])
    airport, month = raw["ADEP_mvt"].to_numpy(), raw["month"].to_numpy()
    old = base[TARGET].to_numpy(dtype=float)
    new, touched, cells = old.copy(), np.zeros(len(old), dtype=bool), {}
    for key, c in report["scored_months"].items():
        if not c["eligible"]:
            continue
        a, m = key.split(" ")
        k = (airport == a) & (month == m) & (old > 0) & (old <= CAP)
        if int(k.sum()) != c["rows"]:
            raise ValueError(f"{key}: the rows differ from the governing run's.")
        new[k] = np.maximum(old[k] - c["shift_s"], FLOOR)
        touched |= k
        cells[key] = {"rows": int(k.sum()), "shift_s": c["shift_s"]}
    provenance = {"experiment_id": decl["experiment_id"], "dry_run": dry_run is not None, "selected_by": f"the declared procedure, used on the user's decision: {decl['user_decision']['words']}",
                  "recipe": "v46 with a level shift on the eligible airport-months' predictions in (0, 7200] s, from EUROCONTROL's published airport-month taxi-out aggregates; floor -12 s",
                  "incumbent": {"submission": Path(V46[0]).name, "official_rmse_seconds": V46[2], "sha256": V46[1]}, "cells": cells,
                  "external_data": decl["external_data"], "declaration": {DECLARATION: {"sha256": sha256_file(ROOT / DECLARATION), "object_hash": object_hash(decl)}},
                  "results": {REPORT: sha256_file(ROOT / REPORT)}, "code": {p: sha256_file(ROOT / p) for p in ["tools/run_published_means.py", "tools/screen_published_means.py"]},
                  "official_scores_used": "none", "audit_used": False, "floor_seconds": FLOOR}
    receipt = export_submission(pd.DataFrame({ID: base[ID].to_numpy(), "prediction": new}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(base[ID]).to_numpy().astype(float)
    change = exported - old
    if ((change != 0) & ~touched).any():
        raise ValueError("A row outside the eligible cells changed.")
    n = len(old)
    record = {"created_at_utc": utc_now(), "version": version, "dry_run": dry_run is not None, "output": output.as_posix(), "sha256": sha256_file(output), "receipt": receipt, "cells": cells,
              "rows_changed": int((change != 0).sum()), "mean_change_s": float(change[change != 0].mean()), "max_abs_change_s": float(np.abs(change).max()),
              "sum_squared_change": float((change ** 2).sum()), "rmse_expected_if_estimates_hold": report["scored_months_summary"]["rmse_expected_if_estimates_hold"],
              "rmse_if_every_move_is_noise": float(np.sqrt((n * V46[2] ** 2 + (change ** 2).sum()) / n))}
    write_json((ROOT / f"docs/published_means/export_v{version}.json") if dry_run is None else out_dir / f"export_v{version}.json", record)
    print({k: v for k, v in record.items() if k not in ("receipt",)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--version", type=int, default=47)
    parser.add_argument("--dry-run", type=Path)
    args = parser.parse_args()
    if args.score:
        if (ROOT / REPORT).exists():
            raise ValueError("Preserve the original results.")
        score()
    elif args.export:
        export(args.version, args.dry_run)
    else:
        raise SystemExit(__doc__)
