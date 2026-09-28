"""Export v42 = the uploaded v40 with the pop_signals mixture delta (configs/pop_signals_v1.json) on the ranking file.

Construction: the declared candidate's classifier bag and normal component are fitted on all twelve 2025 months' population rows
(the round's protocol, the declared seeds); the ranking population rows with a schedule interval at or below the mixture threshold get
value = v40 + 0.75 * (mixture_new - mixture_v9_shipped), floored at -12 s; every other row keeps its uploaded v40 integer. The v9
identity 0.25 * v7 + 0.75 * mixture is re-checked on the shipped population predictions first. The report lists every changed row.
Nothing is uploaded.

  --dry-run DIR [--candidate NAME]                         (any declared candidate; writes outside runs/submissions)
  --version 42 [--candidate NAME --user-override TEXT]     (the selected candidate, or the user's override)
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026_final import pop_signals as ps  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from run_pop_signals import CODE, MONTH_FILES, load_table  # noqa: E402

DECL = "configs/pop_signals_v1.json"
V40 = ("runs/submissions/vigorous-whistle_v40.parquet", "da5b20c328ab6be5fca4a21c37cb7c80210397fba62d8395557a9b84d77d162a", 224.4588)
SHIPPED = "runs/lirf_mixture/export_v9/population_predictions.parquet"
RAW, TEMPLATE = "data/raw/ranking.parquet", "data/raw/submitting.parquet"
FLOOR, WEIGHT = -12.0, 0.75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", default=DECL)
    parser.add_argument("--candidate", help="default: the report's selected candidate")
    parser.add_argument("--version", type=int, default=42)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    parser.add_argument("--models", default="runs/pop_signals", help="directory for the fitted boosters and the detail table")
    parser.add_argument("--reference", default="shipped", choices=["shipped", "control"],
                        help="delta against v9's shipped mixture (the whole new classifier replaces v9's) or against the control bag (the feature effect only)")
    args = parser.parse_args()
    config = read_json(ROOT / args.declaration)
    if not config.get("declared_at_utc"):
        raise ValueError("The declaration is not recorded.")
    report = read_json(ROOT / config["report"])
    selected = report.get("selected_candidate")
    candidate = args.candidate or selected
    if candidate not in config["candidates"]:
        raise ValueError(f"{candidate!r} is not a declared candidate.")
    if args.dry_run is None:
        if candidate != selected and not args.user_override:
            raise ValueError("The rule did not select this candidate; the real export needs --user-override with the user's words.")
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/pop_signals/export_v{args.version}.json"
        model_dir = ROOT / args.models / f"export_v{args.version}"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path, model_dir = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json", out_dir / "models"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    base_file, base_sha, base_rmse = V40
    if sha256_file(ROOT / base_file) != base_sha:
        raise ValueError("The uploaded v40 changed.")
    mixture_config = read_json(ROOT / config["mixture_declaration"])
    threshold = float(mixture_config["mixture"]["schedule_interval_threshold_seconds"])
    spec = config["candidates"][candidate]
    names = ps.candidate_names(spec)

    # fitting rows: all declared months
    train = load_table(config, mixture_config)
    ranking_raw = pd.read_parquet(ROOT / RAW, columns=ps.RAW_COLUMNS)
    rank = ps.population_table(ranking_raw, mixture_config)
    if "cs_rate" in names:
        fill = train["mode"].eq(0).to_numpy(dtype=float)
        smoothing = float(config["callsign_rate"]["smoothing"])
        train["cs_rate"] = ps.callsign_rates(train["flight"], train["lomo_month"], fill, train["flight"], train["lomo_month"], smoothing)
        rank["cs_rate"] = ps.callsign_rates(train["flight"], train["lomo_month"], fill, rank["flight"], None, smoothing)
    classifier = {**mixture_config["classifier"], "params": {**mixture_config["classifier"]["params"], "num_threads": int(config.get("threads", 4))}}
    bag = ps.fit_bag(train, train["mode"].to_numpy(), train["movement_date_utc"].to_numpy(), names, classifier, config["seeds"])
    component = ps.normal_component(train["label"].to_numpy(dtype=float), train["schedule_interval_s"].to_numpy(dtype=float), train["mode"].to_numpy(),
                                    mixture_config)
    mixture_new, p_fill = ps.mixture_for(bag, component, rank)
    # the control (v9's features, the same twelve months and seeds): splits the delta into the refit part and the feature part
    control_bag = ps.fit_bag(train, train["mode"].to_numpy(), train["movement_date_utc"].to_numpy(), ps.candidate_names(config["control"]), classifier, config["seeds"])
    mixture_control, p_fill_control = ps.mixture_for(control_bag, component, rank)

    # the shipped v9 mixture on the same rows, and the v9 identity
    shipped = pd.read_parquet(ROOT / SHIPPED)
    detail = rank[[ID, "schedule_interval_s"]].assign(mixture_new=mixture_new, p_fill_new=p_fill, mixture_control=mixture_control, p_fill_control=p_fill_control).merge(
        shipped[[ID, "schedule_interval_s", "p_fill", "mixture", "v7", "v9"]].rename(columns={"schedule_interval_s": "interval_shipped", "p_fill": "p_fill_v9",
                                                                                              "mixture": "mixture_v9"}), on=ID, how="outer", validate="one_to_one", indicator=True)
    if not detail["_merge"].eq("both").all() or len(detail) != len(shipped):
        raise ValueError("The ranking population differs from v9's shipped population rows.")
    if np.max(np.abs(detail["schedule_interval_s"] - detail["interval_shipped"])) > 1e-6:
        raise ValueError("Schedule intervals differ from the shipped population predictions.")
    in_scope = detail["schedule_interval_s"].to_numpy(dtype=float) <= threshold
    identity = np.max(np.abs(detail["v9"] - (0.25 * detail["v7"] + WEIGHT * detail["mixture_v9"]))[in_scope])
    if identity > 1e-6:
        raise ValueError(f"v9 is not 0.25 * v7 + 0.75 * mixture on the in-scope rows (max {identity}).")

    v40 = pd.read_parquet(ROOT / base_file)
    ids = v40[ID].to_numpy()
    base = v40[TARGET].to_numpy(dtype=float)
    reference = "mixture_v9" if args.reference == "shipped" else "mixture_control"
    delta_table = detail.loc[in_scope, [ID, "mixture_new", "mixture_v9", "mixture_control"]]
    aligned = pd.DataFrame({ID: ids}).merge(delta_table, on=ID, how="left", validate="one_to_one")
    delta = WEIGHT * (aligned["mixture_new"] - aligned[reference]).to_numpy(dtype=float)
    scope_detail = detail.loc[in_scope]
    split = {"reference": args.reference,
             "sum_sq_refit_part_0.75x(control-shipped)": float(np.sum((WEIGHT * (scope_detail["mixture_control"] - scope_detail["mixture_v9"])) ** 2)),
             "sum_sq_feature_part_0.75x(new-control)": float(np.sum((WEIGHT * (scope_detail["mixture_new"] - scope_detail["mixture_control"])) ** 2)),
             "sum_sq_total_0.75x(new-shipped)": float(np.sum((WEIGHT * (scope_detail["mixture_new"] - scope_detail["mixture_v9"])) ** 2)),
             "mean_p_fill": {"v9_shipped": float(scope_detail["p_fill_v9"].mean()), "control": float(scope_detail["p_fill_control"].mean()), "new": float(scope_detail["p_fill_new"].mean())}}
    changed = np.isfinite(delta)
    if changed.sum() != int(in_scope.sum()):
        raise ValueError("Not every in-scope population row is a departure of the file.")
    values = np.where(changed, np.maximum(base + np.nan_to_num(delta), FLOOR), base)

    model_dir.mkdir(parents=True, exist_ok=True)
    for member in bag["members"]:
        member["booster"].save_model(str(model_dir / f"classifier_seed{member['seed']}.txt"), num_iteration=member["rounds"])
    detail.drop(columns="_merge").to_parquet(model_dir / "population_detail.parquet", index=False)
    provenance = {"experiment_id": config["experiment_id"], "candidate": candidate, "dry_run": args.dry_run is not None,
                  "selected_by": ("the declared rule (tools/run_pop_signals.py)" if candidate == selected else f"the user's decision: {args.user_override}"),
                  "recipe": f"v40 (uploaded integers) + {WEIGHT} * (mixture[{candidate}] - mixture[{'v9 shipped' if args.reference == 'shipped' else 'control bag, same months and seeds'}]) "
                            f"on LIRF population rows with interval <= {threshold:.0f} s, floor {FLOOR}",
                  "incumbent": {"submission": Path(base_file).name, "official_rmse_seconds": base_rmse, "sha256": base_sha},
                  "declaration": {args.declaration: {"sha256": sha256_file(ROOT / args.declaration), "object_hash": object_hash(config)},
                                  config["mixture_declaration"]: {"sha256": sha256_file(ROOT / config["mixture_declaration"])}},
                  "report": {config["report"]: {"sha256": sha256_file(ROOT / config["report"])}},
                  "code": {p: sha256_file(ROOT / p) for p in CODE + ["tools/export_pop_signals.py"]},
                  "inputs": {RAW: sha256_file(ROOT / RAW), SHIPPED: sha256_file(ROOT / SHIPPED), **{str(p.relative_to(ROOT)): sha256_file(p) for p in MONTH_FILES.values()}},
                  "fit": {"rows": bag["fitting_rows"], "mode_counts": bag["mode_counts"], "rounds": [m["rounds"] for m in bag["members"]], "seeds": config["seeds"],
                          "feature_names": names, "normal_component": component},
                  "audit_used": False, "external_data": "none for this layer"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / TEMPLATE, output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - base
    if ((change != 0) & ~changed).any():
        raise ValueError("Rows outside the population scope changed.")
    meta = ranking_raw.set_index(ID).reindex(ids[changed])
    det = detail.set_index(ID).reindex(ids[changed])
    text = lambda v: None if pd.isna(v) else str(v)  # noqa: E731
    rows = [{"MVT_ID_mvt": float(i), "takeoff_utc": str(meta.at[i, TIME]), "flight": text(meta.at[i, "FLIGHT_mvt"]), "stand": text(meta.at[i, "STAND_mvt"]),
             "interval_s": float(det.at[i, "schedule_interval_s"]), "p_fill_v9": float(det.at[i, "p_fill_v9"]), "p_fill_new": float(det.at[i, "p_fill_new"]),
             "mixture_v9": float(det.at[i, "mixture_v9"]), "mixture_new": float(det.at[i, "mixture_new"]), "v40": int(base[k]), f"v{args.version}": int(exported[k]),
             "change_s": float(change[k])} for i, k in zip(ids[changed], np.flatnonzero(changed))]
    n = len(ids)
    sse40 = n * base_rmse ** 2
    moved = change[changed]
    summary = {"rows_changed": int((change != 0).sum()), "rows_in_scope": int(changed.sum()), "sum_squared_change": float(np.sum(moved ** 2)),
               "mean_abs_change_s": float(np.mean(np.abs(moved))), "max_abs_change_s": float(np.max(np.abs(moved))),
               "rows_up": int((moved > 0).sum()), "rows_down": int((moved < 0).sum()),
               "rmse_if_every_change_is_fully_right": float(np.sqrt(max(sse40 - np.sum(moved ** 2), 0) / n)),
               "rmse_if_every_change_is_fully_wrong": float(np.sqrt((sse40 + np.sum(moved ** 2)) / n)),
               "lomo_relative_sse_change_of_candidate": report["results"][candidate].get("relative_sse_change"), "delta_split": split}
    write_json(report_path, {"created_at_utc": utc_now(), "version": args.version, "candidate": candidate, "dry_run": args.dry_run is not None,
                             "output": output.as_posix(), "receipt_sha256": receipt.get("sha256"), "models": model_dir.as_posix(), **summary, "rows": rows,
                             "user_override": args.user_override})
    print(summary, receipt.get("sha256"))


if __name__ == "__main__":
    main()
