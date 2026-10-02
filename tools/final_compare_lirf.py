"""Checks the final-file runner's LIRF layers (members stage's masked LIRF network, v53m / v54m / v55m) on a run of the ranking file against
what shipped.

- The stack: every earlier column (v30 .. v46, v51m, v52m) equals the stack detail from before the LIRF layers (--before), exactly.
- The member: the lm25_32 seed mean (runner CPU float32) against the shipped per-seed refit12 ranking predictions (GPU TF32), on every row
  and on the LIRF non-population rows the layers read.
- The files: the runner's v53m / v54m / v55m integer files against the shipped model parts (runs/adsb_ctx3_lirf_mask/export_dry/..._v53,
  runs/lirf_dark/export_dry/..._v54, runs/lirf_combo/export_dry/..._v55): differing rows (all, LIRF route rows, the rest), the largest
  difference, and each layer's step alone (runner vN - runner vN-1 against the shipped files' own step; v52m's step base is the shipped
  v52 model part, runs/adsb_ctx3_mask/export_dry/..._v52).
Usage: tools/final_compare_lirf.py --root R --before OLD_STACK_DETAIL --versions 952 953 954 955 --output OUT.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_final import lirf_members  # noqa: E402

SHIPPED = {"v52m": ("runs/adsb_ctx3_mask/export_dry/vigorous-whistle_v52.parquet", "16e9f33a3dae2f330dca0477274cb6120ad4ef33ae563425561989c4a2e99965"),
           "v53m": ("runs/adsb_ctx3_lirf_mask/export_dry/vigorous-whistle_v53.parquet", "1cb22c38a869247db1a26e3f19d7efd0aefb4fd6e958ca3697935b8b18a88218"),
           "v54m": ("runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet", "b0fbbd1902ecbb30cd61ef5cc933a726defcacbdc812efdda59189cd49abb932"),
           "v55m": ("runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet", "266b8c936f2625578e4980f1b11c3a26cc63594a13916510fa3fe86925ec800f")}
EARLIER = ["v30", "v31", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v45", "v46", "v51m", "v52m"]
LAYERS = ["v52m", "v53m", "v54m", "v55m"]


def integers(path: Path, ids: np.ndarray) -> np.ndarray:
    values = pd.read_parquet(path).set_index(ID)[TARGET].reindex(ids)
    if values.isna().any():
        raise ValueError(f"{path} does not cover every row.")
    return values.to_numpy().astype(np.int64)


def spread(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    if not len(d):
        return {"rows": 0}
    return {"rows": int(len(d)), "rows_differing": int((d != 0).sum()), "max_abs": float(np.abs(d).max()), "rms": float(np.sqrt(np.mean(d ** 2))),
            "p99_abs": float(np.quantile(np.abs(d), 0.99))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--versions", type=int, nargs=4, required=True, help="the runner's v52m v53m v54m v55m file versions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    for name, (path, sha) in SHIPPED.items():
        if sha256_file(ROOT / path) != sha:
            raise ValueError(f"{path} changed.")
    detail = pd.read_parquet(root / "stack" / "detail.parquet")
    ids = detail[ID].to_numpy()
    before = pd.read_parquet(args.before).set_index(ID).reindex(ids)
    earlier = {c: float(np.max(np.abs(detail[c].to_numpy(dtype=float) - before[c].to_numpy(dtype=float)))) for c in EARLIER if c in before}
    report = {"created_at_utc": utc_now(), "root": root.as_posix(), "earlier_layers_max_abs_vs_before": earlier,
              "earlier_layers_unchanged": all(v == 0.0 for v in earlier.values()) and len(earlier) == len(EARLIER)}

    scope = detail["scope"].to_numpy()
    components = pd.read_parquet(root / "recipe" / "detail.parquet", columns=[ID, "ens_ctx_lirf", "ens_v22_lirf_q"]).set_index(ID).reindex(ids)
    lirf_route = np.isin(scope, [lirf_members.UNMATCHED, lirf_members.MATCHED])
    ens = pd.read_parquet(root / "members" / "ensembles.parquet").set_index(ID).reindex(ids)
    frame = pd.read_parquet(root / "members" / "members.parquet").set_index(ID).reindex(ids)
    shipped_member = np.mean([pd.read_parquet(ROOT / learner / "refit12" / "ranking_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
                              for learner in lirf_members.learners()], axis=0)
    runner_member = ens["lirf_lm25_32"].to_numpy(dtype=float)
    if not np.array_equal(runner_member, frame["lirf_mask"].to_numpy(dtype=float)):
        raise ValueError("members.parquet's lirf_mask is not the ensemble's seed mean.")
    report["lirf_member_runner_cpu_vs_shipped_gpu_s"] = {"all_rows": spread(runner_member - shipped_member),
                                                         "lirf_nonpop_rows": spread((runner_member - shipped_member)[lirf_route])}

    runner_files = {name: root / "final" / f"vigorous-whistle_v{v}.parquet" for name, v in zip(LAYERS, args.versions)}
    runs = {name: integers(path, ids) for name, path in runner_files.items()}
    shipped = {name: integers(ROOT / SHIPPED[name][0], ids) for name in LAYERS}
    lirf = detail["airport"].eq("LIRF").to_numpy()
    files = {}
    for name in LAYERS:
        d = runs[name] - shipped[name]
        files[name] = {"runner_file": runner_files[name].as_posix(), "runner_sha256": sha256_file(runner_files[name]), "shipped_file": SHIPPED[name][0],
                       "runner_vs_shipped": spread(d), "lirf_nonpop_rows": spread(d[lirf_route]), "other_rows": spread(d[~lirf_route])}
    steps = {}
    for previous, name in zip(LAYERS[:-1], LAYERS[1:]):
        r, s = runs[name] - runs[previous], shipped[name] - shipped[previous]
        steps[f"{name}_minus_{previous}"] = {"runner_rows": int((r != 0).sum()), "shipped_rows": int((s != 0).sum()),
                                             "runner_rows_outside_lirf": int(((r != 0) & ~lirf).sum()), "shipped_rows_outside_lirf": int(((s != 0) & ~lirf).sum()),
                                             "difference_of_steps": spread(r - s)}
    report["files"], report["steps"] = files, steps
    largest = {}
    for name in LAYERS[1:]:
        d = runs[name] - shipped[name]
        new = (d != 0) & lirf_route & ((runs[LAYERS[0]] - shipped[LAYERS[0]]) == 0)
        order = np.argsort(-np.abs(np.where(new, d, 0)))[:5]
        largest[name] = [{"MVT_ID_mvt": float(ids[i]), "month": str(detail["month"].iat[i]), "scope": str(scope[i]), "runner": int(runs[name][i]),
                          "shipped": int(shipped[name][i]), "runner_member": float(runner_member[i]), "shipped_member": float(shipped_member[i]),
                          "ens_ctx_lirf": float(components["ens_ctx_lirf"].iat[i]), "lirf_tree": float(components["ens_v22_lirf_q"].iat[i])}
                         for i in order if new[i]]
    report["largest_new_lirf_differences"] = largest
    stack_rows = read_json(root / "stages" / "stack.json")["rows"]
    report["stack_record_rows"] = {k: v for k, v in stack_rows.items() if k in LAYERS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print({"earlier_layers_unchanged": report["earlier_layers_unchanged"], "member": report["lirf_member_runner_cpu_vs_shipped_gpu_s"]["lirf_nonpop_rows"]})
    for name in LAYERS:
        print(name, files[name]["runner_vs_shipped"], "| lirf:", files[name]["lirf_nonpop_rows"].get("rows_differing"), "max", files[name]["lirf_nonpop_rows"].get("max_abs"))
    print("steps:", steps)


if __name__ == "__main__":
    main()
