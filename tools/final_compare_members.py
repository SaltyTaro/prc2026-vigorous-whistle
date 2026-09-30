"""Checks the final-file runner's member layers (members stage, v51m / v52m) on a run of the ranking file against what shipped.

- The members: each ensemble's seed mean against the shipped per-seed ranking predictions (GPU TF32 there, CPU float32 here), and
  the routed combo_all / mask_unmatched against the seed means of the member tables v51 / v52 were exported from.
- The stack: every earlier layer column (v30 .. v46) equals the stack detail from before the members stage (--before), exactly.
- The files: the runner's v51m / v52m integer files against the shipped member files (runs/adsb_ctx3_combo/export_dry/..._v51.parquet,
  runs/adsb_ctx3_mask/export_dry/..._v52.parquet): differing rows, the largest difference, and the member step alone (v51m - v46 and
  v52m - v51m against the shipped files' own steps).
Usage: tools/final_compare_members.py --root R --before OLD_STACK_DETAIL --v51m-version N --v52m-version M --output OUT.json
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
from prc2026_final import members  # noqa: E402

SHIPPED = {"v46": ("runs/submissions/vigorous-whistle_v46.parquet", "8abdd87ad5098d7753927f407ffd7e8cd342aa84a2864392e1b64c0dde015af9"),
           "v51m": ("runs/adsb_ctx3_combo/export_dry/vigorous-whistle_v51.parquet", "bd8f73aa3f9b15efb941de27848693e81298b26b3f4f10159bacb3ceb4f46cd5"),
           "v52m": ("runs/adsb_ctx3_mask/export_dry/vigorous-whistle_v52.parquet", "16e9f33a3dae2f330dca0477274cb6120ad4ef33ae563425561989c4a2e99965")}
EARLIER = ["v30", "v31", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v45", "v46"]


def integers(path: Path, ids: np.ndarray) -> np.ndarray:
    values = pd.read_parquet(path).set_index(ID)[TARGET].reindex(ids)
    if values.isna().any():
        raise ValueError(f"{path} does not cover every row.")
    return values.to_numpy().astype(np.int64)


def spread(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    return {"rows_differing": int((d != 0).sum()), "max_abs": float(np.abs(d).max()), "rms": float(np.sqrt(np.mean(d ** 2))),
            "p99_abs": float(np.quantile(np.abs(d), 0.99))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--v51m-version", type=int, required=True)
    parser.add_argument("--v52m-version", type=int, required=True)
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
              "earlier_layers_unchanged": all(v == 0.0 for v in earlier.values())}

    # members against the shipped per-seed predictions and the exported member tables
    ens = pd.read_parquet(root / "members" / "ensembles.parquet").set_index(ID).reindex(ids)
    frame = pd.read_parquet(root / "members" / "members.parquet").set_index(ID).reindex(ids)
    seeds = {}
    for ensemble, learners in members.learners().items():
        shipped = np.mean([pd.read_parquet(ROOT / learner / "refit12" / "ranking_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
                           for learner in learners], axis=0)
        seeds[ensemble] = spread(ens[ensemble].to_numpy(dtype=float) - shipped)
    report["seed_means_runner_cpu_vs_shipped_gpu_s"] = seeds
    for column, decl in [("combo_all", members.COMBO), ("mask_unmatched", members.MASK)]:
        tables = read_json(ROOT / decl)["member"]["tables"]["ranking"]
        shipped = np.mean([pd.read_parquet(ROOT / t).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float) for t in tables], axis=0)
        runner = frame[column].to_numpy(dtype=float)
        same_nan = np.array_equal(np.isnan(runner), np.isnan(shipped))
        finite = np.isfinite(runner) & np.isfinite(shipped)
        report[f"{column}_vs_exported_member_tables_s"] = {"same_rows_without_value": bool(same_nan), **spread(runner[finite] - shipped[finite])}

    # the integer files
    files = {}
    v46 = integers(ROOT / SHIPPED["v46"][0], ids)
    runner_files = {"v51m": root / "final" / f"vigorous-whistle_v{args.v51m_version}.parquet", "v52m": root / "final" / f"vigorous-whistle_v{args.v52m_version}.parquet"}
    shipped_files = {name: integers(ROOT / SHIPPED[name][0], ids) for name in ["v51m", "v52m"]}
    runs = {name: integers(path, ids) for name, path in runner_files.items()}
    runner46 = np.rint(detail["v46"].to_numpy(dtype=float)).astype(np.int64)
    for name in ["v51m", "v52m"]:
        files[name] = {"runner_file": runner_files[name].as_posix(), "runner_sha256": sha256_file(runner_files[name]), "shipped_file": SHIPPED[name][0],
                       "runner_vs_shipped": spread(runs[name] - shipped_files[name]),
                       "runner_v46_vs_uploaded_v46": spread(runner46 - v46)}
    files["member_step"] = {"v51m_minus_v46": {"runner_rows": int((runs["v51m"] != runner46).sum()), "shipped_rows": int((shipped_files["v51m"] != v46).sum()),
                                               "difference_of_steps": spread((runs["v51m"] - runner46) - (shipped_files["v51m"] - v46))},
                            "v52m_minus_v51m": {"runner_rows": int((runs["v52m"] != runs["v51m"]).sum()), "shipped_rows": int((shipped_files["v52m"] != shipped_files["v51m"]).sum()),
                                                "difference_of_steps": spread((runs["v52m"] - runs["v51m"]) - (shipped_files["v52m"] - shipped_files["v51m"]))}}
    report["files"] = files
    report["stack_record_rows"] = {k: v for k, v in read_json(root / "stages" / "stack.json")["rows"].items() if k in ("v46", "v51m", "v52m")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print({"earlier_layers_unchanged": report["earlier_layers_unchanged"], "seed_means": {k: round(v["rms"], 4) for k, v in seeds.items()}})
    for name in ["v51m", "v52m"]:
        print(name, files[name]["runner_vs_shipped"], "| runner v46 vs uploaded v46:", files[name]["runner_v46_vs_uploaded_v46"]["rows_differing"])
    print("member steps:", files["member_step"])


if __name__ == "__main__":
    main()
