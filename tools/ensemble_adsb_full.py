"""Average the predictions of several full-data ADS-B learners fold by fold (label-free; a plain mean of finite values).

Each learner directory holds seasonal/, forward/ (predictions.parquet: ID, adsb_full) and refit/ (ranking_predictions.parquet).
The output directory gets the same layout plus a run record with the source hashes. Rows where every learner is NaN stay NaN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID

FILES = {"seasonal": "predictions.parquet", "forward": "predictions.parquet", "refit": "ranking_predictions.parquet"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learners", nargs="+", required=True, help="learner run directories, e.g. runs/adsb_full_nolirf runs/adsb_full_cat")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_full_ens"))
    args = parser.parse_args()
    out = ROOT / args.output
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing ensemble: {out}")
    sources = {}
    for fold, name in FILES.items():
        frames = []
        for learner in args.learners:
            path = ROOT / learner / fold / name
            frame = pd.read_parquet(path, columns=[ID, "adsb_full"]).set_index(ID)["adsb_full"]
            frames.append(frame.rename(learner))
            sources.setdefault(learner, {})[fold] = {"predictions_sha256": sha256_file(path), "run_sha256": sha256_file(ROOT / learner / fold / "run.json"),
                                                    "record": {k: v for k, v in read_json(ROOT / learner / fold / "run.json").items() if k in ["best_iteration", "rounds", "score_rmse", "score_rmse_by_month", "fitted_rows"]}}
        wide = pd.concat(frames, axis=1)
        if wide.index.duplicated().any() or any(len(f) != len(wide) for f in frames):
            raise ValueError(f"Learner predictions for {fold} do not share the same identities.")
        mean = wide.mean(axis=1, skipna=True)
        (out / fold).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: wide.index.to_numpy(), "adsb_full": mean.to_numpy()}).to_parquet(out / fold / name, index=False, compression="zstd")
        print(fold, {"rows": int(len(wide)), "all_nan_rows": int(mean.isna().sum()), "learners": len(frames),
                     "pairwise_correlation": float(wide.corr().to_numpy()[np.triu_indices(len(frames), 1)].mean()) if len(frames) > 1 else None})
    write_json(out / "run.json", {"created_at_utc": utc_now(), "learners": args.learners, "rule": "mean of finite learner predictions per row", "sources": sources})


if __name__ == "__main__":
    main()
