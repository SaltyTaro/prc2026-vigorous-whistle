"""Reload every declared network's twelve-month refit and re-predict a sample of ranking rows; report the largest difference per seed.

The same check as step 1 of tools/verify_adsb_nn.py, without the export: used to compare the reload on the fitting platform (Linux
workers) with the reload on the workstation (Windows), both on the exact ADS-B ranking table the networks used. Writes a JSON report.
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from verify_adsb_nn import ranking_sample  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_nn_v1.json"))
    parser.add_argument("--adsb-cache", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from prc2026_nn.train import load, predict
    spec = read_json(ROOT / args.config)["network"]
    records = {l: read_json(ROOT / l / "refit12" / "run.json") for l in spec["learners"]}
    cache_sha = {r["inputs"]["adsb_cache_sha256"]["ranking.parquet"] for r in records.values()}
    if len(cache_sha) != 1:
        raise ValueError("The seeds used different ADS-B ranking tables.")
    ids = pd.read_parquet(ROOT / spec["learners"][0] / "refit12" / "ranking_predictions.parquet")[ID].to_numpy()
    frame = ranking_sample(ids, args.sample, 20260923, spec["variant_settings"]["features"]["extra_features"], ROOT / args.adsb_cache, cache_sha.pop())
    out = {}
    for learner in spec["learners"]:
        run = ROOT / learner / "refit12"
        model, pre = load(run)
        x_num, x_cat = pre.transform(frame)
        again = predict(model, pre, x_num, x_cat)
        saved = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["adsb_full"].reindex(frame[ID]).to_numpy(dtype=float)
        delta = np.abs(again - saved)
        out[learner] = {"max_abs_difference_seconds": float(delta.max()), "mean_abs_difference_seconds": float(delta.mean()), "rows_above_1e-3_seconds": int((delta > 1e-3).sum())}
    report = {"created_at_utc": utc_now(), "platform": platform.platform(), "torch": torch.__version__, "sample_rows": int(len(frame)), "learners": out,
              "max_over_seeds_seconds": max(v["max_abs_difference_seconds"] for v in out.values())}
    write_json(ROOT / args.output, report)
    print(report["platform"], report["max_over_seeds_seconds"], {l.split("_")[-1]: round(v["max_abs_difference_seconds"], 6) for l, v in out.items()})


if __name__ == "__main__":
    main()
