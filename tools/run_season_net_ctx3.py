"""Seasonal-weight lever (fork S; configs/season_net_design_v1.json): tools/run_adsb_ctx3.py, unchanged, with the training call swapped for
tools/season_net_train.train_weighted and a per-row seasonal weight on the main squared error.

The variant must carry a `season` block {"kernel": "gaussian_circular_month", "a", "sigma_months", "target_month"}; a variant without one is
refused (use tools/run_adsb_ctx3.py). Per fitting row (in the runner's own fitting order, captured from its single Preprocessor.fit call):
    d = min(|m - t|, 12 - |m - t|),  m = the row's UTC movement month (bundle catalog), t = target_month
    w = 1 + a * exp(-d^2 / (2 * sigma^2)),  then w / mean(w) over the fitting rows
The weights by month, the row counts and the code hashes are written to <output>/<fold>/season.json before training starts; run.json is the
runner's own (its variant_sha256 covers the season block). Arguments: exactly tools/run_adsb_ctx3.py's. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TIME  # noqa: E402

CODE = ["tools/run_season_net_ctx3.py", "tools/season_net_train.py", "tools/run_adsb_ctx3.py", "src/prc2026_ctx3/train.py"]


def month_weights(months: np.ndarray, spec: dict) -> tuple[np.ndarray, dict]:
    if spec.get("kernel") != "gaussian_circular_month":
        raise ValueError(f"Unknown season kernel {spec.get('kernel')}.")
    a, sigma, t = float(spec["a"]), float(spec["sigma_months"]), int(spec["target_month"])
    if not (1 <= t <= 12) or sigma <= 0 or a < 0:
        raise ValueError("Bad season settings.")
    d = np.abs(months.astype(int) - t)
    d = np.minimum(d, 12 - d)
    raw = 1.0 + a * np.exp(-(d.astype(float) ** 2) / (2.0 * sigma ** 2))
    w = raw / raw.mean()
    table = {}
    for m in sorted(set(months.tolist())):
        k = months == m
        table[f"{int(m):02d}"] = {"rows": int(k.sum()), "distance": int(d[k][0]), "raw": float(raw[k][0]), "weight": float(w[k][0])}
    return w, table


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("--config", "--variant", "--output", "--fold"):
        parser.add_argument(name, required=True)
    known, _ = parser.parse_known_args()
    import run_adsb_ctx3 as R
    import prc2026_ctx3.train as T
    import torch
    from prc2026_nn.preprocess import Preprocessor
    from season_net_train import train_weighted
    config = read_json(ROOT / known.config)
    variant = R.resolve_variant(config, known.variant)
    spec = variant.get("season")
    if not spec:
        raise ValueError(f"{known.variant} has no season block; use tools/run_adsb_ctx3.py.")
    out = ROOT / known.output / known.fold
    captured: dict = {}

    original_load_bundle = R.load_bundle

    def load_bundle(root, base):
        result = original_load_bundle(root, base)
        captured["catalog"] = result[3]
        return result

    R.load_bundle = load_bundle
    original_fit = Preprocessor.fit.__func__

    def fit(cls, frame, *args, **kwargs):
        if "fit_ids" in captured:
            raise ValueError("Preprocessor.fit was called more than once; the fitting order is ambiguous.")
        captured["fit_ids"] = frame[ID].to_numpy()
        return original_fit(cls, frame, *args, **kwargs)

    Preprocessor.fit = classmethod(fit)

    def train(model, pre, x_num, x_cat, context, target, training, seed, **kwargs):
        ids = captured["fit_ids"]
        if len(ids) != len(target):
            raise ValueError("Captured fitting rows do not align with the target.")
        when = captured["catalog"].set_index(ID)[TIME].reindex(ids)
        if when.isna().any():
            raise ValueError("Fitting rows without a catalog time.")
        months = when.dt.month.to_numpy()
        w, table = month_weights(months, spec)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "season.json", {"created_at_utc": utc_now(), "variant": known.variant, "fold": known.fold, "season": spec,
                                         "fitting_rows": int(len(ids)), "weight_mean": float(w.mean()), "weight_min": float(w.min()),
                                         "weight_max": float(w.max()), "effective_sample_fraction": float(w.mean() ** 2 / np.mean(w ** 2)),
                                         "by_month": table, "code": {p: sha256_file(ROOT / p) for p in CODE}})
        print({"season": spec, "by_month": table}, flush=True)
        row_weight = torch.from_numpy(w.astype(np.float32)).to(target.device)
        return train_weighted(model, pre, x_num, x_cat, context, target, training, seed, row_weight, **kwargs)

    T.train = train
    R.main()
    if not (out / "run.json").exists() or not (out / "season.json").exists():
        raise ValueError("The run did not complete with its season record.")


if __name__ == "__main__":
    main()
