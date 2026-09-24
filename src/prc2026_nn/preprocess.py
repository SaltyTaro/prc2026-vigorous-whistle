"""Fitting-row preprocessing: numeric quantile bin edges, categorical vocabularies, target scaling.

Numeric values are kept raw (float32, NaN preserved); the network encodes them piecewise-linearly against the bin edges
estimated here, and gets a separate missing-value flag per feature. Categorical values map to 0 (missing), 1 (rare in the
fitting rows, or unseen) or 2.. (a category with at least `min_count` fitting rows).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_TOKEN = "__MISSING__"
MISSING_INDEX, RARE_INDEX = 0, 1


def _as_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_TOKEN)


@dataclass
class Preprocessor:
    numeric: list[str]
    categorical: list[str]
    edges: np.ndarray          # (F, bins + 1) float64; padded by repeating the last real edge
    n_bins: np.ndarray         # (F,) number of real bins per feature, >= 1
    vocab: dict[str, list[str]]
    target_mean: float
    target_scale: float
    info: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, frame: pd.DataFrame, numeric: list[str], categorical: list[str], target: np.ndarray, bins: int = 32, min_count: int = 10) -> "Preprocessor":
        """Estimate everything from the fitting rows. `target` is the training target (already clipped if the recipe clips)."""
        edges = np.zeros((len(numeric), bins + 1))
        n_bins = np.zeros(len(numeric), dtype=np.int64)
        constant, empty = [], []
        for j, name in enumerate(numeric):
            values = frame[name].to_numpy(dtype=float, na_value=np.nan)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                e = np.array([0.0, 1.0])
                empty.append(name)
            else:
                e = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
                if len(e) == 1:
                    e = np.array([e[0], e[0] + 1.0])
                    constant.append(name)
            n_bins[j] = len(e) - 1
            edges[j, :len(e)] = e
            edges[j, len(e):] = e[-1]
        vocab = {}
        for name in categorical:
            counts = _as_strings(frame[name]).value_counts()
            kept = sorted(str(v) for v, c in counts.items() if c >= min_count and v != MISSING_TOKEN)
            vocab[name] = kept
        target = np.asarray(target, dtype=float)
        mean, scale = float(target.mean()), float(target.std())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Degenerate training target.")
        info = {"bins": bins, "min_count": min_count, "rows": int(len(frame)), "constant_features": constant, "all_missing_features": empty,
                "categories_kept": {k: len(v) for k, v in vocab.items()}}
        return cls(list(numeric), list(categorical), edges, n_bins, vocab, mean, scale, info)

    def cardinalities(self) -> list[int]:
        return [len(self.vocab[name]) + 2 for name in self.categorical]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(N, F) float32 raw numeric values with NaN, and (N, C) int64 category indices."""
        x_num = np.empty((len(frame), len(self.numeric)), dtype=np.float32)
        for j, name in enumerate(self.numeric):
            x_num[:, j] = frame[name].to_numpy(dtype="float32", na_value=np.nan)
        x_num[~np.isfinite(x_num)] = np.nan
        x_cat = np.empty((len(frame), len(self.categorical)), dtype=np.int64)
        for j, name in enumerate(self.categorical):
            values = _as_strings(frame[name])
            codes = pd.Categorical(values.where(values.isin(self.vocab[name])), categories=self.vocab[name]).codes.astype(np.int64)
            index = np.where(codes >= 0, codes + 2, RARE_INDEX)
            index[values.eq(MISSING_TOKEN).to_numpy()] = MISSING_INDEX
            x_cat[:, j] = index
        return x_num, x_cat

    def scale_target(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=float) - self.target_mean) / self.target_scale).astype(np.float32)

    def unscale(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.target_scale + self.target_mean

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "preprocessor.npz", edges=self.edges, n_bins=self.n_bins)
        meta = {"numeric": self.numeric, "categorical": self.categorical, "vocab": self.vocab, "target_mean": self.target_mean,
                "target_scale": self.target_scale, "info": self.info}
        (directory / "preprocessor.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "Preprocessor":
        arrays = np.load(directory / "preprocessor.npz")
        meta = json.loads((directory / "preprocessor.json").read_text(encoding="utf-8"))
        return cls(meta["numeric"], meta["categorical"], arrays["edges"], arrays["n_bins"], meta["vocab"], meta["target_mean"], meta["target_scale"], meta["info"])
