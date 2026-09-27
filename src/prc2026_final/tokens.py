"""S5: the context networks' token split for a movement file (tools/build_ctx_tokens.py::build_one on an arbitrary file).

One token per movement of the file (departures carry no label or block time; arrivals keep in-block and taxi-in as the ranking file
does), ADS-B segments from the learners' pass-2 matches of the file's months, K = 128 neighbours within the frozen window. Written as
<root>/tokens/final.npz; the networks read it through prc2026_ctx.tokens.TokenStore(<root>/tokens, ["final"]).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import read_json
from prc2026_ctx.tokens import RAW_COLUMNS, build_tokens, encode_codes, neighbours, save_file

from .common import ROOT

sys.path.insert(0, str(ROOT / "tools"))
from build_ctx_tokens import content_hash  # noqa: E402

NAME = "final"


def build(directory: Path, movements: pd.DataFrame, matches: pd.DataFrame) -> dict:
    k = int(read_json(ROOT / "data/ctx/tokens_v1/manifest.json")["k"])
    tokens = encode_codes(build_tokens(movements[RAW_COLUMNS], matches))
    dep_ids, self_rows, index = neighbours(tokens, k)
    save_file(directory, NAME, tokens, dep_ids, self_rows, index)
    counts = (index >= 0).sum(axis=1)
    return {"k": k, "tokens": int(len(tokens)), "departures": int(len(dep_ids)), "arrivals": int((~tokens["is_dep"]).sum()),
            "matched_departures": int(np.nansum(tokens["adsb_matched"].to_numpy())),
            "neighbours_per_departure": {"mean": float(counts.mean()), "full_k_share": float((counts == k).mean()), "none": int((counts == 0).sum())},
            "content_sha256": content_hash(directory / f"{NAME}.npz")}


def frozen_ranking_content() -> str:
    return read_json(ROOT / "data/ctx/tokens_v1/manifest.json")["files"]["ranking"]["content_sha256"]
