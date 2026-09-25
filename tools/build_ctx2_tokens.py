"""Build round-2 context token files with a configurable neighbour window (src/prc2026_ctx2/tokens.py); label-free.

Tokens are the round-1 tokens (prc2026_ctx.tokens.build_tokens / encode_codes, same label contract: departure tokens never carry
block or taxi time); only the neighbour selection takes --k, --before-min, --after-min and --centre-min. Sources, match tables and
output layout as tools/build_ctx_tokens.py (manifest.json with raw/match hashes and array content hashes).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026_ctx.tokens import RAW_COLUMNS, build_tokens, encode_codes, save_file  # noqa: E402
from prc2026_ctx2.tokens import neighbours_window  # noqa: E402
from build_ctx_tokens import SOURCES, content_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--before-min", type=float, required=True)
    parser.add_argument("--after-min", type=float, required=True)
    parser.add_argument("--centre-min", type=float, required=True)
    args = parser.parse_args()
    output, matches_dir = ROOT / args.output, ROOT / args.matches_dir
    window = {"before": 60 * args.before_min, "after": 60 * args.after_min, "centre": 60 * args.centre_min}
    files = {}
    for name, months, raw_name in SOURCES:
        started = time.time()
        raw = pd.read_parquet(ROOT / "data/raw" / raw_name, columns=RAW_COLUMNS)
        matches = pd.concat([pd.read_parquet(matches_dir / f"matches2_{m}.parquet") for m in months], ignore_index=True)
        tokens = encode_codes(build_tokens(raw, matches))
        dep_ids, self_rows, index = neighbours_window(tokens, args.k, window["before"], window["after"], window["centre"])
        save_file(output, name, tokens, dep_ids, self_rows, index)
        counts = (index >= 0).sum(axis=1)
        files[name] = {"source": raw_name, "raw_sha256": sha256_file(ROOT / "data/raw" / raw_name),
                       "matches_sha256": {m: sha256_file(matches_dir / f"matches2_{m}.parquet") for m in months},
                       "tokens": int(len(tokens)), "departures": int(len(dep_ids)),
                       "neighbours_per_departure": {"mean": float(counts.mean()), "p05": float(np.quantile(counts, 0.05)), "max": int(counts.max()),
                                                    "full_k_share": float((counts == args.k).mean()), "none": int((counts == 0).sum())},
                       "npz_sha256": sha256_file(output / f"{name}.npz"), "content_sha256": content_hash(output / f"{name}.npz"), "seconds": round(time.time() - started, 1)}
        print(name, files[name]["neighbours_per_departure"], flush=True)
    write_json(output / "manifest.json", {"created_at_utc": utc_now(), "k": args.k, "window_seconds": window,
                                          "code_sha256": {p: sha256_file(ROOT / p) for p in ["src/prc2026_ctx/tokens.py", "src/prc2026_ctx2/tokens.py", "tools/build_ctx2_tokens.py", "tools/build_ctx_tokens.py"]},
                                          "files": files})


if __name__ == "__main__":
    main()
