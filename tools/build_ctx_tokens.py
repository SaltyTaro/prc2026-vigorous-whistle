"""Build the context network's token files (src/prc2026_ctx/tokens.py) for every supplied source file; label-free.

One file per source: the eleven 2025 training months, December 2025 and the ranking file. ADS-B surface segments come from the
committee learners' pass-2 match directory (data/external/adsblol/matches, the same tables as their ADS-B features). Output:
<output>/<source name>.npz and manifest.json (input hashes, K, window, counts). Departure labels never enter a token.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026_ctx.tokens import AFTER_S, BEFORE_S, CENTRE_S, RAW_COLUMNS, build_tokens, encode_codes, neighbours, save_file  # noqa: E402
from run_adsb_learner import RANKING_MONTHS, TRAINING_MONTHS, source_file  # noqa: E402

SOURCES = [(m, [m], source_file(m)) for m in [*TRAINING_MONTHS, "2025-12"]] + [("ranking", RANKING_MONTHS, "ranking.parquet")]


def build_one(name: str, months: list[str], raw_name: str, matches_dir: Path, output: Path, k: int) -> dict:
    started = time.time()
    raw = pd.read_parquet(ROOT / "data/raw" / raw_name, columns=RAW_COLUMNS)
    matches = pd.concat([pd.read_parquet(matches_dir / f"matches2_{m}.parquet") for m in months], ignore_index=True)
    tokens = encode_codes(build_tokens(raw, matches))
    dep_ids, self_rows, index = neighbours(tokens, k)
    save_file(output, name, tokens, dep_ids, self_rows, index)
    counts = (index >= 0).sum(axis=1)
    return {"source": raw_name, "raw_sha256": sha256_file(ROOT / "data/raw" / raw_name),
            "matches_sha256": {m: sha256_file(matches_dir / f"matches2_{m}.parquet") for m in months},
            "tokens": int(len(tokens)), "departures": int(len(dep_ids)), "arrivals": int((~tokens["is_dep"]).sum()),
            "matched_departures": int(np.nansum(tokens["adsb_matched"].to_numpy())),
            "neighbours_per_departure": {"mean": float(counts.mean()), "p05": float(np.quantile(counts, 0.05)), "max": int(counts.max()),
                                         "full_k_share": float((counts == k).mean()), "none": int((counts == 0).sum())},
            "npz_sha256": sha256_file(output / f"{name}.npz"), "content_sha256": content_hash(output / f"{name}.npz"), "seconds": round(time.time() - started, 1)}


def content_hash(path: Path) -> str:
    """SHA-256 over every array's name, dtype, shape and bytes (independent of the zip container's metadata)."""
    digest = hashlib.sha256()
    with np.load(path) as z:
        for key in sorted(z.files):
            a = np.ascontiguousarray(z[key])
            digest.update(f"{key}|{a.dtype.str}|{a.shape}".encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ctx/tokens_v1"))
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--only", nargs="*", default=None, help="source names to build (default: all)")
    args = parser.parse_args()
    output, matches_dir = ROOT / args.output, ROOT / args.matches_dir
    files = {}
    for name, months, raw_name in SOURCES:
        if args.only and name not in args.only:
            continue
        files[name] = build_one(name, months, raw_name, matches_dir, output, args.k)
        print(name, {k: files[name][k] for k in ["tokens", "departures", "matched_departures", "neighbours_per_departure", "seconds"]}, flush=True)
    write_json(output / ("manifest.json" if not args.only else f"manifest_{'_'.join(args.only)}.json"),
               {"created_at_utc": utc_now(), "k": args.k, "window_seconds": {"before": BEFORE_S, "after": AFTER_S, "centre": CENTRE_S},
                "code_sha256": {p: sha256_file(ROOT / p) for p in ["src/prc2026_ctx/tokens.py", "tools/build_ctx_tokens.py"]}, "files": files})


if __name__ == "__main__":
    main()
