"""Routed member tables of the combination (configs/adsb_ctx3_combo_dev_v1.json), one per seed and leaf.

For every declared seed and leaf: the dl32 network's prediction on rows with adsb_matched = 1 and the m25dl32 network's prediction on
the other rows, written as MVT_ID_mvt, adsb_full to the declared member table. Both networks must come from the declared network
configuration, fold and seed, and predict the same rows. The match flag comes from the label-free ADS-B feature tables
(runs/adsb_nn/features), the ones every network reads. No label is read. Nothing is uploaded.
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
from prc2026.features import ID  # noqa: E402

DECLARATION = "configs/adsb_ctx3_combo_dev_v1.json"
MATCHED_VARIANT, OTHER_VARIANT = "dl32", "m25dl32"


def matched_flags(cache: Path) -> pd.Series:
    parts = [pd.read_parquet(p, columns=[ID, "adsb_matched"]) for p in sorted(cache.glob("adsb_*.parquet"))]
    table = pd.concat(parts, ignore_index=True)
    if table[ID].duplicated().any() or table["adsb_matched"].isna().any():
        raise ValueError("The ADS-B feature tables hold duplicate or incomplete rows.")
    return table.set_index(ID)["adsb_matched"].eq(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nets", type=Path, default=Path("runs/adsb_ctx3_combo/nets"))
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features"))
    parser.add_argument("--leaves", nargs="*", help="default: every declared leaf")
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx3_combo/member_build.json"))
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    fits = decl["fits"]
    if fits["variants"] != [MATCHED_VARIANT, OTHER_VARIANT]:
        raise ValueError("The declared variants changed.")
    network_sha = sha256_file(ROOT / fits["network_config"])
    flags = matched_flags(ROOT / args.adsb_cache)
    leaves = args.leaves or list(fits["folds"])
    built = {}
    for leaf in leaves:
        fold, name = fits["folds"][leaf]["fold"], fits["folds"][leaf]["file"]
        reference = None
        for seed, target in zip(fits["seeds"], decl["member"]["tables"][leaf]):
            if f"/s{seed}/" not in target:
                raise ValueError("The declared member tables do not follow the declared seeds.")
            frames = {}
            for variant in (MATCHED_VARIANT, OTHER_VARIANT):
                directory = ROOT / args.nets / f"{variant}_s{seed}" / fold
                record = read_json(directory / "run.json")
                if (record["config_sha256"] != network_sha or record["variant"] != variant or record["seed"] != seed or record["fold"] != fold
                        or record.get("smoke_limit_rows") is not None or record["inputs"]["device"] != "cuda"):
                    raise ValueError(f"{directory} is not a declared fit.")
                frames[variant] = pd.read_parquet(directory / name)
            ids = frames[MATCHED_VARIANT][ID].to_numpy()
            if not np.array_equal(ids, frames[OTHER_VARIANT][ID].to_numpy()):
                raise ValueError(f"{leaf} seed {seed}: the two networks predict different rows.")
            if reference is None:
                reference = ids
            elif not np.array_equal(reference, ids):
                raise ValueError(f"{leaf}: seed {seed} predicts different rows than the first seed.")
            matched = flags.reindex(ids)
            if matched.isna().any():
                raise ValueError(f"{leaf}: rows without a match flag.")
            matched = matched.to_numpy(dtype=bool)
            values = np.where(matched, frames[MATCHED_VARIANT]["adsb_full"].to_numpy(dtype=float), frames[OTHER_VARIANT]["adsb_full"].to_numpy(dtype=float))
            out = ROOT / target
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: ids, "adsb_full": values}).to_parquet(out, index=False, compression="zstd")
            built[target] = {"rows": int(len(ids)), "matched_rows": int(matched.sum()), "finite": int(np.isfinite(values).sum()), "sha256": sha256_file(out)}
        print(leaf, "seeds", len(fits["seeds"]), "rows", int(len(reference)), flush=True)
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, {"created_at_utc": utc_now(), "declaration_sha256": sha256_file(ROOT / DECLARATION), "network_config_sha256": network_sha, "leaves": leaves,
                                    "tables": built, "tool_sha256": sha256_file(ROOT / "tools/build_combo_member.py"), "audit_used": False,
                                    "note": "routing by the label-free match flag; no label read"})


if __name__ == "__main__":
    main()
