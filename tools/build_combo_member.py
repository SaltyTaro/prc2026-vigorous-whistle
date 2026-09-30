"""Routed member tables of a network-member round (default: the combination, configs/adsb_ctx3_combo_dev_v1.json), one per seed and leaf.

For every declared seed and leaf: the declaration's member.route names the variant used on rows with adsb_matched = 1 ("matched") and on
the other rows ("unmatched"); a null leaves those rows without a member value (NaN). The combination (no route in its declaration) is
dl32 on matched rows and m25dl32 on the others; the masked round (configs/adsb_ctx3_mask_dev_v1.json) is m25_32 on unmatched rows only.
Written as MVT_ID_mvt, adsb_full to the declared member table. Every network must come from the declared network configuration, fold
and seed, and all predict the same rows. The match flag comes from the label-free ADS-B feature tables the fits read (hash-checked).
No label is read. Nothing is uploaded.
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
COMBO_ROUTE = {"matched": "dl32", "unmatched": "m25dl32"}  # the combination's declaration predates the route field


def matched_flags(cache: Path) -> tuple[pd.Series, dict]:
    """Match flag per row over every table of the cache (twelve months and the ranking rows), and the tables' hashes."""
    paths = sorted(cache.glob("adsb_*.parquet"))
    parts = [pd.read_parquet(p, columns=[ID, "adsb_matched"]) for p in paths]
    table = pd.concat(parts, ignore_index=True)
    if table[ID].duplicated().any() or table["adsb_matched"].isna().any():
        raise ValueError("The ADS-B feature tables hold duplicate or incomplete rows.")
    return table.set_index(ID)["adsb_matched"].eq(1), {p.name: sha256_file(p) for p in paths}


def check_cache(record: dict, hashes: dict, directory: Path) -> None:
    """The flags must come from the very tables the fit read (run.json records their hashes by raw file name)."""
    for raw, sha in record["inputs"]["adsb_cache_sha256"].items():
        name = f"adsb_{raw}"
        if name not in hashes:
            raise ValueError(f"{directory}: the local cache lacks {name}.")
        if hashes[name] != sha:
            raise ValueError(f"{directory}: {name} differs from the table the fit read.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", default=DECLARATION)
    parser.add_argument("--nets", type=Path, help="default: the declaration's fits.outputs directory (runs/<round>/nets)")
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_ctx2/inputs/adsb_features_arcw0"),
                        help="the ADS-B feature tables the fits read (the arcw0 build, twelve months and the ranking rows; hash-checked against every run.json)")
    parser.add_argument("--leaves", nargs="*", help="default: every declared leaf")
    parser.add_argument("--report", type=Path, help="default: <export_report_dir>/member_build.json")
    args = parser.parse_args()
    decl = read_json(ROOT / args.declaration)
    fits = decl["fits"]
    route = decl["member"].get("route", COMBO_ROUTE)
    used = [v for v in (route["matched"], route["unmatched"]) if v]
    if sorted(set(used)) != sorted(fits["variants"]):
        raise ValueError("The member's route does not use exactly the declared variants.")
    nets = args.nets or Path(fits["outputs"].split("/nets/")[0]) / "nets"
    report = args.report or Path(decl["export_report_dir"]) / "member_build.json"
    network_sha = sha256_file(ROOT / fits["network_config"])
    flags, cache_hashes = matched_flags(ROOT / args.adsb_cache)
    leaves = args.leaves or list(fits["folds"])
    built = {}
    for leaf in leaves:
        fold, name = fits["folds"][leaf]["fold"], fits["folds"][leaf]["file"]
        reference = None
        for seed, target in zip(fits["seeds"], decl["member"]["tables"][leaf]):
            if f"/s{seed}/" not in target:
                raise ValueError("The declared member tables do not follow the declared seeds.")
            frames = {}
            for variant in used:
                directory = ROOT / nets / f"{variant}_s{seed}" / fold
                record = read_json(directory / "run.json")
                if (record["config_sha256"] != network_sha or record["variant"] != variant or record["seed"] != seed or record["fold"] != fold
                        or record.get("smoke_limit_rows") is not None or record["inputs"]["device"] != "cuda"):
                    raise ValueError(f"{directory} is not a declared fit.")
                check_cache(record, cache_hashes, directory)
                frames[variant] = pd.read_parquet(directory / name)
            ids = frames[used[0]][ID].to_numpy()
            if any(not np.array_equal(ids, frames[v][ID].to_numpy()) for v in used[1:]):
                raise ValueError(f"{leaf} seed {seed}: the networks predict different rows.")
            if reference is None:
                reference = ids
            elif not np.array_equal(reference, ids):
                raise ValueError(f"{leaf}: seed {seed} predicts different rows than the first seed.")
            matched = flags.reindex(ids)
            if matched.isna().any():
                raise ValueError(f"{leaf}: rows without a match flag.")
            matched = matched.to_numpy(dtype=bool)
            side = {k: (frames[v]["adsb_full"].to_numpy(dtype=float) if v else np.full(len(ids), np.nan)) for k, v in route.items()}
            values = np.where(matched, side["matched"], side["unmatched"])
            out = ROOT / target
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: ids, "adsb_full": values}).to_parquet(out, index=False, compression="zstd")
            built[target] = {"rows": int(len(ids)), "matched_rows": int(matched.sum()), "finite": int(np.isfinite(values).sum()), "sha256": sha256_file(out)}
        print(leaf, "seeds", len(fits["seeds"]), "rows", int(len(reference)), flush=True)
    (ROOT / report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / report, {"created_at_utc": utc_now(), "declaration": args.declaration, "declaration_sha256": sha256_file(ROOT / args.declaration),
                               "route": route, "network_config_sha256": network_sha, "leaves": leaves,
                                    "adsb_cache": {"path": str(args.adsb_cache), "sha256": cache_hashes}, "tables": built,
                                    "tool_sha256": sha256_file(ROOT / "tools/build_combo_member.py"), "audit_used": False,
                                    "note": "routing by the label-free match flag of the tables the fits read; no label read"})


if __name__ == "__main__":
    main()
