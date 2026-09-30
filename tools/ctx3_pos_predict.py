"""Predictions of the positional batch's saved models (configs/adsb_ctx3_pos_v1.json): the scorer check and reading R3's inputs.

For every declared run (runs/adsb_ctx3_pos/design/<variant>_s<seed>/design), on the workstation CPU, with the run's token blanking
(context.blank_tokens) re-applied:
  october              the design fold's monitoring rows (the ids of the run's stop_predictions.parquet), with the tables the fit read;
                       compared with stop_predictions.parquet (the declared scorer check: RMS difference at most 0.05 s)
  <month>_natural      every departure of July / December 2025 with the tables the fits read (July: hash-checked against each run's
                       record; December: against the combination round's refit12 record) and the natural tokens (data/ctx/tokens_v1)
  <month>_natural_cov  the same with the coverage study's own natural tables (the code path of its thinned tables; a check)
  <month>_thinned      coverage-regime attempt a13: the thinned committee and v19 tables and the thinned token file
                       (data/ctx/tokens_thin_a13/<month>, tools/build_ctx_tokens.py on the thinned pass-2 matches)
Output: runs/adsb_ctx3_pos/transfer/<variant>_s<seed>/<reading>.parquet (ID, adsb_full) and transfer/predictions.json. No label is read.
Usage: tools/ctx3_pos_predict.py [--threads 12] [--runs R] [--only VARIANT_SEED ...]
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
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_adsb.features_v19 import FEATURE_NAMES_V19  # noqa: E402
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, source_file  # noqa: E402
from run_adsb_ctx3 import blank_tokens  # noqa: E402

DECLARATION = "configs/adsb_ctx3_pos_v1.json"
NATURAL_CACHE = "runs/adsb_ctx2/inputs/adsb_features_arcw0"   # the tables the fits read (hash-checked below)
NATURAL_V19 = "data/external/adsblol/features_v19"
DECEMBER_BUILD = "data/december/d198dea10625b0a75977"
DECEMBER_REFERENCE = "runs/adsb_ctx3_combo/nets/dl32_s601/refit12/run.json"  # a twelve-month run that read December's tables
REGIME = {"2025-07": "runs/coverage_regime/a13_jul/features", "2025-12": "runs/coverage_regime/a13_dec/features"}
TOKENS = {"natural": "data/ctx/tokens_v1", "thinned": "data/ctx/tokens_thin_a13/{month}"}
BATCH = 50000


def runs_of(decl: dict, runs_root: Path) -> list[tuple[str, int, Path]]:
    return [(v, s, runs_root / f"{v}_s{s}" / "design") for v in decl["runs"] for s in decl["runs"][v]]


def table(month: str, committee_dir: Path, v19_dir: Path) -> tuple[pd.DataFrame, dict]:
    name = source_file(month).replace(".parquet", "")
    committee_path, v19_path = committee_dir / f"adsb_{name}.parquet", v19_dir / f"features_v19_{month}.parquet"
    committee, v19 = pd.read_parquet(committee_path), pd.read_parquet(v19_path)
    if list(committee.columns) != [ID, *ADSB_NAMES] or [c for c in v19.columns if c != ID] != list(FEATURE_NAMES_V19):
        raise ValueError(f"Unexpected table columns for {month}.")
    return committee.merge(v19, on=ID, how="left", validate="one_to_one"), {committee_path.as_posix(): sha256_file(committee_path),
                                                                             v19_path.as_posix(): sha256_file(v19_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx3_pos/design"))
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_ctx3_pos/transfer"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_ctx.tokens import TokenStore
    from prc2026_ctx3.train import Context, Store, load, predict
    device = torch.device("cpu")
    decl = read_json(ROOT / DECLARATION)
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    runs = [r for r in runs_of(decl, ROOT / args.runs) if args.only is None or f"{r[0]}_s{r[1]}" in args.only]
    missing = [f"{v}_s{s}" for v, s, d in runs if not (d / "run.json").exists()]
    if missing:
        raise ValueError(f"Runs not finished or not mirrored: {missing}")
    first = read_json(runs[0][2] / "run.json")["inputs"]
    reference = read_json(ROOT / DECEMBER_REFERENCE)["inputs"]

    # frames: bundle features + ADS-B / v19 tables per reading
    def bundle_rows(month: str) -> pd.DataFrame:
        if month == "2025-12":
            meta = read_json(ROOT / DECEMBER_BUILD / "manifest.json")
            if meta["build_id"] != reference["december_build_id"] or meta["timeline_bundle_id"] != manifest["bundle_id"]:
                raise ValueError("December build is not the one the twelve-month runs read.")
            features = pd.read_parquet(ROOT / DECEMBER_BUILD / "features" / source_file(month), columns=[ID, *BUNDLE_NAMES])
            if identity_hash(features[ID]) != meta["ids_sha256"]:
                raise ValueError("December identities changed.")
            return features
        ids = catalog.loc[catalog["source_file"].eq(source_file(month)), ID]
        features = pd.read_parquet(base / "features" / source_file(month), columns=[ID, *BUNDLE_NAMES])
        features = features.loc[features[ID].isin(ids)].reset_index(drop=True)
        if len(features) != len(ids):
            raise ValueError(f"{month}: bundle rows do not match the catalog.")
        return features

    recorded = {"2025-07": (first["adsb_cache_sha256"], first["extra_feature_tables_sha256"]), "2025-10": (first["adsb_cache_sha256"], first["extra_feature_tables_sha256"]),
                "2025-12": (reference["adsb_cache_sha256"], reference["extra_feature_tables_sha256"])}

    def month_frames(month: str) -> tuple[dict, dict]:
        """{reading: (token regime, frame)} of one month, and the table hashes."""
        rows = bundle_rows(month)
        regimes = {"natural": (ROOT / NATURAL_CACHE, ROOT / NATURAL_V19)}
        if month in REGIME:
            regimes["natural_cov"] = (ROOT / REGIME[month] / "natural" / "committee", ROOT / REGIME[month] / "natural" / "v19")
            regimes["thinned"] = (ROOT / REGIME[month] / "thinned" / "committee", ROOT / REGIME[month] / "thinned" / "v19")
        frames, hashes_by = {}, {}
        for regime, (cdir, vdir) in regimes.items():
            t, hashes = table(month, cdir, vdir)
            if regime == "natural":
                cache_hashes, v19_hashes = recorded[month]
                name = source_file(month)
                if hashes[(cdir / f"adsb_{name.replace('.parquet', '')}.parquet").as_posix()] != cache_hashes[name] or                         hashes[(vdir / f"features_v19_{month}.parquet").as_posix()] != v19_hashes[f"features_v19_{month}.parquet"]:
                    raise ValueError(f"{month}: natural tables differ from the ones the fits read.")
            frame = rows.merge(t, on=ID, how="left", validate="one_to_one")
            if frame["adsb_matched"].isna().any():
                raise ValueError(f"{month} {regime}: ADS-B table does not cover every departure.")
            key = "october" if month == "2025-10" else f"{month}_{regime}"
            frames[key] = ("thinned" if regime == "thinned" else "natural", frame)
            hashes_by[key] = hashes
        return frames, hashes_by

    out_root = ROOT / args.output
    receipt = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "inputs": {},
               "tokens": {"natural": sha256_file(ROOT / TOKENS["natural"] / "manifest.json"),
                          **{f"thinned_{m}": sha256_file(ROOT / TOKENS["thinned"].format(month=m) / f"manifest_{m}.json") for m in REGIME}},
               "code": {p: sha256_file(ROOT / p) for p in ["tools/ctx3_pos_predict.py", "tools/run_adsb_ctx3.py", "src/prc2026_ctx3/train.py"]},
               "runs": {}}
    if (out_root / "predictions.json").exists():
        receipt["runs"] = read_json(out_root / "predictions.json").get("runs", {})
    records = {}
    for variant, seed, run_dir in runs:
        record = read_json(run_dir / "run.json")
        if sha256_file(run_dir / "model.pt") != record["model_sha256"]:
            raise ValueError(f"{run_dir}: model differs from its record.")
        records[f"{variant}_s{seed}"] = (run_dir, record)
        receipt["runs"].setdefault(f"{variant}_s{seed}", {"model_sha256": record["model_sha256"],
                                                          "blank_tokens": record["variant_settings"]["context"].get("blank_tokens", []), "files": {}})
    for month in ["2025-10", "2025-07", "2025-12"]:
        started = time.time()
        frames, hashes_by = month_frames(month)
        receipt["inputs"].update(hashes_by)
        stores = {}
        for name, (run_dir, record) in records.items():
            blank = tuple(record["variant_settings"]["context"].get("blank_tokens", []))
            model, pre, settings = load(run_dir, device)
            target = out_root / name
            target.mkdir(parents=True, exist_ok=True)
            info = receipt["runs"][name]
            for key, (token_regime, frame) in frames.items():
                stop = None
                if key == "october":
                    stop = pd.read_parquet(run_dir / "stop_predictions.parquet")
                    frame = frame.set_index(ID).reindex(stop[ID]).reset_index()
                    if frame["adsb_matched"].isna().any():
                        raise ValueError("October rows missing.")
                if (month, token_regime, blank) not in stores:
                    host = TokenStore(ROOT / TOKENS[token_regime].format(month=month), [month])
                    blank_tokens(host, list(blank))
                    stores[(month, token_regime, blank)] = (host, Store.build(host, device))
                host, store = stores[(month, token_regime, blank)]
                values = np.empty(len(frame))
                for start in range(0, len(frame), BATCH):
                    chunk = frame.iloc[start:start + BATCH]
                    xn, xc = pre.transform(chunk)
                    values[start:start + BATCH] = predict(model, pre, torch.from_numpy(xn), torch.from_numpy(xc),
                                                          Context.build(store, *host.lookup(chunk[ID].to_numpy()), settings))
                if not np.isfinite(values).all():
                    raise ValueError(f"{name} {key}: non-finite predictions.")
                path = target / f"{key}.parquet"
                pd.DataFrame({ID: frame[ID].to_numpy(), "adsb_full": values}).to_parquet(path, index=False, compression="zstd")
                info["files"][key] = sha256_file(path)
                if stop is not None:
                    d = values - stop["adsb_full"].to_numpy(dtype=float)
                    rms = float(np.sqrt(np.mean(d ** 2)))
                    info["scorer_check"] = {"rms_s": rms, "max_abs_s": float(np.max(np.abs(d))), "rows": int(len(d)), "passes": bool(rms <= 0.05)}
                    print(name, "scorer check", info["scorer_check"], flush=True)
        write_json(out_root / "predictions.json", receipt)
        print(month, "done in", round(time.time() - started, 1), "s", {k: len(v[1]) for k, v in frames.items()}, flush=True)
        del frames, stores


if __name__ == "__main__":
    main()
