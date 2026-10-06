"""Transfer predictions of the LIRF own-points design (configs/lirf_own_design_v1.json) on the workstation CPU, LIRF rows only.

tools/ctx3_lirf_predict.py's readings with two changes: the 15 own-aircraft columns (data/external/adsblol/features_v19own) are joined
to every frame, and January 2025 is read as well. Models (round-1 prc2026_ctx and prc2026_ctx3 alike):
  saved C   runs/adsb_ctx/lirf_design/ctx_lirf_s1..s4/design          (the shipped LIRF network family's design seeds)
  saved M   runs/adsb_ctx3_lirf/design/lm25_32_s21..s24/design        (the masked LIRF member's design seeds)
  new       runs/lirf_own/design/<variant>_s<seed>/design              (the declaration's fits, controls included)
Readings:
  october            the design fold's monitoring rows (LIRF); checked against each run's stop_predictions (RMS at most 0.05 s)
  2025-01_natural    January 2025 with the tables the fits read (LIRF naturally dark)
  2025-07_natural    July 2025, natural tables
  2025-07_thinned    coverage-regime attempt a13: thinned committee / v19 tables and tokens; own columns in the unidentified state
                     (p_has_inb 0, the other 14 NaN) on every row
  2025-12_natural    December 2025 (the twelve-month runs' tables)
Own tables of natural readings: features_v19own/<month>, hash-checked against its manifest; its 11 v19 columns must equal the v19
table the reading uses. No label is read. Output: runs/lirf_own/transfer/<model>/<reading>.parquet (ID, adsb_full) and
runs/lirf_own/transfer/predictions.json (inputs, model hashes, scorer checks, reproduction of runs/adsb_ctx3_lirf/transfer).
Usage: tools/lirf_own_predict.py [--threads 8] [--only NAME ...] [--readings R ...]
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

DECLARATION = "configs/lirf_own_design_v1.json"
NATURAL_CACHE = "runs/adsb_ctx2/inputs/adsb_features_arcw0"
NATURAL_V19 = "data/external/adsblol/features_v19"
OWN_DIR = "data/external/adsblol/features_v19own"
DECEMBER_BUILD = "data/december/d198dea10625b0a75977"
DECEMBER_REFERENCE = "runs/adsb_ctx3_combo/nets/dl32_s601/refit12/run.json"
REGIME = {"2025-07": "runs/coverage_regime/a13_jul/features"}
TOKENS = {"natural": "data/ctx/tokens_v1", "thinned": "data/ctx/tokens_thin_a13/{month}"}
OLD_TRANSFER = "runs/adsb_ctx3_lirf/transfer"
READINGS = ["october", "2025-01_natural", "2025-07_natural", "2025-07_thinned", "2025-12_natural"]
BATCH = 50000


def models(decl: dict) -> list[tuple[str, str, Path]]:
    """(name, kind, run directory): kind ctx1 (round 1) or ctx3."""
    out = [(f"ctx_lirf_s{s}", "ctx1", ROOT / f"runs/adsb_ctx/lirf_design/ctx_lirf_s{s}/design") for s in (1, 2, 3, 4)]
    out += [(f"lm25_32_s{s}", "ctx3", ROOT / f"runs/adsb_ctx3_lirf/design/lm25_32_s{s}/design") for s in (21, 22, 23, 24)]
    for variant, spec in decl["fits"].items():
        kind = "ctx1" if spec["config"].endswith("ctx1_v1.json") else "ctx3"
        out += [(f"new_{variant}_s{s}", kind, ROOT / f"runs/lirf_own/design/{variant}_s{s}/design") for s in spec["seeds"]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--readings", nargs="*", default=READINGS)
    parser.add_argument("--output", type=Path, default=Path("runs/lirf_own/transfer"))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_ctx.tokens import TokenStore
    from prc2026_ctx import train as ctx1
    from prc2026_ctx3 import train as ctx3
    from run_adsb_ctx3 import blank_tokens
    device = torch.device("cpu")
    decl = read_json(ROOT / DECLARATION)
    own_names = list(decl["own_columns"])
    own_manifest = read_json(ROOT / OWN_DIR / "manifest.json")["files"]
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    runs = [m for m in models(decl) if args.only is None or m[0] in args.only]
    missing = [n for n, k, d in runs if not (d / "run.json").exists()]
    if missing:
        raise ValueError(f"Runs not finished or not mirrored: {missing}")
    saved_m = read_json(ROOT / "runs/adsb_ctx3_lirf/design/lm25_32_s21/design/run.json")["inputs"]
    reference = read_json(ROOT / DECEMBER_REFERENCE)["inputs"]

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

    def tables(month: str, regime: str) -> tuple[pd.DataFrame, dict]:
        name = source_file(month).replace(".parquet", "")
        if regime == "natural":
            cdir, vdir = ROOT / NATURAL_CACHE, ROOT / NATURAL_V19
        else:
            cdir, vdir = ROOT / REGIME[month] / "thinned" / "committee", ROOT / REGIME[month] / "thinned" / "v19"
        cpath, vpath = cdir / f"adsb_{name}.parquet", vdir / f"features_v19_{month}.parquet"
        committee, v19 = pd.read_parquet(cpath), pd.read_parquet(vpath)
        if list(committee.columns) != [ID, *ADSB_NAMES] or [c for c in v19.columns if c != ID] != list(FEATURE_NAMES_V19):
            raise ValueError(f"Unexpected table columns for {month} {regime}.")
        hashes = {cpath.as_posix(): sha256_file(cpath), vpath.as_posix(): sha256_file(vpath)}
        if regime == "natural":
            rec = reference if month == "2025-12" else saved_m
            if hashes[cpath.as_posix()] != rec["adsb_cache_sha256"][source_file(month)] or \
                    hashes[vpath.as_posix()] != rec["extra_feature_tables_sha256"][f"features_v19_{month}.parquet"]:
                raise ValueError(f"{month}: natural tables differ from the ones the fits read.")
        frame = committee.merge(v19, on=ID, how="left", validate="one_to_one")
        opath = ROOT / OWN_DIR / f"features_v19_{month}.parquet"
        own = pd.read_parquet(opath)
        hashes[opath.as_posix()] = sha256_file(opath)
        if hashes[opath.as_posix()] != own_manifest[f"features_v19_{month}.parquet"]["sha256"]:
            raise ValueError(f"{month}: own table differs from its manifest.")
        if regime == "natural":
            check = frame[[ID, *FEATURE_NAMES_V19]].merge(own[[ID, *FEATURE_NAMES_V19]], on=ID, how="left", suffixes=("", "_own"), validate="one_to_one")
            for c in FEATURE_NAMES_V19:
                a, b = check[c].to_numpy(dtype=float), check[f"{c}_own"].to_numpy(dtype=float)
                if not np.array_equal(a, b, equal_nan=True):
                    raise ValueError(f"{month}: v19 column {c} differs between the v19 and own tables.")
            frame = frame.merge(own[[ID, *own_names]], on=ID, how="left", validate="one_to_one")
            frame["p_has_inb"] = frame["p_has_inb"].fillna(0)
        else:
            frame["p_has_inb"] = 0.0
            for c in own_names[1:]:
                frame[c] = np.nan
        return frame, hashes

    def month_frames(month: str, wanted: list[str]) -> tuple[dict, dict]:
        rows = bundle_rows(month)
        frames, hashes_by = {}, {}
        regimes = [r for r in ("natural", "thinned") if (month == "2025-10" and r == "natural") or f"{month}_{r}" in wanted]
        for regime in regimes:
            t, hashes = tables(month, regime)
            frame = rows.merge(t, on=ID, how="left", validate="one_to_one")
            frame = frame.loc[frame["airport"].astype(str).eq("LIRF")].reset_index(drop=True)
            if frame["adsb_matched"].isna().any():
                raise ValueError(f"{month} {regime}: ADS-B table does not cover every departure.")
            key = "october" if month == "2025-10" else f"{month}_{regime}"
            frames[key] = (regime, frame)
            hashes_by[key] = hashes
        return frames, hashes_by

    out_root = ROOT / args.output
    receipt_path = out_root / "predictions.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else {"runs": {}, "inputs": {}}
    receipt.update({"updated_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
                    "tokens": {"natural": sha256_file(ROOT / TOKENS["natural"] / "manifest.json"),
                               "thinned_2025-07": sha256_file(ROOT / TOKENS["thinned"].format(month="2025-07") / "manifest_2025-07.json")},
                    "code": {p: sha256_file(ROOT / p) for p in ["tools/lirf_own_predict.py", "tools/ctx3_lirf_predict.py", "src/prc2026_ctx/train.py", "src/prc2026_ctx3/train.py"]}})
    loaded = {}
    for name, kind, run_dir in runs:
        record = read_json(run_dir / "run.json")
        model_hash = sha256_file(run_dir / "model.pt")
        if record.get("model_sha256") and record["model_sha256"] != model_hash:
            raise ValueError(f"{run_dir}: model differs from its record.")
        if kind == "ctx1":
            model, pre, settings = ctx1.load(run_dir)
        else:
            model, pre, settings = ctx3.load(run_dir, device)
        blank = tuple(record.get("variant_settings", {}).get("context", {}).get("blank_tokens", []))
        loaded[name] = (kind, run_dir, model, pre, settings, blank)
        receipt["runs"].setdefault(name, {"files": {}})
        receipt["runs"][name].update({"run_dir": run_dir.relative_to(ROOT).as_posix(), "kind": kind, "model_sha256": model_hash,
                                      "extra_features": record["inputs"]["extra_features"]})
    by_month = {}
    for r in args.readings:
        by_month.setdefault("2025-10" if r == "october" else r.split("_")[0], []).append(r)
    for month, wanted in by_month.items():
        started = time.time()
        frames, hashes_by = month_frames(month, wanted)
        receipt["inputs"].update(hashes_by)
        stores = {}
        for name, (kind, run_dir, model, pre, settings, blank) in loaded.items():
            target = out_root / name
            target.mkdir(parents=True, exist_ok=True)
            info = receipt["runs"][name]
            for key, (regime, frame) in frames.items():
                stop = None
                if key == "october":
                    stop = pd.read_parquet(run_dir / "stop_predictions.parquet")
                    stop = stop.loc[stop[ID].isin(frame[ID])].reset_index(drop=True)
                    frame = frame.set_index(ID).reindex(stop[ID]).reset_index()
                    if frame["adsb_matched"].isna().any():
                        raise ValueError("October rows missing.")
                skey = (month, regime, blank)
                if skey not in stores:
                    host = TokenStore(ROOT / TOKENS[regime].format(month=month), [month])
                    blank_tokens(host, list(blank))
                    stores[skey] = (host, ctx3.Store.build(host, device))
                host, store = stores[skey]
                values = np.empty(len(frame))
                for start in range(0, len(frame), BATCH):
                    chunk = frame.iloc[start:start + BATCH]
                    xn, xc = pre.transform(chunk)
                    rows_, nb = host.lookup(chunk[ID].to_numpy())
                    if kind == "ctx1":
                        values[start:start + BATCH] = ctx1.predict(model, pre, xn, xc, ctx1.Context.build(host, rows_, nb, settings))
                    else:
                        values[start:start + BATCH] = ctx3.predict(model, pre, torch.from_numpy(xn), torch.from_numpy(xc), ctx3.Context.build(store, rows_, nb, settings))
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
                if not name.startswith("new_"):
                    old_name = name if kind == "ctx3" else f"baseline/{name}"
                    old = ROOT / OLD_TRANSFER / old_name / f"{key}.parquet"
                    if old.exists():
                        o = pd.read_parquet(old).set_index(ID)["adsb_full"].reindex(frame[ID]).to_numpy(dtype=float)
                        d = values - o
                        rms = float(np.sqrt(np.nanmean(d ** 2)))
                        info.setdefault("reproduces_old_transfer", {})[key] = {"rms_s": rms, "rows_missing": int(np.isnan(o).sum()), "passes": bool(rms <= 0.05 and not np.isnan(o).any())}
                        print(name, key, "old transfer", info["reproduces_old_transfer"][key], flush=True)
        write_json(receipt_path, receipt)
        print(month, "done in", round(time.time() - started, 1), "s", {k: len(v[1]) for k, v in frames.items()}, flush=True)
        del frames, stores


if __name__ == "__main__":
    main()
