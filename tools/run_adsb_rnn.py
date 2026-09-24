"""adsb_rnn: a neural-network member for the residual stage -- the one component of the v25 recipe that has never had a network.

The residual stage learns label - v9 (cross-fitted v9, scratch/v24) on matched rows outside LIRF (scope matched_no_lirf) and adds
it to v9; v25 uses the row-wise mean of a month-weighted LightGBM (adj2) and a CatBoost (cat) residual (adsb_v25). This tool fits the
network of src/prc2026_nn (unchanged: piecewise-linear embeddings, MLP, fixed epoch budget, final-epoch weights) on the same target,
rows and cross-fit provenance, on the residual stage's twelve-month leave-one-month-out (tools/run_adsb_v25.py's protocol):

  --held-out M            fit on the in-scope rows of the eleven other 2025 months, predict month M's in-scope rows (zero elsewhere);
                          no held-out label is read during training (no monitor); the held-out scope RMSE is recorded after the fit
  --held-out 2025-10 --design
                          the same fit for October, with October monitored every epoch (the design protocol; October is not a
                          recipe month)

Variant settings: `defaults` deep-merged with `variants.<name>` from the JSON config. `features.set`:
  nb    the residual trees' 33 columns (adsb_v25 residual_model.feature_sets.nb): airport, v9, proxy_s, schedule_interval_s, hour,
        stand, runway and 26 label-free ADS-B departure/neighbour features; categorical airport, stand, runway
  wide  nb plus the timeline bundle's 276 model features (airport/stand/runway taken once, from nb) and the 11 v19 features:
        the committee learners' contract with v9, the clocks and the residual stage's columns. Wide-only columns enter under the
        prefix "w:" (the bundle's numeric `month` would otherwise collide with the residual table's month label)

Output: <output>/<variant>_s<seed>/<M>/{model.pt, preprocessor.*, architecture.json, predictions.parquet (ID, in_scope, residual), run.json}.
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from run_adsb_v11 import scope_mask  # noqa: E402
from run_adsb_v24 import ALL_MONTHS, adsb_features, load_frame, month_table, raw_file  # noqa: E402

SCOPE = "matched_no_lirf"
WIDE = "w:"
DESIGN_MONTH = "2025-10"
CODE_FILES = ["src/prc2026_nn/__init__.py", "src/prc2026_nn/preprocess.py", "src/prc2026_nn/model.py", "src/prc2026_nn/train.py",
              "tools/run_adsb_rnn.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py", "src/prc2026_adsb/features.py"]


def code_receipt() -> dict:
    hashes = {p: sha256_file(ROOT / p) for p in CODE_FILES}
    return {"files": hashes, "sha256": object_hash(hashes)}


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def resolve_variant(config: dict, name: str) -> dict:
    if name not in config["variants"]:
        raise ValueError(f"Undeclared variant {name}.")
    return merge(config["defaults"], config["variants"][name])


def columns_for(config: dict, variant: dict) -> tuple[list[str], list[str], list[str]]:
    """(numeric, categorical, wide-only names) of the variant's feature set."""
    nb, nb_categorical = list(config["residual_stage"]["nb_features"]), list(config["residual_stage"]["categorical"])
    kind = variant["features"]["set"]
    if kind == "nb":
        names, extra, categorical_pool = nb, [], set(nb_categorical)
    elif kind == "wide":
        from prc2026_timeline.features import CATEGORICAL_COLUMNS
        from run_adsb_learner import BUNDLE_NAMES
        extra = [WIDE + c for c in BUNDLE_NAMES if c not in nb] + [WIDE + c for c in config["wide"]["v19_features"] if c not in nb]
        names, categorical_pool = nb + extra, set(nb_categorical) | {WIDE + c for c in CATEGORICAL_COLUMNS}
    else:
        raise ValueError(f"Undeclared feature set {kind}.")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate feature names.")
    return [n for n in names if n not in categorical_pool], [n for n in names if n in categorical_pool], extra


def wide_columns(config: dict, month: str, ids: pd.Series, extra: list[str], bundle: tuple) -> pd.DataFrame:
    """The wide-only columns for the given IDs of one month: timeline bundle features (December from the opened December build) and the
    v19 features; every ID must be covered."""
    manifest, base = bundle
    v19_names = config["wide"]["v19_features"]
    bundle_cols = [c[len(WIDE):] for c in extra if c[len(WIDE):] not in v19_names]
    v19_cols = [c[len(WIDE):] for c in extra if c[len(WIDE):] in v19_names]
    if month == "2025-12":
        december = ROOT / config["wide"]["december_build"]
        meta = read_json(december / "manifest.json")
        if meta["month"] != month or meta["timeline_bundle_id"] != manifest["bundle_id"]:
            raise ValueError("December build does not belong to this timeline bundle.")
        path = december / "features" / raw_file(month)
    else:
        path = base / "features" / raw_file(month)
    frame = pd.DataFrame({ID: ids.to_numpy()})
    features = pd.read_parquet(path, columns=[ID, *bundle_cols]).rename(columns={c: WIDE + c for c in bundle_cols})
    frame = frame.merge(features, on=ID, how="left", validate="one_to_one", indicator=True)
    v19 = pd.read_parquet(ROOT / config["wide"]["v19_dir"] / f"features_v19_{month}.parquet", columns=[ID, *v19_cols]).rename(columns={c: WIDE + c for c in v19_cols})
    frame = frame.merge(v19, on=ID, how="left", validate="one_to_one", indicator="_v19")
    if not (frame["_merge"].eq("both").all() and frame["_v19"].eq("both").all()):
        raise ValueError(f"Wide feature tables do not cover the in-scope rows of {month}.")
    return frame.drop(columns=["_merge", "_v19"])


def month_rows(config: dict, month: str, cache: Path, extra: list[str], bundle: tuple | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all rows of the month: ID and the scope flag, in-scope rows with label, v9 and every feature). Label-free features only."""
    stage = config["residual_stage"]
    table = month_table(load_frame(stage, month, "cross_fit"), adsb_features(month, ROOT / stage["matches_dirs"]["v11"], cache), month)
    inside = scope_mask(table, SCOPE)
    rows = table.loc[inside].reset_index(drop=True)
    if extra:
        rows = rows.merge(wide_columns(config, month, rows[ID], extra, bundle), on=ID, how="left", validate="one_to_one")
    return pd.DataFrame({ID: table[ID].to_numpy(), "in_scope": inside}), rows


def load_bundle_identity() -> tuple:
    from prc2026_timeline.bundle import load_bundle
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, _, _, _ = load_bundle(ROOT, base)
    return manifest, base


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_network(rows: pd.DataFrame, numeric: list[str], categorical: list[str], variant: dict, seed: int, monitor_rows: pd.DataFrame | None = None, log=print):
    """Fit the network on label - v9 of the given in-scope rows; `monitor_rows` only records the per-epoch RMSE (never steers training)."""
    from prc2026_nn.preprocess import Preprocessor
    from prc2026_nn.train import build_model, train
    target = (rows[TARGET] - rows["v9"]).to_numpy(dtype=float)
    clip = variant["preprocess"].get("target_clip")
    fit_target = np.clip(target, -clip, clip) if clip is not None else target
    pre = Preprocessor.fit(rows, numeric, categorical, fit_target, bins=variant["preprocess"]["bins"], min_count=variant["preprocess"]["min_count"])
    x_num, x_cat = pre.transform(rows)
    monitor = None
    if monitor_rows is not None:
        m_num, m_cat = pre.transform(monitor_rows)
        monitor = (m_num, m_cat, (monitor_rows[TARGET] - monitor_rows["v9"]).to_numpy(dtype=float))
    model = build_model(pre, variant["architecture"], seed)
    t0 = time.time()
    history = train(model, pre, x_num, x_cat, pre.scale_target(fit_target), variant["training"], seed, monitor=monitor, log=log)
    return model, pre, history, round(time.time() - t0, 1)


def predict_rows(model, pre, rows: pd.DataFrame) -> np.ndarray:
    from prc2026_nn.train import predict
    values = np.empty(len(rows))
    for start in range(0, len(rows), 200000):
        xn, xc = pre.transform(rows.iloc[start:start + 200000])
        values[start:start + 200000] = predict(model, pre, xn, xc)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--held-out", required=True, choices=ALL_MONTHS)
    parser.add_argument("--design", action="store_true", help="monitor the held-out month every epoch (October only)")
    parser.add_argument("--output", type=Path, required=True, help="learner directory; the held-out month is a subdirectory")
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="code-path test: 30,000 training rows, one epoch; refuses to write under runs/")
    args = parser.parse_args()
    started = time.time()
    if args.design and args.held_out != DESIGN_MONTH:
        raise ValueError("The design protocol reads October only.")
    if args.smoke and "runs" in args.output.parts:
        raise ValueError("A smoke test must write outside runs/.")
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_nn.train import save

    config = read_json(ROOT / args.config)
    if config.get("audit_used"):
        raise ValueError("Configuration does not respect its own contract.")
    stage_parent = config["residual_stage"]["declaration"]
    if sha256_file(ROOT / stage_parent["path"]) != stage_parent["sha256"]:
        raise ValueError("The residual-stage declaration changed.")
    variant = resolve_variant(config, args.variant)
    if args.smoke:
        variant = merge(variant, {"training": {"epochs": 1}})
    out = ROOT / args.output / args.held_out
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    numeric, categorical, extra = columns_for(config, variant)
    bundle = load_bundle_identity() if extra else None
    cache = ROOT / args.cache
    frames = {m: month_rows(config, m, cache, extra, bundle) for m in ALL_MONTHS}
    train_rows = pd.concat([frames[m][1] for m in ALL_MONTHS if m != args.held_out], ignore_index=True)
    if args.smoke:
        train_rows = train_rows.sample(n=30000, random_state=args.seed).reset_index(drop=True)
    held_all, held_rows = frames[args.held_out]
    print({"train_rows": len(train_rows), "held_rows": len(held_rows), "numeric": len(numeric), "categorical": len(categorical), "seconds": round(time.time() - started, 1)}, flush=True)

    model, pre, history, fit_seconds = fit_network(train_rows, numeric, categorical, variant, args.seed, held_rows if args.design else None, log=lambda e: print(e, flush=True))
    save(model, pre, variant["architecture"], out)
    residual = predict_rows(model, pre, held_rows)
    column = pd.Series(residual, index=held_rows[ID].to_numpy()).reindex(held_all[ID]).fillna(0.0).to_numpy()
    frame = pd.DataFrame({ID: held_all[ID].to_numpy(), "in_scope": held_all["in_scope"].to_numpy(), "residual": np.where(held_all["in_scope"], column, 0.0)})
    frame.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    label, v9 = held_rows[TARGET].to_numpy(dtype=float), held_rows["v9"].to_numpy(dtype=float)
    stage = config["residual_stage"]
    record = {"created_at_utc": utc_now(), "stage": "design" if args.design else "development", "held_out": args.held_out, "variant": args.variant,
              "variant_settings": variant, "variant_sha256": object_hash(variant), "seed": args.seed, "config": args.config.as_posix(),
              "config_sha256": sha256_file(ROOT / args.config), "smoke": args.smoke, "scope": SCOPE, "target": "label - cross-fit v9",
              "fitted_rows": int(len(train_rows)), "fitted_months": [m for m in ALL_MONTHS if m != args.held_out],
              "fitted_rows_by_month": {m: int(len(frames[m][1])) for m in ALL_MONTHS if m != args.held_out},
              "held_rows": int(len(held_all)), "held_rows_in_scope": int(len(held_rows)), "held_ids_sha256": identity_hash(held_all[ID]),
              "features": {"numeric": numeric, "categorical": categorical}, "epochs": int(variant["training"]["epochs"]), "history": history,
              "fit_seconds": fit_seconds, "preprocessing": pre.info, "target_mean": pre.target_mean, "target_scale": pre.target_scale,
              "parameters": int(sum(p.numel() for p in model.parameters())), "model_sha256": sha256_file(out / "model.pt"),
              "held_scope_rmse": {"v9": rmse(label, v9), "v9_plus_network": rmse(label, v9 + residual)},
              "inputs": {"code": code_receipt(), "residual_stage_declaration": stage_parent,
                         "scratch": {m: sha256_file(ROOT / stage["cross_fit"]["scratch"] / f"{m}.parquet") for m in ALL_MONTHS},
                         "feature_cache": {m: sha256_file(cache / f"{Path(stage['matches_dirs']['v11']).name}_{m}.parquet") for m in ALL_MONTHS},
                         "timeline_bundle_id": bundle[0]["bundle_id"] if bundle else None, "torch": torch.__version__, "threads": args.threads},
              "audit_used": False}
    if args.design:
        record["best_monitor_epoch"] = int(min(history, key=lambda e: e["monitor_rmse"])["epoch"])
    record["seconds"] = round(time.time() - started, 1)
    record["environment"] = environment(ROOT)
    write_json(out / "run.json", record)
    print({k: record.get(k) for k in ["held_out", "variant", "seed", "fitted_rows", "held_rows_in_scope", "held_scope_rmse", "best_monitor_epoch", "fit_seconds", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
