"""adsb_rnn twelve-month refit of one residual-network seed (used only if the selection needs it; queued behind the development months).

The declared network's variant and seed (configs/adsb_rnn_v1.json residual_networks) fitted exactly as in development
(tools/run_adsb_rnn.py's functions, imported unchanged) on the in-scope rows of all twelve 2025 months (label - cross-fit v9,
matched_no_lirf), predicting the ranking's in-scope rows from the saved label-free ranking table (runs/adsb_rnn/ranking_nb.parquet,
tools/rnn_ranking_table.py; its hash is checked against its receipt). Rows outside the scope get residual 0.
Output: <learner>/refit12/{model files, ranking_predictions.parquet (ID, in_scope, residual), run.json}. Nothing is uploaded.
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
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from run_adsb_rnn import (WIDE, code_receipt, columns_for, fit_network, load_bundle_identity, month_rows, predict_rows,  # noqa: E402
                          resolve_variant)
from run_adsb_v24 import ALL_MONTHS  # noqa: E402

RANKING_MONTHS = ["2026-01", "2026-07"]


def ranking_rows(config: dict, table_path: Path, extra: list[str], bundle: tuple | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ID and scope flag of every ranking departure, the in-scope rows with every feature)."""
    receipt = read_json(table_path.with_suffix(".json"))
    if sha256_file(table_path) != receipt["table_sha256"]:
        raise ValueError("The saved ranking table changed.")
    table = pd.read_parquet(table_path)
    rows = table.loc[table["in_scope"]].reset_index(drop=True)
    if extra:
        _, base = bundle
        v19_names = config["wide"]["v19_features"]
        bundle_cols = [c[len(WIDE):] for c in extra if c[len(WIDE):] not in v19_names]
        v19_cols = [c[len(WIDE):] for c in extra if c[len(WIDE):] in v19_names]
        features = pd.read_parquet(base / "features" / "ranking.parquet", columns=[ID, *bundle_cols]).rename(columns={c: WIDE + c for c in bundle_cols})
        v19 = pd.concat([pd.read_parquet(ROOT / config["wide"]["v19_dir"] / f"features_v19_{m}.parquet", columns=[ID, *v19_cols]) for m in RANKING_MONTHS],
                        ignore_index=True).rename(columns={c: WIDE + c for c in v19_cols})
        rows = rows.merge(features, on=ID, how="left", validate="one_to_one", indicator=True).merge(v19, on=ID, how="left", validate="one_to_one", indicator="_v19")
        if not (rows["_merge"].eq("both").all() and rows["_v19"].eq("both").all()):
            raise ValueError("Wide feature tables do not cover the ranking's in-scope rows.")
        rows = rows.drop(columns=["_merge", "_v19"])
    return table[[ID, "in_scope"]], rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_rnn_v1.json"))
    parser.add_argument("--network", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ranking-table", type=Path, default=Path("runs/adsb_rnn/ranking_nb.parquet"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_nn.train import save

    declaration = read_json(ROOT / args.declaration)
    spec = declaration["residual_networks"][args.network]
    if args.seed not in spec["seeds"]:
        raise ValueError("Undeclared seed.")
    config = read_json(ROOT / spec["config"])
    if sha256_file(ROOT / spec["config"]) != spec["config_sha256"] or code_receipt()["sha256"] != spec["code_sha256"]:
        raise ValueError("The network configuration or code changed since the declaration.")
    variant = resolve_variant(config, spec["variant"])
    if object_hash(variant) != spec["variant_sha256"]:
        raise ValueError("The variant changed since the declaration.")
    learner = spec["learners"][spec["seeds"].index(args.seed)]
    out = ROOT / learner / "refit12"
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    numeric, categorical, extra = columns_for(config, variant)
    bundle = load_bundle_identity() if extra else None
    cache = ROOT / args.cache
    frames = {m: month_rows(config, m, cache, extra, bundle) for m in ALL_MONTHS}
    train_rows = pd.concat([frames[m][1] for m in ALL_MONTHS], ignore_index=True)
    ranking_all, ranking_in = ranking_rows(config, ROOT / args.ranking_table, extra, bundle)
    print({"train_rows": len(train_rows), "ranking_rows_in_scope": len(ranking_in), "seconds": round(time.time() - started, 1)}, flush=True)
    model, pre, history, fit_seconds = fit_network(train_rows, numeric, categorical, variant, args.seed, None, log=lambda e: print(e, flush=True))
    save(model, pre, variant["architecture"], out)
    residual = predict_rows(model, pre, ranking_in)
    column = pd.Series(residual, index=ranking_in[ID].to_numpy()).reindex(ranking_all[ID]).fillna(0.0).to_numpy()
    frame = pd.DataFrame({ID: ranking_all[ID].to_numpy(), "in_scope": ranking_all["in_scope"].to_numpy(), "residual": np.where(ranking_all["in_scope"], column, 0.0)})
    frame.to_parquet(out / "ranking_predictions.parquet", index=False, compression="zstd")
    record = {"created_at_utc": utc_now(), "stage": "refit12", "network": args.network, "variant": spec["variant"], "variant_settings": variant,
              "variant_sha256": spec["variant_sha256"], "seed": args.seed, "config": spec["config"], "config_sha256": spec["config_sha256"],
              "declaration": args.declaration.as_posix(), "declaration_sha256": sha256_file(ROOT / args.declaration), "smoke": False,
              "scope": config["residual_stage"]["scope"], "target": "label - cross-fit v9", "fitted_rows": int(len(train_rows)), "fitted_months": ALL_MONTHS,
              "fitted_rows_by_month": {m: int(len(frames[m][1])) for m in ALL_MONTHS}, "ranking_rows": int(len(frame)), "ranking_rows_in_scope": int(len(ranking_in)),
              "ranking_ids_sha256": identity_hash(frame[ID]), "ranking_table_sha256": sha256_file(ROOT / args.ranking_table),
              "features": {"numeric": numeric, "categorical": categorical}, "epochs": int(variant["training"]["epochs"]), "history": history,
              "fit_seconds": fit_seconds, "preprocessing": pre.info, "target_mean": pre.target_mean, "target_scale": pre.target_scale,
              "parameters": int(sum(p.numel() for p in model.parameters())), "model_sha256": sha256_file(out / "model.pt"),
              "prediction_quantiles": {str(q): float(np.quantile(residual, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]},
              "inputs": {"code": code_receipt(), "refit_tool": sha256_file(ROOT / "tools/refit_adsb_rnn.py"), "torch": torch.__version__, "threads": args.threads},
              "audit_used": False}
    record["seconds"] = round(time.time() - started, 1)
    record["environment"] = environment(ROOT)
    write_json(out / "run.json", record)
    print({k: record.get(k) for k in ["network", "seed", "fitted_rows", "ranking_rows_in_scope", "prediction_quantiles", "fit_seconds", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
