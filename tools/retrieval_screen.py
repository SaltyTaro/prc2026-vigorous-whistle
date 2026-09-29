"""Retrieval screen: a go/no-go reading before a retrieval committee member is built. Scratch diagnostic, October only.

Question: do the OUT-OF-SAMPLE residuals of the most similar departures of other months say anything about a departure's own
residual? A retrieval network can only add to the committee if they do.

One job = one held-out month M of the seasonal fold's fit partition (2025-02..06, 08, 09):
  1. fit the committee network (configs/adsb_nn_network_v1.json variant nnb: recipe, 314-column contract and data path of
     tools/run_adsb_nn.py, imported unchanged) on the other six fit months, LIRF excluded as for every committee learner;
  2. predict and embed (the backbone's output, the layer in front of the linear head) the held-out month's rows (the candidates,
     never seen by this model) and the October rows (the queries: the fold's early-stop partition, never fitted);
  3. for every October row find its K nearest candidates and write the mean of their out-of-sample residual (label - prediction),
     of the residual clipped to +-1800 s, of their prediction and of their distance, for k in KS, under two metrics (l2: Euclidean in
     the embedding; l2w: Euclidean after scaling every unit by |head weight|) and two scopes (any candidate; same airport only).
A query therefore never sees a candidate of its own month, and every candidate residual comes from a model that did not fit the
candidate: the regime a ranking row is in (it can only retrieve 2025 rows of other months).

Rows read: the seasonal fold's fit and early-stop partitions only. No scoring month (2025-01/07/11/12) and no ranking row is
loaded. Output: <output>/<M>/{run.json, queries.parquet, candidates.parquet[, neighbours.npz]}. Nothing is exported or uploaded.
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_timeline.features import CATEGORICAL_COLUMNS  # noqa: E402
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, TRAINING_MONTHS  # noqa: E402
from run_adsb_nn import build_tables, keep_mask, resolve_variant, rmse  # noqa: E402
from run_adsb_learner import attach  # noqa: E402

FIT_MONTHS = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09"]
OCTOBER = "2025-10"
KS = (4, 16, 64)
METRICS = ("l2", "l2w")
SCOPES = ("any", "same")
CLIP = 1800.0
PENALTY = 1.0e12
CODE_FILES = ["tools/retrieval_screen.py", "tools/run_adsb_nn.py", "tools/run_adsb_learner.py", "src/prc2026_nn/preprocess.py", "src/prc2026_nn/model.py",
              "src/prc2026_nn/train.py", "src/prc2026_adsb/features.py"]


def embed(model, x_num: np.ndarray, x_cat: np.ndarray, batch: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """The backbone's output (N, width) float32 and the scaled prediction (N,) float64 of a k = 1 network."""
    import torch
    if model.k != 1:
        raise ValueError("The screen reads a single-member network.")
    model.eval()
    out, scaled = [], []
    with torch.no_grad():
        for start in range(0, len(x_num), batch):
            xn, xc = torch.from_numpy(x_num[start:start + batch]), torch.from_numpy(x_cat[start:start + batch])
            h, nan = model.numeric(xn)
            z = torch.cat([h, nan.float()] + [emb(xc[:, j]) for j, emb in enumerate(model.embeddings)], dim=1)
            e = model.backbone(z)
            out.append(e.numpy().astype(np.float32, copy=True))
            scaled.append((e @ model.head_weight[0] + model.head_bias[0]).double().numpy())
    return np.concatenate(out), np.concatenate(scaled)


def nearest(queries: np.ndarray, candidates: np.ndarray, query_group: np.ndarray, candidate_group: np.ndarray, k: int, block: int = 2048) -> dict:
    """The k nearest candidates of every query (squared Euclidean distance), over all candidates ("any") and over the candidates of
    the query's own group ("same"). Returns {scope: (index (N, k) int32, distance (N, k) float32)}, nearest first."""
    import torch
    q, c = torch.from_numpy(queries), torch.from_numpy(candidates)
    qg, cg = torch.from_numpy(np.array(query_group, dtype=np.int64)), torch.from_numpy(np.array(candidate_group, dtype=np.int64))
    c2 = (c * c).sum(dim=1)
    ct = c.T.contiguous()
    out = {s: (np.empty((len(q), k), dtype=np.int32), np.empty((len(q), k), dtype=np.float32)) for s in SCOPES}
    with torch.no_grad():
        for start in range(0, len(q), block):
            stop = min(start + block, len(q))
            qb = q[start:stop]
            d2 = ((qb * qb).sum(dim=1, keepdim=True) + c2.unsqueeze(0) - 2.0 * (qb @ ct)).clamp_(min=0.0)
            values, index = torch.topk(d2, k, dim=1, largest=False, sorted=True)
            out["any"][0][start:stop], out["any"][1][start:stop] = index.numpy(), values.sqrt().numpy()
            d2 += (qg[start:stop].unsqueeze(1) != cg.unsqueeze(0)).float() * PENALTY
            values, index = torch.topk(d2, k, dim=1, largest=False, sorted=True)
            if bool((values >= PENALTY).any()):
                raise ValueError("A query has fewer than k candidates at its own airport.")
            out["same"][0][start:stop], out["same"][1][start:stop] = index.numpy(), values.sqrt().numpy()
    return out


def summarise(index: np.ndarray, distance: np.ndarray, residual: np.ndarray, prediction: np.ndarray, same_group: np.ndarray, prefix: str) -> dict:
    """Neighbour means at every k of KS (the neighbours are sorted nearest first)."""
    columns = {}
    res, pred, clipped = residual[index], prediction[index], np.clip(residual, -CLIP, CLIP)[index]
    for k in KS:
        columns[f"res_{prefix}_{k}"] = res[:, :k].mean(axis=1).astype(np.float32)
        columns[f"resc_{prefix}_{k}"] = clipped[:, :k].mean(axis=1).astype(np.float32)
        columns[f"prednb_{prefix}_{k}"] = pred[:, :k].mean(axis=1).astype(np.float32)
        columns[f"dist_{prefix}_{k}"] = distance[:, :k].mean(axis=1).astype(np.float32)
        columns[f"same_{prefix}_{k}"] = same_group[:, :k].mean(axis=1).astype(np.float32)
    return columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", required=True, choices=FIT_MONTHS)
    parser.add_argument("--output", type=Path, default=Path("runs/retrieval/screen"))
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_nn_network_v1.json"))
    parser.add_argument("--variant", default="nnb")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--extra-features", type=Path, default=Path("data/external/adsblol/features_v19"))
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save-neighbours", action="store_true", help="also write the l2 neighbour indices and distances (largest k)")
    parser.add_argument("--limit-rows", type=int, default=None, help="smoke test only: subsample fitting, candidate and query rows (never a result)")
    parser.add_argument("--epochs", type=int, default=None, help="smoke test only: override the epoch budget (never a result)")
    args = parser.parse_args()
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_nn.preprocess import Preprocessor
    from prc2026_nn.train import build_model, predict, train

    smoke = args.limit_rows is not None or args.epochs is not None
    out = ROOT / args.output / (args.held_out + ("_smoke" if smoke else ""))
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = read_json(ROOT / args.config)
    if config.get("audit_used"):
        raise ValueError("Configuration does not respect its own contract.")
    variant = resolve_variant(config, args.variant)
    if args.epochs is not None:
        variant["training"]["epochs"] = args.epochs
    extra_names = list(variant["features"]["extra_features"])
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *extra_names]
    numeric = [n for n in names if n not in CATEGORICAL_COLUMNS]
    categorical = [n for n in names if n in CATEGORICAL_COLUMNS]
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    tables, input_hashes = build_tables(TRAINING_MONTHS, args, extra_names)

    assignment = pd.read_parquet(parent / "splits" / "seasonal.parquet")
    parts, truths = {}, {}
    for role in ("fit", "early_stop"):
        features, truth = load_partition(base, parent, catalog, assignment, role, BUNDLE_NAMES)
        if identity_hash(features[ID]) != original["folds"]["seasonal"]["identity_sha256"][role]:
            raise ValueError("Protected split populations changed.")
        parts[role] = attach(features, tables, catalog)
        truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)
    del features
    month = {role: catalog.set_index(ID)[TIME].reindex(parts[role][ID]).dt.strftime("%Y-%m").to_numpy() for role in parts}
    if sorted(set(month["fit"])) != FIT_MONTHS or set(month["early_stop"]) != {OCTOBER}:
        raise ValueError("The seasonal fold's partitions are not the expected months.")

    keep = {role: keep_mask(parts[role], variant["rows"], truths[role]) for role in parts}
    fit_rows = keep["fit"] & (month["fit"] != args.held_out)
    candidate_rows = keep["fit"] & (month["fit"] == args.held_out)
    query_rows = keep["early_stop"].copy()
    if args.limit_rows is not None:
        rng = np.random.default_rng(args.seed)
        for mask in (fit_rows, candidate_rows, query_rows):
            chosen = np.flatnonzero(mask)
            drop = rng.choice(chosen, size=max(0, len(chosen) - args.limit_rows), replace=False)
            mask[drop] = False

    fit_frame = parts["fit"].loc[fit_rows].reset_index(drop=True)
    target = truths["fit"][fit_rows]
    pre = Preprocessor.fit(fit_frame, numeric, categorical, target, bins=variant["preprocess"]["bins"], min_count=variant["preprocess"]["min_count"])
    x_num, x_cat = pre.transform(fit_frame)
    del fit_frame
    query_frame = parts["early_stop"].loc[query_rows].reset_index(drop=True)
    candidate_frame = parts["fit"].loc[candidate_rows].reset_index(drop=True)
    xq_num, xq_cat = pre.transform(query_frame)
    xc_num, xc_cat = pre.transform(candidate_frame)
    query_label, candidate_label = truths["early_stop"][query_rows], truths["fit"][candidate_rows]
    print({"held_out": args.held_out, "fitted_rows": int(fit_rows.sum()), "candidates": int(candidate_rows.sum()), "queries": int(query_rows.sum()),
           "features": len(names), "seconds": round(time.time() - started, 1)}, flush=True)

    model = build_model(pre, variant["architecture"], args.seed)
    t0 = time.time()
    history = train(model, pre, x_num, x_cat, pre.scale_target(target), variant["training"], args.seed, monitor=(xq_num, xq_cat, query_label),
                    log=lambda e: print(e, flush=True))
    fit_seconds = round(time.time() - t0, 1)
    del x_num, x_cat

    eq, zq = embed(model, xq_num, xq_cat)
    ec, zc = embed(model, xc_num, xc_cat)
    query_pred, candidate_pred = pre.unscale(zq), pre.unscale(zc)
    check = float(np.max(np.abs(query_pred - predict(model, pre, xq_num, xq_cat))))
    if check > 1e-2:
        raise ValueError(f"The embedding path does not reproduce the network's prediction ({check} s).")
    residual = candidate_label - candidate_pred
    airports = sorted(set(query_frame["airport"].astype(str)) | set(candidate_frame["airport"].astype(str)))
    code = {a: i for i, a in enumerate(airports)}
    qg = query_frame["airport"].astype(str).map(code).to_numpy(dtype=np.int64)
    cg = candidate_frame["airport"].astype(str).map(code).to_numpy(dtype=np.int64)
    head = np.abs(model.head_weight[0].detach().numpy()).astype(np.float32)
    head = head / float(head.mean())

    t1 = time.time()
    columns = {ID: query_frame[ID].to_numpy(), "pred": query_pred}
    neighbours = {}
    for metric in METRICS:
        scale = head if metric == "l2w" else np.ones_like(head)
        found = nearest(np.ascontiguousarray(eq * scale), np.ascontiguousarray(ec * scale), qg, cg, max(KS))
        for scope in SCOPES:
            index, distance = found[scope]
            columns.update(summarise(index, distance, residual, candidate_pred, cg[index] == qg[:, None], f"{metric}_{scope}"))
            if args.save_neighbours and metric == "l2":
                neighbours[f"index_{scope}"], neighbours[f"distance_{scope}"] = index, distance
        print({"metric": metric, "seconds": round(time.time() - t1, 1)}, flush=True)
    queries = pd.DataFrame(columns)
    queries.to_parquet(out / "queries.parquet", index=False, compression="zstd")
    pd.DataFrame({ID: candidate_frame[ID].to_numpy(), "airport": candidate_frame["airport"].astype(str).to_numpy(), "label": candidate_label,
                  "pred": candidate_pred}).to_parquet(out / "candidates.parquet", index=False, compression="zstd")
    if neighbours:
        np.savez_compressed(out / "neighbours.npz", **neighbours)

    record = {"created_at_utc": utc_now(), "kind": "retrieval_screen", "held_out": args.held_out, "variant": args.variant, "variant_settings": variant,
              "variant_sha256": object_hash(variant), "seed": args.seed, "config": args.config.as_posix(), "config_sha256": sha256_file(ROOT / args.config),
              "fitted_rows": int(fit_rows.sum()), "candidate_rows": int(candidate_rows.sum()), "query_rows": int(query_rows.sum()),
              "fit_months": [m for m in FIT_MONTHS if m != args.held_out], "history": history, "fit_seconds": fit_seconds, "knn_seconds": round(time.time() - t1, 1),
              "october_rmse": rmse(query_label, query_pred), "candidate_rmse": rmse(candidate_label, candidate_pred), "embedding_check_seconds": check,
              "ks": list(KS), "metrics": list(METRICS), "scopes": list(SCOPES), "clip": CLIP, "smoke": smoke, "limit_rows": args.limit_rows,
              "inputs": {"bundle_id": manifest["bundle_id"], **input_hashes, "code": {p: sha256_file(ROOT / p) for p in CODE_FILES}, "features": len(names),
                         "torch": torch.__version__, "threads": args.threads, "stop_ids_sha256": identity_hash(parts["early_stop"][ID])},
              "outputs": {"queries_sha256": sha256_file(out / "queries.parquet"), "candidates_sha256": sha256_file(out / "candidates.parquet")},
              "seconds": round(time.time() - started, 1), "environment": environment(ROOT), "audit_used": False}
    write_json(out / "run.json", record)
    print({k: record[k] for k in ["held_out", "fitted_rows", "candidate_rows", "query_rows", "october_rmse", "candidate_rmse", "fit_seconds", "knn_seconds", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
