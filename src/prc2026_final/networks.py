"""S8: the network ensembles, each the plain mean of its declared seeds' saved twelve-month refits (<learner>/refit12).

Frame: the timeline bundle's 276 features + the learners' 27 ADS-B features + the 11 v19 features (each Preprocessor selects its own
columns). Context networks read the run's token split (tokens.build) through prc2026_ctx.tokens.TokenStore:
- CPU-trained context nets (ens_ctx, ens_ctx_e24, ens_ctx_e24b, ens_ctx_lirf): prc2026_ctx.train.load/predict + Context.build (as
  tools/run_adsb_ctx.py predicted the ranking);
- GPU-trained e32 nets (ens_ctx_e32, record device cuda): prc2026_ctx3.train.load/predict on the CPU (float32; the TF32 GPU values
  differ by ~0.09 s RMS per net, docs/adsb_ctx3/verification_v30.json);
- tabular MLPs (ens_new, ens_nn, ens_nnl): prc2026_nn.train.load/predict.
v30's recipe needs ens_ctx_e24, ens_ctx_e24b, ens_ctx_e32 (committee mix) and ens_ctx_lirf (LIRF blends); ens_ctx, ens_new, ens_nn and
ens_nnl are v30-detail columns the recipe does not use (computed only on request, for comparison).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.artifacts import read_json, sha256_file
from prc2026.features import ID

from .common import ROOT

RECIPE_ENSEMBLES = ["ens_ctx_e24", "ens_ctx_e24b", "ens_ctx_e32", "ens_ctx_lirf"]
DETAIL_ONLY = ["ens_ctx", "ens_new", "ens_nn", "ens_nnl"]
BATCH = 50000


def ensemble_learners() -> dict[str, list[str]]:
    decl5 = read_json(ROOT / "configs/adsb_ctx_v5.json")
    decl_b = read_json(ROOT / "configs/adsb_ctx_lirf_v1.json")
    rnn = read_json(ROOT / "configs/adsb_rnn_v1.json")
    out = {name: list(net["learners"]) for name, net in decl5["networks"].items()}
    out["ens_ctx_lirf"] = list(decl_b["networks"]["ens_ctx_lirf"]["learners"])
    out.update({name: list(spec["learners"]) for name, spec in rnn["ensembles"].items()})
    return out


def kind_of(record: dict) -> str:
    variant = str(record.get("variant", ""))
    if record["inputs"].get("device") == "cuda":
        return "ctx3"
    if "tokens" in record["inputs"] or variant.startswith("ctx"):
        return "ctx"
    return "nn"


def predict_learner(learner: str, frame: pd.DataFrame, host, threads: int, cache: dict) -> tuple[np.ndarray, dict]:
    """`host`: the run's TokenStore (None for tabular-only runs); `cache` keeps the torch view of the token arrays across learners."""
    import torch
    torch.set_num_threads(threads)
    run = ROOT / learner / "refit12"
    record = read_json(run / "run.json")
    kind = kind_of(record)
    values = np.empty(len(frame))
    ids = frame[ID].to_numpy()
    if kind == "nn":
        from prc2026_nn.train import load, predict
        model, pre = load(run)
        for start in range(0, len(frame), BATCH):
            xn, xc = pre.transform(frame.iloc[start:start + BATCH])
            values[start:start + BATCH] = predict(model, pre, xn, xc)
    else:
        if host is None:
            raise ValueError(f"{learner}: a context network needs the run's token split.")
        if kind == "ctx":
            from prc2026_ctx.train import Context, load, predict
            model, pre, settings = load(run)
            for start in range(0, len(frame), BATCH):
                chunk = frame.iloc[start:start + BATCH]
                xn, xc = pre.transform(chunk)
                values[start:start + BATCH] = predict(model, pre, xn, xc, Context.build(host, *host.lookup(chunk[ID].to_numpy()), settings))
        else:
            from prc2026_ctx3.train import Context, Store, load, predict
            device = torch.device("cpu")
            model, pre, settings = load(run, device)
            store = cache.setdefault("ctx3_store", Store.build(host, device))
            for start in range(0, len(frame), BATCH):
                chunk = frame.iloc[start:start + BATCH]
                xn, xc = pre.transform(chunk)
                values[start:start + BATCH] = predict(model, pre, torch.from_numpy(xn), torch.from_numpy(xc),
                                                      Context.build(store, *host.lookup(chunk[ID].to_numpy()), settings))
    if not np.isfinite(values).all():
        raise ValueError(f"{learner}: non-finite predictions.")
    info = {"kind": kind, "trained_on": "gpu" if kind == "ctx3" else "cpu", "seed": record.get("seed"), "model_sha256": sha256_file(run / "model.pt"),
            "record_model_sha256": record.get("model_sha256"), "rows": int(len(ids))}
    if info["model_sha256"] != info["record_model_sha256"]:
        raise ValueError(f"{learner}: saved model differs from its refit record.")
    return values, info
