"""Training, prediction and save/load for the context network.

The recipe is the committee network's (prc2026_nn.train): AdamW with decoupled weight decay on weight matrices only, linear
warm-up then cosine decay to `min_lr_fraction` over a FIXED epoch budget, gradient-norm clipping; the monitor only records
the per-epoch RMSE and the saved model is the final epoch's weights. Each batch gathers its rows' context tokens from the
token store (tokens.TokenStore arrays) and describes them relative to the focal row (model.token_features).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from prc2026_nn.preprocess import Preprocessor
from prc2026_nn.train import learning_rate

from .model import ContextNet, token_features

ARCHITECTURE_KEYS = ("d_num", "cat_cap", "width", "depth", "dropout", "tok_dim", "heads", "max_tokens")
CONTEXT_KEYS = ("use_adsb", "departures_only")


@dataclass
class Context:
    """Token arrays (shared) and, per row of the matrix being fitted or predicted, its self row and neighbour rows."""
    times: torch.Tensor
    values: torch.Tensor
    codes: torch.Tensor
    self_rows: torch.Tensor
    nb_rows: torch.Tensor
    use_adsb: bool = True
    departures_only: bool = False

    @classmethod
    def build(cls, store, self_rows: np.ndarray, nb_rows: np.ndarray, settings: dict) -> "Context":
        return cls(torch.from_numpy(store.times), torch.from_numpy(store.values), torch.from_numpy(store.codes),
                   torch.from_numpy(np.asarray(self_rows, dtype=np.int64)), torch.from_numpy(np.asarray(nb_rows, dtype=np.int32)),
                   bool(settings.get("use_adsb", True)), bool(settings.get("departures_only", False)))

    def batch(self, index: torch.Tensor, max_tokens: int):
        nb = self.nb_rows[index][:, :max_tokens].long()
        return token_features(self.times, self.values, self.codes, self.self_rows[index], nb, self.use_adsb, self.departures_only)


def build_model(pre: Preprocessor, architecture: dict, seed: int) -> ContextNet:
    torch.manual_seed(seed)
    return ContextNet(pre.edges, pre.n_bins, pre.cardinalities(), **{k: architecture[k] for k in ARCHITECTURE_KEYS})


@torch.no_grad()
def predict(model: ContextNet, pre: Preprocessor, x_num: np.ndarray, x_cat: np.ndarray, context: Context, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = np.empty(len(x_num), dtype=np.float64)
    for start in range(0, len(x_num), batch):
        stop = min(start + batch, len(x_num))
        index = torch.arange(start, stop)
        tok, tok_nan, wake, valid = context.batch(index, model.max_tokens)
        z = model(torch.from_numpy(x_num[start:stop]), torch.from_numpy(x_cat[start:stop]), tok, tok_nan, wake, valid)[:, 0]
        out[start:stop] = z.double().numpy()
    return pre.unscale(out)


def train(model: ContextNet, pre: Preprocessor, x_num: np.ndarray, x_cat: np.ndarray, context: Context, target: np.ndarray, training: dict,
          seed: int, monitor: tuple | None = None, log=print) -> list[dict]:
    """Fit in place. `target` is scaled; `monitor` = (x_num, x_cat, Context, label seconds)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    xn, xc, y = torch.from_numpy(x_num), torch.from_numpy(x_cat), torch.from_numpy(target)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    plain = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": training["weight_decay"]}, {"params": plain, "weight_decay": 0.0}], lr=training["lr"])
    n, batch = len(target), int(training["batch_size"])
    steps_per_epoch = math.ceil(n / batch)
    total = steps_per_epoch * int(training["epochs"])
    warmup = max(1, int(round(training["warmup_epochs"] * steps_per_epoch)))
    history, step, rate = [], 0, training["lr"]
    for epoch in range(int(training["epochs"])):
        started = time.time()
        model.train()
        order = torch.randperm(n, generator=generator)
        running, seen = 0.0, 0
        for s in range(steps_per_epoch):
            index = order[s * batch:(s + 1) * batch]
            rate = learning_rate(step, total, warmup, training["lr"], training["min_lr_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            tok, tok_nan, wake, valid = context.batch(index, model.max_tokens)
            out = model(xn[index], xc[index], tok, tok_nan, wake, valid)
            loss = ((out[:, 0] - y[index]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            running += float(loss.detach()) * len(index)
            seen += len(index)
            step += 1
        entry = {"epoch": epoch + 1, "train_loss_scaled": running / seen, "lr_end": rate, "seconds": round(time.time() - started, 1)}
        if monitor is not None:
            values = predict(model, pre, monitor[0], monitor[1], monitor[2])
            entry["monitor_rmse"] = float(np.sqrt(np.mean((values - monitor[3]) ** 2)))
        history.append(entry)
        log(entry)
    return history


def save(model: ContextNet, pre: Preprocessor, architecture: dict, context_settings: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pre.save(directory)
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "architecture.json").write_text(json.dumps({"architecture": {k: architecture[k] for k in ARCHITECTURE_KEYS},
                                                             "context": {k: bool(context_settings.get(k, k == "use_adsb")) for k in CONTEXT_KEYS}},
                                                            indent=1, sort_keys=True), encoding="utf-8")


def load(directory: Path) -> tuple[ContextNet, Preprocessor, dict]:
    pre = Preprocessor.load(directory)
    spec = json.loads((directory / "architecture.json").read_text(encoding="utf-8"))
    model = ContextNet(pre.edges, pre.n_bins, pre.cardinalities(), **spec["architecture"])
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
    model.eval()
    return model, pre, spec["context"]
