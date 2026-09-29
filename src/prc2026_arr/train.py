"""Training of the two-task network: base's loop, with one arrival batch added to every step.

Held equal to base (prc2026_nn.train.train): the epoch budget, the steps per epoch (from the departure rows), the learning-rate
schedule, the optimiser and its parameter groups, the gradient clip, and the departure batches (the permutation sequence of the
seed). Each step adds one arrival batch from an independent permutation; the loss is the departure batch's mean squared error
plus `lam` times the arrival batch's, each in its own standardised unit. With `arrivals=None` every step is base's and the run
ends with base's parameters exactly.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from prc2026_nn.preprocess import MISSING_TOKEN, Preprocessor, _as_strings
from prc2026_nn.train import ARCHITECTURE_KEYS, learning_rate, predict

from .model import TwoTaskNet


def build_model(pre: Preprocessor, architecture: dict, seed: int) -> TwoTaskNet:
    torch.manual_seed(seed)
    return TwoTaskNet(pre.edges, pre.n_bins, pre.cardinalities(), **{k: architecture[k] for k in ARCHITECTURE_KEYS})


def pooled_vocabulary(pre: Preprocessor, frames: list, min_count: int) -> dict:
    """Categories with at least `min_count` rows over all frames together (base: the departure fitting rows alone)."""
    vocab = {}
    for name in pre.categorical:
        counts = None
        for frame in frames:
            c = _as_strings(frame[name]).value_counts()
            counts = c if counts is None else counts.add(c, fill_value=0)
        vocab[name] = sorted(str(v) for v, c in counts.items() if c >= min_count and v != MISSING_TOKEN)
    return vocab


def arrival_batches(rows: int, size: int, generator: torch.Generator):
    """Full batches from successive permutations; a permutation's tail is left to the next permutation's draw."""
    if rows < size:
        raise ValueError("Fewer arrival rows than one batch.")
    while True:
        order = torch.randperm(rows, generator=generator)
        for start in range(0, rows - size + 1, size):
            yield order[start:start + size]


@torch.no_grad()
def predict_arrival(model: TwoTaskNet, x_num: np.ndarray, x_cat: np.ndarray, mean: float, scale: float, batch: int = 16384) -> np.ndarray:
    model.eval()
    out = np.empty(len(x_num), dtype=np.float64)
    for start in range(0, len(x_num), batch):
        z = model.arrival(torch.from_numpy(x_num[start:start + batch]), torch.from_numpy(x_cat[start:start + batch])).mean(dim=1)
        out[start:start + batch] = z.double().numpy()
    return out * scale + mean


def train(model: TwoTaskNet, pre: Preprocessor, x_num: np.ndarray, x_cat: np.ndarray, target: np.ndarray, training: dict, seed: int,
          arrivals: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, lam: float = 0.0, arrival_batch: int = 2048, generator_offset: int = 7919,
          monitor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, log=print) -> list[dict]:
    """Fit in place. `target` and `arrivals[2]` are scaled targets; `monitor` = (x_num, x_cat, departure label seconds)."""
    if training.get("ema_decay"):
        raise ValueError("The two-task loop follows the base recipe, which has no weight averaging.")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    xn, xc, y = torch.from_numpy(x_num), torch.from_numpy(x_cat), torch.from_numpy(target)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and n != "adapter"]
    plain = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and n != "adapter")]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": training["weight_decay"]}, {"params": plain, "weight_decay": 0.0}], lr=training["lr"])
    n, batch = len(target), int(training["batch_size"])
    steps_per_epoch = math.ceil(n / batch)
    total = steps_per_epoch * int(training["epochs"])
    warmup = max(1, int(round(training["warmup_epochs"] * steps_per_epoch)))
    if arrivals is not None:
        an, ac, ay = (torch.from_numpy(a) for a in arrivals)
        draw = arrival_batches(len(ay), arrival_batch, torch.Generator().manual_seed(seed + generator_offset))
    history, step = [], 0
    for epoch in range(int(training["epochs"])):
        started = time.time()
        model.train()
        order = torch.randperm(n, generator=generator)
        running, seen, running_arr, seen_arr = 0.0, 0, 0.0, 0
        for s in range(steps_per_epoch):
            index = order[s * batch:(s + 1) * batch]
            rate = learning_rate(step, total, warmup, training["lr"], training["min_lr_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            if arrivals is None:
                out = model(xn[index], xc[index])
                loss = ((out - y[index].unsqueeze(1)) ** 2).mean()
                departure_loss = loss
            else:
                other = next(draw)
                out, out_arr = model.heads(torch.cat([xn[index], an[other]]), torch.cat([xc[index], ac[other]]), len(index))
                departure_loss = ((out - y[index].unsqueeze(1)) ** 2).mean()
                arrival_loss = ((out_arr - ay[other].unsqueeze(1)) ** 2).mean()
                loss = departure_loss + lam * arrival_loss
                running_arr += float(arrival_loss.detach()) * len(other)
                seen_arr += len(other)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            running += float(departure_loss.detach()) * len(index)
            seen += len(index)
            step += 1
        entry = {"epoch": epoch + 1, "train_loss_scaled": running / seen, "lr_end": rate, "seconds": round(time.time() - started, 1)}
        if seen_arr:
            entry["arrival_loss_scaled"] = running_arr / seen_arr
        if monitor is not None:
            values = predict(model, pre, monitor[0], monitor[1])
            entry["monitor_rmse"] = float(np.sqrt(np.mean((values - monitor[2]) ** 2)))
        history.append(entry)
        log(entry)
    return history


def save(model: TwoTaskNet, pre: Preprocessor, architecture: dict, arrival_scaling: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pre.save(directory)
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "architecture.json").write_text(json.dumps({k: architecture[k] for k in ARCHITECTURE_KEYS}, indent=1, sort_keys=True), encoding="utf-8")
    (directory / "arrival_scaling.json").write_text(json.dumps(arrival_scaling, indent=1, sort_keys=True), encoding="utf-8")


def load(directory: Path) -> tuple[TwoTaskNet, Preprocessor, dict]:
    pre = Preprocessor.load(directory)
    architecture = json.loads((directory / "architecture.json").read_text(encoding="utf-8"))
    model = TwoTaskNet(pre.edges, pre.n_bins, pre.cardinalities(), **architecture)
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
    model.eval()
    return model, pre, json.loads((directory / "arrival_scaling.json").read_text(encoding="utf-8"))
