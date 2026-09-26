"""Training, prediction and save/load for the round-3 context network (GPU sprint).

The recipe is round 2's (prc2026_ctx2.train, itself round 1's and the committee network's): AdamW with decoupled weight decay on
weight matrices only, linear warm-up then cosine decay over a FIXED epoch budget, gradient-norm clipping, the final epoch's weights
saved, the monitor only recording. Three optional additions, each off by default:

- masking: at every training step, each fitted row whose focal ADS-B match flag is 1 is drawn with probability `p` (from a
  generator of its own, so the shuffling and dropout streams are unchanged) and its focal-flight ADS-B inputs are replaced by what an
  unmatched row carries (the declared columns NaN, the match flag 0). Its label is unchanged. Prediction never masks.
- auxiliary heads: on the masked rows only, the auxiliary head (prc2026_ctx3.model.ContextNet3) predicts standardised focal ADS-B
  quantities the masked inputs no longer show; its mean squared error over the finite entries, times `weight`, joins the loss.
- weight-averaged readouts, which never influence training: an exponential moving average of the parameters (decay
  min(ema_decay, (1 + t) / (10 + t)) at update t) and the plain mean of the end-of-epoch parameters over the last `tail_epochs`
  epochs. Both are returned as state dicts next to the final weights.

With masking p = 0 and no auxiliary head, the fitted final weights equal prc2026_ctx2.train's for the same inputs and seed (test).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from prc2026_ctx2.train import ARCHITECTURE_KEYS as ARCHITECTURE_KEYS2, CONTEXT_KEYS, Context, Store  # noqa: F401  (re-exported)
from prc2026_nn.preprocess import Preprocessor
from prc2026_nn.train import learning_rate

from .model import ContextNet3

ARCHITECTURE_KEYS = (*ARCHITECTURE_KEYS2, "n_aux")
MASK_SEED_OFFSET = 1_000_003


def build_model(pre: Preprocessor, architecture: dict, seed: int, device: torch.device) -> ContextNet3:
    torch.manual_seed(seed)
    return ContextNet3(pre.edges, pre.n_bins, pre.cardinalities(), **{k: architecture.get(k, 0) if k == "n_aux" else architecture[k]
                                                                     for k in ARCHITECTURE_KEYS}).to(device)


@torch.no_grad()
def predict(model: ContextNet3, pre: Preprocessor, x_num: torch.Tensor, x_cat: torch.Tensor, context: Context, batch: int = 8192) -> np.ndarray:
    """x_num / x_cat are device tensors aligned with the context rows; the main head only."""
    model.eval()
    out = np.empty(len(x_num), dtype=np.float64)
    for start in range(0, len(x_num), batch):
        stop = min(start + batch, len(x_num))
        index = torch.arange(start, stop, device=x_num.device)
        tok, tok_nan, wake, valid = context.batch(index, model.max_tokens)
        z = model(x_num[start:stop], x_cat[start:stop], tok, tok_nan, wake, valid)[:, 0]
        out[start:stop] = z.double().cpu().numpy()
    return pre.unscale(out)


def mask_rows(xb: torch.Tensor, draw: torch.Tensor, p: float, columns: torch.Tensor, matched_column: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of the batch whose match flag is 1 and whose uniform draw is below p get the unmatched form of the focal ADS-B inputs.
    Returns the (copied if changed) batch and the boolean mask of masked rows."""
    masked = (xb[:, matched_column] == 1.0) & (draw < p)
    rows = masked.nonzero(as_tuple=True)[0]
    if len(rows):
        xb = xb.clone()
        xb[rows[:, None], columns[None, :]] = float("nan")
        xb[rows, matched_column] = 0.0
    return xb, masked


def aux_loss(outputs: torch.Tensor, targets: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the auxiliary outputs over the finite target entries of masked rows (0 when there are none)."""
    ok = masked[:, None] & torch.isfinite(targets)
    err = (outputs - torch.nan_to_num(targets, nan=0.0)) ** 2 * ok
    return err.sum() / ok.sum().clamp(min=1)


def ema_update(ema: list[torch.Tensor], params: list[torch.Tensor], update: int, decay: float) -> None:
    """In place: ema <- d * ema + (1 - d) * params with d = min(decay, (1 + update) / (10 + update)), `update` counting from 1."""
    d = min(float(decay), (1.0 + update) / (10.0 + update))
    with torch.no_grad():
        torch._foreach_lerp_(ema, [p.detach() for p in params], 1.0 - d)


def train(model: ContextNet3, pre: Preprocessor, x_num: torch.Tensor, x_cat: torch.Tensor, context: Context, target: torch.Tensor, training: dict,
          seed: int, monitor: tuple | None = None, log=print, masking: dict | None = None, aux: dict | None = None,
          readouts: dict | None = None) -> tuple[list[dict], dict]:
    """Fit in place; all tensors on the model's device. `monitor` = (x_num, x_cat, Context, label seconds as numpy).

    masking = {"p", "columns" (LongTensor of x_num columns), "matched" (x_num column of the match flag)}; aux = {"targets" ((N, n_aux)
    tensor, NaN where unknown), "weight"}, requires masking; readouts = {"ema_decay", "tail_epochs"}. Returns (history, averaged state
    dicts by readout name)."""
    device = x_num.device
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    p = float(masking["p"]) if masking else 0.0
    if aux is not None and (p <= 0 or not model.n_aux):
        raise ValueError("Auxiliary heads need masked rows and a network with an auxiliary head.")
    mask_generator = torch.Generator(device=device).manual_seed(seed + MASK_SEED_OFFSET) if p > 0 else None
    columns = masking["columns"].to(device) if p > 0 else None
    decay = [p_ for _, p_ in model.named_parameters() if p_.ndim >= 2]
    plain = [p_ for _, p_ in model.named_parameters() if p_.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": training["weight_decay"]}, {"params": plain, "weight_decay": 0.0}], lr=training["lr"])
    n, batch = len(target), int(training["batch_size"])
    steps_per_epoch = math.ceil(n / batch)
    epochs = int(training["epochs"])
    total = steps_per_epoch * epochs
    warmup = max(1, int(round(training["warmup_epochs"] * steps_per_epoch)))
    names = [name for name, _ in model.named_parameters()]
    params = [p_ for _, p_ in model.named_parameters()]
    ema = [p_.detach().clone() for p_ in params] if readouts else None
    tail_epochs = int(readouts["tail_epochs"]) if readouts else 0
    tail = [torch.zeros_like(p_) for p_ in params] if tail_epochs else None
    history, step, rate = [], 0, training["lr"]
    for epoch in range(epochs):
        started = time.time()
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        running = torch.zeros((), device=device, dtype=torch.float64)
        running_aux = torch.zeros((), device=device, dtype=torch.float64)
        masked_rows = torch.zeros((), device=device, dtype=torch.int64)
        for s in range(steps_per_epoch):
            index = order[s * batch:(s + 1) * batch]
            rate = learning_rate(step, total, warmup, training["lr"], training["min_lr_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            tok, tok_nan, wake, valid = context.batch(index, model.max_tokens)
            xb = x_num[index]
            masked = None
            if p > 0:
                draw = torch.rand(len(index), generator=mask_generator, device=device)
                xb, masked = mask_rows(xb, draw, p, columns, int(masking["matched"]))
                masked_rows += masked.sum()
            if aux is not None:
                out, aux_out = model.forward_aux(xb, x_cat[index], tok, tok_nan, wake, valid)
                main = ((out[:, 0] - target[index]) ** 2).mean()
                extra = aux_loss(aux_out, aux["targets"][index], masked)
                loss = main + float(aux["weight"]) * extra
                running_aux += extra.detach().double() * len(index)
            else:
                out = model(xb, x_cat[index], tok, tok_nan, wake, valid)
                main = loss = ((out[:, 0] - target[index]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            running += main.detach().double() * len(index)
            step += 1
            if ema is not None:
                ema_update(ema, params, step, readouts["ema_decay"])
        if tail is not None and epoch >= epochs - tail_epochs:
            with torch.no_grad():
                torch._foreach_add_(tail, [p_.detach() for p_ in params])
        entry = {"epoch": epoch + 1, "train_loss_scaled": float(running) / n, "lr_end": rate, "seconds": round(time.time() - started, 1)}
        if p > 0:
            entry["masked_rows"] = int(masked_rows)
        if aux is not None:
            entry["aux_loss"] = float(running_aux) / n
        if monitor is not None:
            values = predict(model, pre, monitor[0], monitor[1], monitor[2])
            entry["monitor_rmse"] = float(np.sqrt(np.mean((values - monitor[3]) ** 2)))
        history.append(entry)
        log(entry)
    averaged = {}
    if readouts:
        base = model.state_dict()
        averaged["ema"] = {k: v.detach().clone() for k, v in base.items()}
        for name, value in zip(names, ema):
            averaged["ema"][name] = value.clone()
        if tail is not None:
            averaged["tail"] = {k: v.detach().clone() for k, v in base.items()}
            for name, value in zip(names, tail):
                averaged["tail"][name] = value / tail_epochs
    return history, averaged


def save(model: ContextNet3, pre: Preprocessor, architecture: dict, context_settings: dict, directory: Path, averaged: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pre.save(directory)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / "model.pt")
    for name, state in (averaged or {}).items():
        torch.save({k: v.detach().cpu() for k, v in state.items()}, directory / f"model_{name}.pt")
    (directory / "architecture.json").write_text(json.dumps({"architecture": {k: architecture.get(k, 0) if k == "n_aux" else architecture[k] for k in ARCHITECTURE_KEYS},
                                                             "context": {k: bool(context_settings.get(k, k == "use_adsb")) for k in CONTEXT_KEYS},
                                                             "readouts": sorted(averaged or {})}, indent=1, sort_keys=True), encoding="utf-8")


def load(directory: Path, device: torch.device, readout: str | None = None) -> tuple[ContextNet3, Preprocessor, dict]:
    """The final weights (readout None) or a saved weight-averaged readout ("ema", "tail")."""
    pre = Preprocessor.load(directory)
    spec = json.loads((directory / "architecture.json").read_text(encoding="utf-8"))
    model = ContextNet3(pre.edges, pre.n_bins, pre.cardinalities(), **spec["architecture"]).to(device)
    model.load_state_dict(torch.load(directory / ("model.pt" if readout is None else f"model_{readout}.pt"), weights_only=True, map_location=device))
    model.eval()
    return model, pre, spec["context"]
