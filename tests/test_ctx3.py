import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from prc2026_ctx.tokens import TokenStore, build_tokens, encode_codes, neighbours, save_file
from prc2026_ctx2.model import ContextNet2
from prc2026_ctx2 import train as train2
from prc2026_ctx3.model import ContextNet3
from prc2026_ctx3.train import Context, Store, aux_loss, build_model, ema_update, load, mask_rows, predict, save, train
from prc2026_nn.preprocess import Preprocessor
from test_ctx import matches_for, raw_movements

ARCH = {"d_num": 4, "cat_cap": 4, "width": 32, "depth": 2, "dropout": 0.1, "tok_dim": 16, "heads": 4, "max_tokens": 12,
        "sa_layers": 0, "sa_heads": 4, "sa_ff": 2, "sa_dropout": 0.0}
TRAINING = {"epochs": 3, "batch_size": 64, "lr": 3e-3, "weight_decay": 1e-5, "warmup_epochs": 0.5, "min_lr_fraction": 0.05, "grad_clip": 5.0}


def fixture(tmp_path, k=12):
    raw = raw_movements(n=1200, seed=5)
    tokens = encode_codes(build_tokens(raw, matches_for(raw)))
    dep_ids, self_rows, index = neighbours(tokens, k=k)
    index[:3] = -1
    save_file(tmp_path, "a", tokens, dep_ids, self_rows, index)
    store = TokenStore(tmp_path, ["a"])
    rng = np.random.default_rng(1)
    n = len(dep_ids)
    matched = (rng.random(n) < 0.6).astype(float)
    taxi = np.where(matched == 1, rng.gamma(4.0, 150.0, n), np.nan)
    stopped = np.where(matched == 1, rng.gamma(1.0, 60.0, n), np.nan)
    frame = pd.DataFrame({"a": rng.normal(size=n), "adsb_matched": matched, "adsb_taxi": taxi, "adsb_stopped": stopped,
                          "c": np.where(np.arange(n) % 2, "x", "y")})
    target = frame["a"].to_numpy() * 100 + 600 + np.nan_to_num(taxi) * 0.5
    pre = Preprocessor.fit(frame, ["a", "adsb_matched", "adsb_taxi", "adsb_stopped"], ["c"], target, bins=8, min_count=1)
    return store, dep_ids, frame, target, pre


def tensors(store, dep_ids, frame, pre):
    device = torch.device("cpu")
    context = Context.build(Store.build(store, device), *store.lookup(dep_ids), {"use_adsb": True})
    xn, xc = (torch.from_numpy(v) for v in pre.transform(frame))
    return context, xn, xc


def test_no_aux_head_is_round_two_network(tmp_path):
    store, dep_ids, frame, target, pre = fixture(tmp_path)
    context, xn, xc = tensors(store, dep_ids, frame, pre)
    torch.manual_seed(7)
    two = ContextNet2(pre.edges, pre.n_bins, pre.cardinalities(), **ARCH)
    torch.manual_seed(7)
    three = ContextNet3(pre.edges, pre.n_bins, pre.cardinalities(), **ARCH, n_aux=0)
    torch.manual_seed(7)
    with_aux = ContextNet3(pre.edges, pre.n_bins, pre.cardinalities(), **ARCH, n_aux=3)
    s2, s3, sa = two.state_dict(), three.state_dict(), with_aux.state_dict()
    assert list(s2) == list(s3) and all(torch.equal(s2[k], s3[k]) for k in s2)
    assert set(sa) - set(s2) == {"aux_head.weight", "aux_head.bias"} and all(torch.equal(s2[k], sa[k]) for k in s2)
    feats = context.batch(torch.arange(len(xn)), ARCH["max_tokens"])
    for m in (two, three, with_aux):
        m.eval()
    with torch.no_grad():
        out2 = two(xn, xc, *feats)
        assert torch.equal(out2, three(xn, xc, *feats))
        main, aux = with_aux.forward_aux(xn, xc, *feats)
        assert torch.equal(main, out2) and aux.shape == (len(xn), 3)


def test_training_with_everything_off_equals_round_two(tmp_path):
    store, dep_ids, frame, target, pre = fixture(tmp_path)
    context, xn, xc = tensors(store, dep_ids, frame, pre)
    y = torch.from_numpy(pre.scale_target(target))
    m2 = train2.build_model(pre, ARCH, seed=3, device=torch.device("cpu"))
    h2 = train2.train(m2, pre, xn, xc, train2.Context.build(train2.Store.build(store, torch.device("cpu")), *store.lookup(dep_ids), {"use_adsb": True}),
                      y, TRAINING, seed=3, log=lambda e: None)
    m3 = build_model(pre, ARCH, seed=3, device=torch.device("cpu"))
    h3, averaged = train(m3, pre, xn, xc, context, y, TRAINING, seed=3, log=lambda e: None, readouts={"ema_decay": 0.99, "tail_epochs": 2})
    s2, s3 = m2.state_dict(), m3.state_dict()
    assert all(torch.equal(s2[k], s3[k]) for k in s2)
    assert [e["train_loss_scaled"] for e in h2] == [e["train_loss_scaled"] for e in h3]
    assert set(averaged) == {"ema", "tail"} and not torch.equal(averaged["ema"]["head.weight"], s3["head.weight"])


def test_tail_mean_is_mean_of_last_epochs(tmp_path):
    store, dep_ids, frame, target, pre = fixture(tmp_path)
    context, xn, xc = tensors(store, dep_ids, frame, pre)
    model = build_model(pre, ARCH, seed=4, device=torch.device("cpu"))
    snaps = []
    _, averaged = train(model, pre, xn, xc, context, torch.from_numpy(pre.scale_target(target)), TRAINING, seed=4,
                        log=lambda e: snaps.append({k: v.clone() for k, v in model.state_dict().items()}), readouts={"ema_decay": 0.99, "tail_epochs": 2})
    for k, v in averaged["tail"].items():
        assert torch.allclose(v, (snaps[1][k] + snaps[2][k]) / 2, atol=1e-7)


def test_ema_update_formula():
    ema = [torch.tensor([1.0, 2.0])]
    params = [torch.tensor([3.0, 6.0])]
    ema_update(ema, params, update=1, decay=0.999)      # d = min(0.999, 2/11)
    d = 2.0 / 11.0
    assert torch.allclose(ema[0], torch.tensor([d * 1 + (1 - d) * 3, d * 2 + (1 - d) * 6]))
    ema = [torch.tensor([1.0])]
    ema_update(ema, [torch.tensor([3.0])], update=100000, decay=0.9)
    assert torch.allclose(ema[0], torch.tensor([0.9 * 1 + 0.1 * 3]))


def test_mask_rows_only_matched_and_drawn():
    xb = torch.tensor([[1.0, 1.0, 5.0, 7.0], [1.0, 0.0, float("nan"), float("nan")], [2.0, 1.0, 6.0, 8.0], [3.0, 1.0, 9.0, 4.0]])
    draw = torch.tensor([0.1, 0.1, 0.9, 0.2])
    out, masked = mask_rows(xb, draw, 0.5, torch.tensor([2, 3]), matched_column=1)
    assert masked.tolist() == [True, False, False, True]
    assert torch.isnan(out[0, 2:]).all() and out[0, 1] == 0 and out[0, 0] == 1
    assert torch.isnan(out[3, 2:]).all() and out[3, 1] == 0 and out[3, 0] == 3
    assert torch.equal(out[2], xb[2]) and out[1, 1] == 0
    assert xb[0, 1] == 1 and xb[0, 2] == 5          # the input batch is not modified


def test_aux_loss_uses_masked_finite_entries_only():
    outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    targets = torch.tensor([[0.0, float("nan")], [0.0, 0.0], [0.0, 0.0]])
    masked = torch.tensor([True, False, True])
    assert torch.isclose(aux_loss(outputs, targets, masked), torch.tensor((1.0 + 25.0 + 36.0) / 3))
    assert aux_loss(outputs, targets, torch.tensor([False, False, False])) == 0


def test_masked_aux_training_save_load(tmp_path):
    store, dep_ids, frame, target, pre = fixture(tmp_path)
    context, xn, xc = tensors(store, dep_ids, frame, pre)
    arch = {**ARCH, "n_aux": 2}
    model = build_model(pre, arch, seed=5, device=torch.device("cpu"))
    matched = frame["adsb_matched"].eq(1).to_numpy()
    aux_targets = np.stack([np.where(matched, np.log1p(frame[c].to_numpy()), np.nan) for c in ("adsb_taxi", "adsb_stopped")], axis=1)
    aux_targets = (aux_targets - np.nanmean(aux_targets, axis=0)) / np.nanstd(aux_targets, axis=0)
    masking = {"p": 0.5, "columns": torch.tensor([pre.numeric.index("adsb_taxi"), pre.numeric.index("adsb_stopped")]), "matched": pre.numeric.index("adsb_matched")}
    history, averaged = train(model, pre, xn, xc, context, torch.from_numpy(pre.scale_target(target)), TRAINING, seed=5, log=lambda e: None,
                              masking=masking, aux={"targets": torch.from_numpy(aux_targets.astype(np.float32)), "weight": 0.3},
                              readouts={"ema_decay": 0.99, "tail_epochs": 2}, monitor=(xn, xc, context, target))
    assert all(e["masked_rows"] > 0 and np.isfinite(e["aux_loss"]) and np.isfinite(e["monitor_rmse"]) for e in history)
    # about half of the matched rows are masked in each epoch
    assert 0.35 < history[0]["masked_rows"] / matched.sum() < 0.65
    values = predict(model, pre, xn, xc, context)
    save(model, pre, arch, {"use_adsb": True}, tmp_path / "m", averaged)
    again, pre2, settings = load(tmp_path / "m", torch.device("cpu"))
    assert np.allclose(predict(again, pre2, xn, xc, context), values, atol=1e-6)
    ema_model, _, _ = load(tmp_path / "m", torch.device("cpu"), readout="ema")
    model.load_state_dict(averaged["ema"])
    assert np.allclose(predict(ema_model, pre2, xn, xc, context), predict(model, pre, xn, xc, context), atol=1e-6)
    with pytest.raises(ValueError):
        train(build_model(pre, ARCH, seed=5, device=torch.device("cpu")), pre, xn, xc, context, torch.from_numpy(pre.scale_target(target)), TRAINING, seed=5,
              log=lambda e: None, aux={"targets": torch.zeros(len(xn), 2), "weight": 0.3})


def test_blank_tokens_blanks_only_the_named_quantities(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from run_adsb_ctx3 import blank_tokens
    from prc2026_ctx.model import TOKEN_FEATURES, token_features
    store, dep_ids, frame, target, pre = fixture(tmp_path)
    before_t, before_v = store.times.copy(), store.values.copy()
    self_rows, nb = store.lookup(dep_ids)
    args = (torch.from_numpy(store.times), torch.from_numpy(store.values), torch.from_numpy(store.codes), torch.from_numpy(self_rows).long(),
            torch.from_numpy(nb).long())
    x0, nan0, _, _ = token_features(*args)
    blank_tokens(store, ["first_seen", "adsb_gate"])
    assert np.isnan(store.times[:, 6]).all() and np.isnan(store.values[:, 3]).all()
    keep_t, keep_v = [j for j in range(before_t.shape[1]) if j != 6], [j for j in range(before_v.shape[1]) if j != 3]
    assert np.array_equal(store.times[:, keep_t], before_t[:, keep_t], equal_nan=True)
    assert np.array_equal(store.values[:, keep_v], before_v[:, keep_v], equal_nan=True)
    args = (torch.from_numpy(store.times), torch.from_numpy(store.values), torch.from_numpy(store.codes), torch.from_numpy(self_rows).long(),
            torch.from_numpy(nb).long())
    x1, nan1, _, _ = token_features(*args)
    changed = [TOKEN_FEATURES[j] for j in range(len(TOKEN_FEATURES)) if not torch.equal(nan0[..., j], nan1[..., j]) or not torch.equal(x0[..., j], x1[..., j])]
    assert set(changed) <= {"first_seen", "adsb_gate"} and nan1[..., TOKEN_FEATURES.index("first_seen")].all()
    with pytest.raises(ValueError):
        blank_tokens(store, ["rel_motion"])
