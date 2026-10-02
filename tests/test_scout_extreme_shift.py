import numpy as np
import pandas as pd

from prc2026_scout import extreme_shift as xs


def rows_for(day: str, airport: str, n: int, shift: float, pub_delta: float, n_pub_extra: int, seed: int, extra_rows_ifr: bool = True):
    """One airport-day: n rows with a = 1500 s, prediction 900 s, labels 900 + shift; the published total = own true total + pub_delta."""
    rng = np.random.default_rng(seed)
    a = np.full(n, 1500.0)
    stack = np.full(n, 900.0)
    label = stack + shift + rng.normal(0, 20, n)
    pos_true = np.maximum(a - label, 0)
    pub = float(pos_true.sum() + pub_delta)
    return pd.DataFrame({"month": "m", "airport": airport, "bday": day, "block_day": day, "a": a, "stack": stack, "label": label, "pos_true": pos_true,
                         "pub": pub, "n_pub": n + n_pub_extra, "in_figure": extra_rows_ifr, "moved": True, "extra": False, "v48": stack + 0.2 * shift, "v48_shift": 0.2 * shift})


def training_days():
    frames = [rows_for(f"d{k}", "X", 200, s, 0.0, 0, k) for k, s in enumerate(np.linspace(-150, 150, 40))]
    return pd.concat(frames, ignore_index=True)


def test_fit_finds_the_slope_and_calibrates_a_reliable_airport():
    days = xs.day_frame(training_days(), ["airport", "bday"])
    p = xs.fit(days)
    assert p["stats"]["X"]["corr"] > 0.99 and 0.9 < p["beta"]["X"] <= 1.0 and abs(p["c0"]["X"]) < 1.0


def test_only_extreme_eligible_days_change_and_sign_safety_holds():
    p = xs.fit(xs.day_frame(training_days(), ["airport", "bday"]))
    use = pd.concat([rows_for("small", "X", 200, 30.0, 0.0, 0, 1),            # |I| < 80: unchanged
                     rows_for("big", "X", 200, 120.0, 0.0, 0, 2),             # dn = 0, |I| >= 80: changed
                     rows_for("safe", "X", 200, 120.0, 0.0, 2, 3),            # dn > 0, published below implied (labels above prediction): safe
                     rows_for("unsafe", "X", 200, -120.0, 0.0, 2, 4)],        # dn > 0, published above implied: missing flights could explain it
                    ignore_index=True)
    days = xs.day_frame(use, ["airport", "bday"])
    ch = xs.shifts(days, p)
    assert set(ch.index.get_level_values("bday")) == {"big", "safe"}
    assert abs(ch.loc[("X", "big"), "shift_new"] - 120.0) < 15.0
    new, mask = xs.apply(use, ch, ["airport", "bday"])
    assert mask[use["bday"].isin(["big", "safe"]).to_numpy()].all() and not mask[use["bday"].isin(["small", "unsafe"]).to_numpy()].any()
    assert np.array_equal(new[~mask], use["v48"].to_numpy()[~mask])
    assert abs(np.mean(new[(use["bday"] == "big").to_numpy()]) - (900 + 120)) < 15.0


def test_rows_outside_the_figure_keep_v48():
    p = xs.fit(xs.day_frame(training_days(), ["airport", "bday"]))
    use = rows_for("big", "X", 200, 120.0, 0.0, 0, 5)
    use.loc[use.index[:10], "in_figure"] = False
    use["n_pub"] = 190
    new, mask = xs.apply(use, xs.shifts(xs.day_frame(use, ["airport", "bday"]), p), ["airport", "bday"])
    assert not mask[:10].any() and mask[10:].all()
