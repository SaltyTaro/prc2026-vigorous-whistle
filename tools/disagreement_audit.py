"""Label-free component-disagreement audit (27 September 2026).

Question: do the two sides of v34's recipe disagree more often, or by more, on the ranking months (January / July 2026) than on the
development months, and does that explain part of the ranking error the per-cell error map (docs/error_map) cannot?

Sides, per row:
  non-LIRF   base = v9 + residual stage (matched rows), mix = committee mix (0.25 committee + 0.75 context networks); D = mix - base
  LIRF       v9 against the LIRF blend column (net/tree at the scope's proportions); D = column - v9; population rows are v9-fixed

Ranking rows use runs/adsb_ctx/export_v30/ranking_values.parquet (no labels exist or are read). Development rows use the fair set
(runs/fair_dev/rows.parquet: equal-footing fold provenance, labels known) with v34's weights rebuilt by tools/run_fair_weights.build.
The projection multiplies the ranking's row counts per (airport group x |D| bin) by the development mean squared error of the same
cell (December alone, and the four months pooled), and compares with the per-cell projection that ignores D.

  --report docs/disagreement/audit.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from run_fair_weights import build, candidate_weights  # noqa: E402

RANKING = "runs/adsb_ctx/export_v30/ranking_values.parquet"
FAIR = "runs/fair_dev/rows.parquet"
V34 = "runs/submissions/vigorous-whistle_v34.parquet"
OFFICIAL_V34 = 228.5933
BINS = [0, 60, 150, 300, 600, 1200, 2400, np.inf]
LABELS = ["<60", "60-150", "150-300", "300-600", "600-1200", "1200-2400", ">2400"]
V34_LIRF = {"lirf_nonpop_matched": {"v9": 0.15, "net": 0.4, "tree": 0.45}, "lirf_nonpop_unmatched": {"v9": 0.0, "net": 0.35, "tree": 0.25}}


def group_of(airport: np.ndarray, matched: np.ndarray, lirf_pop: np.ndarray) -> np.ndarray:
    kind = np.where(matched, "matched", "unmatched")
    out = np.char.add(np.char.add(airport.astype(str), ":"), kind)
    return np.where(lirf_pop, "LIRF:population", out)


def sides(airport, matched, v9, residual_or_base, mix, lirf_net, lirf_tree, lirf_m, lirf_u, is_base: bool):
    base = residual_or_base if is_base else v9 + np.where(matched & (airport != "LIRF"), residual_or_base, 0.0)
    m_col = (0.4 * lirf_net + 0.45 * lirf_tree) / 0.85
    u_col = (0.35 * lirf_net + 0.25 * lirf_tree) / 0.60
    a = np.where(airport == "LIRF", v9, base)
    b = np.where(airport == "LIRF", np.where(lirf_m, m_col, np.where(lirf_u, u_col, v9)), mix)
    return a, b


def ranking_frame() -> pd.DataFrame:
    r = pd.read_parquet(ROOT / RANKING)
    airport = r["airport"].to_numpy()
    scope = r["scope"].to_numpy()
    lirf_m, lirf_u = scope == "lirf_nonpop_matched", scope == "lirf_nonpop_unmatched"
    matched = (scope == "matched_no_lirf") | lirf_m
    a, b = sides(airport, matched, r["v9"].to_numpy(float), r["residual"].to_numpy(float), r["committee_mix"].to_numpy(float),
                 r["ens_ctx_lirf"].to_numpy(float), r["ens_v22_lirf_q"].to_numpy(float), lirf_m, lirf_u, is_base=False)
    v34 = pd.read_parquet(ROOT / V34).set_index(ID)[TARGET].reindex(r[ID]).to_numpy(float)
    if np.isnan(v34).any():
        raise ValueError("v34 does not cover the ranking rows.")
    return pd.DataFrame({ID: r[ID].to_numpy(), "period": r["month"].to_numpy(), "airport": airport, "matched": matched,
                         "group": group_of(airport, matched, (airport == "LIRF") & (scope == "fixed")),
                         "v9": r["v9"].to_numpy(float), "a": a, "b": b, "d": b - a, "pred": v34,
                         "committee": r["committee"].to_numpy(float), "nets": (0.25 * r["ens_ctx_e24"] + 0.25 * r["ens_ctx_e24b"] + 0.5 * r["ens_ctx_e32"]).to_numpy(float)})


def dev_frame() -> pd.DataFrame:
    f = pd.read_parquet(ROOT / FAIR)
    control = read_json(ROOT / "configs/fair_weights_v1.json")["control"]
    outer, lirf = candidate_weights(control, {"lirf_v9": {"lirf_nonpop_unmatched": 0.0}})
    if lirf != V34_LIRF:
        raise ValueError(f"v34's LIRF weights do not rebuild: {lirf}")
    airport, matched = f["airport"].to_numpy(), f["matched"].to_numpy()
    lirf_m, lirf_u = f["lirf_m"].to_numpy() & (airport == "LIRF"), f["lirf_u"].to_numpy() & (airport == "LIRF")
    a, b = sides(airport, matched, f["v9"].to_numpy(float), f["base"].to_numpy(float), f["mix"].to_numpy(float),
                 f["lirf_net"].to_numpy(float), f["lirf_tree"].to_numpy(float), lirf_m, lirf_u, is_base=True)
    return pd.DataFrame({ID: f[ID].to_numpy(), "period": f["month"].to_numpy(), "day": f["day"].to_numpy(), "airport": airport, "matched": matched,
                         "group": group_of(airport, matched | lirf_m, (airport == "LIRF") & ~lirf_m & ~lirf_u),
                         "v9": f["v9"].to_numpy(float), "a": a, "b": b, "d": b - a, "pred": build(f, outer, lirf), "label": f[TARGET if TARGET in f else "label"].to_numpy(float)})


def rates(frame: pd.DataFrame) -> dict:
    out = {}
    ad = np.abs(frame["d"].to_numpy())
    for period, idx in frame.groupby("period").indices.items():
        x = ad[idx]
        out[period] = {"rows": int(len(idx)), "rms_d": float(np.sqrt(np.mean(x ** 2))), "p99_abs_d": float(np.quantile(x, 0.99)),
                       "p999_abs_d": float(np.quantile(x, 0.999)), **{f"share_abs_d_gt_{t}": float(np.mean(x > t)) for t in [300, 600, 1200, 2400]},
                       "pred_gt_3h": int(np.sum(frame["pred"].to_numpy()[idx] > 10800)), "pred_gt_1h": int(np.sum(frame["pred"].to_numpy()[idx] > 3600)),
                       "pred_lt_0": int(np.sum(frame["pred"].to_numpy()[idx] < 0))}
    return out


def by_group(frame: pd.DataFrame) -> dict:
    out = {}
    for (period, group), g in frame.groupby(["period", "group"]):
        ad = np.abs(g["d"].to_numpy())
        out.setdefault(group, {})[period] = {"rows": int(len(g)), "rms_d": float(np.sqrt(np.mean(ad ** 2))), "mean_d": float(g["d"].mean()),
                                             "share_gt_600": float(np.mean(ad > 600)), "share_gt_1200": float(np.mean(ad > 1200))}
    return out


def dev_bins(dev: pd.DataFrame) -> dict:
    """Per |D| bin on the development months: rows, SSE share, RMSE of each side and of v34, and the least-squares weight on side b."""
    out = {}
    dev = dev.assign(bin=pd.cut(np.abs(dev["d"]), BINS, labels=LABELS, right=False))
    for period, p in [*dev.groupby("period"), ("pooled", dev)]:
        sse = float(np.sum((p["label"] - p["pred"]) ** 2))
        rows = {}
        for b, g in p.groupby("bin", observed=True):
            y, a, bb, pr = (g[c].to_numpy() for c in ["label", "a", "b", "pred"])
            d = bb - a
            w = float(np.sum((y - a) * d) / np.sum(d ** 2)) if np.sum(d ** 2) > 0 else float("nan")
            rows[str(b)] = {"rows": int(len(g)), "sse_share": float(np.sum((y - pr) ** 2) / sse), "rmse_a": float(np.sqrt(np.mean((y - a) ** 2))),
                            "rmse_b": float(np.sqrt(np.mean((y - bb) ** 2))), "rmse_v34": float(np.sqrt(np.mean((y - pr) ** 2))), "ls_weight_b": w,
                            "b_closer_share": float(np.mean(np.abs(y - bb) < np.abs(y - a)))}
        out[period] = rows
    return out


def projection(dev: pd.DataFrame, rank: pd.DataFrame, source_periods: list[str], use_bins: bool) -> dict:
    """Project v34's ranking RMSE from development MSE per cell; cells without development rows fall back to the group MSE."""
    src = dev[dev["period"].isin(source_periods)].copy()
    for f in (src, rank):
        f["bin"] = pd.cut(np.abs(f["d"]), BINS, labels=LABELS, right=False).astype(str) if use_bins else "all"
    src["se"] = (src["label"] - src["pred"]) ** 2
    cell = src.groupby(["group", "bin"])["se"].agg(["mean", "size"])
    grp = src.groupby("group")["se"].mean()
    apt = src.groupby("airport")["se"].mean()  # groups absent from development (e.g. LTFM:matched) fall back to the airport
    out = {}
    for period, p in rank.groupby("period"):
        counts = p.groupby(["group", "bin"]).size()
        sse, fallback = 0.0, 0
        for (g, b), n in counts.items():
            if (g, b) in cell.index and cell.loc[(g, b), "size"] >= 30:
                sse += n * cell.loc[(g, b), "mean"]
            else:
                sse += n * (grp[g] if g in grp.index else apt[g.split(":")[0]])
                fallback += n
        out[period] = {"rows": int(len(p)), "projected_rmse": float(np.sqrt(sse / len(p))), "rows_on_group_fallback": int(fallback)}
    total_sse = sum(out[p]["projected_rmse"] ** 2 * out[p]["rows"] for p in out)
    out["both"] = {"rows": int(len(rank)), "projected_rmse": float(np.sqrt(total_sse / len(rank)))}
    return out


def main() -> None:
    rank, dev = ranking_frame(), dev_frame()
    report = {"created_at_utc": utc_now(), "labels_read": "development months only (fair set); no ranking label exists",
              "inputs": {p: sha256_file(ROOT / p) for p in [RANKING, FAIR, V34]}, "official_v34": OFFICIAL_V34,
              "rates": {"development": rates(dev), "ranking": rates(rank)},
              "by_group": {"development": by_group(dev), "ranking": by_group(rank)},
              "dev_bins": dev_bins(dev)}
    dev_rmse = {m: float(np.sqrt(np.mean((g["label"] - g["pred"]) ** 2))) for m, g in dev.groupby("period")}
    report["dev_v34_rmse"] = dev_rmse
    report["projection"] = {f"{name}:{'bins' if b else 'cells'}": projection(dev, rank, periods, b)
                            for name, periods in [("december", ["2025-12"]), ("pooled", ["2025-01", "2025-07", "2025-11", "2025-12"]),
                                                  ("same_month", None)] if periods for b in [False, True]}
    for b in [False, True]:  # January from 2025-01, July from 2025-07
        jan = projection(dev, rank[rank["period"] == "2026-01"], ["2025-01"], b)["2026-01"]
        jul = projection(dev, rank[rank["period"] == "2026-07"], ["2025-07"], b)["2026-07"]
        both = float(np.sqrt((jan["projected_rmse"] ** 2 * jan["rows"] + jul["projected_rmse"] ** 2 * jul["rows"]) / (jan["rows"] + jul["rows"])))
        report["projection"][f"same_month:{'bins' if b else 'cells'}"] = {"2026-01": jan, "2026-07": jul, "both": {"projected_rmse": both}}
    out = ROOT / (sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else "docs/disagreement/audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    rank.to_parquet(ROOT / "runs/disagreement_rank.parquet", index=False)
    dev.to_parquet(ROOT / "runs/disagreement_dev.parquet", index=False)
    print("dev v34 RMSE", {k: round(v, 3) for k, v in dev_rmse.items()})
    print(pd.DataFrame({**{f"dev {k}": v for k, v in report["rates"]["development"].items()}, **{f"rank {k}": v for k, v in report["rates"]["ranking"].items()}}).round(4).to_string())
    for k, v in report["projection"].items():
        print(k, {p: round(e["projected_rmse"], 2) for p, e in v.items()})


if __name__ == "__main__":
    main()
