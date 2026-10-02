"""The LIRF layers v53 / v54 / v55 add to v52's model part, on any file's rows (layer sets v53m / v54m / v55m in stack.py).

M = the 16-seed mean of the masked LIRF context network lm25_32 (configs/adsb_ctx3_lirf_mask_dev_v1.json, seeds 801-816, twelve-month
refits; the members stage predicts it on every row). Each layer adds a delta, against the shipped arithmetic, on one LIRF route of the
recipe detail (v9, ens_ctx_lirf = the shipped LIRF network, ens_v22_lirf_q = the LIRF tree), through the frozen code that built the
ranking exports:
- v53m (configs/adsb_ctx3_lirf_mask_dev_v1.json, w55 selected by its rule; tools/export_lirf_member.py): on lirf_nonpop_unmatched rows,
  v34's unmatched blend with v45's gate u600 on network' = (1 - w) ens_ctx_lirf + w M minus the same on ens_ctx_lirf.
- v54m (configs/lirf_dark_reread_v1.json, w70_s750 selected by its rule; tools/export_lirf_dark.py, tools/run_lirf_dark.values): on
  lirf_nonpop_unmatched rows of dark LIRF months only (a month whose LIRF non-population rows have an ADS-B match share below 5%,
  read from the file's own scopes; the ranking file: July 2026), the selected (w, s) minus v53's (0.55, 0.35/0.60), gate u600.
- v55m (configs/lirf_combo_reread_v1.json, mw40 selected by its rule; tools/export_lirf_combo.py, tools/run_lirf_combo.values): on
  lirf_nonpop_matched rows, v34's matched blend with v45's gate m600 on network' minus the same on ens_ctx_lirf. v55 was not uploaded.
v53 / v54 then add the published daily totals (tools/export_daily_totals.py --base), which is not part of this runner: v53m / v54m /
v55m are the open-data model parts. No label is read here.
"""
from __future__ import annotations

import sys

import numpy as np

from prc2026.artifacts import read_json

from . import lirf_gate
from .common import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import run_lirf_combo  # noqa: E402
import run_lirf_dark  # noqa: E402
import run_lirf_member  # noqa: E402

MASK_DEV, DARK, COMBO = "configs/adsb_ctx3_lirf_mask_dev_v1.json", "configs/lirf_dark_reread_v1.json", "configs/lirf_combo_reread_v1.json"
SHIPPED = {MASK_DEV: "w55", DARK: "w70_s750", COMBO: "mw40"}
UNMATCHED, MATCHED = "lirf_nonpop_unmatched", "lirf_nonpop_matched"
DARK_SHARE = 0.05
CODE = ["src/prc2026_final/lirf_members.py", "src/prc2026_final/lirf_gate.py", "tools/run_lirf_member.py", "tools/run_lirf_dark.py", "tools/run_lirf_combo.py",
        "tools/export_lirf_member.py", "tools/export_lirf_dark.py", "tools/export_lirf_combo.py"]


def selections() -> dict:
    """Each declaration's candidate as its report selected it (and as it shipped): {declaration: (name, spec)}."""
    out = {}
    for path, shipped in SHIPPED.items():
        decl = read_json(ROOT / path)
        selected = read_json(ROOT / decl["report"])["selected_candidate"]
        if selected != shipped:
            raise ValueError(f"{path}: the report selects {selected!r}, the runner builds {shipped!r}.")
        out[path] = (shipped, dict(decl["candidates"][shipped]))
    return out


def learners() -> list[str]:
    """The masked LIRF network's declared learners' directories (each holds refit12/)."""
    fits = read_json(ROOT / MASK_DEV)["fits"]
    if fits["variants"] != ["lm25_32"]:
        raise ValueError(f"{MASK_DEV} declares {fits['variants']}, not lm25_32 alone.")
    base = fits["outputs"].split("/nets/")[0]
    return [f"{base}/nets/lm25_32_s{seed}" for seed in fits["seeds"]]


def dark_months(scope: np.ndarray, month: np.ndarray) -> dict:
    """{month: {matched, unmatched, match_share, dark}} over the LIRF non-population scopes (label-free)."""
    scope, month = np.asarray(scope), np.asarray(month)
    out = {}
    for m in sorted(set(month[np.isin(scope, [UNMATCHED, MATCHED])])):
        nm, nu = int(((scope == MATCHED) & (month == m)).sum()), int(((scope == UNMATCHED) & (month == m)).sum())
        out[m] = {"matched": nm, "unmatched": nu, "match_share": nm / (nm + nu), "dark": bool(nm / (nm + nu) < DARK_SHARE)}
    return out


def deltas(v9: np.ndarray, net: np.ndarray, tree: np.ndarray, member: np.ndarray, scope: np.ndarray, month: np.ndarray) -> tuple[dict, dict]:
    """({layer: (rows, delta on those rows)}, record) for v53m / v54m / v55m, each against the shipped arithmetic before it."""
    v9, net, tree, member = (np.asarray(a, dtype=float) for a in (v9, net, tree, member))
    scope, month = np.asarray(scope), np.asarray(month)
    chosen = selections()
    w53 = float(chosen[MASK_DEV][1]["member_weight"])
    w54, s54 = float(chosen[DARK][1]["member_weight"]), float(chosen[DARK][1]["network_share"])
    w55 = float(chosen[COMBO][1]["member_weight"])
    if run_lirf_dark.CONTROL["member_weight"] != w53:
        raise ValueError("The dark re-read's control is not v53's member weight.")
    gate = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    if gate["unmatched"] != run_lirf_member.GATE or gate["matched"] != run_lirf_combo.GATE:
        raise ValueError("The LIRF member scorers' gates are not v45's u600_m600.")
    shares = dark_months(scope, month)
    dark = [m for m, v in shares.items() if v["dark"]]
    out, record = {}, {"weights": {"v53m": w53, "v54m": [w54, s54], "v55m": w55}, "match_shares": shares, "dark_months": dark}
    for name, rows in [("v53m", scope == UNMATCHED), ("v54m", (scope == UNMATCHED) & np.isin(month, dark)), ("v55m", scope == MATCHED)]:
        if not np.isfinite(member[rows]).all():
            raise ValueError(f"{name}: the masked LIRF member is missing on a route row.")
        s = {"v9": v9[rows], "net": net[rows], "tree": tree[rows], "member": member[rows]}
        if name == "v53m":
            control, fire0 = lirf_gate.gated(s["v9"], s["net"], s["tree"], "unmatched", run_lirf_member.GATE)
            candidate, fire1 = lirf_gate.gated(s["v9"], (1 - w53) * s["net"] + w53 * s["member"], s["tree"], "unmatched", run_lirf_member.GATE)
        elif name == "v54m":
            control, fire0 = run_lirf_dark.values(s, w53, run_lirf_dark.CONTROL["network_share"], 600.0)
            candidate, fire1 = run_lirf_dark.values(s, w54, s54, 600.0)
        else:
            control, fire0 = run_lirf_combo.values(s["v9"], s["net"], s["tree"], None, 0.0)
            candidate, fire1 = run_lirf_combo.values(s["v9"], s["net"], s["tree"], s["member"], w55)
        out[name] = (rows, candidate - control, control)
        record[name] = {"route_rows": int(rows.sum()), "gate_rows_control": int(fire0.sum()), "gate_rows": int(fire1.sum())}
    return out, record


__all__ = ["MASK_DEV", "DARK", "COMBO", "SHIPPED", "DARK_SHARE", "CODE", "selections", "learners", "dark_months", "deltas"]
