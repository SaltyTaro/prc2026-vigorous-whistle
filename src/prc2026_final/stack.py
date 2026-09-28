"""S11-S13: the layers from v30 to v35 / v36 / v37 / v38, each through the frozen code that first built it for the ranking file.

- repair (v31, docs/input_shift/lfpg_eobt_sensitivity.md). A label-free detector flags an airport-month when its share of departures
  with EOBT_1 > IOBT lies more than one 2025 span above the 2025 maximum (reference: the same airport's twelve 2025 months, read from
  data/raw/training_2025-*.parquet; span = maximum - minimum). The audit's finding was LFPG July 2026 at 35% against 4-7%. In a flagged
  airport-month, EOBT_1 := IOBT on the departures with EOBT_1 > IOBT (the sensitivity run's edit) and the chain features..recipe runs
  again on that input; v31 = max(v30 + 0.5 x (v30 on the edited input - v30), -12), as tools/export_lfpg_eobt_blend.py (weight 0.5).
  No flag: v31 = v30.
- rematch (the callsign fix's input): tools/adsb_rematch_nonm.py, unchanged, with its REPO pointed at this workspace, on the file's
  departures without an NM flight record (FLIGHT_ID_mvt missing) at EDDF / EHAM, the declared scope of configs/callsign_fix_v1.json.
- stack: the recipe deltas on the file's own recipe detail, as the exports took them from the dry-run detail:
  v33 = max(v31 + export_v33.recipe(detail, 1.0, 0.6, 0.7) - export_v33.recipe(detail, 0.7, 0.7, 0.7), -12);
  v34 = max(v33 + export_fair_weights.recipe(detail, V33_OUTER, V34_LIRF) - recipe(detail, V33_OUTER, V33_LIRF), -12);
  then tools/export_stack.layers itself, for each layer set:
  v35 = layers(v34, ..., queue 0.75, regular): u300 gate (export_disagreement_gate.gate_delta), callsign fix 50%
        (run_callsign_fix.scope_table / offsets), queue trust 0.75 on regular gate matches (run_queue_trust segments / offsets_all);
  v36 = layers(v34, ..., queue 0.75, regular, mix_up=600, peer=True): the same after export_stack.gated_detail's mu600 mix gate
        (network share 1.0 where nets - committee > 600 s; recipe and u300 recomputed on the gated mix), then export_stack.peer_lifts
        (configs/higher_bundle_v2.json mu_pc_pcc: +100 s no-clock / +50 s untracked non-LIRF rows in airport-hours whose NM proxies
        run > 600 s above the month's normal, >= 5 peers).
  v37 = layers(v34, ..., queue 0.75, regular, stretch=1500): v35, then export_stack.tail_stretch after every other layer
        (configs/tail_stretch_v1.json v35_t1500: value + k x (value - 1500) above 1,500 s on non-LIRF rows outside the callsign and
        queue layers, k untracked / tracked from docs/disagreement/tail_stretch_v1_results.json's four-month fit; floor -12).
  v38 = layers(v34, ..., queue 0.75, regular, stretch=1500, stretch_lower=900): v35, then the two-sided stretch on every non-LIRF row
        outside the callsign and queue layers (configs/tail_stretch_v2.json two_sided_900): u = value + k1 x max(value - 1500, 0),
        value = max(u + k2 x max(900 - u, 0), -12), k1 / k2 per untracked / tracked from tail_stretch_v2_results.json.
  v39 = v38 with the fill pins (configs/fill_pin_v1.json, src/prc2026_final/fill_pin.py): the file's own LIRF departures with a
        schedule interval above 43,200 s that are either the v9 population (no NM record, no aircraft type) or carry a one-day-early
        NM off-block clock take their schedule interval; no model, no fitted coefficient (the user's decision of 28 September).
  export_stack's three ranking-bound readers (scope_table, segments, ranking_peers) answer from the run's own tables during the
  call (run_tables): the file's rematch, its residual match tables and tools/peer_congestion.peer_table on the file itself.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import object_hash, read_json, sha256_file
from prc2026.features import ID, PHASE, TIME

from .common import ROOT, save_frame, source_file
from .fill_pin import COLUMNS as FILL_COLUMNS, SCOPES as FILL_SCOPES, apply_pins, candidate_pins, fill_pins

sys.path.insert(0, str(ROOT / "tools"))
import export_v33  # noqa: E402
from export_disagreement_gate import V34_LIRF  # noqa: E402
from export_fair_weights import FLOOR, V30_OUTER, V33_LIRF, V33_OUTER, recipe  # noqa: E402
from peer_congestion import COLUMNS as PEER_COLUMNS, peer_table  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_queue_trust import SEGMENT, offsets_all  # noqa: E402

REFERENCE_MONTHS = [f"2025-{m:02}" for m in range(1, 13)]
EOBT_COLUMNS = [PHASE, "ADEP_mvt", TIME, "EOBT_1_flt", "IOBT_flt"]
LFPG_WEIGHT = 0.5  # v31: tools/export_lfpg_eobt_blend.py --weight 0.5 (docs/input_shift/export_v31.json)
REMATCH_AIRPORTS = ["EDDF", "EHAM"]  # configs/callsign_fix_v1.json: no-NM departures at EDDF / EHAM
REMATCH_COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_ID_mvt"]
REMATCH_OUTPUT_COLUMNS = ["MVT_ID_mvt", "airport", "flight", "mapped", "takeoff"] + [f"{m}_{k}" for m in ["iata_mapped", "key"] for k in
                                                                                   ["icao", "gate", "motion_start", "segment_start", "last_ground",
                                                                                    "taxi_path_m", "taxi_stops", "taxi_stopped_s", "at_known_stand"]]
DECLARATIONS = {"gate": "configs/disagreement_gate_v1.json", "callsign": "configs/callsign_fix_v1.json", "queue": "configs/queue_trust_v1.json",
                "mix": "configs/mix_gate_v1.json", "peer": "configs/higher_bundle_v2.json", "stretch": "configs/tail_stretch_v1.json",
                "stretch2": "configs/tail_stretch_v2.json", "fill_pin": "configs/fill_pin_v1.json", "fill_pin2": "configs/fill_pin_v2.json",
                "fill_pin3": "configs/fill_pin_v3.json"}
RESULTS = ["docs/disagreement/tail_stretch_v1_results.json", "docs/disagreement/tail_stretch_v2_results.json"]  # the stretches' k (four-month fits), read by export_stack.tail_stretch
CODE = ["tools/export_stack.py", "tools/peer_congestion.py", "tools/export_disagreement_gate.py", "tools/export_fair_weights.py", "tools/export_v33.py",
        "tools/run_callsign_fix.py", "tools/run_queue_trust.py", "tools/adsb_rematch_nonm.py"]
LAYER_SETS = ["v35", "v36", "v37", "v38", "v39", "v40", "v41"]
V35 = {"gate": "u300", "callsign": "u300_cs50", "queue": "qt75", "queue_source": "regular",
       "record": "docs/disagreement/export_v35.json (tools/export_stack.py --queue 0.75 --queue-source regular --version 35, user override)"}
V36 = {**V35, "peer": "mu_pc_pcc",
       "record": "docs/disagreement/export_v36.json (tools/export_stack.py --queue 0.75 --queue-source regular --mix-up 600 --peer-lift --version 36, user override)"}
V37 = {**V35, "stretch": "v35_t1500",
       "record": "docs/disagreement/export_v37.json (tools/export_stack.py --queue 0.75 --queue-source regular --stretch 1500 --version 37, user override)"}
V38 = {**V35, "stretch": "two_sided_900",
       "record": "docs/disagreement/export_v38.json (tools/export_stack.py --queue 0.75 --queue-source regular --stretch 1500 --stretch-lower 900 --version 38, user override)"}
V39 = {**V38, "fill_pin": "all_twelve",
       "record": "docs/fill_pin/export_v39.json (tools/export_fill_pin.py --version 39, the user's decision on the declared candidates)"}
V40 = {**V38, "fill_pin": "evidenced_nine",
       "record": "docs/fill_pin/export_v40.json (tools/export_fill_pin.py --declaration configs/fill_pin_v2.json --candidate evidenced_nine --version 40, the user's decision)"}
V41 = {**V40, "fill_pin3": "nosos431_one_day",
       "record": "docs/fill_pin/export_v41.json (tools/export_fill_pin.py --declaration configs/fill_pin_v3.json --candidate nosos431_one_day --base-version 40 --version 41, the user's decision)"}


def pins(raw_path: Path) -> pd.DataFrame:
    """src/prc2026_final/fill_pin.fill_pins on the movement file itself (label-free: schedule, takeoff and NM clock columns only)."""
    return fill_pins(pd.read_parquet(raw_path, columns=FILL_COLUMNS))


# ---------------------------------------------------------------- repair (v31)

def eobt_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Per departure airport and movement month: departures, departures with EOBT_1 > IOBT (a missing clock counts as not later), share."""
    dep = frame.loc[frame[PHASE].eq("DEP")]
    table = pd.DataFrame({"airport": dep["ADEP_mvt"].to_numpy(), "month": dep[TIME].dt.strftime("%Y-%m").to_numpy(),
                          "later": (dep["EOBT_1_flt"] > dep["IOBT_flt"]).fillna(False).to_numpy(dtype=bool)})
    out = table.groupby(["airport", "month"]).agg(departures=("later", "size"), later=("later", "sum")).reset_index()
    out["share"] = out["later"] / out["departures"]
    return out


def reference_shares() -> pd.DataFrame:
    """The 2025 reference per airport: minimum, maximum and span of the twelve monthly shares."""
    frames = [pd.read_parquet(ROOT / "data/raw" / source_file(m), columns=EOBT_COLUMNS) for m in REFERENCE_MONTHS]
    shares = eobt_shares(pd.concat(frames, ignore_index=True))
    shares = shares[shares["month"].isin(REFERENCE_MONTHS)]
    ref = shares.groupby("airport")["share"].agg(reference_min="min", reference_max="max", reference_months="size").reset_index()
    ref["span"] = ref["reference_max"] - ref["reference_min"]
    ref["threshold"] = ref["reference_max"] + ref["span"]
    return ref


def detect(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Flagged (airport, month) pairs of the file and the full table (share, 2025 reference, threshold, margin = share - threshold)."""
    reference = reference_shares() if reference is None else reference
    table = eobt_shares(frame).merge(reference, on="airport", how="left", validate="many_to_one")
    table["margin"] = table["share"] - table["threshold"]
    table["flagged"] = (table["reference_months"] == len(REFERENCE_MONTHS)) & (table["margin"] > 0)
    return [(a, m) for a, m in table.loc[table["flagged"], ["airport", "month"]].itertuples(index=False)], table


def reset_input(raw: pd.DataFrame, flagged: list[tuple[str, str]]) -> tuple[pd.DataFrame, np.ndarray]:
    """The raw file with EOBT_1 := IOBT on the flagged airport-months' departures whose EOBT_1 > IOBT; (edited frame, edited row mask)."""
    month = raw[TIME].dt.strftime("%Y-%m")
    in_scope = np.zeros(len(raw), dtype=bool)
    for airport, m in flagged:
        in_scope |= (raw[PHASE].eq("DEP") & raw["ADEP_mvt"].eq(airport) & month.eq(m)).to_numpy(dtype=bool)
    edit = in_scope & (raw["EOBT_1_flt"] > raw["IOBT_flt"]).fillna(False).to_numpy(dtype=bool)
    out = raw.copy()
    out.loc[edit, "EOBT_1_flt"] = out.loc[edit, "IOBT_flt"]
    return out, edit


def same_content(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.columns) == list(b.columns) and len(a) == len(b) and all(a[c].equals(b[c]) for c in a.columns)


# ---------------------------------------------------------------- rematch (v35 callsign fix input)

def rematch_departures(departures: pd.DataFrame) -> pd.DataFrame:
    keep = departures["FLIGHT_ID_mvt"].isna() & departures["ADEP_mvt"].isin(REMATCH_AIRPORTS)
    return departures.loc[keep, REMATCH_COLUMNS].reset_index(drop=True)


def rematch(deps_path: Path, out_path: Path) -> None:
    """tools/adsb_rematch_nonm.py's main, unchanged, with REPO = this workspace (stands: matches_v11; surface: data/external/adsblol/surface)."""
    import adsb_rematch_nonm as frozen
    saved = frozen.REPO, sys.argv
    frozen.REPO, sys.argv = ROOT, ["adsb_rematch_nonm.py", str(deps_path), str(out_path)]
    try:
        frozen.main()
    finally:
        frozen.REPO, sys.argv = saved
    out = pd.read_parquet(out_path)
    if out.empty and not len(out.columns):  # no departure had a surface day: the frozen tool writes a frame without columns
        save_frame(pd.DataFrame({c: pd.Series(dtype=float) for c in REMATCH_OUTPUT_COLUMNS}), out_path)




# ---------------------------------------------------------------- stack (v31 -> v35 / v36)

def queue_segments(regular: pd.DataFrame, rematch_out: pd.DataFrame, callsign_ids: set) -> pd.DataFrame:
    """run_queue_trust.segments on the run's own tables: regular gate matches (the residual match dir, matches_v11's stands) plus the
    callsign fix's scope rows with their segment shape; same columns, rule and duplicate guard."""
    regular = regular.loc[regular["gate"].astype(bool), SEGMENT].assign(source="regular")
    rem = rematch_out[rematch_out[ID].isin(callsign_ids)].rename(columns={"iata_mapped_motion_start": "motion_start", "iata_mapped_taxi_stops": "taxi_stops",
                                                                         "iata_mapped_taxi_stopped_s": "taxi_stopped_s", "iata_mapped_taxi_path_m": "taxi_path_m"})
    seg = pd.concat([regular, rem[SEGMENT].assign(source="callsign")], ignore_index=True)
    if seg[ID].duplicated().any():
        raise ValueError("A departure has both a regular and a callsign segment.")
    seg["taxi"] = seg["takeoff"] - seg["motion_start"]
    seg["in_scope"] = (seg["taxi_stops"] >= 5) & (seg["taxi_path_m"] >= 5000) & (seg["taxi_stopped_s"] / seg["taxi"] <= 0.75) & (seg["taxi"] > 3000)
    return seg.set_index(ID)


def peers(raw_path: Path) -> pd.DataFrame:
    """tools/peer_congestion.peer_table on the movement file itself (its departures' own NM clocks; label-free)."""
    return peer_table(pd.read_parquet(raw_path, columns=PEER_COLUMNS))


def layer_specs() -> dict:
    """export_stack.layers' arguments per layer set, from the declarations (v35 / v36 / v37 / v38 = the user's overrides recorded in
    docs/disagreement/export_v35.json .. export_v38.json)."""
    queue = float(read_json(ROOT / DECLARATIONS["queue"])["candidates"][V35["queue"]]["weight"])
    bundle = read_json(ROOT / DECLARATIONS["peer"])["candidates"][V36["peer"]]
    stretch = read_json(ROOT / DECLARATIONS["stretch"])["candidates"][V37["stretch"]]
    if stretch["control"] != "v35":
        raise ValueError("v37's stretch is declared on the v35 stack.")
    two = read_json(ROOT / DECLARATIONS["stretch2"])["candidates"][V38["stretch"]]
    base = {"queue": queue, "queue_source": V35["queue_source"], "mix_up": None, "peer": False, "stretch": None, "stretch_lower": None}
    return {"v35": base, "v36": {**base, "mix_up": float(bundle["net_up"]), "peer": True}, "v37": {**base, "stretch": float(stretch["t"])},
            "v38": {**base, "stretch": float(two["t_upper"]), "stretch_lower": float(two["t_lower"])}}


@contextmanager
def run_tables(callsign: pd.DataFrame, segments: pd.DataFrame, peer: pd.DataFrame):
    """tools/export_stack with its three ranking-bound readers answering from the run's own tables: scope_table (the declared ranking
    rematch outputs), segments (matches_v11's 2026 months) and ranking_peers (data/raw/ranking.parquet). Everything else is its own code."""
    import export_stack
    patches = {"scope_table": lambda decl: callsign.copy(), "segments": lambda months: segments.copy(), "ranking_peers": lambda: peer.copy()}
    saved = {name: getattr(export_stack, name) for name in patches}
    try:
        for name, function in patches.items():
            setattr(export_stack, name, function)
        yield export_stack
    finally:
        for name, function in saved.items():
            setattr(export_stack, name, function)


def compute(detail: pd.DataFrame, repaired_v30: np.ndarray | None, rematch_path: Path, regular: pd.DataFrame,
            peer: pd.DataFrame, fill: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`detail`: the recipe stage's detail (v30's columns); `repaired_v30`: v30 on the repaired input in detail order (None: no repair);
    `regular`: the run's residual match tables; `peer`: peers() of the file; `fill`: pins() of the file (v39's fill pins).
    Returns v30 .. v34 and every layer set's values and masks."""
    ids = detail[ID].to_numpy()
    v30 = detail["v30"].to_numpy(dtype=float)
    # the detail must be v30's recipe under both frozen rebuilds (export_v33 at 0.7; export_fair_weights at v30's outer and LIRF weights)
    checks = {"export_v33_recipe_at_0.7": float(np.max(np.abs(export_v33.recipe(detail, 0.7, 0.7, 0.7) - v30))),
              "fair_weights_recipe_at_v30": float(np.max(np.abs(recipe(detail, V30_OUTER, V33_LIRF) - v30)))}
    if not max(checks.values()) <= 1e-9:
        raise ValueError(f"The detail is not v30's recipe: {checks}")
    effect = np.zeros(len(ids)) if repaired_v30 is None else np.asarray(repaired_v30, dtype=float) - v30
    if not np.isfinite(effect).all():
        raise ValueError("The repair effect has non-finite values.")
    # tools/export_disagreement_gate.rebuild_v34's arithmetic, with the file's own v30 in place of the recorded ranking v30
    v31 = np.maximum(v30 + LFPG_WEIGHT * effect, FLOOR)
    v33 = np.maximum(v31 + export_v33.recipe(detail, 1.0, 0.6, 0.7) - export_v33.recipe(detail, 0.7, 0.7, 0.7), FLOOR)
    at33, at34 = recipe(detail, V33_OUTER, V33_LIRF), recipe(detail, V33_OUTER, V34_LIRF)
    v34 = np.maximum(v33 + at34 - at33, FLOOR)

    rematch_out = pd.read_parquet(rematch_path)
    callsign = scope_table({"rematch": {"outputs": {rematch_path.as_posix(): sha256_file(rematch_path)}}})
    segments = queue_segments(regular, rematch_out, set(callsign[ID]))
    if not np.array_equal(np.sort(peer[ID].to_numpy()), np.sort(ids)):
        raise ValueError("The peer table does not cover exactly the file's departures.")
    frame = pd.DataFrame({ID: ids, "airport": detail["airport"].to_numpy(), "month": detail["month"].to_numpy(), "scope": detail["scope"].to_numpy(),
                          "v30": v30, "repair_effect": effect, "v31": v31, "v33": v33, "v34": v34})
    specs = layer_specs()
    rows: dict = {"repair_effect_nonzero": int((effect != 0).sum())}
    with run_tables(callsign, segments, peer) as export_stack:
        for name, spec in specs.items():
            values, masks = export_stack.layers(v34, detail, at34, ids, spec["queue"], spec["queue_source"], spec["mix_up"], spec["peer"], spec["stretch"],
                                                spec["stretch_lower"])
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {name} values.")
            frame[name] = values
            for layer, mask in masks.items():
                frame[f"{name}_{layer}"] = np.asarray(mask, dtype=bool)
            rows[name] = {layer: int(np.sum(mask)) for layer, mask in masks.items()}
            rows[name]["floored"] = int((values == FLOOR).sum())
    # v39 = v38 with the fill pins (configs/fill_pin_v1.json): the file's own in-scope LIRF departures at their schedule interval
    v39, pinned = apply_pins(frame["v38"].to_numpy(dtype=float), ids, fill, detail["airport"].to_numpy())
    frame["v39"], frame["v39_fill_pin"] = v39, pinned
    scope = pd.DataFrame({ID: ids}).merge(fill[[ID, "scope"]], on=ID, how="left", validate="one_to_one")["scope"] if len(fill) else pd.Series([None] * len(ids))
    rows["v39"] = {"fill_pin": int(pinned.sum()), **{f"fill_pin_{s}": int(scope.eq(s).sum()) for s in FILL_SCOPES}, "floored": int((v39 == FLOOR).sum())}
    # v40 = v38 with configs/fill_pin_v2.json's evidenced_nine: population rows only, in the declared interval bands
    nine = read_json(ROOT / DECLARATIONS["fill_pin2"])["candidates"][V40["fill_pin"]]
    v40, pinned40 = apply_pins(frame["v38"].to_numpy(dtype=float), ids, candidate_pins(fill, nine), detail["airport"].to_numpy())
    frame["v40"], frame["v40_fill_pin"] = v40, pinned40
    rows["v40"] = {"fill_pin": int(pinned40.sum()), "floored": int((v40 == FLOOR).sum())}
    # v41 = v40 with configs/fill_pin_v3.json's nosos431_one_day: population rows in its band at the one-day value (86,400 s + the 2025 one-day taxi mean)
    one_day = read_json(ROOT / DECLARATIONS["fill_pin3"])["candidates"][V41["fill_pin3"]]
    v41, pinned41 = apply_pins(v40, ids, candidate_pins(fill, one_day), detail["airport"].to_numpy())
    frame["v41"], frame["v41_one_day"] = v41, pinned41
    rows["v41"] = {"fill_pin": int(pinned40.sum()), "one_day": int(pinned41.sum()), "floored": int((v41 == FLOOR).sum())}
    specs = {**specs, "v39": {"base": "v38", "fill_pin": DECLARATIONS["fill_pin"], "pinned_value": "schedule interval (takeoff minus scheduled departure) in seconds"},
             "v40": {"base": "v38", "fill_pin": DECLARATIONS["fill_pin2"], "candidate": V40["fill_pin"], "scopes": nine["scopes"],
                     "population_interval_bands_s": nine["population_interval_bands_s"], "pinned_value": "schedule interval in seconds"},
             "v41": {"base": "v40", "fill_pin": DECLARATIONS["fill_pin3"], "candidate": V41["fill_pin3"], "scopes": one_day["scopes"],
                     "population_interval_bands_s": one_day["population_interval_bands_s"], "pinned_value": one_day["pinned_value"]}}
    record = {"recipe_checks_max_abs": checks, "lfpg_weight": LFPG_WEIGHT, "floor_seconds": FLOOR, "v33_outer": V33_OUTER, "v33_lirf": V33_LIRF,
              "v34_lirf": V34_LIRF, "layer_sets": specs, "shipped_as": {"v35": V35, "v36": V36, "v37": V37, "v38": V38, "v39": V39, "v40": V40, "v41": V41},
              "declarations": {p: {"sha256": sha256_file(ROOT / p), "object_hash": object_hash(read_json(ROOT / p))} for p in DECLARATIONS.values()},
              "results": {p: sha256_file(ROOT / p) for p in RESULTS},
              "stretch_k": {"v37": read_json(ROOT / RESULTS[0])["candidates"][V37["stretch"]]["k_all_months"],
                            "v38": read_json(ROOT / RESULTS[1])["candidates"][V38["stretch"]]["k_all_months"]},
              "code": {p: sha256_file(ROOT / p) for p in CODE}, "offsets_callsign_s": offsets(), "offsets_queue_s": offsets_all(),
              "rows": rows, "rematch_rows": int(len(rematch_out)), "callsign_scope_rows": int(len(callsign)),
              "peer": {"departures": int(len(peer)), "rows_with_5_peers": int((peer["peer_n"] >= 5).sum()), "no_clock": int(peer["no_clock"].sum())}}
    return frame, record
