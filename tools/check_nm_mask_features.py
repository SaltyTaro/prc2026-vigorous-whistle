"""Audit masked inputs, every retained predictor and a stratified reconstruction."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import write_json, sha256_file, utc_now
from prc2026.features import ID, TARGET, BLOCK
from prc2026_phase5r.features import GROUND_NAMES
from prc2026_nm_mask.bundle import load_bundle
from prc2026_nm_mask.features import (AIRPORT_INPUTS, UNCHANGED_NAMES, MASKED_FIELDS,
    masked_row_features, masked_ground_features, eligible_route)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/nm_mask/local_feature_verification.json"))
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise ValueError("Use a fresh verification receipt.")
    base, manifest = load_bundle(ROOT)
    rng = np.random.default_rng(20260915)
    samples, complete_rows, unchanged_values, offsets_checked = [], 0, 0, 0
    for source in sorted(manifest["coverage"]):
        new = pd.read_parquet(base / "features" / source)
        inherited = pd.read_parquet(ROOT / manifest["weather_bundle"] / "features" / source,
                                   columns=[ID, *UNCHANGED_NAMES, "conditional_schedule_offset_s"])
        pd.testing.assert_frame_equal(new[[ID, *UNCHANGED_NAMES]], inherited[[ID, *UNCHANGED_NAMES]], check_exact=True)
        routing = pd.read_parquet(base / "routing" / source)
        np.testing.assert_array_equal(new[ID], routing[ID])
        natural = eligible_route(routing)
        np.testing.assert_array_equal(new.conditional_schedule_offset_s.to_numpy()[natural],
                                      inherited.conditional_schedule_offset_s.to_numpy()[natural])
        if not new.nm_missing.eq(1).all() or not new.nm_clock_count.eq(0).all():
            raise ValueError("NM information remains in the counterfactual view.")
        positions = set(rng.choice(len(new), min(100, len(new)), replace=False))
        positions.update([0, len(new) - 1])
        positions.update(np.flatnonzero(natural)[:10])
        samples.append(new.iloc[sorted(positions)].copy())
        complete_rows += len(new)
        unchanged_values += len(new) * len(UNCHANGED_NAMES)
        offsets_checked += int(natural.sum())
    sampled = pd.concat(samples, ignore_index=True)
    context = pd.read_parquet(base / "inputs/airport_context.parquet")
    if list(context) != AIRPORT_INPUTS or set(context) & (set(MASKED_FIELDS) | {TARGET, BLOCK}):
        raise ValueError("A prohibited field entered the airport-only source snapshot.")
    chosen = context.loc[context[ID].isin(sampled[ID])]
    rows = masked_row_features(chosen)
    rows = sampled[[ID]].merge(rows, on=ID, validate="one_to_one", sort=False)
    pd.testing.assert_frame_equal(sampled[list(rows)].reset_index(drop=True), rows, check_exact=True)
    ground = masked_ground_features(context, sampled[ID])
    pd.testing.assert_frame_equal(sampled[[ID, *GROUND_NAMES]].reset_index(drop=True), ground, check_exact=True)
    report = {"created_at_utc": utc_now(), "passed": True, "bundle_id": manifest["bundle_id"],
        "manifest_sha256": sha256_file(base / "manifest.json"), "complete_feature_rows": complete_rows,
        "unchanged_inherited_values_checked": unchanged_values, "original_conditional_offsets_checked": offsets_checked,
        "source_context_rows": len(context), "source_projection": AIRPORT_INPUTS,
        "reconstructed_sample_rows": len(sampled), "reconstructed_row_values": len(sampled) * (len(rows.columns) - 1),
        "reconstructed_ground_values": len(sampled) * len(GROUND_NAMES), "maximum_difference": 0.0,
        "departure_labels_read": False, "audit_used": False, "verifier_code_sha256": sha256_file(Path(__file__))}
    write_json(output, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
