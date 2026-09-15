"""Check all static rows and independently reconstruct bounded route-queue samples."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TIME
from prc2026.phase1 import validate_inputs
from prc2026_rome_geometry.bundle import load_bundle, load_maps, read_context
from prc2026_rome_geometry.common import configurations
from prc2026_rome_geometry.features import checked_context, STATIC_NAMES, QUEUE_NAMES, seconds


def independent_queue(context, query, mapped):
    """Full-array masks, separate from the producer's sorted event windows."""
    query_key = (query.stand, query.runway)
    if query_key not in mapped["routes"]:
        return {name:np.nan for name in QUEUE_NAMES}
    route = mapped["route_index"][query_key]
    peers = np.array([mapped["route_index"].get((stand, runway), -1) for stand, runway in zip(context.stand,context.runway)])
    weights = np.zeros(len(context))
    valid = peers >= 0
    weights[valid] = mapped["shared_fraction"][route,peers[valid]]
    different = context[ID].ne(query[ID]).to_numpy()
    if pd.notna(query.FLIGHT_ID_mvt):
        different &= context.FLIGHT_ID_mvt.ne(query.FLIGHT_ID_mvt).to_numpy()
    different &= weights > 0
    anchor = query.anchor.timestamp()
    starts, ends = seconds(context.offblock), seconds(context[TIME])
    proxy = context.proxy_s.to_numpy(dtype=float)
    ages = anchor - starts
    pushed = different & (starts < anchor) & (ages <= 21600)
    result = {}
    for window in [15,60]:
        inside = pushed & (ages <= window * 60)
        result[f"geo_shared_push_count_{window}m"] = inside.sum()
        result[f"geo_shared_push_equivalent_{window}m"] = weights[inside].sum()
        result[f"geo_shared_push_max_fraction_{window}m"] = weights[inside].max() if inside.any() else 0.
        completed = different & (ends >= anchor - window * 60) & (ends < anchor)
        result[f"geo_shared_service_count_{window}m"] = completed.sum()
        finite = proxy[completed & np.isfinite(proxy)]
        result[f"geo_shared_service_proxy_mean_s_{window}m"] = finite.mean() if len(finite) else np.nan
    result["geo_shared_push_last_s"] = ages[pushed].min() if pushed.any() else np.nan
    active = pushed & (ends >= anchor) & (proxy > 0) & (proxy <= 21600)
    result["geo_shared_active_count"] = active.sum()
    result["geo_shared_active_equivalent"] = weights[active].sum()
    result["geo_shared_active_max_age_s"] = ages[active].max() if active.any() else np.nan
    result["geo_shared_active_mean_age_s"] = ages[active].mean() if active.any() else np.nan
    return result


def verify():
    config,_ = configurations(ROOT)
    base,manifest,previous,_,_,_,extra = load_bundle(ROOT)
    if manifest["descriptor"]["input_files"] != validate_inputs(ROOT):
        raise ValueError("Original data receipt changed.")
    raw,by_source = read_context(ROOT,config,read_json(previous / "manifest.json"))
    context = checked_context(raw,config)
    queries = pd.DataFrame({ID:extra[ID]}).merge(context,on=ID,validate="one_to_one",sort=False)
    maps,_ = load_maps(ROOT,config)
    reconstructed = pd.DataFrame(np.nan,index=queries.index,columns=STATIC_NAMES)
    reconstructed["geo_stand_available"] = 0.
    reconstructed["geo_route_available"] = 0.
    for stamp,part in queries.groupby("snapshot",sort=True):
        mapped = maps[stamp]
        route_keys = list(zip(part.stand,part.runway))
        records = []
        for stand,key in zip(part.stand,route_keys):
            item = mapped["routes"].get(key)
            value = item["features"] if item else mapped["parking"].get(stand)
            records.append([value[name] if value else (0. if name in ["geo_stand_available","geo_route_available"] else np.nan) for name in STATIC_NAMES])
        reconstructed.loc[part.index] = np.asarray(records)
    np.testing.assert_allclose(reconstructed.to_numpy(dtype="float32"),extra[STATIC_NAMES].to_numpy(),atol=0,rtol=0,equal_nan=True)
    indexes = set(np.linspace(0,len(extra)-1,min(160,len(extra)),dtype=int))
    for _,part in queries.groupby(queries[TIME].dt.strftime("%Y-%m"),sort=True):
        for mask in [part.proxy_s.lt(0),part.proxy_s.gt(21600),part.offblock.isna(),
                     extra.loc[part.index,"geo_route_available"].eq(0)]:
            selected = part.index[np.asarray(mask)]
            indexes.update(selected[:12])
        indexes.update(part.index[[0,-1]])
    deltas=[]
    for index in sorted(indexes):
        query = queries.loc[index]
        # Only nearby peers can enter any declared window; widen to seven hours
        # without using labels, then apply the independent exact masks.
        anchor=query.anchor
        nearby=((context.offblock>=anchor-pd.Timedelta(hours=7))&(context.offblock<anchor)) | ((context[TIME]>=anchor-pd.Timedelta(hours=1))&(context[TIME]<anchor))
        sample=context.loc[nearby].reset_index(drop=True)
        values=independent_queue(sample,query,maps[query.snapshot])
        expected=np.array([values[name] for name in QUEUE_NAMES],dtype="float32")
        actual=extra.loc[index,QUEUE_NAMES].to_numpy(dtype="float32")
        np.testing.assert_allclose(expected,actual,atol=1e-6,rtol=1e-6,equal_nan=True)
        finite=np.isfinite(expected)&np.isfinite(actual)
        deltas.append(float(np.max(np.abs(expected[finite]-actual[finite]))) if finite.any() else 0.)
    return {"created_at_utc":utc_now(),"passed":True,"bundle_id":manifest["bundle_id"],
        "manifest_sha256":sha256_file(base / "manifest.json"),"all_static_feature_rows_checked":len(extra),
        "static_values_checked":len(extra)*len(STATIC_NAMES),"independent_queue_sample_rows":len(indexes),
        "independent_queue_maximum_difference":max(deltas,default=0.),"coverage":manifest["coverage"],
        "parent_rows_unchanged":sum(manifest["parent_rows_unchanged"].values()),"audit_used":False,
        "departure_labels_read":False,"verifier_code_sha256":sha256_file(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("docs/rome_geometry/feature_verification.json"))
    args=parser.parse_args()
    output=ROOT/args.output
    if output.exists(): raise ValueError("Preserve prior feature verification.")
    report=verify()
    write_json(output,report)
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,dict)},indent=2))


if __name__=="__main__": main()
