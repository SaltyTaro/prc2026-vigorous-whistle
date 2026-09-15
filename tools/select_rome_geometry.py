"""Evaluate all four declared complete recipes against the frozen v7 reference."""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from prc2026.artifacts import object_hash,read_json,sha256_file,utc_now,verify_receipts,write_json
from prc2026.features import ID
from prc2026.validation import evaluate,identity_hash
from prc2026_nmclock.features import route_mask
from prc2026_rome_geometry.bundle import load_bundle
from prc2026_rome_geometry.common import configurations,complete_reference,aligned_values,model_code_receipt
from prc2026_rome_geometry.training import checked_runs
from prc2026_rome_geometry.inference import compose
from select_phase4 import paired_days


def collect(root,runs,surface_runs,timeline_runs):
    _,config=configurations(root)
    base,manifest,previous,original_base,original,catalog,extra=load_bundle(root)
    complete=checked_runs(root,runs,config,manifest)
    if set(complete)!={(fold,variant) for fold in config["development_folds"] for variant in config["variants"]}:
        raise ValueError("All six declared fits must finish before selection.")
    references,truths,features,components,eligibility,reference_metrics,run_receipts={},{},{},{},{},{},{}
    for fold in config["development_folds"]:
        frame,truth,reference,airport,metrics=complete_reference(root,previous,original_base,original,catalog,fold,surface_runs,timeline_runs)
        references[fold],truths[fold],features[fold],reference_metrics[fold]=reference,truth,frame,metrics
        geo=frame[[ID]].merge(extra[[ID,"geo_route_available"]],on=ID,how="left",validate="one_to_one",sort=False)
        eligible=frame.airport.eq("LIRF").to_numpy() & route_mask(frame) & geo.geo_route_available.eq(1).to_numpy()
        eligibility[fold]=eligible
        components[fold]={"original":airport.prediction.to_numpy()}
        ref_dir=runs/"references"/fold
        verify_receipts(ref_dir,read_json(ref_dir/"reference.json")["artifacts"])
        pd.testing.assert_frame_equal(pd.read_parquet(ref_dir/"v7.parquet"),reference,check_exact=True)
        run_receipts[fold]={}
        for variant in config["variants"]:
            run,report=complete[(fold,variant)]
            if report["complete_score_ids_sha256"]!=identity_hash(frame[ID]):
                raise ValueError("Complete scoring IDs differ from the original protocol.")
            pred=pd.read_parquet(run/"component_predictions.parquet")
            expected=frame.loc[frame.airport.eq("LIRF").to_numpy()&route_mask(frame),[ID]]
            pred=aligned_values(expected,pred)
            values=airport.prediction.to_numpy().copy()
            values[frame.airport.eq("LIRF").to_numpy()&route_mask(frame)]=pred.prediction.to_numpy()
            if variant=="control":
                difference=float(np.max(np.abs(values-components[fold]["original"])))
                if difference>config["control_reload_maximum_difference_seconds"] or report["control_maximum_difference_seconds"]>config["control_reload_maximum_difference_seconds"]:
                    raise ValueError("The reproduction control failed.")
            components[fold][variant]=values
            run_receipts[fold][variant]={"run_id":run.name,"run_sha256":sha256_file(run/"run.json"),"iterations":report["iterations"],
                "resources":report["resources"],"control_difference":report["control_maximum_difference_seconds"]}
    candidates,predictions={},{}
    for variant in config["candidate_variants"]:
        for weight in config["candidate_weights"]:
            name=f"{variant}_v7_{int(weight*100)}"
            folds,ratios={},[]
            for fold in config["development_folds"]:
                values=compose(references[fold].prediction,components[fold]["original"],components[fold][variant],eligibility[fold],weight)
                prediction=pd.DataFrame({ID:features[fold][ID],"prediction":values})
                if not np.array_equal(prediction.prediction.to_numpy()[~eligibility[fold]],references[fold].prediction.to_numpy()[~eligibility[fold]]):
                    raise ValueError("An ineligible prediction changed.")
                metrics=evaluate(truths[fold],prediction,original["ranking_month_weights"] if fold=="seasonal" else None)
                ratio=metrics["overall"]["rmse_seconds"]/reference_metrics[fold]["overall"]["rmse_seconds"]
                folds[fold]={"metrics":metrics,"rmse_ratio_to_v7":ratio,"eligible_rows":int(eligibility[fold].sum()),"unchanged_rows":int((~eligibility[fold]).sum())}
                ratios.append(ratio)
                predictions[(name,fold)]=prediction
            candidates[name]={"variant":variant,"new_weight":weight,"folds":folds,"mean_relative_rmse":float(np.mean(ratios)),
                              "eligible":all(value<=1-config["minimum_improvement_fraction_each_fold"] for value in ratios)}
    allowed=[name for name,item in candidates.items() if item["eligible"]]
    selected=min(allowed,key=lambda name:(candidates[name]["mean_relative_rmse"],name)) if allowed else None
    paired,refit={},None
    if selected:
        variant=candidates[selected]["variant"]
        refit=int(np.rint(np.median([run_receipts[fold][variant]["iterations"] for fold in config["development_folds"]])))
        paired={fold:paired_days(truths[fold],predictions[(selected,fold)],references[fold],config["seed"]) for fold in config["development_folds"]}
    report={"created_at_utc":utc_now(),"configuration":config,"config_sha256":object_hash(config),
        "bundle_id":manifest["bundle_id"],"feature_manifest_sha256":sha256_file(base/"manifest.json"),"model_code_sha256":model_code_receipt(root)["sha256"],
        "selection_code_sha256":sha256_file(Path(__file__)),"reference_metrics":reference_metrics,"development_runs":run_receipts,
        "native_fit_count":6,"candidate_count":4,"candidates":candidates,"selected_candidate":selected,
        "refit_iterations":refit,"selected_paired_days":paired,"paired_day_reference":"v7; inherited helper field names are descriptive only",
        "audit_used":False,"target_filter":"none","prediction_postprocess":"none","submission_uploaded":False}
    return report,predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=Path("runs/rome_geometry"))
    parser.add_argument("--surface-runs",type=Path,default=Path("runs/surface_airports"))
    parser.add_argument("--timeline-runs",type=Path,default=Path("runs/timeline"))
    args=parser.parse_args()
    runs=ROOT/args.runs
    destination=ROOT/"docs/rome_geometry/development_results.json"
    if destination.exists() or (runs/"selection.json").exists(): raise ValueError("Preserve the declared selection.")
    report,predictions=collect(ROOT,runs,ROOT/args.surface_runs,ROOT/args.timeline_runs)
    directory=runs/"candidate_predictions"
    directory.mkdir(exist_ok=False)
    for (name,fold),frame in predictions.items(): frame.to_parquet(directory/f"{fold}_{name}.parquet",index=False,compression="zstd")
    report["candidate_prediction_sha256"]={path.relative_to(runs).as_posix():sha256_file(path) for path in sorted(directory.glob("*.parquet"))}
    write_json(runs/"selection.json",report)
    write_json(destination,report)
    print({"selected_candidate":report["selected_candidate"],"scores":{name:{fold:data["metrics"]["overall"]["rmse_seconds"] for fold,data in row["folds"].items()} for name,row in report["candidates"].items()}})


if __name__=="__main__": main()
