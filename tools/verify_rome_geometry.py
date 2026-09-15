"""Reconstruct complete candidates, fitting vocabularies and every focal native output."""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from prc2026.artifacts import read_json,sha256_file,utc_now,write_json
from prc2026.features import ID
from prc2026.validation import identity_hash
from prc2026_nmclock.features import route_mask
from prc2026_rome_geometry.bundle import load_bundle
from prc2026_rome_geometry.common import configurations,focal_partition,model_code_receipt
from prc2026_rome_geometry.encoding import fit_encoder
from prc2026_rome_geometry.training import checked_runs
from prc2026_rome_geometry.inference import node_prediction
from select_rome_geometry import collect
from verify_phase5 import same_metrics


def verify(runs,surface_runs,timeline_runs,feature_report):
    _,config=configurations(ROOT)
    base,manifest,previous,original_base,original,catalog,extra=load_bundle(ROOT)
    features_checked=read_json(feature_report)
    if not features_checked["passed"] or features_checked["manifest_sha256"]!=sha256_file(base/"manifest.json"):
        raise ValueError("The feature extension must pass independent verification.")
    saved=read_json(runs/"selection.json")
    rebuilt,predictions=collect(ROOT,runs,surface_runs,timeline_runs)
    for key in ["config_sha256","bundle_id","feature_manifest_sha256","model_code_sha256","selection_code_sha256",
                "native_fit_count","candidate_count","selected_candidate","refit_iterations"]:
        if rebuilt[key]!=saved[key]: raise ValueError("Selection no longer reproduces: "+key)
    for key in ["candidates","reference_metrics","selected_paired_days"]:
        same_metrics(rebuilt[key],saved[key])
    for name,digest in saved["candidate_prediction_sha256"].items():
        if sha256_file(runs/name)!=digest: raise ValueError("A complete candidate file changed.")
    for (name,fold),frame in predictions.items():
        pd.testing.assert_frame_equal(frame,pd.read_parquet(runs/"candidate_predictions"/f"{fold}_{name}.parquet"),check_exact=True)
    complete=checked_runs(ROOT,runs,config,manifest)
    selected=read_json(ROOT/"docs/surface_airports/development_results.json")
    native={}
    for fold in config["development_folds"]:
        assignment=pd.read_parquet(original_base/"splits"/(fold+".parquet"))
        parts={}
        for role in ["fit","early_stop","score"]:
            data,_=focal_partition(ROOT,previous,original_base,catalog,assignment,role,extra,original)
            parts[role]=data.loc[route_mask(data)].reset_index(drop=True)
        old_run=surface_runs/selected["development_runs"][fold]["run_id"]
        old_node=read_json(old_run/"nodes.json")["LIRF"]
        native[fold]={}
        for variant in config["variants"]:
            run,report=complete[(fold,variant)]
            node=read_json(run/"node.json")
            if node["settings"]!=config["settings"] or node["variant"]!=variant or node["target"]!="nm_residual" or node["target_filter"]!="none":
                raise ValueError("A native learner changed its declared contract.")
            if not 1<=node["iterations"]<=config["settings"]["lightgbm"]["rounds"]:
                raise ValueError("Invalid stopping count.")
            for role in ["fit","early_stop"]:
                if node[role+"_rows"]!=len(parts[role]) or node[role+"_ids_sha256"]!=identity_hash(parts[role][ID]):
                    raise ValueError("A model used different fitting/stopping IDs.")
                pd.testing.assert_frame_equal(parts[role][[ID]],pd.read_parquet(run/(role+"_ids.parquet")),check_exact=True)
            if variant=="control":
                for key in ["fit_rows","fit_ids_sha256","early_stop_rows","early_stop_ids_sha256","iterations","settings","parameters"]:
                    if node[key]!=old_node[key]: raise ValueError("Control differs from frozen airport component: "+key)
            expected_encoder=fit_encoder(parts["fit"],variant)
            if expected_encoder!=read_json(run/node["encoder_path"]):
                raise ValueError("Categorical vocabularies are not fitting-only.")
            expected=parts["score"][[ID]].merge(pd.read_parquet(run/"component_predictions.parquet"),on=ID,validate="one_to_one",how="left").prediction.to_numpy()
            actual=node_prediction(run,node,parts["score"],config["threads"])
            difference=float(np.max(np.abs(actual-expected)))
            if not np.isfinite(difference) or difference>1e-8:
                raise ValueError("A native component fails complete focal reload.")
            native[fold][variant]={"rows":len(actual),"maximum_difference_seconds":difference,"iterations":node["iterations"],
                "fitting_ids_verified":True,"stopping_ids_verified":True,"fitting_only_encoder_verified":True,
                "control_difference":report["control_maximum_difference_seconds"]}
    return {"created_at_utc":utc_now(),"passed":True,"audit_used":False,"bundle_id":manifest["bundle_id"],
        "selection_sha256":sha256_file(runs/"selection.json"),"model_code_sha256":model_code_receipt(ROOT)["sha256"],
        "feature_verification_sha256":sha256_file(feature_report),"verifier_code_sha256":sha256_file(Path(__file__)),
        "native_fit_count":6,"candidate_count":4,"complete_candidate_fold_metrics_reproduced":8,
        "selected_candidate":saved["selected_candidate"],"refit_iterations":saved["refit_iterations"],
        "native_predictions":native,"all_ineligible_predictions_equal_v7":True,
        "complete_rmse_seconds":{name:{fold:data["metrics"]["overall"]["rmse_seconds"] for fold,data in item["folds"].items()} for name,item in saved["candidates"].items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=Path("runs/rome_geometry"))
    parser.add_argument("--surface-runs",type=Path,default=Path("runs/surface_airports"))
    parser.add_argument("--timeline-runs",type=Path,default=Path("runs/timeline"))
    parser.add_argument("--feature-report",type=Path,default=Path("docs/rome_geometry/feature_verification.json"))
    parser.add_argument("--output",type=Path,default=Path("docs/rome_geometry/verification.json"))
    args=parser.parse_args()
    output=ROOT/args.output
    if output.exists(): raise ValueError("Preserve the original native verification.")
    report=verify(ROOT/args.runs,ROOT/args.surface_runs,ROOT/args.timeline_runs,ROOT/args.feature_report)
    write_json(output,report)
    print({"passed":report["passed"],"selected_candidate":report["selected_candidate"],"native_prediction_rows":sum(item["rows"] for fold in report["native_predictions"].values() for item in fold.values())})


if __name__=="__main__": main()
