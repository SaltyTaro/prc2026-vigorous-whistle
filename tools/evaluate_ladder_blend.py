"""Evaluate or independently verify the eight separately declared cross-campaign blends."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.validation import evaluate, identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.features import FEATURE_NAMES
from prc2026_ladder.inference import compose, node_prediction as original_node, route_mask
from prc2026_ladder.training import aligned_predictions, model_code_receipt as ladder_code
from prc2026_ladder_scale.training import model_code_receipt as scale_code
from prc2026_ladder_blend.recipes import convex, model_code_receipt, validate_config
from verify_ladder_scale import independent_node_values
from verify_phase5 import same_metrics
from select_phase4 import paired_days


def parent_inputs(root, config):
    result, evidence = {}, {}
    for campaign, entry in config["parent_selections"].items():
        path = root/entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("A declared parent selection changed.")
        selection = read_json(path)
        code = ladder_code(root) if campaign == "ladder" else scale_code(root)
        if selection["audit_used"] or selection["selected_candidate"] is not None or selection["model_code_sha256"] != code["sha256"]:
            raise ValueError("Expected both completed unpromoted parent campaigns.")
        verified = path.parent/"verification.json"
        receipt = read_json(verified)
        if not receipt["passed"] or receipt["selection_sha256"] != entry["sha256"]:
            raise ValueError("Every preceding campaign must pass native verification.")
        result[campaign] = selection
        evidence[campaign] = {"selection_sha256": entry["sha256"], "azure_verification_sha256": sha256_file(verified),
                              "model_code_sha256": code["sha256"]}
    return result, evidence


def query_sample(features, route):
    routed = features.loc[route_mask(features, route)].reset_index(drop=True)
    if route == "clock_missing":
        return routed
    extreme = (routed["nmclock_proxy_s"].lt(0) | routed["nmclock_proxy_s"].gt(10800)
               | routed["nmclock_clock_difference_s"].abs().gt(3600))
    indices = np.unique(np.r_[np.linspace(0, len(routed)-1, min(1000,len(routed)),dtype=int),
                              np.flatnonzero(extreme), routed.groupby("airport",sort=True).head(1).index])
    return routed.iloc[indices].reset_index(drop=True)


def parent_predictions(directory, selection, names, fold, features, truth, original, campaign):
    reference_dir = directory/"references"/fold
    reference_receipt = read_json(reference_dir/"reference.json")
    verify_receipts(reference_dir, reference_receipt["artifacts"])
    base = aligned_predictions(features, pd.read_parquet(reference_dir/"v6.parquet"))
    arrays, native, source_receipts = {}, {}, {}
    for name in ["v6_global","v6_airports"]:
        arrays[name] = aligned_predictions(features, pd.read_parquet(reference_dir/(name+".parquet")))["prediction"].to_numpy()
    needed = sorted({name for candidate_name in names
                     for weights in selection["candidates"][candidate_name]["routes"].values()
                     for name in weights if name not in {"v6","v6_global","v6_airports"}})
    for name in needed:
        saved = selection["development_runs"][name][fold]
        run = directory/saved["run_id"]
        if sha256_file(run/"run.json") != saved["run_sha256"]:
            raise ValueError("A native parent run changed.")
        verify_receipts(run, read_json(run/"run.json")["artifacts"])
        full = aligned_predictions(features,pd.read_parquet(run/"predictions.parquet"))
        arrays[name] = full["prediction"].to_numpy()
        node = read_json(run/"node.json")
        query = query_sample(features,node["spec"]["route"])
        predicted = (original_node(run,node,query,16) if campaign == "ladder"
                     else independent_node_values(run,node,query,16))
        recorded = query[[ID]].merge(full,on=ID,validate="one_to_one")["prediction"].to_numpy()
        difference = float(np.max(np.abs(predicted-recorded)))
        if difference > 1e-8:
            raise ValueError("A native parent prediction fails independent reconstruction.")
        native[name] = {"rows":len(query),"maximum_difference_seconds":difference}
        source_receipts[name] = {"run_sha256":saved["run_sha256"],"prediction_sha256":sha256_file(run/"predictions.parquet")}
    output = {}
    for name in names:
        candidate = selection["candidates"][name]
        values = compose(base["prediction"],arrays,route_mask(features,"clock_available"),
                         candidate["routes"],candidate["new_weight"])
        frame = pd.DataFrame({ID:features[ID],"prediction":values})
        metrics = evaluate(truth,frame,original["ranking_month_weights"] if fold=="seasonal" else None)
        same_metrics(metrics,candidate["folds"][fold]["metrics"])
        output[name] = frame
    return base,output,native,source_receipts


def compute(root, bundle, config_path, ladder_runs, scale_runs):
    config = read_json(config_path)
    validate_config(config)
    if sha256_file(config_path) != read_json(root/"docs/ladder_blend/declaration.json")["configuration_sha256"]:
        raise ValueError("The cross-campaign declaration changed.")
    manifest,parent,original,catalog = load_bundle(root,bundle)
    if manifest["bundle_id"] != config["feature_bundle_id"]:
        raise ValueError("The frozen bundle changed.")
    selections,evidence = parent_inputs(root,config)
    names = {
        "ladder":sorted({row["ladder_candidate"] for row in config["candidates"].values()}),
        "scale":sorted({row["scale_candidate"] for row in config["candidates"].values()}),
    }
    predictions, truths, references, baselines, native, sources, identities = {},{},{},{},{},{},{}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent/"splits"/(fold+".parquet"))
        features,truths[fold] = load_partition(bundle,parent,catalog,assignment,"score",FEATURE_NAMES)
        if identity_hash(features[ID]) != original["folds"][fold]["identity_sha256"]["score"]:
            raise ValueError("The complete frozen scoring IDs changed.")
        identities[fold] = {"rows":len(features),"ids_sha256":identity_hash(features[ID])}
        parent_values = {}
        native[fold],sources[fold] = {},{}
        for campaign,directory in [("ladder",ladder_runs),("scale",scale_runs)]:
            base,parent_values[campaign],native[fold][campaign],sources[fold][campaign] = parent_predictions(
                directory,selections[campaign],names[campaign],fold,features,truths[fold],original,campaign)
            if fold in references:
                pd.testing.assert_frame_equal(base,references[fold],check_exact=True)
            else:
                references[fold] = base
        baselines[fold] = evaluate(truths[fold],references[fold],original["ranking_month_weights"] if fold=="seasonal" else None)
        for name,spec in config["candidates"].items():
            a = parent_values["ladder"][spec["ladder_candidate"]]["prediction"]
            b = parent_values["scale"][spec["scale_candidate"]]["prediction"]
            predictions[name,fold] = pd.DataFrame({ID:features[ID],"prediction":convex(a,b,spec["ladder_weight"])})
    candidates = {}
    for name,spec in config["candidates"].items():
        folds,ratios = {},[]
        for fold in config["development_folds"]:
            metrics = evaluate(truths[fold],predictions[name,fold],original["ranking_month_weights"] if fold=="seasonal" else None)
            ratio = metrics["overall"]["rmse_seconds"]/baselines[fold]["overall"]["rmse_seconds"]
            ratios.append(ratio)
            folds[fold] = {"metrics":metrics,"rmse_ratio_to_v6":ratio}
        candidates[name] = {"spec":spec,"folds":folds,"mean_relative_rmse":float(np.mean(ratios)),
                            "eligible":all(value<=1-config["minimum_improvement_fraction_each_fold"] for value in ratios)}
    eligible = [name for name,row in candidates.items() if row["eligible"]]
    chosen = min(eligible,key=lambda name:(candidates[name]["mean_relative_rmse"],name)) if eligible else None
    paired,iterations = {},{}
    if chosen:
        for fold in config["development_folds"]:
            paired[fold] = paired_days(truths[fold],predictions[chosen,fold],references[fold],config["seed"])
        for campaign,components in [("ladder",["cat_missing_ground"]),("scale",["clock_scale300","missing_scale300"])]:
            for component in components:
                counts=[selections[campaign]["development_runs"][component][fold]["iterations"] for fold in config["development_folds"]]
                iterations[campaign+"/"+component] = int(np.rint(np.median(counts)))
    return {
        "created_at_utc":utc_now(),"configuration":config,"config_sha256":object_hash(config),
        "model_code":model_code_receipt(root),"bundle_id":manifest["bundle_id"],"audit_used":False,
        "parent_evidence":evidence,"native_parent_checks":native,"parent_native_sources":sources,
        "complete_scoring_populations":identities,"reference_metrics":baselines,
        "candidate_count":len(candidates),"new_native_development_fits":0,
        "candidates":candidates,"selected_candidate":chosen,"refit_iterations":iterations,
        "selected_paired_days":paired,
        "refit_rule":"Active new components use ties-to-even rounded medians of their two original October-selected counts; refit January-November while protecting all December flight groups. Imported v6 models remain unchanged.",
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=Path("data/nmclock/72e86725e23367ede5b5"))
    parser.add_argument("--config",type=Path,default=Path("configs/ladder_blend_v1.json"))
    parser.add_argument("--ladder-runs",type=Path,default=Path("runs/ladder"))
    parser.add_argument("--scale-runs",type=Path,default=Path("runs/ladder_scale"))
    parser.add_argument("--selection",type=Path,default=Path("runs/ladder_blend/selection.json"))
    parser.add_argument("--output",type=Path,default=Path("docs/ladder_blend/development_results.json"))
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    if (root/args.output).exists() or (not args.verify_only and (root/args.selection).exists()):
        raise ValueError("Preserve existing selection and verification evidence.")
    with ResourceMonitor() as monitor:
        report = compute(root,root/args.bundle,root/args.config,root/args.ladder_runs,root/args.scale_runs)
    if args.verify_only:
        recorded = read_json(root/args.selection)
        keys = sorted(set(report)-{"created_at_utc"})
        same_metrics({key:report[key] for key in keys},{key:recorded[key] for key in keys})
        report = {"created_at_utc":utc_now(),"passed":True,"selection_sha256":sha256_file(root/args.selection),
                  "model_code_sha256":report["model_code"]["sha256"],"audit_used":False,"candidate_count":8,
                  "native_parent_checks":report["native_parent_checks"],"selected_candidate":report["selected_candidate"],
                  "complete_scoring_populations":report["complete_scoring_populations"]}
    else:
        report["resources"] = monitor.stats
        write_json(root/args.selection,report)
    write_json(root/args.output,report)
    if args.verify_only:
        print({"passed":True,"selected":report["selected_candidate"],"candidate_count":8})
    else:
        leading=sorted(report["candidates"],key=lambda name:report["candidates"][name]["mean_relative_rmse"])
        print({"selected":report["selected_candidate"],"candidate_count":8,"refit_iterations":report["refit_iterations"],
               "leading":[{"name":name,"eligible":report["candidates"][name]["eligible"],
                           "scores":{fold:report["candidates"][name]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
                                     for fold in ["seasonal","forward"]}} for name in leading[:4]]})


if __name__=="__main__":
    main()
