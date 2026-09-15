"""Fit LightGBM on scaled residuals, selecting October iterations in physical seconds."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import lightgbm as lgb
from prc2026.artifacts import ResourceMonitor, write_json
from prc2026.features import ID, TARGET
from prc2026.phase1 import emit
from prc2026.validation import identity_hash
from prc2026_models.experiments import predict
from prc2026_phase5r.baseline import aligned_truth
from .inference import encoding, node_prediction
from .targets import offset_scale, physical_rmse, target_and_weight


def fit_node(run: Path, name, variant, features, truth, stop=None, stop_truth=None, iterations=None):
    if (stop is None) != (stop_truth is None) or (iterations is not None and stop is not None):
        raise ValueError("A fixed refit cannot inspect stopping data.")
    if iterations is not None and iterations < 1:
        raise ValueError("Positive fixed iterations required.")
    if variant["kind"] != "lightgbm":
        raise ValueError("The target-scale declaration permits CPU LightGBM only.")
    spec = {key: deepcopy(variant[key]) for key in ["kind", "features", "route", "target"] if key in variant}
    settings = variant["settings"]
    paths = {key+"_path": name+"_"+key+(".txt" if key == "model" else ".json")
             for key in ["model", "encoder", "history", "importance"]}
    if any((run/path).exists() for path in paths.values()):
        raise ValueError("Preserve existing native files.")
    with ResourceMonitor() as monitor:
        enc = encoding(variant["schema"])
        encoder = enc.fit_encoder(features, spec)
        x = enc.transform(features, encoder)
        targets = aligned_truth(features, truth)[TARGET].to_numpy(dtype=float)
        y, weights = target_and_weight(features, targets, variant)
        offset, scale = offset_scale(features, variant)
        train = lgb.Dataset(x, label=y, weight=weights, categorical_feature=encoder["categorical_columns"])
        native = settings["lightgbm"]
        rounds = native["rounds"] if iterations is None else iterations
        params = dict(native["parameters"])
        params.update(num_threads=settings["threads"], seed=settings["seed"], feature_fraction_seed=settings["seed"],
                      bagging_seed=settings["seed"], data_random_seed=settings["seed"], metric="None")
        history, kwargs = {}, {}
        callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history)]
        if stop is not None:
            xs = enc.transform(stop, encoder)
            actual_stop = aligned_truth(stop, stop_truth)[TARGET].to_numpy(dtype=float)
            ys, ws = target_and_weight(stop, actual_stop, variant)
            stop_offset, stop_scale = offset_scale(stop, variant)
            validation = lgb.Dataset(xs, label=ys, weight=ws, reference=train, categorical_feature=encoder["categorical_columns"])
            def physical_metric(values, dataset):
                if dataset is not validation:
                    raise ValueError("Unexpected data supplied to October stopping metric.")
                return "rmse_seconds", physical_rmse(values, actual_stop, stop_offset, stop_scale), False
            kwargs.update(valid_sets=[validation], valid_names=["early_stop"], feval=physical_metric)
            callbacks.append(lgb.early_stopping(native["early_stopping_rounds"], first_metric_only=True))
        emit("ladder_scale_training", component=name, rows=len(features), predictors=len(encoder["feature_names"]), maximum_iterations=rounds)
        model = lgb.train(params, train, num_boost_round=rounds, callbacks=callbacks, **kwargs)
        best = int(model.best_iteration or model.current_iteration())
        if iterations is not None and best != iterations:
            raise ValueError("Refit differs from the declared fixed count.")
        model.save_model(str(run/paths["model_path"]), num_iteration=best)
        write_json(run/paths["encoder_path"], encoder)
        write_json(run/paths["history_path"], history)
        write_json(run/paths["importance_path"], dict(zip(encoder["feature_names"], map(float, model.feature_importance(importance_type="gain")))))
        card = {
            "model_kind": "lightgbm", "schema": variant["schema"], "spec": spec,
            "target_transform": variant["target_transform"], "settings": settings, "parameters": params,
            "iterations": best, **paths,
            "fit_rows": len(features), "fit_ids_sha256": identity_hash(features[ID]),
            "early_stop_rows": 0 if stop is None else len(stop),
            "early_stop_ids_sha256": None if stop is None else identity_hash(stop[ID]),
            "target_filter": "none", "stopping_metric": "complete routed RMSE in seconds",
            "weight_identity": "weight * (z - zhat)^2 = (y - offset - scale*zhat)^2 / 300^2",
            "weight_quantiles": list(map(float, np.quantile(weights, [0, .5, .99, 1]))),
            "scale_quantiles_seconds": list(map(float, np.quantile(scale, [0, .5, .99, 1]))),
        }
        sample = (features if stop is None else stop).iloc[:1000]
        sample_offset, sample_scale = offset_scale(sample, variant)
        before = sample_offset+sample_scale*predict(model, enc.transform(sample, encoder), "lightgbm", settings["threads"])
        delta = float(np.max(np.abs(before-node_prediction(run, card, sample, settings["threads"]))))
        if delta > 1e-8:
            raise ValueError("The scaled native model fails reload in physical seconds.")
        card["max_model_reload_difference_seconds"] = delta
        if stop is not None:
            card["early_stop_rmse_seconds"] = physical_rmse(
                predict(model, xs, "lightgbm", settings["threads"]), actual_stop, stop_offset, stop_scale)
    card["resources"] = monitor.stats
    return card

