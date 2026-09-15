"""The unchanged v7 airport LightGBM settings with explicit feature ablations."""

import numpy as np
from prc2026.artifacts import ResourceMonitor, write_json
from prc2026.features import ID, TARGET
from prc2026.phase1 import emit
from prc2026.validation import identity_hash
from prc2026_models.experiments import predict
from prc2026_nmclock.features import route_mask, target_offset
from prc2026_phase5r.baseline import aligned_truth
from .encoding import fit_encoder, transform
from .inference import node_prediction


def fit_node(run, variant, settings, features, truth, stopping=None, stopping_truth=None, iterations=None):
    if (stopping is None) != (stopping_truth is None) or (iterations is not None and stopping is not None):
        raise ValueError("Fixed refitting cannot inspect stopping labels.")
    if iterations is not None and iterations < 1:
        raise ValueError("Positive fixed tree count required.")
    for frame in [features] + ([] if stopping is None else [stopping]):
        if not frame.airport.eq("LIRF").all() or not route_mask(frame).all():
            raise ValueError("Fit only original clock-present LIRF rows.")
    paths = {"model_path":"model.txt", "encoder_path":"encoder.json", "history_path":"history.json", "importance_path":"importance.json"}
    if any((run / name).exists() for name in paths.values()):
        raise ValueError("Never overwrite fitted components.")
    with ResourceMonitor() as monitor:
        import lightgbm as lgb
        encoder = fit_encoder(features, variant)
        x = transform(features, encoder)
        y = aligned_truth(features, truth)[TARGET].to_numpy(dtype=float) - target_offset(features, "nm_residual")
        native = settings["lightgbm"]
        params = dict(native["parameters"])
        params.update(num_threads=settings["threads"], seed=settings["seed"], feature_fraction_seed=settings["seed"],
                      bagging_seed=settings["seed"], data_random_seed=settings["seed"], metric="rmse")
        train = lgb.Dataset(x, label=y, categorical_feature=encoder["categorical_columns"])
        history, kwargs = {}, {}
        callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history)]
        if stopping is not None:
            xs = transform(stopping, encoder)
            ys = aligned_truth(stopping, stopping_truth)[TARGET].to_numpy(dtype=float) - target_offset(stopping, "nm_residual")
            valid = lgb.Dataset(xs, label=ys, reference=train, categorical_feature=encoder["categorical_columns"])
            kwargs.update(valid_sets=[valid], valid_names=["early_stop"])
            callbacks.append(lgb.early_stopping(native["early_stopping_rounds"], first_metric_only=True))
        rounds = native["rounds"] if iterations is None else iterations
        emit("rome_geometry_training", variant=variant, fit_rows=len(x), predictors=len(encoder["feature_names"]), maximum_iterations=rounds)
        model = lgb.train(params, train, num_boost_round=rounds, callbacks=callbacks, **kwargs)
        best = int(model.best_iteration or model.current_iteration())
        if iterations is not None and best != iterations:
            raise ValueError("Refit tree count differs from its declaration.")
        model.save_model(str(run / paths["model_path"]), num_iteration=best)
        write_json(run / paths["encoder_path"], encoder)
        write_json(run / paths["history_path"], history)
        write_json(run / paths["importance_path"], dict(zip(encoder["feature_names"], map(float, model.feature_importance(importance_type="gain")))))
        features[[ID]].to_parquet(run / "fit_ids.parquet", index=False, compression="zstd")
        if stopping is not None:
            stopping[[ID]].to_parquet(run / "early_stop_ids.parquet", index=False, compression="zstd")
        node = {"schema":"rome_geometry", "model_kind":"lightgbm", "variant":variant, "target":"nm_residual",
            "settings":settings, "parameters":params, "iterations":best, **paths,
            "fit_rows":len(features), "fit_ids_sha256":identity_hash(features[ID]),
            "early_stop_rows":len(stopping) if stopping is not None else 0,
            "early_stop_ids_sha256":identity_hash(stopping[ID]) if stopping is not None else None,
            "target_filter":"none", "prediction_postprocess":"none"}
        sample = (features if stopping is None else stopping).iloc[:1000]
        before = predict(model, transform(sample, encoder), "lightgbm", settings["threads"]) + target_offset(sample, "nm_residual")
        difference = float(np.max(np.abs(before - node_prediction(run, node, sample, settings["threads"]))))
        if difference > 1e-9:
            raise ValueError("Native reload changes a prediction.")
        node["native_reload_maximum_difference_seconds"] = difference
    node["resources"] = monitor.stats
    return node
