"""Weighted native models with fitting-only encoders and fixed October stopping."""
from copy import deepcopy
from pathlib import Path
import importlib.metadata
import numpy as np

from prc2026.artifacts import read_json, write_json
from prc2026_weather.encoding import fit_encoder, transform
from .features import offset


def environment():
    return {name: importlib.metadata.version(name) for name in
            ["numpy", "pandas", "pyarrow", "lightgbm", "catboost", "scipy", "psutil", "tzdata"]}


def fit_native(fit, target, weights, stopping, stopping_target, variant):
    kind = variant["kind"]
    spec = {"kind": kind, "features": "ground_weather"}
    encoder = fit_encoder(fit, spec)
    encoder["fitted_on"] = "fitting_partition_only"
    xfit, xstop = transform(fit, encoder), transform(stopping, encoder)
    settings = variant["settings"]
    parameters = deepcopy(settings[kind]["parameters"])
    if not (np.isfinite(target).all() and np.isfinite(stopping_target).all()
            and np.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError("Native targets and positive fitting weights must be finite.")
    if kind == "lightgbm":
        import lightgbm as lgb
        parameters.update(seed=settings["seed"], num_threads=settings["threads"],
                          bagging_seed=settings["seed"], feature_fraction_seed=settings["seed"],
                          data_random_seed=settings["seed"])
        train = lgb.Dataset(xfit, label=target, weight=weights,
                            categorical_feature=encoder["categorical_columns"], free_raw_data=False)
        stop = lgb.Dataset(xstop, label=stopping_target, reference=train,
                           categorical_feature=encoder["categorical_columns"], free_raw_data=False)
        history = {}
        model = lgb.train(parameters, train, num_boost_round=settings[kind]["rounds"],
            valid_sets=[stop], valid_names=["early_stop"], callbacks=[
                lgb.early_stopping(settings[kind]["early_stopping_rounds"], verbose=False),
                lgb.record_evaluation(history), lgb.log_evaluation(period=0)])
        iterations = int(model.best_iteration or model.current_iteration())
    elif kind == "catboost":
        from catboost import CatBoostRegressor, Pool
        parameters.update(iterations=settings[kind]["rounds"], random_seed=settings["seed"],
                          thread_count=settings["threads"])
        model = CatBoostRegressor(**parameters)
        model.fit(Pool(xfit, label=target, weight=weights, cat_features=encoder["categorical_columns"]),
                  eval_set=Pool(xstop, label=stopping_target, cat_features=encoder["categorical_columns"]),
                  early_stopping_rounds=settings[kind]["early_stopping_rounds"], use_best_model=True, verbose=False)
        iterations = int(model.tree_count_)
        history = model.get_evals_result()
    else:
        raise ValueError("Unsupported native model.")
    return model, encoder, history, iterations, parameters


def predict_native(model, encoder, features, iterations):
    values = transform(features, encoder)
    if encoder["kind"] == "lightgbm":
        residual = model.predict(values, num_iteration=iterations, num_threads=16)
    else:
        residual = model.predict(values, ntree_end=iterations, thread_count=16)
    result = np.asarray(residual, dtype=float) + offset(features)
    if not np.isfinite(result).all():
        raise ValueError("Every signed prediction must be finite.")
    return result


def save_native(directory, model, encoder, history, iterations, parameters):
    filename = "model.txt" if encoder["kind"] == "lightgbm" else "model.cbm"
    if encoder["kind"] == "lightgbm":
        model.save_model(str(directory / filename), num_iteration=iterations)
    else:
        model.save_model(str(directory / filename))
    write_json(directory / "encoder.json", encoder)
    write_json(directory / "history.json", history)
    node = {"kind": encoder["kind"], "model": filename, "encoder": "encoder.json", "iterations": iterations,
            "native_parameters": parameters, "target_offset": "NM-masked conditional_schedule_offset_s",
            "prediction_postprocess": "none"}
    write_json(directory / "node.json", node)
    return node


def load_native(directory):
    node = read_json(directory / "node.json")
    encoder = read_json(directory / node["encoder"])
    if node["kind"] == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(directory / node["model"]))
        count = model.num_trees()
    else:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(directory / node["model"]))
        count = model.tree_count_
    if count != node["iterations"]:
        raise ValueError("Saved native tree count differs from the selected count.")
    return model, encoder, node
