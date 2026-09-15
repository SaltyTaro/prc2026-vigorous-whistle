"""Native timeline residuals; the supplied NM offset remains unbounded."""
from prc2026.artifacts import read_json
from prc2026_models.experiments import load_model, predict
from prc2026_nmclock.features import route_mask, target_offset
from .encoding import transform


def node_prediction(run, node, features, threads):
    if not route_mask(features).all():
        raise ValueError("Timeline native models require available NM clocks.")
    encoder = read_json(run/node["encoder_path"])
    model = load_model(run/node["model_path"], node["model_kind"])
    return predict(model, transform(features, encoder), node["model_kind"], threads)+target_offset(features, node["spec"]["target"])
