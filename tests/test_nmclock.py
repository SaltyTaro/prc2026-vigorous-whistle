from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TIME, TARGET, BLOCK
from prc2026.validation import identity_hash
from prc2026_nmclock.features import (CLOCK_NAMES, FEATURE_NAMES, CATEGORICAL_COLUMNS,
    INPUT_COLUMNS, clock_features, route_mask, target_offset)
from prc2026_nmclock.encoding import fit_encoder, transform
from prc2026_nmclock.inference import copy_file_tree, node_prediction, pipeline_prediction
from prc2026_nmclock.training import fit_node, settings_for, validate_config


ROOT = Path(__file__).resolve().parents[1]


def raw_clocks():
    return pd.DataFrame({
        ID: [4., 2., 8., 9.], TIME: pd.to_datetime(['2025-01-02T00:10Z']*4, utc=True),
        'AOBT_3_flt': pd.to_datetime(['2025-01-01T23:50Z', None, None, '2025-01-02T00:20Z'], utc=True),
        'LOBT_flt': pd.to_datetime(['2025-01-01T23:49Z', '2025-01-01T23:40Z', None, '2025-01-02T00:20Z'], utc=True),
        'SCHED_TIME_UTC_mvt': pd.to_datetime(['2025-01-01T23:30Z']*4, utc=True),
        'EOBT_1_flt': pd.to_datetime(['2025-01-01T23:35Z']*4, utc=True),
        'IOBT_flt': pd.to_datetime(['2025-01-01T23:25Z']*4, utc=True),
    })


def test_clock_signs_midnight_and_missing_fallback():
    x = clock_features(raw_clocks())
    assert list(x) == [ID, *CLOCK_NAMES]
    assert x[ID].tolist() == [4., 2., 8., 9.]
    assert x.nmclock_source.tolist() == [1., 2., 0., 1.]
    np.testing.assert_allclose(x.nmclock_proxy_s, [1200., 1800., np.nan, -600.], equal_nan=True)
    assert x.nmclock_aobt_movement_day_delta.iloc[0] == 1
    assert x.nmclock_aobt_minus_schedule_s.iloc[0] == 1200
    assert x.nmclock_aobt_minus_eobt1_s.iloc[0] == 900
    assert x.nmclock_aobt_minus_iobt_s.iloc[0] == 1500
    assert x.nmclock_clock_difference_s.iloc[0] == 60
    assert x.nmclock_agree_60s.iloc[0] == 1
    assert x.nmclock_proxy_mod_day_s.iloc[-1] == 85800
    assert route_mask(x).tolist() == [True, True, False, True]


def test_departure_labels_airport_block_and_arrival_clock_cannot_enter():
    raw = raw_clocks()
    expected = clock_features(raw)
    for name in [TARGET, BLOCK, 'ARVT_3_flt']:
        assert name not in INPUT_COLUMNS
        raw[name] = 123456789
    pd.testing.assert_frame_equal(clock_features(raw), expected, check_exact=True)
    pd.testing.assert_frame_equal(clock_features(raw.iloc[::-1]), expected.iloc[::-1].reset_index(drop=True), check_exact=True)
    raw.loc[0, 'AOBT_3_flt'] += pd.Timedelta(seconds=125)
    assert clock_features(raw).nmclock_proxy_s.iloc[0] == expected.nmclock_proxy_s.iloc[0]-125


def test_clock_inputs_reject_ambiguous_time_duplicates_and_december():
    raw = raw_clocks()
    with pytest.raises(ValueError, match='Unique'):
        clock_features(pd.concat([raw, raw.iloc[:1]], ignore_index=True))
    bad = raw.copy()
    bad[TIME] = bad[TIME].dt.tz_localize(None)
    with pytest.raises(ValueError, match='UTC'):
        clock_features(bad)
    bad = raw.copy()
    bad[TIME] = pd.Timestamp('2025-12-01T12:00Z')
    with pytest.raises(ValueError, match='Reserved audit'):
        clock_features(bad)


def test_target_offset_preserves_signed_values_and_squared_seconds_loss():
    frame = clock_features(raw_clocks()).iloc[[0, 1, 3]].reset_index(drop=True)
    y = np.array([-300., 120000., 15.])
    residual = np.array([100., -50., 7.])
    offset = target_offset(frame, 'nm_residual')
    assert offset[-1] == -600
    np.testing.assert_array_equal(target_offset(frame, 'direct'), np.zeros(3))
    np.testing.assert_allclose((y-(offset+residual))**2, ((y-offset)-residual)**2, rtol=0, atol=0)
    with pytest.raises(ValueError, match='finite clock'):
        target_offset(clock_features(raw_clocks()), 'nm_residual')
    frame.loc[0, 'nmclock_available'] = 0
    with pytest.raises(ValueError, match='disagree'):
        route_mask(frame)


def model_fixture():
    rng = np.random.default_rng(18)
    n = 240
    frame = pd.DataFrame({ID: np.arange(n, dtype=float), **{
        name: rng.choice(['A', 'B'], n) if name in CATEGORICAL_COLUMNS else rng.normal(size=n)
        for name in FEATURE_NAMES}})
    frame['airport'] = np.where(np.arange(n) % 2, 'LIRF', 'LFPG')
    frame['nmclock_available'] = 1.
    frame['nmclock_proxy_s'] = np.linspace(-600, 3000, n)
    y = frame['nmclock_proxy_s'].to_numpy()+50*frame['wx_speed_mps'].to_numpy()+350
    y[0], y[1] = -250., 120000.
    return frame, pd.DataFrame({ID: frame[ID], TARGET: y})


@pytest.mark.parametrize('kind', ['lightgbm', 'catboost'])
@pytest.mark.parametrize('target', ['direct', 'nm_residual'])
def test_native_clock_models_keep_all_routed_labels_and_reload(tmp_path, kind, target):
    frame, truth = model_fixture()
    cfg = json.loads((ROOT/'configs/nmclock_v1.json').read_text())
    spec = {'kind': kind, 'features': 'core_weather_nmclock', 'target': target, 'route': 'clock_available'}
    settings = settings_for(cfg, spec)
    settings['threads'] = 1
    settings[kind]['rounds'], settings[kind]['early_stopping_rounds'] = 12, 3
    settings[kind]['parameters'].update({'min_data_in_leaf': 10, 'num_leaves': 7} if kind == 'lightgbm' else {'depth': 3})
    fit, stop = frame.iloc[:180], frame.iloc[180:].copy()
    stop['operator'] = 'STOP_ONLY'
    encoder = fit_encoder(fit, spec)
    assert 'STOP_ONLY' not in encoder['vocabularies']['operator']
    assert transform(stop, encoder)['operator'].astype(str).eq('__MISSING__').all()
    assert ID not in encoder['feature_names'] and TARGET not in encoder['feature_names']
    node = fit_node(tmp_path, 'clock', spec, settings, fit, truth.iloc[:180].iloc[::-1], stop, truth.iloc[180:])
    assert node['fit_rows'] == 180 and node['fit_ids_sha256'] == identity_hash(fit[ID])
    assert node['max_model_reload_difference_seconds'] == 0
    values = node_prediction(tmp_path, node, stop, 1)
    assert values.shape == (60,) and np.isfinite(values).all()
    with pytest.raises(ValueError, match='fixed refit'):
        fit_node(tmp_path, 'invalid', spec, settings, fit, truth.iloc[:180], stop, truth.iloc[180:], iterations=3)
    no_clock = fit.copy()
    no_clock.loc[0, ['nmclock_available', 'nmclock_proxy_s']] = [0., np.nan]
    with pytest.raises(ValueError, match='clock-available'):
        fit_node(tmp_path, 'bad_route', spec, settings, no_clock, truth.iloc[:180])


@pytest.mark.parametrize('weight', [.25, 1.])
def test_pipeline_keeps_missing_clock_fallback_exact(monkeypatch, weight):
    import prc2026_nmclock.inference as inference
    features = clock_features(raw_clocks())
    def new_node(run, node, frame, threads):
        assert route_mask(frame).all()
        return frame[ID].to_numpy()+100
    monkeypatch.setattr(inference, 'node_prediction', new_node)
    monkeypatch.setattr(inference, 'weather_prediction', lambda run, pipeline, frame, threads: frame[ID].to_numpy()-100)
    pipeline = {'kind': 'nmclock_patch', 'weight': weight, 'base': {'kind': 'v5', 'pipeline': {}},
                'replacement': {'kind': 'nmclock_node', 'node': {}}}
    result = pipeline_prediction(Path('.'), pipeline, features, 1)
    expected = features[ID].to_numpy()-100
    expected[route_mask(features)] += weight*200
    np.testing.assert_array_equal(result, expected)
    assert result[2] == -92


def test_native_file_copy_preserves_sharing_and_rejects_escape(tmp_path):
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.mkdir(); destination.mkdir()
    (source/'model.txt').write_text('native model')
    tree = {'first': {'model_path': 'model.txt'}, 'second': {'model': 'model.txt'}}
    cloned = copy_file_tree(destination, source, tree, 'v5_')
    assert cloned['first']['model_path'] == cloned['second']['model'] == 'v5_model.txt'
    assert len(list(destination.iterdir())) == 1 and tree['first']['model_path'] == 'model.txt'
    with pytest.raises(ValueError, match='flat files'):
        copy_file_tree(destination, source, {'model_path': '../outside.txt'}, 'bad_')


def test_nmclock_protocol_and_anchor_guards():
    cfg = json.loads((ROOT/'configs/nmclock_v1.json').read_text())
    protocol = json.loads((ROOT/'configs/validation_v1.json').read_text())
    validate_config(cfg, protocol)
    bad = deepcopy(cfg)
    bad['refit_months'].append('2025-12')
    with pytest.raises(ValueError, match='December'):
        validate_config(bad, protocol)
    bad = deepcopy(cfg)
    bad['variants']['cat_direct']['airport_anchor'] = 'lgb_residual'
    with pytest.raises(ValueError, match='matching target'):
        validate_config(bad, protocol)
    bad = deepcopy(cfg)
    bad['candidate_families']['bad'] = {'lgb_direct': .7}
    with pytest.raises(ValueError, match='sum to one'):
        validate_config(bad, protocol)
