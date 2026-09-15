"""Describe clock coverage and covariate shifts without reading departure labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID
from prc2026_nmclock.features import route_mask


COLUMNS = [ID, 'airport', 'month', 'nm_missing', 'nmclock_available', 'nmclock_proxy_s',
           'nmclock_clock_difference_s', 'nmclock_source', 'nmclock_both_present']


def describe(frame):
    available = route_mask(frame)
    proxy = frame.loc[available, 'nmclock_proxy_s'].to_numpy(dtype=float)
    difference = frame['nmclock_clock_difference_s'].to_numpy(dtype=float)
    both = frame['nmclock_both_present'].eq(1).to_numpy()
    nm_missing = frame['nm_missing'].eq(1).to_numpy()
    quantiles = [0, .01, .1, .5, .9, .99, 1]
    return {
        'rows': len(frame), 'clock_rows': int(available.sum()),
        'clock_fraction': float(available.mean()), 'missing_clock_rows': int((~available).sum()),
        'nm_origin_missing_rows': int(nm_missing.sum()),
        'nm_origin_missing_but_clock_present_rows': int((nm_missing & available).sum()),
        'nm_origin_present_but_clock_missing_rows': int((~nm_missing & ~available).sum()),
        'lobt_only_rows': int(frame['nmclock_source'].eq(2).sum()),
        'both_clocks_rows': int(both.sum()),
        'clocks_disagree_over_60s_rows': int((both & (np.abs(difference) > 60)).sum()),
        'negative_proxy_rows': int((proxy < 0).sum()),
        'over_three_hour_proxy_rows': int((proxy > 10800).sum()),
        'proxy_quantiles_seconds': {str(q): float(v) for q,v in zip(quantiles, np.quantile(proxy, quantiles))} if len(proxy) else {},
    }


def analyze(root, base):
    manifest = read_json(base/'manifest.json')
    frames, hashes, file_boundary_rows = [], {}, 0
    receipts = {item['path']: item for item in manifest['artifacts']}
    for source, expected_rows in manifest['rows'].items():
        relative = 'features/'+source
        path = base/relative
        hashes[source] = sha256_file(path)
        if hashes[source] != receipts[relative]['sha256']:
            raise ValueError('Clock feature artifact changed.')
        frame = pd.read_parquet(path, columns=COLUMNS)
        if len(frame) != expected_rows or frame[ID].duplicated().any():
            raise ValueError('Clock covariate population changed.')
        ranking = source == 'ranking.parquet'
        if ranking:
            if not frame['month'].isin([1, 7]).all():
                raise ValueError('Undeclared ranking month.')
        else:
            month = int(source.split('_')[1][5:7])
            if not 1 <= month <= 11 or not frame['month'].isin(range(1, 12)).all():
                raise ValueError('Protected audit or undeclared development month.')
            file_boundary_rows += int(frame['month'].ne(month).sum())
        frame['population'] = 'ranking' if ranking else 'jan_nov_refit'
        frame['period'] = ('2026-' if ranking else '2025-')+frame['month'].astype(int).astype(str).str.zfill(2)
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    if all_rows[ID].duplicated().any():
        raise ValueError('Overlapping source populations.')
    populations = {}
    for name, frame in all_rows.groupby('population', sort=True):
        populations[name] = {'overall': describe(frame),
            'by_airport': {str(k): describe(v) for k,v in frame.groupby('airport', sort=True)},
            'by_period': {str(k): describe(v) for k,v in frame.groupby('period', sort=True)}}
    return {'created_at_utc': utc_now(), 'code_sha256': sha256_file(Path(__file__)),
            'feature_manifest_sha256': sha256_file(base/'manifest.json'), 'bundle_id': manifest['bundle_id'],
            'input_feature_sha256': hashes, 'columns_read': COLUMNS, 'target_columns_read': [],
            'rows_whose_actual_month_differs_from_filename': file_boundary_rows,
            'period_assignment': 'Inherited actual UTC movement month; source filenames are not split assignments.',
            'audit_used': False, 'populations': populations,
            'interpretation': 'Descriptive covariates only; no model, feature recipe, weight, prediction or selection is changed.'}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/nmclock/coverage_analysis.json'))
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/'data/nmclock/latest.json')['bundle'])
    report = analyze(root, base)
    write_json(root/args.output, report)
    print({name: row['overall'] for name,row in report['populations'].items()})


if __name__ == '__main__':
    main()
