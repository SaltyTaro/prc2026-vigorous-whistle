"""Summarize the declared NM-clock batch from saved complete metrics and receipts."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json


def summarize(root, runs, selection_path):
    selection = read_json(selection_path)
    if selection['audit_used']:
        raise ValueError('The independent audit cannot enter this summary.')
    comparisons = {}
    resources = []
    for variant, folds in selection['development_runs'].items():
        comparisons[variant] = {}
        for fold, saved in folds.items():
            path = runs/saved['run_id']
            if sha256_file(path/'run.json') != saved['run_sha256']:
                raise ValueError('Development run changed.')
            run = read_json(path/'run.json')
            resources.append(run['resources'])
            metric = saved['metrics']['overall']['rmse_seconds']
            reference = selection['reference_metrics'][fold]['overall']['rmse_seconds']
            comparisons[variant][fold] = {'rmse_seconds': metric, 'improvement_seconds': reference-metric,
                                         'improvement_fraction': 1-metric/reference}
    selected = selection['selected_candidate']
    result = {'created_at_utc': utc_now(), 'code_sha256': sha256_file(Path(__file__)),
              'selection_sha256': sha256_file(selection_path), 'audit_used': False,
              'development_runs': comparisons, 'candidate_count': len(selection['candidates']),
              'eligible_candidates': [k for k,v in selection['candidates'].items() if v['eligible']],
              'selected_candidate': selected, 'fit_resources': {
                  'total_wall_seconds': sum(r['wall_seconds'] for r in resources),
                  'total_cpu_seconds': sum(r['cpu_seconds'] for r in resources),
                  'maximum_sampled_rss_bytes': max(r['peak_process_rss_bytes_sampled'] for r in resources)},
              'interpretation': 'These are reused complete development sets, not official leaderboard results or the reserved December audit.'}
    if not selected:
        return result
    chosen = selection['candidates'][selected]
    diagnostics = {}
    for fold, row in chosen['folds'].items():
        old, new = selection['reference_metrics'][fold], row['metrics']
        delta = {'rows': new['overall']['rows'], 'rmse_seconds': new['overall']['rmse_seconds'],
                 'rmse_improvement_seconds': old['overall']['rmse_seconds']-new['overall']['rmse_seconds'],
                 'rmse_improvement_fraction': 1-new['overall']['rmse_seconds']/old['overall']['rmse_seconds'],
                 'paired_days': row['paired_days']}
        for grouping in ['by_airport', 'by_nm_missing', 'by_target_band', 'by_month']:
            delta[grouping] = {}
            for key, item in new[grouping].items():
                baseline = old[grouping][key]
                if item['rows'] != baseline['rows']:
                    raise ValueError('Diagnostic comparison changed its evaluation population.')
                delta[grouping][key] = {'rows': item['rows'], 'rmse_seconds': item['rmse_seconds'],
                    'rmse_delta_seconds': item['rmse_seconds']-baseline['rmse_seconds'],
                    'sse_delta': item['sse']-baseline['sse'], 'new_sse_fraction': item['sse']/new['overall']['sse']}
        delta['airports_improved'] = sum(v['rmse_delta_seconds'] < 0 for v in delta['by_airport'].values())
        missing_delta = delta['by_nm_missing']['True']['sse_delta']
        if not np.isclose(missing_delta, 0, rtol=0, atol=1e-4):
            raise ValueError('The missing-clock group no longer retains v5.')
        tail_share = delta['by_target_band']['over_3h']['new_sse_fraction']
        delta['oracle_perfect_above_three_hours_rmse_reduction_fraction'] = 1-np.sqrt(1-tail_share)
        diagnostics[fold] = delta
    result['selected_diagnostics'] = diagnostics
    result['refit_iterations'] = selection['refit_iterations']
    result['family_weights'] = chosen['family_weights']
    result['new_candidate_weight'] = chosen['new_weight']
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/nmclock'))
    parser.add_argument('--selection', type=Path, default=Path('runs/nmclock/selection.json'))
    parser.add_argument('--output', type=Path, default=Path('docs/nmclock/analysis.json'))
    args = parser.parse_args()
    report = summarize(root, root/args.runs, root/args.selection)
    write_json(root/args.output, report)
    print({k: report[k] for k in ['candidate_count', 'eligible_candidates', 'selected_candidate', 'fit_resources']})


if __name__ == '__main__':
    main()
