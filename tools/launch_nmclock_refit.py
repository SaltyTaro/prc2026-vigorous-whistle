"""Launch the fixed arcw0 refit after matching Azure and Windows verification."""
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_nmclock.training import model_code_receipt
from refit_nmclock import eligible_recipe


def main():
    if ROOT != Path('/home/azureuser/prc-flight-2026') or socket.gethostname() != 'arcw0':
        raise ValueError('This refit is restricted to the authorized arcw0 workspace.')
    output = ROOT/'runs/nmclock'
    if (output/'refit_launch.json').exists() or (output/'refit.log').exists():
        raise ValueError('Refit launch evidence already exists; inspect it without duplicating the job.')
    selection_path = output/'selection.json'
    selection, config = read_json(selection_path), read_json(ROOT/'configs/nmclock_v1.json')
    manifest = read_json(ROOT/read_json(ROOT/'data/nmclock/latest.json')['bundle']/'manifest.json')
    code = model_code_receipt(ROOT)
    eligible_recipe(config, selection, manifest, code)
    verification_hashes = {}
    for name in ['verification.json', 'local_verification.json']:
        path = ROOT/'docs/nmclock'/name
        report = read_json(path)
        if (not report['passed'] or report['audit_used'] or report['refits']
                or report['selection_sha256'] != sha256_file(selection_path)
                or report['model_code_sha256'] != code['sha256']
                or report['bundle_id'] != manifest['bundle_id']
                or report['selected_candidate'] != selection['selected_candidate']
                or len(report['development_scores']) != 8
                or report['blend_fold_scores_checked'] != 40):
            raise ValueError('Matching complete development verification is required on both platforms.')
        verification_hashes[name] = sha256_file(path)
    environment = dict(os.environ)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        environment[key] = str(config['threads'])
    with (output/'refit.log').open('xb') as log:
        process = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u', 'tools/refit_nmclock.py'],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'created_at_utc': utc_now(), 'pid': process.pid, 'host': socket.gethostname(),
        'workspace': str(ROOT), 'log': 'runs/nmclock/refit.log',
        'selected_candidate': selection['selected_candidate'], 'selection_sha256': sha256_file(selection_path),
        'verification_sha256': verification_hashes, 'launch_code_sha256': sha256_file(Path(__file__)),
        'refit_code_sha256': sha256_file(ROOT/'tools/refit_nmclock.py')}
    write_json(output/'refit_launch.json', receipt)
    print(receipt)


if __name__ == '__main__':
    main()
