"""Package only this campaign's code, declarations and field-use evidence."""
import argparse
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_nmclock.bundle import verify_field_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('infra/azure/.local/nmclock-code-initial.tar.gz'))
    args = parser.parse_args()
    verify_field_review(ROOT, read_json(ROOT/'configs/features_nmclock_v1.json'))
    files = [*sorted((ROOT/'src/prc2026_nmclock').glob('*.py')),
             *sorted((ROOT/'tools').glob('*nmclock*.py')),
             *sorted((ROOT/'tests').glob('test_nmclock*.py')),
             ROOT/'requirements-nmclock.txt', ROOT/'configs/nmclock_v1.json',
             ROOT/'configs/features_nmclock_v1.json', ROOT/'infra/azure/nmclock.sh',
             ROOT/'docs/nmclock/README.md', ROOT/'docs/nmclock/field_use_review.json', ROOT/'docs/nmclock/declaration.json']
    for item in read_json(ROOT/'docs/nmclock/field_use_review.json')['sources']:
        path = ROOT/item['path']
        if sha256_file(path) != item['sha256']:
            raise ValueError('Archived public evidence changed.')
        files.append(path)
    output = ROOT/args.output
    if output.exists() or not output.resolve().is_relative_to((ROOT/'infra/azure/.local').resolve()):
        raise ValueError('Choose a new archive under infra/azure/.local.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(set(files)):
            if not path.is_file() or path.is_symlink():
                raise ValueError('Unexpected package input.')
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    receipt = {'created_at_utc': utc_now(), 'sha256': sha256_file(output), 'size_bytes': output.stat().st_size,
               'files': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(set(files))}}
    write_json(output.with_suffix('.json'), receipt)
    print({'package': output.name, 'sha256': receipt['sha256'], 'bytes': receipt['size_bytes'], 'files': len(receipt['files'])})


if __name__ == '__main__':
    main()
