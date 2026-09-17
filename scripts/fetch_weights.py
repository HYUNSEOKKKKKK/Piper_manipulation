#!/usr/bin/env python3
"""Download or verify checkpoints, without importing Torch or opening pickle files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify(path, spec):
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        raise ValueError(f"{path}: missing or incorrect size (expected {spec['bytes']} bytes)")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != spec['sha256']:
        raise ValueError(f'{path}: SHA-256 mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true', help='Check existing files; never download')
    mode.add_argument('--from-dir', type=Path, help='Import files extracted from the official archive')
    mode.add_argument('--list', action='store_true', help='Print sources and sizes; do not download')
    args = parser.parse_args()
    specs = json.loads((ROOT / 'third_party/weights.json').read_text())['files']
    for spec in specs:
        target = args.weights_dir / spec['name']
        if args.list:
            print(f"{spec['name']} ({spec['bytes'] / 1048576:.1f} MiB): {spec['url']}")
            continue
        if args.verify or target.exists():
            verify(target, spec)
            print(f"Verified: {target}")
            continue
        args.weights_dir.mkdir(parents=True, exist_ok=True)
        # Only an entirely downloaded AND verified file gets the final filename.
        with tempfile.NamedTemporaryFile(dir=args.weights_dir, suffix='.download', delete=False) as f:
            temporary = Path(f.name)
        try:
            if args.from_dir:
                source = args.from_dir / spec['name']
                verify(source, spec)
                shutil.copyfile(source, temporary)
            else:
                print(f"Downloading {spec['name']} from {spec['source']}...", flush=True)
                request = urllib.request.Request(spec['url'], headers={'User-Agent': 'Piper_manipulation/0.1'})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as out:
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > spec['bytes']:
                            raise ValueError(f"{spec['name']}: download exceeds expected size")
                        out.write(block)
            verify(temporary, spec)
            temporary.replace(target)
            print(f'Verified: {target}')
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
