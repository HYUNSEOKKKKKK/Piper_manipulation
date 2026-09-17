#!/usr/bin/env python3
"""Fetch exact source revisions without installing packages or starting hardware."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=('perception', 'robot', 'all'), default='all')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    repos = json.loads((ROOT / 'third_party/dependencies.json').read_text())['repositories']
    for repo in repos:
        if args.group not in ('all', repo['group']):
            continue
        dest = ROOT / repo['directory']
        print(f"{repo['name']} @ {repo['commit']} -> {dest}", flush=True)
        if args.dry_run:
            continue
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            run('git', 'clone', '--filter=blob:none', '--no-checkout', repo['url'], str(dest))
            if repo['name'] == 'MobileSAM':
                run('git', 'sparse-checkout', 'set', '--cone', 'MobileSAMv2/mobilesamv2',
                    'MobileSAMv2/tinyvit', 'MobileSAMv2/ultralytics', cwd=dest)
            run('git', 'checkout', '--detach', repo['commit'], cwd=dest)
        if run('git', 'rev-parse', 'HEAD', cwd=dest) != repo['commit']:
            raise SystemExit(f'{dest}: revision mismatch; refusing to overwrite an existing checkout')
        if (dest / '.gitmodules').exists():
            # Gitlinks in the pinned parent fix the mesh revision; never use --remote.
            run('git', 'submodule', 'update', '--init', '--recursive', cwd=dest)
        if repo['name'] == 'MobileSAM':
            for patch in sorted((ROOT / 'third_party/patches').glob('mobilesam*.patch')):
                reversed_check = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)],
                    cwd=dest, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if reversed_check.returncode == 0:
                    print(f'  already applied: {patch.name}')
                else:
                    run('git', 'apply', '--check', str(patch), cwd=dest)
                    run('git', 'apply', str(patch), cwd=dest)
                    print(f'  applied: {patch.name}')
    if args.group in ('all', 'robot') and not args.dry_run:
        link = ROOT / '.workspace/src/piper_pnp'
        target = ROOT / 'ros2/piper_pnp'
        if link.exists() or link.is_symlink():
            if link.resolve() != target:
                raise SystemExit(f'Refusing to replace {link}')
        else:
            link.symlink_to(target, target_is_directory=True)
    print('Done. No robot commands were sent.')


if __name__ == '__main__':
    main()
