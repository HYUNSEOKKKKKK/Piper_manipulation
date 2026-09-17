#!/usr/bin/env python3
"""Audit tracked release files; does not read ignored weights or local credentials."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    files = [name for name in files if name]
    if not files:
        raise SystemExit('No tracked files; stage the intended release first.')
    errors = []
    total = 0
    forbidden = {'.pt', '.pth', '.onnx', '.engine', '.so', '.zip', '.bag', '.db3', '.mcap', '.mp4'}
    for name in files:
        path = ROOT / name
        if path.is_symlink():
            errors.append(f'{name}: release must not contain workstation symlinks')
            continue
        data = path.read_bytes()
        total += len(data)
        if path.suffix.lower() in forbidden or '.state.json' in name:
            errors.append(f'{name}: runtime or binary artifact')
        if len(data) > 5 * 1024 * 1024:
            errors.append(f'{name}: exceeds project 5 MiB source-file budget')
        text = data.decode('utf-8', errors='replace')
        if re.search(r'/home/[A-Za-z0-9_.-]+/', text):
            errors.append(f'{name}: hard-coded home directory')
        if re.search(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}', text) or re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----', text):
            errors.append(f'{name}: potential credential')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=name)
            except SyntaxError as error:
                errors.append(f'{name}: {error}')
    manifest = json.loads((ROOT / 'third_party/dependencies.json').read_text())
    for spec in manifest['repositories']:
        if not re.fullmatch('[a-f0-9]{40}', spec['commit']):
            errors.append(f"{spec['name']}: source revision is not pinned")
    for spec in json.loads((ROOT / 'third_party/weights.json').read_text())['files']:
        if not re.fullmatch('[a-f0-9]{64}', spec['sha256']) or spec['bytes'] <= 0:
            errors.append(f"{spec['name']}: invalid weight integrity metadata")
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Release audit passed: {len(files)} tracked files, {total / 1048576:.2f} MiB uncompressed.')


if __name__ == '__main__':
    main()
