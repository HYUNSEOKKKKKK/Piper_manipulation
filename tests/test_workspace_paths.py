"""Resolve the project and its Python without borrowing another research env."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('active,override,expected', [
    ('piper_manipulation', None, 'conda'),
    ('defm', None, 'project'),
    ('piper_manipulation', '/explicit/python', 'explicit'),
])
def test_common_paths_from_foreign_directory(tmp_path, active, override, expected):
    prefix = tmp_path / 'environment'
    python = prefix / 'bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o755)
    environment = os.environ.copy()
    for key in ('PIPER_WS', 'PIPER_PERCEPTION_PYTHON', 'CONDA_DEFAULT_ENV', 'CONDA_PREFIX'):
        environment.pop(key, None)
    environment.update(CONDA_DEFAULT_ENV=active, CONDA_PREFIX=str(prefix))
    if override:
        environment['PIPER_PERCEPTION_PYTHON'] = override
    lines = subprocess.check_output(
        ['bash', '--noprofile', '--norc', '-c',
         'source "$1"; printf "%s\\n" "$PIPER_ROOT" "$PIPER_WS" "$PIPER_PERCEPTION_PYTHON"',
         '_', str(ROOT / 'scripts/common.sh')],
        cwd=tmp_path, env=environment, text=True).splitlines()
    assert Path(lines[0]).resolve() == ROOT
    assert Path(lines[1]).resolve() == ROOT / '.workspace'
    target = {'conda': str(python), 'project': str(ROOT / '.venv-perception/bin/python'),
              'explicit': override}[expected]
    assert lines[2] == target
