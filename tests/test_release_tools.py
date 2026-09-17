"""Checkpoint integrity rejects corrupted files before any model deserialization."""
import hashlib
from pathlib import Path
import pytest
from fetch_weights import verify


def test_checkpoint_hash_required_even_when_size_matches(tmp_path):
    path = tmp_path / 'model.pt'
    path.write_bytes(b'valid checkpoint fixture')
    spec = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    verify(path, spec)
    path.write_bytes(b'bad!! checkpoint fixture')
    with pytest.raises(ValueError, match='SHA-256'):
        verify(path, spec)


def test_truncated_or_missing_checkpoint_rejected(tmp_path):
    path = tmp_path / 'model.pt'
    spec = dict(bytes=10, sha256='0'*64)
    with pytest.raises(ValueError, match='missing or incorrect size'):
        verify(path, spec)
    path.write_bytes(b'partial')
    with pytest.raises(ValueError, match='missing or incorrect size'):
        verify(path, spec)
