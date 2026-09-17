"""Real PTY regression: stop keys must not remain hidden until the next c."""
import os
import pty
import time
import tty

import pytest

from piper_pnp.console_input import TerminalKeyReader, describe_key, key_action


@pytest.fixture
def terminal():
    master, slave = pty.openpty()
    tty.setcbreak(slave)
    try:
        yield master, TerminalKeyReader(slave)
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize('stop_key', [b'\n', b' ', b'\t', b'\x1b', b'x'])
def test_stop_key_in_same_read_is_not_delayed_until_next_c(terminal, stop_key):
    master, reader = terminal
    os.write(master, b'c'+stop_key)
    assert key_action(reader.read_key()) == 'continue'
    assert key_action(reader.read_key(timeout=0)) == 'stop'
    assert reader.read_key(timeout=.01) is None
    os.write(master, b'c')
    assert key_action(reader.read_key()) == 'continue'
    assert reader.read_key(timeout=.01) is None


def test_two_independent_c_keys_never_generate_a_stop(terminal):
    master, reader = terminal
    for _ in range(2):
        os.write(master,b'c')
        assert reader.read_key() == 'c'
        time.sleep(.01)
        assert reader.read_key(timeout=.01) is None


def test_fragmented_hangul_key_and_following_stop(terminal):
    master, reader = terminal
    encoded = 'ㅊ'.encode('utf-8')
    os.write(master,encoded[:1])
    assert reader.read_key() is None
    os.write(master,encoded[1:]+b' ')
    assert key_action(reader.read_key()) == 'continue'
    assert key_action(reader.read_key(timeout=0)) == 'stop'


@pytest.mark.parametrize('key, action', [
    ('c','continue'),('ㅊ','continue'),('r','resume'),('ㄱ','resume'),
    ('q','quit'),('ㅂ','quit'),('\n','stop'),('\r','stop'),(' ','stop'),
    ('C','stop'),('\x03','stop'),('\x1b','stop'),('\x00','stop'),('�','stop'),
])
def test_key_bindings_preserve_stop_behavior(key, action):
    assert key_action(key) == action


def test_invalid_bytes_produce_a_stop_key(terminal):
    master,reader = terminal
    os.write(master,b'\xff')
    assert key_action(reader.read_key()) == 'stop'


def test_disconnect_is_reported_instead_of_spinning_forever():
    read_fd,write_fd = os.pipe()
    reader = TerminalKeyReader(read_fd)
    os.close(write_fd)
    try:
        with pytest.raises(EOFError): reader.read_key()
    finally:
        os.close(read_fd)


def test_stop_reason_distinguishes_enter_space_and_escape():
    assert describe_key('\n') == 'Enter (LF) / U+000A'
    assert describe_key(' ') == 'Space / U+0020'
    assert describe_key('\x1b') == 'Escape / U+001B'
