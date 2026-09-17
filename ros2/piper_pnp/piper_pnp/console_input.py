"""Unbuffered terminal input with explicit UTF-8 decoding and key diagnostics."""
import codecs
from collections import deque
import os
import select


class TerminalKeyReader:
    """Drain decoded characters before waiting for another file-descriptor event.

    TextIOWrapper.read(1) can prefetch the next key. select(stdin) then misses
    that buffered key and delays it until a later physical keypress.
    """
    def __init__(self, fd):
        self.fd = fd
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.pending = deque()

    def read_key(self, timeout=.2):
        if self.pending:
            return self.pending.popleft()
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        data = os.read(self.fd, 128)
        if not data:
            raise EOFError('terminal input closed')
        self.pending.extend(self.decoder.decode(data))
        return self.pending.popleft() if self.pending else None


def key_action(key):
    if key in ('c', 'ㅊ'):
        return 'continue'
    if key in ('r', 'ㄱ'):
        return 'resume'
    if key in ('q', 'ㅂ'):
        return 'quit'
    if key in ('h', 'ㅗ'):
        return 'reset'
    return 'stop'


def describe_key(key):
    names = {'\n': 'Enter (LF)', '\r': 'Enter (CR)', ' ': 'Space',
             '\t': 'Tab', '\x1b': 'Escape', '\x03': 'Ctrl+C',
             '\x7f': 'Backspace', '\x00': 'NUL'}
    name = names.get(key, repr(key))
    return f'{name} / ' + ' '.join(f'U+{ord(char):04X}' for char in key)


class ConsoleKeyState:
    """Only h followed by a distinct y can confirm an empty-workspace reset."""
    def __init__(self):
        self.reset_pending = False

    def action(self, key):
        action = key_action(key)
        if action == 'reset':
            self.reset_pending = True
            return 'reset_prompt'
        if self.reset_pending:
            self.reset_pending = False
            if key in ('y', 'ㅛ'):
                return 'reset_confirm'
            return 'quit' if action == 'quit' else 'stop'
        return action
