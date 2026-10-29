"""Shared schema and deterministic file utilities; no model dependencies."""
import hashlib
import json
from pathlib import Path

VERSION = '2.0.0'
ATTRS = ('color', 'texture', 'shape')
COLORS = {'red': (211, 55, 60), 'yellow': (229, 183, 37), 'blue': (48, 111, 205),
          'green': (43, 151, 88), 'purple': (141, 74, 185), 'orange': (234, 121, 33)}
TEXTURES = ('solid', 'stripes', 'dots', 'grid')
SHAPES = ('circle', 'triangle', 'square', 'star', 'pentagon', 'hexagon')
VALUES = (tuple(COLORS), TEXTURES, SHAPES)


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(canonical(r) + '\n' for r in rows), encoding='utf-8')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(obj):
    return f'{obj[0]} {obj[1]} {obj[2]}'


def parse_answer(text, kind):
    text = (text or '').strip()
    if kind == 'attributes':
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return None
        if not isinstance(value, dict) or set(value) != set(ATTRS):
            return None
        return value if all(value[a] in v for a, v in zip(ATTRS, VALUES)) else None
    text = text.upper()
    allowed = {'binary': ('YES', 'NO'), 'choice2': ('A', 'B'),
               'choice4': ('A', 'B', 'C', 'D'), 'recall': tuple(x.upper() for x in TEXTURES)}[kind]
    return text if text in allowed else None


def correct(pred, answer):
    return pred == answer
