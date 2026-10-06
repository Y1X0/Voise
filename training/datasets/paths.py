"""Configuration-based path mapping (data stays byte-identical; only WHERE it is read from changes).

Manifests, feature-cache indexes and teacher-unit references store LOGICAL paths (e.g.
/data/mls/mls_english/..., data/features/..., data/teacher/units/...). On another machine (e.g. a
Kaggle notebook, where private datasets are mounted read-only under /kaggle/input/<slug>/) the same
files live elsewhere. A path map translates logical prefixes to physical ones at the moment a file is
opened; manifests, indexes, hashes and the data selection are unchanged.

  VOISE_PATH_MAP = JSON object {"<logical prefix>": "<physical prefix>", ...} or a path to such a JSON
  file (train.py / kaggle.py --path-map set it). Longest matching prefix wins; whole path components
  only ("data/features" maps "data/features/x" but not "data/features2/x"). Unmapped paths are
  returned unchanged, so without a map nothing changes. DataLoader workers inherit the variable.
"""
import json
import os

ENV = "VOISE_PATH_MAP"
_cache = {"src": None, "items": ()}


def load(spec):
    """spec: JSON text, a JSON file path, or a dict -> [(logical, physical)] sorted longest first."""
    if isinstance(spec, dict):
        m = spec
    elif spec and spec.lstrip().startswith("{"):
        m = json.loads(spec)
    elif spec:
        with open(spec, encoding="utf-8") as f:
            m = json.load(f)
    else:
        m = {}
    if not isinstance(m, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in m.items()):
        raise ValueError("path map must be a JSON object of strings {logical_prefix: physical_prefix}")
    items = [(os.path.normpath(k), os.path.normpath(v)) for k, v in m.items() if not k.startswith("_")]
    return tuple(sorted(items, key=lambda kv: -len(kv[0])))


def _items():
    spec = os.environ.get(ENV, "")
    if spec != _cache["src"]:
        _cache["src"], _cache["items"] = spec, load(spec)
    return _cache["items"]


def resolve(path, items=None):
    """Physical path for a logical one (unchanged if no prefix matches)."""
    if not path:
        return path
    items = _items() if items is None else items
    if not items:
        return path
    n = os.path.normpath(path)
    for src, dst in items:
        if n == src or n.startswith(src + os.sep):
            return dst + n[len(src):]
    return path


def activate(spec):
    """Set the map for this process and its children (spec: file path or JSON text)."""
    if spec:
        load(spec)                                   # validate early
        os.environ[ENV] = os.path.abspath(spec) if os.path.exists(spec) else spec
    return _items()
