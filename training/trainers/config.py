"""YAML config -> typed objects. See training/configs/stream_anon_s.yaml."""
import dataclasses
import os
import sys
from dataclasses import dataclass, field

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stream_anon import StreamAnonConfig  # noqa: E402


@dataclass
class Config:
    model: StreamAnonConfig
    data: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)
    optim: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)
    pseudo_speaker: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def load_config(path) -> Config:
    raw = yaml.safe_load(open(path))
    known = {f.name for f in dataclasses.fields(StreamAnonConfig)}
    unknown = set(raw.get("model", {})) - known
    if unknown:
        raise ValueError(f"unknown model keys in {path}: {sorted(unknown)}")
    return Config(model=StreamAnonConfig(**raw.get("model", {})), data=raw.get("data", {}),
                  losses=raw.get("losses", {}), optim=raw.get("optim", {}), stages=raw.get("stages", []),
                  pseudo_speaker=raw.get("pseudo_speaker", {}), raw=raw)
