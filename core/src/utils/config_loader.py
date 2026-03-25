"""Configuration loader with dot-access support."""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """YAML config를 dot-access로 사용할 수 있게 래핑."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [Config(v) if isinstance(v, dict) else v for v in value])
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                result[k] = v
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(path: Optional[str] = None) -> Config:
    """YAML config 파일을 로드하여 Config 객체로 반환."""
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "configs" / "default_config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)
