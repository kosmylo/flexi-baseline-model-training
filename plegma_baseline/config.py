"""Configuration loading."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config. Relative paths resolve against the config file's folder."""
    if path:
        p = Path(path).resolve()
    elif DEFAULT_CONFIG.exists():
        p = DEFAULT_CONFIG
    elif Path("config.yaml").exists():
        p = Path("config.yaml").resolve()
    else:
        raise FileNotFoundError("No config.yaml found. Run from the repository folder or pass --config.")
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, value in cfg["paths"].items():
        q = Path(value)
        cfg["paths"][key] = str(q if q.is_absolute() else (p.parent / q).resolve())
    return cfg
