from __future__ import annotations

from typing import Any, Dict


def get_cfg(cfg: Dict[str, Any], key: str, default):
    return cfg[key] if key in cfg else default
