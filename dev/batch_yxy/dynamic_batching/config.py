"""配置装载：dict/JSON -> BatchPolicy，无隐式默认。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .types import BatchPolicy

_REQUIRED_KEYS = (
    "model_id",
    "policy_version",
    "bucket_upper_bounds",
    "max_batch_size",
    "max_total_tokens",
    "max_wait_ms",
)
_OPTIONAL_KEYS = ("allow_adjacent_bucket_merge", "max_padding_ratio")


def policy_from_config(cfg: Mapping[str, Any]) -> BatchPolicy:
    missing = [k for k in _REQUIRED_KEYS if cfg.get(k) is None]
    if missing:
        raise KeyError(f"策略配置缺少必填字段: {missing}")
    kwargs: dict[str, Any] = {k: cfg[k] for k in _REQUIRED_KEYS}
    kwargs["bucket_upper_bounds"] = tuple(cfg["bucket_upper_bounds"])
    for key in _OPTIONAL_KEYS:
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    return BatchPolicy(**kwargs)


def policy_from_json_file(path: str | Path) -> BatchPolicy:
    return policy_from_config(json.loads(Path(path).read_text(encoding="utf-8")))
