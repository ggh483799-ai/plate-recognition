"""YAML 配置加载（带缓存）。计划书：配置走 YAML 禁止硬编码。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# 项目根 = src/common/ 上溯两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """加载 configs/<name>（可省略 .yaml 后缀），结果缓存。"""
    path = CONFIGS_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def weight_path(name: str) -> Path:
    """返回 weights/ 下某个权重文件的绝对路径。"""
    return WEIGHTS_DIR / name
