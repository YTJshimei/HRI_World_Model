"""Utilities for loading and validating project path configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "读取 YAML 配置需要 PyYAML。请按项目环境管理流程安装 requirements-base.txt。"
    ) from exc


class ConfigError(ValueError):
    """Raised when a project configuration is invalid or incomplete."""


def expand_path(value: str | os.PathLike[str]) -> Path:
    """Expand environment variables and ``~`` in a path value."""
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(
            f"路径值必须是字符串或 os.PathLike，实际类型为 {type(value).__name__}。"
        )

    raw_value = os.fspath(value).strip()
    if not raw_value:
        raise ConfigError("路径值不能为空。")

    return Path(os.path.expandvars(os.path.expanduser(raw_value)))


def load_yaml_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping from *config_path* with actionable errors."""
    path = expand_path(config_path)
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    if not path.is_file():
        raise ConfigError(f"配置路径不是文件：{path}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 格式无效（{path}）：{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件（{path}）：{exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"配置文件顶层必须是键值映射：{path}")
    return loaded


def get_required_path(config: Mapping[str, Any], key: str) -> Path:
    """Return an expanded path for a required configuration key."""
    if key not in config:
        raise ConfigError(f"缺失必需配置项：'{key}'")

    value = config[key]
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(
            f"配置项 '{key}' 必须是路径字符串，实际类型为 {type(value).__name__}。"
        )

    try:
        return expand_path(value)
    except ConfigError as exc:
        raise ConfigError(f"配置项 '{key}' 无效：{exc}") from exc


def check_path_exists(
    path: str | os.PathLike[str], *, label: str = "路径", expect_directory: bool = True
) -> Path:
    """Validate that a path exists and optionally that it is a directory."""
    expanded = expand_path(path)
    if not expanded.exists():
        raise ConfigError(f"{label}不存在：{expanded}")
    if expect_directory and not expanded.is_dir():
        raise ConfigError(f"{label}不是目录：{expanded}")
    if not expect_directory and not expanded.is_file():
        raise ConfigError(f"{label}不是文件：{expanded}")
    return expanded
