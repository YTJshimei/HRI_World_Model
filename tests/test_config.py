from pathlib import Path

import pytest

from src.utils.config import ConfigError, get_required_path, load_yaml_config


def write_config(tmp_path: Path, content: str) -> Path:
    """Create a temporary YAML configuration used by a test."""
    config_file = tmp_path / "paths.yaml"
    config_file.write_text(content, encoding="utf-8")
    return config_file


def test_yaml_can_be_loaded(tmp_path: Path) -> None:
    config_file = write_config(
        tmp_path,
        'project_root: "C:/HRI_World_Model"\n'
        'data_root: "E:/HRI_World_Model_Data"\n',
    )

    config = load_yaml_config(config_file)

    assert isinstance(config, dict)


def test_project_root_is_loaded(tmp_path: Path) -> None:
    config_file = write_config(
        tmp_path,
        'project_root: "C:/HRI_World_Model"\n'
        'data_root: "E:/HRI_World_Model_Data"\n',
    )

    config = load_yaml_config(config_file)

    assert get_required_path(config, "project_root") == Path("C:/HRI_World_Model")


def test_data_root_is_loaded(tmp_path: Path) -> None:
    config_file = write_config(
        tmp_path,
        'project_root: "C:/HRI_World_Model"\n'
        'data_root: "E:/HRI_World_Model_Data"\n',
    )

    config = load_yaml_config(config_file)

    assert get_required_path(config, "data_root") == Path("E:/HRI_World_Model_Data")


def test_missing_required_key_has_clear_error(tmp_path: Path) -> None:
    config_file = write_config(
        tmp_path,
        'project_root: "C:/HRI_World_Model"\n',
    )
    config = load_yaml_config(config_file)

    with pytest.raises(ConfigError, match="缺失必需配置项.*data_root"):
        get_required_path(config, "data_root")
