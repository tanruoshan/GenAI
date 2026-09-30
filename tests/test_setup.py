"""Smoke test: the package installs and the config loads."""
from pathlib import Path

import yaml

import blnrepair


def test_package_imports():
    assert blnrepair.__version__


def test_config_loads():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["seed"] == 42
