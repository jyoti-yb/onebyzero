from copy import deepcopy
from pathlib import Path

import pytest

from src.config import ConfigError, load_config, validate_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "smoke_test.yaml"


def test_smoke_test_config_defaults() -> None:
    config = load_config(CONFIG_PATH)

    assert config["domain"] == {"south": 6.5, "north": 38.5, "west": 66.5, "east": 100.0}
    assert config["gfs"]["cycle"] == "00"
    assert config["gfs"]["model"] == "gfs_0p25"
    assert config["gfs"]["lead_day"] == 1
    assert config["verification"]["rainfall_window_start_utc"] == 3
    assert config["verification"]["rainfall_window_hours"] == 24
    assert config["variables"]["surface"] == ["APCP", "PRMSL", "PWAT", "CAPE"]
    assert config["quality"]["min_valid_grid_fraction"] == 0.90


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("domain", "north", 5.0),
        ("gfs", "cycle", "01"),
        ("verification", "rainfall_window_start_utc", 24),
        ("quality", "rh_max", 101),
        ("quality", "min_valid_grid_fraction", 0),
    ],
)
def test_invalid_config_is_rejected(section: str, key: str, value: object) -> None:
    config = deepcopy(load_config(CONFIG_PATH))
    config[section][key] = value

    with pytest.raises(ConfigError):
        validate_config(config)


def test_missing_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("domain: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(config_path)
