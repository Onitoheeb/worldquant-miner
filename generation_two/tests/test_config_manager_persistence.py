from pathlib import Path

from generation_two.core.config.config_manager import ConfigManager


def test_default_config_path_is_absolute_and_app_relative():
    manager = ConfigManager()

    expected = Path(__file__).resolve().parents[1] / "generation_two_config.json"
    assert Path(manager.config_path).is_absolute()
    assert Path(manager.config_path) == expected


def test_save_creates_parent_directory_and_round_trips(tmp_path):
    config_path = tmp_path / "nested" / "settings" / "generation_two_config.json"
    manager = ConfigManager(config_path=str(config_path))
    manager.update_section("simulation", {"max_wait_time": 420})

    manager.save()

    assert config_path.exists()
    loaded = ConfigManager(config_path=str(config_path))
    assert loaded.get("simulation", "max_wait_time") == 420
