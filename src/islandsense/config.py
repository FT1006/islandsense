"""Configuration loader for IslandSense MVP."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Loads and validates config.yaml."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            # Default: config.yaml in project root
            config_path = Path(__file__).parent.parent.parent / "config.yaml"

        with open(config_path, "r") as f:
            self._data: Dict[str, Any] = yaml.safe_load(f)

        self._validate()

    def _validate(self):
        """Basic validation of required fields."""
        required_sections = ["project", "data", "label", "categories", "jdi", "actions"]
        for section in required_sections:
            if section not in self._data:
                raise ValueError(f"Missing required config section: {section}")

    # Project settings
    @property
    def project_name(self) -> str:
        return self._data["project"]["name"]

    @property
    def horizon_hours(self) -> int:
        return self._data["project"]["horizon_hours"]

    @property
    def bin_hours(self) -> int:
        return self._data["project"]["bin_hours"]

    # Data file paths
    @property
    def data_dir(self) -> Path:
        """Return data directory path."""
        return Path(__file__).parent.parent.parent / "data"

    @property
    def sailings_file(self) -> Path:
        return self.data_dir / self._data["data"]["sailings_file"]

    @property
    def status_file(self) -> Path:
        return self.data_dir / self._data["data"]["status_file"]

    @property
    def metocean_file(self) -> Path:
        return self.data_dir / self._data["data"]["metocean_file"]

    @property
    def tides_file(self) -> Path:
        return self.data_dir / self._data["data"]["tides_file"]

    @property
    def exposure_file(self) -> Path:
        return self.data_dir / self._data["data"]["exposure_file"]

    @property
    def my_sailings_file(self) -> Path:
        return self.data_dir / self._data["data"]["my_sailings_file"]

    # Label settings
    @property
    def disruption_delay_minutes(self) -> int:
        return self._data["label"]["disruption_delay_minutes"]

    # Categories
    @property
    def categories(self) -> Dict[str, Dict[str, str]]:
        return self._data["categories"]

    # JDI bands
    @property
    def jdi_bands(self) -> Dict[str, Dict[str, Any]]:
        return self._data["jdi"]["bands"]

    @property
    def jdi_expected_loss_min(self) -> float:
        return self._data["jdi"]["expected_loss_min"]

    @property
    def jdi_expected_loss_max(self) -> float:
        return self._data["jdi"]["expected_loss_max"]

    # Actions
    @property
    def actions(self) -> list:
        return self._data["actions"]

    # Impact calculation
    @property
    def k_hours_per_unit(self) -> float:
        return self._data["impact"]["k_hours_per_unit"]

    @property
    def units_per_trailer(self) -> float:
        return self._data["impact"]["units_per_trailer"]

    # Model settings
    @property
    def model_type(self) -> str:
        return self._data["model"]["type"]

    @property
    def random_seed(self) -> int:
        return self._data["model"]["random_seed"]

    @property
    def test_size_fraction(self) -> float:
        return self._data["model"]["test_size_fraction"]

    # UI settings
    @property
    def default_window_label(self) -> str:
        return self._data["ui"]["default_window_label"]

    @property
    def show_what_if_sliders(self) -> bool:
        return self._data["ui"]["show_what_if_sliders"]

    @property
    def show_download_brief(self) -> bool:
        return self._data["ui"]["show_download_brief"]


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
