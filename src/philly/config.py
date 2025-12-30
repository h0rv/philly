"""Configuration management for Philly library.

Supports loading configuration from YAML files with priority:
1. ./philly.yml (project-level)
2. ~/.config/philly/config.yml (user-level)
3. Default values
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """Cache configuration settings."""

    enabled: bool = True
    directory: str = "~/.cache/philly"
    ttl: int = 3600
    max_size_mb: float | None = None


class DefaultsConfig(BaseModel):
    """Default behavior configuration."""

    format_preference: list[str] = ["csv", "geojson", "json", "shp"]
    ignore_load_errors: bool = False
    max_concurrency: int = 20
    show_progress: bool = True


class CLIConfig(BaseModel):
    """CLI-specific configuration."""

    output_format: str = "auto"
    quiet: bool = False
    verbose: bool = False
    compact: bool = False


class NetworkConfig(BaseModel):
    """Network request configuration."""

    timeout: int = 120
    retries: int = 3
    user_agent: str = "philly/1.0"


class PhillyConfig(BaseModel):
    """Main configuration for Philly library."""

    cache: CacheConfig = Field(default_factory=CacheConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    @classmethod
    def load_from_file(cls, path: Path) -> "PhillyConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the configuration file

        Returns:
            PhillyConfig instance with loaded settings

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file is not valid YAML
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Handle empty file
        if data is None:
            return cls()

        return cls(**data)

    def save_to_file(self, path: Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where the configuration should be saved
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and write to file
        data = self.model_dump()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def find_config_file() -> Path | None:
    """Find configuration file in priority order.

    Search order:
    1. ./philly.yml (project-level)
    2. ~/.config/philly/config.yml (user-level)

    Returns:
        Path to the configuration file, or None if not found
    """
    # Check project-level config
    project_config = Path.cwd() / "philly.yml"
    if project_config.exists():
        return project_config

    # Check user-level config
    user_config = Path.home() / ".config" / "philly" / "config.yml"
    if user_config.exists():
        return user_config

    return None


def load_config(config_file: str | Path | None = None) -> PhillyConfig:
    """Load configuration from file or use defaults.

    Args:
        config_file: Optional path to configuration file.
                    If None, will search in default locations.

    Returns:
        PhillyConfig instance with loaded or default settings

    Raises:
        FileNotFoundError: If config_file is specified but doesn't exist
        yaml.YAMLError: If the configuration file is not valid YAML
    """
    if config_file is not None:
        # Explicit config file specified
        path = Path(config_file).expanduser()
        return PhillyConfig.load_from_file(path)

    # Search for config file in default locations
    config_path = find_config_file()
    if config_path is not None:
        return PhillyConfig.load_from_file(config_path)

    # No config file found, use defaults
    return PhillyConfig()
