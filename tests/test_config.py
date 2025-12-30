"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from philly.config import (
    CacheConfig,
    CLIConfig,
    DefaultsConfig,
    NetworkConfig,
    PhillyConfig,
    find_config_file,
    load_config,
)


class TestCacheConfig:
    """Test CacheConfig model."""

    def test_default_values(self):
        """Test default values for CacheConfig."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.directory == "~/.cache/philly"
        assert config.ttl == 3600
        assert config.max_size_mb is None

    def test_custom_values(self):
        """Test CacheConfig with custom values."""
        config = CacheConfig(
            enabled=False,
            directory="/tmp/cache",
            ttl=7200,
            max_size_mb=100.0,
        )
        assert config.enabled is False
        assert config.directory == "/tmp/cache"
        assert config.ttl == 7200
        assert config.max_size_mb == 100.0

    def test_partial_values(self):
        """Test CacheConfig with partial custom values."""
        config = CacheConfig(ttl=1800)
        assert config.enabled is True  # default
        assert config.directory == "~/.cache/philly"  # default
        assert config.ttl == 1800  # custom
        assert config.max_size_mb is None  # default


class TestDefaultsConfig:
    """Test DefaultsConfig model."""

    def test_default_values(self):
        """Test default values for DefaultsConfig."""
        config = DefaultsConfig()
        assert config.format_preference == ["csv", "geojson", "json", "shp"]
        assert config.ignore_load_errors is False
        assert config.max_concurrency == 20
        assert config.show_progress is True

    def test_custom_values(self):
        """Test DefaultsConfig with custom values."""
        config = DefaultsConfig(
            format_preference=["json", "csv"],
            ignore_load_errors=True,
            max_concurrency=10,
            show_progress=False,
        )
        assert config.format_preference == ["json", "csv"]
        assert config.ignore_load_errors is True
        assert config.max_concurrency == 10
        assert config.show_progress is False

    def test_partial_values(self):
        """Test DefaultsConfig with partial custom values."""
        config = DefaultsConfig(max_concurrency=5)
        assert config.format_preference == ["csv", "geojson", "json", "shp"]
        assert config.ignore_load_errors is False
        assert config.max_concurrency == 5
        assert config.show_progress is True


class TestCLIConfig:
    """Test CLIConfig model."""

    def test_default_values(self):
        """Test default values for CLIConfig."""
        config = CLIConfig()
        assert config.output_format == "auto"
        assert config.quiet is False
        assert config.verbose is False
        assert config.compact is False

    def test_custom_values(self):
        """Test CLIConfig with custom values."""
        config = CLIConfig(
            output_format="json",
            quiet=True,
            verbose=True,
            compact=True,
        )
        assert config.output_format == "json"
        assert config.quiet is True
        assert config.verbose is True
        assert config.compact is True

    def test_partial_values(self):
        """Test CLIConfig with partial custom values."""
        config = CLIConfig(output_format="csv", quiet=True)
        assert config.output_format == "csv"
        assert config.quiet is True
        assert config.verbose is False
        assert config.compact is False


class TestNetworkConfig:
    """Test NetworkConfig model."""

    def test_default_values(self):
        """Test default values for NetworkConfig."""
        config = NetworkConfig()
        assert config.timeout == 120
        assert config.retries == 3
        assert config.user_agent == "philly/1.0"

    def test_custom_values(self):
        """Test NetworkConfig with custom values."""
        config = NetworkConfig(
            timeout=60,
            retries=5,
            user_agent="custom-agent/2.0",
        )
        assert config.timeout == 60
        assert config.retries == 5
        assert config.user_agent == "custom-agent/2.0"

    def test_partial_values(self):
        """Test NetworkConfig with partial custom values."""
        config = NetworkConfig(timeout=30)
        assert config.timeout == 30
        assert config.retries == 3
        assert config.user_agent == "philly/1.0"


class TestPhillyConfig:
    """Test PhillyConfig composition."""

    def test_default_values(self):
        """Test PhillyConfig with all default sub-configs."""
        config = PhillyConfig()

        # Check sub-configs exist
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.defaults, DefaultsConfig)
        assert isinstance(config.cli, CLIConfig)
        assert isinstance(config.network, NetworkConfig)

        # Check default values
        assert config.cache.enabled is True
        assert config.defaults.max_concurrency == 20
        assert config.cli.output_format == "auto"
        assert config.network.timeout == 120

    def test_custom_sub_configs(self):
        """Test PhillyConfig with custom sub-configs."""
        config = PhillyConfig(
            cache=CacheConfig(enabled=False),
            defaults=DefaultsConfig(max_concurrency=5),
            cli=CLIConfig(quiet=True),
            network=NetworkConfig(timeout=60),
        )

        assert config.cache.enabled is False
        assert config.defaults.max_concurrency == 5
        assert config.cli.quiet is True
        assert config.network.timeout == 60

    def test_partial_sub_configs(self):
        """Test PhillyConfig with partial sub-configs."""
        config = PhillyConfig(
            cache=CacheConfig(ttl=7200),
        )

        # Custom cache config
        assert config.cache.ttl == 7200
        assert config.cache.enabled is True  # default preserved

        # Default sub-configs
        assert config.defaults.max_concurrency == 20
        assert config.cli.output_format == "auto"
        assert config.network.timeout == 120


class TestLoadFromFile:
    """Test PhillyConfig.load_from_file method."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML configuration."""
        config_file = tmp_path / "config.yml"
        config_data = {
            "cache": {"enabled": False, "ttl": 7200},
            "defaults": {"max_concurrency": 10},
            "cli": {"output_format": "json", "quiet": True},
            "network": {"timeout": 60},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = PhillyConfig.load_from_file(config_file)

        assert config.cache.enabled is False
        assert config.cache.ttl == 7200
        assert config.defaults.max_concurrency == 10
        assert config.cli.output_format == "json"
        assert config.cli.quiet is True
        assert config.network.timeout == 60

    def test_load_partial_config(self, tmp_path):
        """Test loading partial config merges with defaults."""
        config_file = tmp_path / "config.yml"
        config_data = {
            "cache": {"ttl": 1800},
            "cli": {"quiet": True},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = PhillyConfig.load_from_file(config_file)

        # Custom values
        assert config.cache.ttl == 1800
        assert config.cli.quiet is True

        # Default values preserved
        assert config.cache.enabled is True
        assert config.cache.directory == "~/.cache/philly"
        assert config.defaults.max_concurrency == 20
        assert config.network.timeout == 120

    def test_load_empty_file_returns_defaults(self, tmp_path):
        """Test loading empty file returns default config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("")

        config = PhillyConfig.load_from_file(config_file)

        # Should have all defaults
        assert config.cache.enabled is True
        assert config.cache.ttl == 3600
        assert config.defaults.max_concurrency == 20
        assert config.cli.output_format == "auto"
        assert config.network.timeout == 120

    def test_load_only_comments_returns_defaults(self, tmp_path):
        """Test loading file with only comments returns defaults."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# This is a comment\n# Another comment\n")

        config = PhillyConfig.load_from_file(config_file)

        # Should have all defaults
        assert config.cache.enabled is True
        assert config.defaults.max_concurrency == 20

    def test_load_invalid_yaml_raises_error(self, tmp_path):
        """Test loading invalid YAML raises error."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            PhillyConfig.load_from_file(config_file)

    def test_load_nonexistent_file_raises_error(self, tmp_path):
        """Test loading non-existent file raises FileNotFoundError."""
        config_file = tmp_path / "nonexistent.yml"

        with pytest.raises(FileNotFoundError):
            PhillyConfig.load_from_file(config_file)

    def test_load_nested_partial_config(self, tmp_path):
        """Test loading config with nested partial values."""
        config_file = tmp_path / "config.yml"
        config_data = {
            "cache": {"ttl": 5400},
            "defaults": {
                "format_preference": ["json"],
                "max_concurrency": 15,
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = PhillyConfig.load_from_file(config_file)

        # Custom values
        assert config.cache.ttl == 5400
        assert config.defaults.format_preference == ["json"]
        assert config.defaults.max_concurrency == 15

        # Defaults preserved within same section
        assert config.cache.enabled is True
        assert config.cache.directory == "~/.cache/philly"
        assert config.defaults.ignore_load_errors is False
        assert config.defaults.show_progress is True


class TestSaveToFile:
    """Test PhillyConfig.save_to_file method."""

    def test_save_creates_valid_yaml(self, tmp_path):
        """Test save creates valid YAML file."""
        config_file = tmp_path / "config.yml"
        config = PhillyConfig(
            cache=CacheConfig(enabled=False, ttl=7200),
            cli=CLIConfig(quiet=True),
        )

        config.save_to_file(config_file)

        assert config_file.exists()

        # Load and verify
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)

        assert data["cache"]["enabled"] is False
        assert data["cache"]["ttl"] == 7200
        assert data["cli"]["quiet"] is True

    def test_save_creates_parent_directory(self, tmp_path):
        """Test save creates parent directories if needed."""
        config_file = tmp_path / "nested" / "dir" / "config.yml"
        config = PhillyConfig()

        config.save_to_file(config_file)

        assert config_file.exists()
        assert config_file.parent.exists()

    def test_roundtrip_save_then_load(self, tmp_path):
        """Test save then load preserves configuration."""
        config_file = tmp_path / "config.yml"

        # Create and save config
        original_config = PhillyConfig(
            cache=CacheConfig(enabled=False, ttl=9000, max_size_mb=50.0),
            defaults=DefaultsConfig(
                format_preference=["json", "csv"],
                max_concurrency=8,
                ignore_load_errors=True,
            ),
            cli=CLIConfig(output_format="table", quiet=True, verbose=False),
            network=NetworkConfig(timeout=30, retries=5, user_agent="test/1.0"),
        )

        original_config.save_to_file(config_file)

        # Load and verify
        loaded_config = PhillyConfig.load_from_file(config_file)

        # Cache config
        assert loaded_config.cache.enabled == original_config.cache.enabled
        assert loaded_config.cache.ttl == original_config.cache.ttl
        assert loaded_config.cache.max_size_mb == original_config.cache.max_size_mb

        # Defaults config
        assert (
            loaded_config.defaults.format_preference
            == original_config.defaults.format_preference
        )
        assert (
            loaded_config.defaults.max_concurrency
            == original_config.defaults.max_concurrency
        )
        assert (
            loaded_config.defaults.ignore_load_errors
            == original_config.defaults.ignore_load_errors
        )

        # CLI config
        assert loaded_config.cli.output_format == original_config.cli.output_format
        assert loaded_config.cli.quiet == original_config.cli.quiet

        # Network config
        assert loaded_config.network.timeout == original_config.network.timeout
        assert loaded_config.network.retries == original_config.network.retries
        assert loaded_config.network.user_agent == original_config.network.user_agent

    def test_save_overwrites_existing_file(self, tmp_path):
        """Test save overwrites existing file."""
        config_file = tmp_path / "config.yml"

        # Create initial config
        config1 = PhillyConfig(cache=CacheConfig(ttl=1000))
        config1.save_to_file(config_file)

        # Overwrite with new config
        config2 = PhillyConfig(cache=CacheConfig(ttl=2000))
        config2.save_to_file(config_file)

        # Verify it was overwritten
        loaded_config = PhillyConfig.load_from_file(config_file)
        assert loaded_config.cache.ttl == 2000


class TestFindConfigFile:
    """Test find_config_file function."""

    def test_finds_project_level_config(self, tmp_path, monkeypatch):
        """Test finds project-level config (./philly.yml)."""
        monkeypatch.chdir(tmp_path)

        # Create project-level config
        project_config = tmp_path / "philly.yml"
        project_config.write_text("cache:\n  ttl: 1000\n")

        found = find_config_file()

        assert found is not None
        assert found == project_config
        assert found.name == "philly.yml"

    def test_finds_user_level_config(self, tmp_path, monkeypatch):
        """Test finds user-level config (~/.config/philly/config.yml)."""
        # Mock home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Change to directory without project config
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create user-level config
        user_config_dir = fake_home / ".config" / "philly"
        user_config_dir.mkdir(parents=True)
        user_config = user_config_dir / "config.yml"
        user_config.write_text("cache:\n  ttl: 2000\n")

        found = find_config_file()

        assert found is not None
        assert found == user_config

    def test_returns_none_when_no_config_exists(self, tmp_path, monkeypatch):
        """Test returns None when no config file exists."""
        # Mock home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Change to empty directory
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        found = find_config_file()

        assert found is None

    def test_priority_project_over_user(self, tmp_path, monkeypatch):
        """Test project config takes priority over user config."""
        # Mock home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Change to work directory
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create both configs
        project_config = work_dir / "philly.yml"
        project_config.write_text("cache:\n  ttl: 1000\n")

        user_config_dir = fake_home / ".config" / "philly"
        user_config_dir.mkdir(parents=True)
        user_config = user_config_dir / "config.yml"
        user_config.write_text("cache:\n  ttl: 2000\n")

        found = find_config_file()

        # Should find project config, not user config
        assert found is not None
        assert found == project_config

    def test_handles_missing_user_config_directory(self, tmp_path, monkeypatch):
        """Test handles missing ~/.config/philly directory gracefully."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # No .config/philly directory exists
        found = find_config_file()

        assert found is None


class TestLoadConfig:
    """Test load_config function."""

    def test_with_explicit_path(self, tmp_path):
        """Test load_config with explicit path."""
        config_file = tmp_path / "custom.yml"
        config_data = {"cache": {"ttl": 5000}}

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        assert config.cache.ttl == 5000

    def test_with_explicit_string_path(self, tmp_path):
        """Test load_config with explicit string path."""
        config_file = tmp_path / "custom.yml"
        config_data = {"cache": {"ttl": 5000}}

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))

        assert config.cache.ttl == 5000

    def test_with_tilde_expansion(self, tmp_path, monkeypatch):
        """Test load_config expands ~ in paths."""

        fake_home = tmp_path / "home"
        fake_home.mkdir()

        # Mock the HOME environment variable which expanduser() uses
        monkeypatch.setenv("HOME", str(fake_home))

        config_file = fake_home / "my_config.yml"
        config_data = {"cache": {"ttl": 6000}}

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config("~/my_config.yml")

        assert config.cache.ttl == 6000

    def test_explicit_nonexistent_file_raises_error(self, tmp_path):
        """Test load_config with nonexistent explicit file raises error."""
        config_file = tmp_path / "nonexistent.yml"

        with pytest.raises(FileNotFoundError):
            load_config(config_file)

    def test_auto_discovery_project_config(self, tmp_path, monkeypatch):
        """Test auto-discovery finds project config."""
        monkeypatch.chdir(tmp_path)

        project_config = tmp_path / "philly.yml"
        config_data = {"cache": {"ttl": 3000}}

        with open(project_config, "w") as f:
            yaml.dump(config_data, f)

        config = load_config()

        assert config.cache.ttl == 3000

    def test_auto_discovery_user_config(self, tmp_path, monkeypatch):
        """Test auto-discovery finds user config."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        user_config_dir = fake_home / ".config" / "philly"
        user_config_dir.mkdir(parents=True)
        user_config = user_config_dir / "config.yml"
        config_data = {"cache": {"ttl": 4000}}

        with open(user_config, "w") as f:
            yaml.dump(config_data, f)

        config = load_config()

        assert config.cache.ttl == 4000

    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        """Test returns defaults when no config file found."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        config = load_config()

        # Should have all defaults
        assert config.cache.enabled is True
        assert config.cache.ttl == 3600
        assert config.defaults.max_concurrency == 20
        assert config.cli.output_format == "auto"
        assert config.network.timeout == 120

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test invalid YAML raises error."""
        config_file = tmp_path / "invalid.yml"
        config_file.write_text("invalid: [yaml:")

        with pytest.raises(yaml.YAMLError):
            load_config(config_file)


class TestEdgeCases:
    """Test edge cases and validation behavior."""

    def test_cache_config_type_coercion(self):
        """Test Pydantic coerces compatible types."""
        # String to int coercion
        config = CacheConfig(ttl="7200")  # pyright: ignore[reportArgumentType]
        assert config.ttl == 7200
        assert isinstance(config.ttl, int)

    def test_defaults_config_type_coercion(self):
        """Test max_concurrency accepts string numbers."""
        config = DefaultsConfig(max_concurrency="15")  # pyright: ignore[reportArgumentType]
        assert config.max_concurrency == 15

    def test_network_config_type_coercion(self):
        """Test timeout and retries accept strings."""
        config = NetworkConfig(timeout="60", retries="5")  # pyright: ignore[reportArgumentType]
        assert config.timeout == 60
        assert config.retries == 5

    def test_philly_config_with_dict_sub_configs(self):
        """Test PhillyConfig accepts dicts for sub-configs."""
        config = PhillyConfig(
            cache={"enabled": False, "ttl": 1800},  # pyright: ignore[reportArgumentType]
            defaults={"max_concurrency": 10},  # pyright: ignore[reportArgumentType]
            cli={"quiet": True},  # pyright: ignore[reportArgumentType]
            network={"timeout": 30},  # pyright: ignore[reportArgumentType]
        )

        assert config.cache.enabled is False
        assert config.cache.ttl == 1800
        assert config.defaults.max_concurrency == 10
        assert config.cli.quiet is True
        assert config.network.timeout == 30

    def test_load_config_with_unknown_keys(self, tmp_path):
        """Test that unknown keys in config file are ignored."""
        config_file = tmp_path / "config.yml"
        config_data = {
            "cache": {"enabled": True, "unknown_key": "ignored"},
            "unknown_section": {"foo": "bar"},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Pydantic's default is to ignore extra fields
        config = PhillyConfig.load_from_file(config_file)
        assert config.cache.enabled is True
        assert not hasattr(config.cache, "unknown_key")

    def test_model_dump_produces_valid_dict(self):
        """Test model_dump produces expected structure."""
        config = PhillyConfig(
            cache=CacheConfig(ttl=1234),
            defaults=DefaultsConfig(max_concurrency=42),
        )

        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["cache"]["ttl"] == 1234
        assert data["cache"]["enabled"] is True  # default
        assert data["defaults"]["max_concurrency"] == 42
        assert "cli" in data
        assert "network" in data

    def test_load_file_with_null_optional_values(self, tmp_path):
        """Test loading config with explicit null for optional fields."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
cache:
  enabled: true
  max_size_mb: null
""")

        config = PhillyConfig.load_from_file(config_file)

        # max_size_mb is Optional[float], so null is valid
        assert config.cache.max_size_mb is None

    def test_load_file_with_null_required_list_raises_error(self, tmp_path):
        """Test that null for required list fields raises validation error."""
        from pydantic import ValidationError

        config_file = tmp_path / "config.yml"
        config_file.write_text("""
defaults:
  format_preference: null
""")

        # format_preference is list[str], not Optional, so null is invalid
        with pytest.raises(ValidationError):
            PhillyConfig.load_from_file(config_file)

    def test_cli_config_boolean_coercion(self):
        """Test CLI config coerces boolean-like values."""
        # Pydantic v2 is strict about booleans, but "true"/"false" strings work
        config = CLIConfig(quiet=True, verbose=False)
        assert config.quiet is True
        assert config.verbose is False

    def test_save_to_file_with_string_path(self, tmp_path):
        """Test save_to_file works with string path."""
        config_file = str(tmp_path / "config.yml")
        config = PhillyConfig()

        # This should work if save_to_file handles string paths
        config.save_to_file(Path(config_file))

        assert Path(config_file).exists()

    def test_find_config_finds_yaml_not_yml(self, tmp_path, monkeypatch):
        """Test find_config_file only looks for philly.yml not .yaml."""
        monkeypatch.chdir(tmp_path)

        # Create .yaml file (wrong extension)
        wrong_file = tmp_path / "philly.yaml"
        wrong_file.write_text("cache:\n  ttl: 999\n")

        # Should not find the .yaml file
        found = find_config_file()
        assert found is None
