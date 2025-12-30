"""Command-line interface for Philly library with proper formatting and progress tracking."""

import asyncio
import sys
from pathlib import Path

import yaml
from fire import Fire

from philly.philly import Philly
from phl.cli import OutputFormatter, ProgressTracker


class ConfigCommands:
    """Configuration management commands."""

    def show(self) -> None:
        """Show current configuration.

        Examples:
            phl config show
        """
        from philly.config import find_config_file, load_config

        config = load_config()
        config_path = find_config_file()

        print(
            f"# Config file: {config_path or 'Using defaults (no config file found)'}"
        )
        print(yaml.dump(config.model_dump(), default_flow_style=False))

    def path(self) -> None:
        """Show config file path.

        Examples:
            phl config path
        """
        from philly.config import find_config_file

        path = find_config_file()
        if path:
            print(path)
        else:
            print("No config file found. Searched:")
            print("  - ./philly.yml")
            print("  - ~/.config/philly/config.yml")

    def init(self, path: str = "./philly.yml") -> None:
        """Create a new config file with defaults.

        Args:
            path: Path where config file should be created (default: ./philly.yml)

        Examples:
            phl config init
            phl config init --path ~/.config/philly/config.yml
        """
        from philly.config import PhillyConfig

        config_path = Path(path)
        if config_path.exists():
            print(f"Config file already exists: {config_path}")
            return

        config = PhillyConfig()
        config.save_to_file(config_path)
        print(f"Created config file: {config_path}")

    def get(self, key: str) -> None:
        """Get a config value (e.g., cache.ttl, defaults.format_preference).

        Args:
            key: Dot-separated path to config value (e.g., "cache.ttl")

        Examples:
            phl config get cache.ttl
            phl config get defaults.format_preference
            phl config get cache.enabled
        """
        from philly.config import load_config

        config = load_config()
        parts = key.split(".")

        value = config
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                print(f"Unknown config key: {key}")
                return

        print(value)


class PhillyCLI:
    """Command-line interface for Philly library.

    Provides formatted output and progress tracking for all Philly operations.

    Examples:
        phl datasets
        phl search "crime" --fuzzy
        phl sample "Crime Incidents" --n 20 --output-format json
        phl load "Crime Incidents" --format csv --limit 100
    """

    def __init__(
        self,
        format: str = "auto",
        quiet: bool = False,
        verbose: bool = False,
        cache: bool = True,
        cache_dir: str = "~/.cache/philly",
        cache_ttl: int = 3600,
        cache_max_size_mb: float | None = None,
        config_file: str | None = None,
    ):
        """Initialize the CLI wrapper.

        Args:
            format: Output format (auto, json, jsonl, csv, tsv, table)
            quiet: Suppress all non-error output
            verbose: Show verbose progress messages
            cache: Enable caching
            cache_dir: Cache directory path
            cache_ttl: Cache TTL in seconds
            cache_max_size_mb: Maximum cache size in MB
            config_file: Path to config file
        """
        self._philly = Philly(
            cache=cache,
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
            cache_max_size_mb=cache_max_size_mb,
            config_file=config_file,
        )
        self._formatter = OutputFormatter(format=format)
        self._progress = ProgressTracker(show_progress=not quiet, quiet=quiet)
        self._verbose = verbose
        self.config = ConfigCommands()

    def datasets(self) -> None:
        """List all available datasets.

        Examples:
            phl datasets
            phl datasets --format json
        """
        self._progress.progress("Loading datasets...")
        data = self._philly.list_datasets()
        self._progress.success(f"Found {len(data)} datasets")
        print(self._formatter.format_output(data))

    def resources(self, dataset: str, names_only: bool = False) -> None:
        """List resources for a dataset.

        Args:
            dataset: Name of the dataset
            names_only: Only show resource names (not full details)

        Examples:
            phl resources "Crime Incidents"
            phl resources "Crime Incidents" --names-only
        """
        self._progress.progress(f"Loading resources for '{dataset}'...")
        data = self._philly.list_resources(dataset, names_only=names_only)
        self._progress.success(f"Found {len(data)} resources")
        print(self._formatter.format_output(data))

    def search(
        self,
        query: str,
        category: str | None = None,
        fuzzy: bool = False,
        limit: int | None = None,
    ) -> None:
        """Search for datasets.

        Args:
            query: Search query string
            category: Filter by category
            fuzzy: Use fuzzy matching
            limit: Maximum number of results

        Examples:
            phl search "crime"
            phl search "crime" --fuzzy
            phl search "crime" --category "Public Safety" --limit 5
        """
        self._progress.progress(f"Searching for '{query}'...")
        data = self._philly.search(query, category=category, fuzzy=fuzzy, limit=limit)
        self._progress.success(f"Found {len(data)} results")
        print(self._formatter.format_output(data))

    def categories(self) -> None:
        """List all dataset categories.

        Examples:
            phl categories
        """
        self._progress.progress("Loading categories...")
        data = self._philly.list_categories()
        self._progress.success(f"Found {len(data)} categories")
        print(self._formatter.format_output(data))

    def info(self, dataset: str, resource: str | None = None) -> None:
        """Get detailed information about a dataset or resource.

        Args:
            dataset: Name of the dataset
            resource: Optional resource name

        Examples:
            phl info "Crime Incidents"
            phl info "Crime Incidents" --resource "crime_incidents_csv"
        """
        self._progress.progress(f"Loading info for '{dataset}'...")
        data = self._philly.info(dataset, resource)
        self._progress.success("Info loaded")
        print(self._formatter.format_output(data))

    def formats(self, dataset: str) -> None:
        """List available formats for a dataset.

        Args:
            dataset: Name of the dataset

        Examples:
            phl formats "Crime Incidents"
        """
        self._progress.progress(f"Loading formats for '{dataset}'...")
        data = self._philly.get_formats(dataset)
        self._progress.success(f"Found {len(data)} formats")
        print(self._formatter.format_output(data))

    def load(
        self,
        dataset: str,
        resource: str | None = None,
        format: str | None = None,
        where: str | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        output_format: str | None = None,
    ) -> None:
        """Load a dataset resource.

        Args:
            dataset: Name of the dataset
            resource: Optional resource name (auto-selected if not provided)
            format: Resource format to load (csv, json, etc.)
            where: SQL WHERE clause for server-side filtering
            columns: List of columns to select
            limit: Maximum number of rows to return
            offset: Number of rows to skip
            output_format: Override CLI output format for this command

        Examples:
            phl load "Crime Incidents"
            phl load "Crime Incidents" --format csv --limit 100
            phl load "Crime Incidents" --where "dispatch_date >= '2024-01-01'" --limit 50
            phl load "Crime Incidents" --output-format json
        """
        if output_format:
            self._formatter = OutputFormatter(format=output_format)

        self._progress.progress(f"Loading '{dataset}'...")

        async def _load():
            return await self._philly.load(
                dataset,
                resource,
                format=format,
                where=where,
                columns=columns,
                limit=limit,
                offset=offset,
            )

        try:
            data = asyncio.run(_load())
            self._progress.success("Data loaded successfully")
            print(self._formatter.format_output(data))
        except Exception as e:
            self._progress.error(str(e))
            sys.exit(1)

    def sample(
        self,
        dataset: str,
        resource: str | None = None,
        n: int = 10,
        output_format: str | None = None,
    ) -> None:
        """Preview data without downloading the entire file.

        Args:
            dataset: Name of the dataset
            resource: Optional resource name (auto-selected if not provided)
            n: Number of rows to sample
            output_format: Override CLI output format for this command

        Examples:
            phl sample "Crime Incidents"
            phl sample "Crime Incidents" --n 20
            phl sample "Crime Incidents" --n 50 --output-format json
        """
        if output_format:
            self._formatter = OutputFormatter(format=output_format)

        self._progress.progress(f"Sampling {n} rows from '{dataset}'...")

        async def _sample():
            return await self._philly.sample(dataset, resource, n=n)

        try:
            data = asyncio.run(_sample())
            self._progress.success(f"Sampled {n} rows")
            print(self._formatter.format_output(data))
        except Exception as e:
            self._progress.error(str(e))
            sys.exit(1)

    def columns(self, dataset: str, resource: str | None = None) -> None:
        """Get column names for a dataset.

        Args:
            dataset: Name of the dataset
            resource: Optional resource name (auto-selected if not provided)

        Examples:
            phl columns "Crime Incidents"
        """
        self._progress.progress(f"Loading columns for '{dataset}'...")

        async def _columns():
            return await self._philly.get_filterable_columns(dataset, resource)

        try:
            data = asyncio.run(_columns())
            self._progress.success(f"Found {len(data)} columns")
            print(self._formatter.format_output(data))
        except Exception as e:
            self._progress.error(str(e))
            sys.exit(1)

    def schema(self, dataset: str, resource: str | None = None) -> None:
        """Get detailed schema for a dataset.

        Args:
            dataset: Name of the dataset
            resource: Optional resource name (auto-selected if not provided)

        Examples:
            phl schema "Crime Incidents"
        """
        self._progress.progress(f"Loading schema for '{dataset}'...")

        async def _schema():
            return await self._philly.get_filter_schema(dataset, resource)

        try:
            data = asyncio.run(_schema())
            self._progress.success(f"Schema loaded with {len(data)} columns")
            print(self._formatter.format_output(data))
        except Exception as e:
            self._progress.error(str(e))
            sys.exit(1)

    def filter_examples(self, dataset: str, resource: str | None = None) -> None:
        """Get example WHERE clauses for a dataset.

        Args:
            dataset: Name of the dataset
            resource: Optional resource name (auto-selected if not provided)

        Examples:
            phl filter-examples "Crime Incidents"
        """
        self._progress.progress(f"Generating filter examples for '{dataset}'...")

        async def _examples():
            return await self._philly.get_filter_examples(dataset, resource)

        try:
            data = asyncio.run(_examples())
            self._progress.success(f"Generated {len(data)} examples")
            print(self._formatter.format_output(data))
        except Exception as e:
            self._progress.error(str(e))
            sys.exit(1)

    def cache_info(
        self, dataset: str | None = None, resource: str | None = None
    ) -> None:
        """Show cache statistics.

        Args:
            dataset: Optional dataset name to filter by
            resource: Optional resource name to filter by

        Examples:
            phl cache-info
            phl cache-info --dataset "Crime Incidents"
        """
        self._progress.progress("Loading cache info...")
        try:
            data = self._philly.cache_info(dataset_name=dataset, resource_name=resource)
            self._progress.success("Cache info loaded")
            print(self._formatter.format_output(data))
        except Exception as e:
            self._progress.error(str(e))
            sys.exit(1)

    def cache_clear(self, dataset: str | None = None) -> None:
        """Clear cache.

        Args:
            dataset: Optional dataset name (clears all if not provided)

        Examples:
            phl cache-clear
            phl cache-clear --dataset "Crime Incidents"
        """
        self._progress.progress("Clearing cache...")
        try:
            self._philly.cache_clear(dataset)
            self._progress.success("Cache cleared")
        except Exception as e:
            self._progress.error(str(e))
            sys.exit(1)

    def check_updates(
        self,
        datasets: list[str] | None = None,
        only_outdated: bool = False,
    ) -> None:
        """Check for updates on cached resources.

        Args:
            datasets: Optional list of dataset names to check
            only_outdated: Only show resources with available updates

        Examples:
            phl check-updates
            phl check-updates --only-outdated
        """
        self._progress.progress("Checking for updates...")
        try:
            data = self._philly.check_updates(
                dataset_names=datasets, only_outdated=only_outdated
            )
            count = len(data)
            if only_outdated:
                self._progress.warning(f"Found {count} outdated resources")
            else:
                self._progress.success(f"Checked {count} resources")
            print(self._formatter.format_output(data))
        except Exception as e:
            self._progress.error(str(e))
            sys.exit(1)


def main() -> None:
    """Entry point for the phl command."""
    Fire(PhillyCLI)


if __name__ == "__main__":
    main()
