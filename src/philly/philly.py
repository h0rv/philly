import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from philly.cache import CacheEntry, FileCache
from philly.config import PhillyConfig, load_config
from philly.filtering import (
    BackendType,
    build_arcgis_query,
    build_carto_query,
    detect_backend,
)
from philly.format_selection import (
    find_resource_by_format,
    get_formats as _get_formats,
    has_format as _has_format,
)
from philly.loaders import load, use_http_client
from philly.metadata import (
    get_dataset_info,
    get_remote_last_modified,
    get_remote_size,
    get_resource_info,
    is_url_available,
)
from philly.models import Dataset, Resource
from philly.sample import (
    format_chunk,
    get_columns_from_sample,
    infer_schema_from_sample,
    sample_csv,
    sample_geojson,
    sample_json,
)
from philly.filters import (
    get_filterable_columns as _get_filterable_columns,
    get_filter_schema as _get_filter_schema,
    get_filter_examples as _get_filter_examples,
    validate_filter as _validate_filter,
)
from philly.search import (
    SearchIndex,
    build_search_index,
    get_by_category as _get_by_category,
    list_categories as _list_categories,
    search as _search,
    search_resources as _search_resources,
)
from philly.streaming import (
    paginated_arcgis_stream,
    paginated_carto_stream,
    stream_csv,
    stream_json_array,
)
from philly.updates import (
    check_single_update,
    check_updates_batch,
    get_outdated_entries,
)


class Philly:
    def __init__(
        self,
        cache: bool = True,
        cache_dir: str = "~/.cache/philly",
        cache_ttl: int = 3600,
        cache_max_size_mb: float | None = None,
        config_file: str | Path | None = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)

        # Load configuration
        self.config: PhillyConfig = load_config(config_file)

        # Apply config to override defaults
        if config_file is not None:
            # If config_file is explicitly provided, use its cache settings
            cache = self.config.cache.enabled
            cache_dir = self.config.cache.directory
            cache_ttl = self.config.cache.ttl
            cache_max_size_mb = self.config.cache.max_size_mb

        self._module_dir: Path = Path(__file__).parent.resolve()
        self._datasets_dir: Path = self._module_dir / "datasets"

        self.datasets: list[Dataset] = sorted(
            [
                Dataset.from_file(str(self._datasets_dir / file))
                for file in self._datasets_dir.glob("*.yaml")
            ],
            key=lambda x: x.title,
        )

        self._datasets_map: dict[str, Dataset] = {
            dataset.title: dataset for dataset in self.datasets
        }

        self.cache_enabled = cache
        self._cache: FileCache | None = None
        if self.cache_enabled:
            self._cache = FileCache(
                cache_dir=cache_dir,
                ttl=cache_ttl,
                max_size_mb=cache_max_size_mb,
            )

        # Search index (built lazily on first search)
        self._search_index: SearchIndex | None = None

    def _get_dataset(self, dataset_name: str) -> Dataset:
        dataset = self._datasets_map.get(dataset_name)

        if not dataset:
            raise ValueError(f"dataset '{dataset_name}' does not exist")

        return dataset

    def _generate_cache_key(
        self,
        dataset_name: str,
        resource_name: str,
        format: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate unique cache key for a resource load"""
        key_data = {
            "dataset": dataset_name,
            "resource": resource_name,
            "format": format,
            "params": kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def list_datasets(self) -> list[str]:
        return [d.title for d in self.datasets]

    def list_resources(
        self,
        dataset_name: str,
        names_only: bool = False,
    ) -> list[str] | list[dict[str, Any]]:
        """List all resources for a dataset.

        Args:
            dataset_name: Name of the dataset
            names_only: If True, return just resource names.
                       If False, return detailed info dicts.

        Returns:
            List of resource names (if names_only=True) or
            List of dicts with name, format, url (if names_only=False)

        Examples:
            >>> philly = Philly()
            >>> names = philly.list_resources("Crime Incidents", names_only=True)
            >>> details = philly.list_resources("Crime Incidents")
        """
        dataset = self._get_dataset(dataset_name)
        resources = dataset.resources or []

        if names_only:
            return [r.name for r in resources]

        return [
            {"name": r.name, "format": str(r.format), "url": r.url} for r in resources
        ]

    def list_all_resources(self) -> list[dict[str, str]]:
        """List all resources across all datasets.

        Returns:
            List of dicts with name, format, dataset keys
        """
        return [
            {"name": r.name, "format": str(r.format), "dataset": d.title}
            for d in self.datasets
            for r in (d.resources or [])
        ]

    async def load(
        self,
        dataset_name: str,
        resource_name: str | None = None,
        format: str | None = None,
        ignore_load_errors: bool = False,
        use_cache: bool = True,
        where: str | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        **kwargs: Any,
    ) -> object | None:
        dataset = self._get_dataset(dataset_name)

        # Auto-select resource by format if resource_name not provided
        if resource_name is None:
            if format is None:
                # Use format preference from config
                format_preference = self.config.defaults.format_preference
                for preferred_format in format_preference:
                    resource = find_resource_by_format(dataset, preferred_format)
                    if resource:
                        break
                else:
                    raise ValueError(
                        f"No resource found for dataset '{dataset_name}' with preferred formats"
                    )
            else:
                resource = find_resource_by_format(dataset, format)
                if not resource:
                    raise ValueError(
                        f"No resource found for dataset '{dataset_name}' with format '{format}'"
                    )
        else:
            resource = dataset.get_resource(resource_name, format=format)

        if not resource.url:
            return None

        url = resource.url

        # Apply server-side filtering if supported
        if where or columns or limit is not None or offset is not None:
            backend = detect_backend(url)
            if backend == BackendType.CARTO:
                url = build_carto_query(
                    url, where=where, columns=columns, limit=limit, offset=offset
                )
            elif backend == BackendType.ARCGIS:
                url = build_arcgis_query(
                    url, where=where, columns=columns, limit=limit, offset=offset
                )
            elif backend in (BackendType.STATIC, BackendType.UNKNOWN):
                # Static files and unknown backends don't support server-side filtering
                self._logger.warning(
                    f"Resource '{resource.name}' is a static file or unsupported backend. "
                    f"Server-side filtering (where, columns, limit, offset) is not supported. "
                    f"The full dataset will be loaded."
                )

        # Generate cache key including filter params
        filter_params = {
            "where": where,
            "columns": columns,
            "limit": limit,
            "offset": offset,
        }
        cache_key = self._generate_cache_key(
            dataset_name,
            resource.name,
            format=format,
            **filter_params,
            **kwargs,
        )

        # Check cache if enabled and not force refresh
        if use_cache and self.cache_enabled and self._cache:
            cached_data = self._cache.get(cache_key)
            if cached_data is not None:
                return cached_data

        # Cache miss or disabled - load from network
        # Create a modified resource with the potentially modified URL
        if url != resource.url:
            from copy import copy

            resource = copy(resource)
            resource.url = url

        data: object | None = None
        try:
            data = await load(resource, ignore_errors=ignore_load_errors)
        except Exception as e:
            if not ignore_load_errors:
                raise e
            self._logger.warning(
                f"Resource {resource.name} could not be loaded (error: {e}). Skipping."
            )
            return None

        if data is None:
            return None

        # Store in cache if enabled
        if self.cache_enabled and self._cache:
            current_time = time.time()
            cache_entry = CacheEntry(
                cache_key=cache_key,
                dataset_name=dataset_name,
                resource_name=resource.name,
                format=str(resource.format),
                url=url,
                cached_at=current_time,
                last_accessed=current_time,
                ttl=self._cache.ttl,
                size_bytes=0,
                query_params={**filter_params, **kwargs}
                if filter_params or kwargs
                else None,
            )
            self._cache.set(cache_key, data, cache_entry)

        return data

    async def load_all(
        self,
        show_progress: bool = False,
        ignore_load_errors: bool = True,
        max_concurrency: int | None = 20,
    ) -> list[object | None]:
        resources = [
            resource
            for dataset in self.datasets
            for resource in (dataset.resources or [])
        ]
        if not resources:
            return []

        concurrency = (
            max_concurrency
            if max_concurrency and max_concurrency > 0
            else len(resources)
        )
        results: list[object | None] = [None] * len(resources)
        semaphore = asyncio.Semaphore(concurrency)

        progress = tqdm(total=len(resources)) if show_progress else None

        async def _load_one(index: int, resource: Resource) -> None:
            async with semaphore:
                if ignore_load_errors:
                    try:
                        results[index] = await load(resource, ignore_errors=True)
                    except Exception as e:
                        self._logger.warning(
                            f"Resource load failed for {resource}: {e}. Skipping."
                        )
                        results[index] = None
                else:
                    results[index] = await load(resource, ignore_errors=False)
            if progress is not None:
                progress.update(1)

        async with httpx.AsyncClient() as client:
            async with use_http_client(client):
                async with asyncio.TaskGroup() as tg:
                    for index, resource in enumerate(resources):
                        tg.create_task(_load_one(index, resource))

        if progress is not None:
            progress.close()

        return results

    def cache_clear(self, dataset_name: str | None = None) -> None:
        """Clear cache entries. If dataset_name is provided, only clear that dataset."""
        if not self._cache:
            raise RuntimeError("Cache is not enabled")
        self._cache.clear(pattern=dataset_name)

    def cache_info(
        self, dataset_name: str | None = None, resource_name: str | None = None
    ) -> dict[str, Any]:
        """Get cache statistics. If dataset_name and resource_name provided, get info for specific resource."""
        if not self._cache:
            raise RuntimeError("Cache is not enabled")
        return self._cache.info(dataset_name=dataset_name, resource_name=resource_name)

    # ============================================================================
    # Search Methods
    # ============================================================================

    def _ensure_search_index(self) -> SearchIndex:
        """Build search index lazily on first use."""
        if self._search_index is None:
            self._search_index = build_search_index(self.datasets)
        return self._search_index

    def search(
        self,
        query: str,
        category: str | None = None,
        fuzzy: bool = False,
        limit: int | None = None,
    ) -> list[str]:
        """Search for datasets by query.

        Args:
            query: Search query string
            category: Optional category filter
            fuzzy: Whether to use fuzzy matching (default: False)
            limit: Maximum number of results to return

        Returns:
            List of dataset titles sorted by relevance

        Examples:
            >>> philly = Philly()
            >>> results = philly.search("crime")
            >>> results = philly.search("crime", category="Public Safety", limit=5)
        """
        index = self._ensure_search_index()
        return _search(index, query, category=category, fuzzy=fuzzy, limit=limit)

    def search_resources(
        self,
        query: str,
        dataset_name: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Search for resources by name.

        Args:
            query: Search query string for resource name
            dataset_name: Optional dataset title filter
            limit: Maximum number of results to return

        Returns:
            List of strings in format "resource_name [dataset_title]"

        Examples:
            >>> philly = Philly()
            >>> results = philly.search_resources("csv")
            >>> results = philly.search_resources("2020", dataset_name="Crime Incidents")
        """
        index = self._ensure_search_index()
        return _search_resources(index, query, dataset_filter=dataset_name, limit=limit)

    def list_categories(self) -> list[str]:
        """Get a sorted list of all categories.

        Returns:
            Sorted list of category names

        Examples:
            >>> philly = Philly()
            >>> categories = philly.list_categories()
        """
        index = self._ensure_search_index()
        return _list_categories(index)

    def get_by_category(self, category: str) -> list[str]:
        """Get all datasets in a specific category.

        Args:
            category: Category name to filter by

        Returns:
            List of dataset titles in the category (sorted)

        Examples:
            >>> philly = Philly()
            >>> datasets = philly.get_by_category("Public Safety")
        """
        index = self._ensure_search_index()
        return _get_by_category(index, category)

    # ============================================================================
    # Metadata/Info Methods
    # ============================================================================

    def info(
        self,
        dataset_name: str,
        resource_name: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed information about a dataset or resource.

        Args:
            dataset_name: Name of the dataset
            resource_name: Optional resource name (if None, returns dataset info)

        Returns:
            Dictionary with metadata

        Examples:
            >>> philly = Philly()
            >>> info = philly.info("Crime Incidents")
            >>> info = philly.info("Crime Incidents", "crime_incidents_csv")
        """
        dataset = self._get_dataset(dataset_name)

        if resource_name is None:
            return get_dataset_info(dataset)
        else:
            resource = dataset.get_resource(resource_name)
            return get_resource_info(dataset, resource)

    def get_size(
        self,
        dataset_name: str,
        resource_name: str,
    ) -> float | None:
        """Get the size of a resource in MB.

        Args:
            dataset_name: Name of the dataset
            resource_name: Name of the resource

        Returns:
            Size in MB, or None if unable to determine

        Examples:
            >>> philly = Philly()
            >>> size_mb = philly.get_size("Crime Incidents", "crime_incidents_csv")
        """
        dataset = self._get_dataset(dataset_name)
        resource = dataset.get_resource(resource_name)

        if not resource.url:
            return None

        return get_remote_size(resource.url)

    def get_last_modified(
        self,
        dataset_name: str,
        resource_name: str,
    ) -> datetime | None:
        """Get the last modified date of a resource.

        Args:
            dataset_name: Name of the dataset
            resource_name: Name of the resource

        Returns:
            Last modified datetime, or None if unable to determine

        Examples:
            >>> philly = Philly()
            >>> last_mod = philly.get_last_modified("Crime Incidents", "crime_incidents_csv")
        """
        dataset = self._get_dataset(dataset_name)
        resource = dataset.get_resource(resource_name)

        if not resource.url:
            return None

        return get_remote_last_modified(resource.url)

    def is_available(
        self,
        dataset_name: str,
        resource_name: str,
    ) -> bool:
        """Check if a resource is available (returns 200 OK).

        Args:
            dataset_name: Name of the dataset
            resource_name: Name of the resource

        Returns:
            True if URL returns 200, False otherwise

        Examples:
            >>> philly = Philly()
            >>> available = philly.is_available("Crime Incidents", "crime_incidents_csv")
        """
        dataset = self._get_dataset(dataset_name)
        resource = dataset.get_resource(resource_name)

        if not resource.url:
            return False

        return is_url_available(resource.url)

    def get_url(
        self,
        dataset_name: str,
        resource_name: str,
    ) -> str:
        """Get the URL of a resource.

        Args:
            dataset_name: Name of the dataset
            resource_name: Name of the resource

        Returns:
            URL string

        Examples:
            >>> philly = Philly()
            >>> url = philly.get_url("Crime Incidents", "crime_incidents_csv")
        """
        dataset = self._get_dataset(dataset_name)
        resource = dataset.get_resource(resource_name)
        return resource.url or ""

    # ============================================================================
    # Format Methods
    # ============================================================================

    def get_formats(self, dataset_name: str) -> list[str]:
        """Get all unique formats available in a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Sorted list of unique format strings

        Examples:
            >>> philly = Philly()
            >>> formats = philly.get_formats("Crime Incidents")
        """
        dataset = self._get_dataset(dataset_name)
        return _get_formats(dataset)

    def has_format(self, dataset_name: str, format: str) -> bool:
        """Check if a dataset has a resource with the specified format.

        Args:
            dataset_name: Name of the dataset
            format: Format to look for (case-insensitive)

        Returns:
            True if format exists in dataset, False otherwise

        Examples:
            >>> philly = Philly()
            >>> has_csv = philly.has_format("Crime Incidents", "csv")
        """
        dataset = self._get_dataset(dataset_name)
        return _has_format(dataset, format)

    # ============================================================================
    # Sample/Preview Methods
    # ============================================================================

    async def sample(
        self,
        dataset_name: str,
        resource_name: str | None = None,
        n: int = 10,
        output_format: str = "records",
    ) -> Any:
        """Preview data without downloading the entire file.

        Args:
            dataset_name: Name of the dataset
            resource_name: Optional resource name (uses format auto-selection if None)
            n: Number of rows/records to sample
            output_format: Output format ("records" or "dataframe")

        Returns:
            Sample data in requested format

        Examples:
            >>> philly = Philly()
            >>> sample = await philly.sample("Crime Incidents", n=10)
            >>> df = await philly.sample("Crime Incidents", n=10, output_format="dataframe")
        """
        dataset = self._get_dataset(dataset_name)

        if resource_name is None:
            # Auto-select resource
            format_preference = self.config.defaults.format_preference
            for preferred_format in format_preference:
                resource = find_resource_by_format(dataset, preferred_format)
                if resource:
                    break
            else:
                raise ValueError(f"No resource found for dataset '{dataset_name}'")
        else:
            resource = dataset.get_resource(resource_name)

        if not resource.url:
            return None

        # Sample based on format
        resource_format = str(resource.format).lower()

        if resource_format == "csv" or resource.url.endswith(".csv"):
            data = await sample_csv(resource.url, n)
        elif resource_format == "geojson" or resource.url.endswith(".geojson"):
            data = await sample_geojson(resource.url, n)
        elif resource_format == "json" or resource.url.endswith(".json"):
            data = await sample_json(resource.url, n)
        else:
            # Try JSON as fallback
            try:
                data = await sample_json(resource.url, n)
            except Exception:
                # Try CSV as fallback
                try:
                    data = await sample_csv(resource.url, n)
                except Exception as e:
                    raise ValueError(
                        f"Unable to sample format '{resource_format}': {e}"
                    ) from e

        return format_chunk(data, output_format)

    async def get_columns(
        self,
        dataset_name: str,
        resource_name: str | None = None,
    ) -> list[str]:
        """Get column names from a dataset.

        Args:
            dataset_name: Name of the dataset
            resource_name: Optional resource name (uses format auto-selection if None)

        Returns:
            List of column names

        Examples:
            >>> philly = Philly()
            >>> columns = await philly.get_columns("Crime Incidents")
        """
        sample_data = await self.sample(
            dataset_name, resource_name, n=1, output_format="records"
        )
        return get_columns_from_sample(sample_data)

    async def get_schema(
        self,
        dataset_name: str,
        resource_name: str | None = None,
    ) -> dict[str, str]:
        """Infer schema from a dataset.

        Args:
            dataset_name: Name of the dataset
            resource_name: Optional resource name (uses format auto-selection if None)

        Returns:
            Dictionary mapping column names to type strings

        Examples:
            >>> philly = Philly()
            >>> schema = await philly.get_schema("Crime Incidents")
        """
        sample_data = await self.sample(
            dataset_name, resource_name, n=100, output_format="records"
        )
        return infer_schema_from_sample(sample_data)

    # ============================================================================
    # Filter Discovery Methods
    # ============================================================================

    async def get_filterable_columns(
        self,
        dataset_name: str,
        resource_name: str | None = None,
    ) -> list[str]:
        """Get list of filterable columns from a dataset.

        Samples the dataset to discover available column names that can be
        used in WHERE clauses.

        Args:
            dataset_name: Name of the dataset
            resource_name: Optional resource name (uses format auto-selection if None)

        Returns:
            List of column names

        Examples:
            >>> philly = Philly()
            >>> columns = await philly.get_filterable_columns("Crime Incidents")
            >>> print(columns)
            ['objectid', 'dc_dist', 'dispatch_date', 'dispatch_time', ...]
        """
        sample_data = await self.sample(
            dataset_name, resource_name, n=1, output_format="records"
        )
        return _get_filterable_columns(sample_data)

    async def get_filter_schema(
        self,
        dataset_name: str,
        resource_name: str | None = None,
        sample_size: int = 100,
    ) -> dict[str, dict[str, Any]]:
        """Get detailed schema information for filter construction.

        Analyzes a sample of the dataset to infer column types, nullable fields,
        example values, and other metadata useful for building filters.

        Args:
            dataset_name: Name of the dataset
            resource_name: Optional resource name (uses format auto-selection if None)
            sample_size: Number of rows to sample for schema inference (default: 100)

        Returns:
            Dictionary mapping column names to metadata dictionaries containing:
            - type: pandas dtype as string
            - nullable: whether null values exist
            - example: first non-null value
            - null_count: count of null values
            - unique_count: count of unique values

        Examples:
            >>> philly = Philly()
            >>> schema = await philly.get_filter_schema("Crime Incidents", sample_size=50)
            >>> print(schema['dispatch_date'])
            {'type': 'object', 'nullable': False, 'example': '2024-01-15', ...}
        """
        sample_data = await self.sample(
            dataset_name, resource_name, n=sample_size, output_format="records"
        )
        return _get_filter_schema(sample_data, sample_size=sample_size)

    async def get_filter_examples(
        self,
        dataset_name: str,
        resource_name: str | None = None,
    ) -> list[str]:
        """Get example WHERE clauses for a dataset.

        Generates example filter expressions based on the dataset's schema,
        demonstrating different query patterns for various column types.

        Args:
            dataset_name: Name of the dataset
            resource_name: Optional resource name (uses format auto-selection if None)

        Returns:
            List of example WHERE clause strings (max 10)

        Examples:
            >>> philly = Philly()
            >>> examples = await philly.get_filter_examples("Crime Incidents")
            >>> for example in examples:
            ...     print(example)
            dispatch_date >= '2024-01-01'
            dc_dist = '01'
            text_general_code LIKE '%THEFT%'
        """
        schema = await self.get_filter_schema(dataset_name, resource_name)
        return _get_filter_examples(schema)

    async def validate_filter(
        self,
        dataset_name: str,
        resource_name: str,
        where: str,
    ) -> dict[str, Any]:
        """Validate a WHERE clause against available columns.

        Performs basic validation to check if a filter uses valid column names
        and doesn't contain dangerous SQL keywords.

        Args:
            dataset_name: Name of the dataset
            resource_name: Name of the resource
            where: WHERE clause string to validate

        Returns:
            Dictionary with 'valid' (bool) and 'error' (str or None)
            May also include 'available_columns' if invalid columns found

        Examples:
            >>> philly = Philly()
            >>> result = await philly.validate_filter(
            ...     "Crime Incidents",
            ...     "crime_csv",
            ...     "dispatch_date >= '2024-01-01'"
            ... )
            >>> print(result)
            {'valid': True, 'error': None}

            >>> result = await philly.validate_filter(
            ...     "Crime Incidents",
            ...     "crime_csv",
            ...     "invalid_column = 'test'"
            ... )
            >>> print(result['valid'])
            False
        """
        columns = await self.get_filterable_columns(dataset_name, resource_name)
        return _validate_filter(where, columns)

    # ============================================================================
    # Streaming Methods
    # ============================================================================

    async def stream(
        self,
        dataset_name: str,
        resource_name: str | None = None,
        chunk_size: int = 10000,
        where: str | None = None,
        columns: list[str] | None = None,
        show_progress: bool = False,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Stream data in chunks to minimize memory usage.

        Args:
            dataset_name: Name of the dataset
            resource_name: Optional resource name (uses format auto-selection if None)
            chunk_size: Number of rows to yield per chunk
            where: Optional SQL WHERE clause for server-side filtering
            columns: Optional list of column names for server-side selection
            show_progress: If True, print progress information

        Yields:
            Lists of dictionaries, where each dict represents a row

        Examples:
            >>> philly = Philly()
            >>> async for chunk in philly.stream("Crime Incidents", chunk_size=1000):
            ...     print(f"Processing {len(chunk)} rows")
        """
        dataset = self._get_dataset(dataset_name)

        if resource_name is None:
            # Auto-select resource
            format_preference = self.config.defaults.format_preference
            for preferred_format in format_preference:
                resource = find_resource_by_format(dataset, preferred_format)
                if resource:
                    break
            else:
                raise ValueError(f"No resource found for dataset '{dataset_name}'")
        else:
            resource = dataset.get_resource(resource_name)

        if not resource.url:
            return

        url = resource.url
        backend = detect_backend(url)

        # Use appropriate streaming method based on backend
        if backend == BackendType.CARTO:
            async for chunk in paginated_carto_stream(
                url,
                chunk_size=chunk_size,
                where=where,
                columns=columns,
                show_progress=show_progress,
            ):
                yield chunk
        elif backend == BackendType.ARCGIS:
            async for chunk in paginated_arcgis_stream(
                url,
                chunk_size=chunk_size,
                where=where,
                columns=columns,
                show_progress=show_progress,
            ):
                yield chunk
        else:
            # Static file - use format-based streaming
            resource_format = str(resource.format).lower()

            if resource_format == "csv" or url.endswith(".csv"):
                async for chunk in stream_csv(
                    url, chunk_size=chunk_size, show_progress=show_progress
                ):
                    yield chunk
            elif resource_format in ("json", "geojson") or url.endswith(
                (".json", ".geojson")
            ):
                async for chunk in stream_json_array(
                    url, chunk_size=chunk_size, show_progress=show_progress
                ):
                    yield chunk
            else:
                raise ValueError(
                    f"Streaming not supported for format '{resource_format}'"
                )

    # ============================================================================
    # Update Checking Methods
    # ============================================================================

    def check_update(
        self,
        dataset_name: str,
        resource_name: str,
    ) -> dict[str, Any]:
        """Check if a cached resource has an update available.

        Args:
            dataset_name: Name of the dataset
            resource_name: Name of the resource

        Returns:
            Dictionary with update information

        Examples:
            >>> philly = Philly()
            >>> update_info = philly.check_update("Crime Incidents", "crime_incidents_csv")
        """
        if not self._cache:
            raise RuntimeError("Cache is not enabled")

        # Find cache entry for this resource
        metadata = self._cache.metadata

        for cache_entry in metadata.values():
            if (
                cache_entry.dataset_name == dataset_name
                and cache_entry.resource_name == resource_name
            ):
                return check_single_update(cache_entry, cache_entry.url)

        raise ValueError(
            f"No cached entry found for '{dataset_name}' / '{resource_name}'"
        )

    def check_updates(
        self,
        dataset_names: list[str] | None = None,
        only_outdated: bool = False,
    ) -> list[dict[str, Any]]:
        """Check for updates on cached resources.

        Args:
            dataset_names: Optional list of dataset names to check (checks all if None)
            only_outdated: If True, only return entries with available updates

        Returns:
            List of update information dictionaries

        Examples:
            >>> philly = Philly()
            >>> updates = philly.check_updates()
            >>> outdated = philly.check_updates(only_outdated=True)
        """
        if not self._cache:
            raise RuntimeError("Cache is not enabled")

        metadata = self._cache.metadata
        return check_updates_batch(
            metadata, dataset_filter=dataset_names, only_outdated=only_outdated
        )

    async def refresh_outdated(self) -> list[str]:
        """Refresh all outdated cache entries.

        Returns:
            List of refreshed dataset/resource names

        Examples:
            >>> philly = Philly()
            >>> refreshed = await philly.refresh_outdated()
        """
        if not self._cache:
            raise RuntimeError("Cache is not enabled")

        metadata = self._cache.metadata
        outdated_entries = get_outdated_entries(metadata)

        refreshed: list[str] = []

        for entry in outdated_entries:
            try:
                # Reload the resource
                await self.load(
                    entry.dataset_name,
                    entry.resource_name,
                    use_cache=True,
                )
                refreshed.append(f"{entry.dataset_name} / {entry.resource_name}")
            except Exception as e:
                self._logger.warning(
                    f"Failed to refresh {entry.dataset_name} / {entry.resource_name}: {e}"
                )

        return refreshed
