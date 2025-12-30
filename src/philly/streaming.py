"""Streaming iterators for large datasets.

This module provides async generators that stream data in chunks to minimize memory usage
when working with large datasets. Supports CSV streaming, JSON array chunking, and
paginated API queries for both Carto and ArcGIS backends.
"""

import csv
import sys
from collections.abc import AsyncIterator
from io import StringIO
from typing import Any

import httpx

from philly.filtering import build_arcgis_query, build_carto_query


async def stream_csv(
    url: str,
    chunk_size: int = 10000,
    show_progress: bool = False,
) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream CSV data in chunks without loading the entire file into memory.

    Downloads CSV incrementally and yields chunks of parsed rows. Ideal for large
    CSV files that would otherwise consume too much memory.

    Args:
        url: URL of the CSV file to stream
        chunk_size: Number of rows to yield per chunk
        show_progress: If True, print progress information to stderr

    Yields:
        Lists of dictionaries, where each dict represents a CSV row with column
        names as keys. Each list contains up to chunk_size rows.

    Example:
        >>> async for chunk in stream_csv("https://example.com/data.csv", chunk_size=1000):
        ...     print(f"Processing {len(chunk)} rows")
        ...     for row in chunk:
        ...         print(row["column_name"])
    """
    async with httpx.AsyncClient() as client:
        buffer = []
        bytes_downloaded = 0
        headers = None

        async with client.stream("GET", url) as response:
            response.raise_for_status()

            # Read line by line
            async for line in response.aiter_lines():
                bytes_downloaded += len(line.encode("utf-8"))

                # First line contains headers
                if headers is None:
                    headers = list(csv.reader([line]))[0]
                    if show_progress:
                        print(
                            f"CSV headers: {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}",
                            file=sys.stderr,
                        )
                    continue

                # Parse line as CSV row
                try:
                    # Use csv.reader to handle quoted fields properly
                    parsed_line = list(csv.reader([line]))[0]
                    if len(parsed_line) == len(headers):
                        row = dict(zip(headers, parsed_line, strict=False))
                        buffer.append(row)
                except (csv.Error, IndexError):
                    # Skip malformed lines
                    continue

                # Yield chunk when buffer is full
                if len(buffer) >= chunk_size:
                    if show_progress:
                        print(
                            f"Downloaded {bytes_downloaded:,} bytes, yielding {len(buffer)} rows",
                            file=sys.stderr,
                        )
                    yield buffer
                    buffer = []

        # Yield remaining rows
        if buffer:
            if show_progress:
                print(
                    f"Downloaded {bytes_downloaded:,} bytes total, yielding final {len(buffer)} rows",
                    file=sys.stderr,
                )
            yield buffer


async def stream_json_array(
    url: str,
    chunk_size: int = 10000,
    show_progress: bool = False,
) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream JSON array data in chunks.

    Since JSON cannot be incrementally parsed reliably, this function fetches
    the entire JSON response and then yields it in chunks. Handles common JSON
    structures including plain arrays, GeoJSON, and objects with data arrays.

    Args:
        url: URL of the JSON file to stream
        chunk_size: Number of items to yield per chunk
        show_progress: If True, print progress information to stderr

    Yields:
        Lists of dictionaries. Each list contains up to chunk_size items.

    Example:
        >>> async for chunk in stream_json_array("https://example.com/data.json"):
        ...     print(f"Processing {len(chunk)} items")

    Note:
        This function must download the entire JSON file before yielding chunks,
        as JSON cannot be reliably parsed incrementally. For true streaming,
        use paginated API endpoints with paginated_carto_stream or
        paginated_arcgis_stream instead.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

        data = response.json()

        # Determine the array to iterate over
        items = None

        if isinstance(data, list):
            # Plain JSON array
            items = data
        elif isinstance(data, dict):
            # Try common keys for nested data arrays
            if "features" in data:  # GeoJSON
                items = data["features"]
            elif "data" in data:
                items = data["data"]
            elif "results" in data:
                items = data["results"]
            elif "records" in data:
                items = data["records"]
            elif "items" in data:
                items = data["items"]
            else:
                # If no known key, try to find the first list value
                for value in data.values():
                    if isinstance(value, list):
                        items = value
                        break

        if items is None:
            raise ValueError(
                f"Unable to find array data in JSON structure. Top-level type: {type(data).__name__}"
            )

        if show_progress:
            print(f"Loaded {len(items):,} items from JSON", file=sys.stderr)

        # Yield in chunks
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            if show_progress:
                print(
                    f"Yielding chunk {i // chunk_size + 1}: items {i:,} to {i + len(chunk):,}",
                    file=sys.stderr,
                )
            yield chunk


async def paginated_carto_stream(
    base_url: str,
    chunk_size: int = 10000,
    where: str | None = None,
    columns: list[str] | None = None,
    show_progress: bool = False,
) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream data from Carto API using pagination.

    Uses Carto's LIMIT/OFFSET pagination to fetch data in chunks without
    loading everything into memory. This is the most memory-efficient way
    to work with large Carto datasets.

    Args:
        base_url: Base Carto API URL (must contain 'q' parameter with SQL query)
        chunk_size: Number of rows to fetch per API request
        where: Optional SQL WHERE clause (without the WHERE keyword)
        columns: Optional list of column names to select
        show_progress: If True, print progress information to stderr

    Yields:
        Lists of dictionaries, where each dict represents a row. Each list
        contains up to chunk_size rows.

    Example:
        >>> url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        >>> async for chunk in paginated_carto_stream(url, where="district = '6'", chunk_size=500):
        ...     print(f"Processing {len(chunk)} rows")

    Note:
        The Carto API URL must include both the 'q' parameter (SQL query) and
        'format' parameter. CSV format is recommended for better performance.
    """
    offset = 0
    total_rows = 0

    async with httpx.AsyncClient() as client:
        while True:
            # Build query URL with pagination
            query_url = build_carto_query(
                base_url,
                where=where,
                columns=columns,
                limit=chunk_size,
                offset=offset,
            )

            if show_progress:
                print(
                    f"Fetching rows {offset:,} to {offset + chunk_size:,}...",
                    file=sys.stderr,
                )

            response = await client.get(query_url)
            response.raise_for_status()

            # Parse response based on format
            content_type = response.headers.get("content-type", "").lower()

            if "csv" in content_type or query_url.endswith("format=csv"):
                # Parse CSV response
                text = response.text
                reader = csv.DictReader(StringIO(text))
                chunk = list(reader)
            elif "json" in content_type:
                # Parse JSON response
                data = response.json()
                if isinstance(data, dict) and "rows" in data:
                    chunk = data["rows"]
                elif isinstance(data, list):
                    chunk = data
                else:
                    chunk = []
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

            if not chunk:
                # No more data
                if show_progress:
                    print(
                        f"Completed: fetched {total_rows:,} total rows", file=sys.stderr
                    )
                break

            total_rows += len(chunk)

            if show_progress:
                print(
                    f"Yielding {len(chunk)} rows (total: {total_rows:,})",
                    file=sys.stderr,
                )

            yield chunk

            # If we got fewer rows than requested, we've reached the end
            if len(chunk) < chunk_size:
                if show_progress:
                    print(
                        f"Completed: fetched {total_rows:,} total rows", file=sys.stderr
                    )
                break

            offset += chunk_size


async def paginated_arcgis_stream(
    base_url: str,
    chunk_size: int = 10000,
    where: str | None = None,
    columns: list[str] | None = None,
    show_progress: bool = False,
) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream data from ArcGIS REST API using pagination.

    Uses ArcGIS's resultRecordCount/resultOffset pagination to fetch data in
    chunks without loading everything into memory. This is the most memory-efficient
    way to work with large ArcGIS datasets.

    Args:
        base_url: Base ArcGIS FeatureServer query URL
        chunk_size: Number of features to fetch per API request
        where: Optional SQL WHERE clause (without the WHERE keyword)
        columns: Optional list of field names to select
        show_progress: If True, print progress information to stderr

    Yields:
        Lists of dictionaries, where each dict represents a feature's attributes.
        Each list contains up to chunk_size features.

    Example:
        >>> url = "https://services.arcgis.com/.../FeatureServer/0/query?where=1=1&outFields=*&f=geojson"
        >>> async for chunk in paginated_arcgis_stream(url, where="STATUS = 'Active'", chunk_size=500):
        ...     print(f"Processing {len(chunk)} features")

    Note:
        The ArcGIS API URL should include 'where', 'outFields', and 'f' (format)
        parameters. GeoJSON format is recommended for geographic data.
    """
    offset = 0
    total_features = 0

    async with httpx.AsyncClient() as client:
        while True:
            # Build query URL with pagination
            query_url = build_arcgis_query(
                base_url,
                where=where,
                columns=columns,
                limit=chunk_size,
                offset=offset,
            )

            if show_progress:
                print(
                    f"Fetching features {offset:,} to {offset + chunk_size:,}...",
                    file=sys.stderr,
                )

            response = await client.get(query_url)
            response.raise_for_status()

            data = response.json()

            # Extract features based on response format
            if "features" in data:
                # GeoJSON or ESRI JSON format
                features = data["features"]
                # Extract attributes from GeoJSON or use the whole feature dict
                if features and "properties" in features[0]:
                    # GeoJSON format
                    chunk = [f["properties"] for f in features]
                elif features and "attributes" in features[0]:
                    # ESRI JSON format
                    chunk = [f["attributes"] for f in features]
                else:
                    # Unknown format, use as-is
                    chunk = features
            elif "rows" in data:
                # Alternative format
                chunk = data["rows"]
            elif isinstance(data, list):
                # Direct array
                chunk = data
            else:
                # No features found
                chunk = []

            if not chunk:
                # No more data
                if show_progress:
                    print(
                        f"Completed: fetched {total_features:,} total features",
                        file=sys.stderr,
                    )
                break

            total_features += len(chunk)

            if show_progress:
                print(
                    f"Yielding {len(chunk)} features (total: {total_features:,})",
                    file=sys.stderr,
                )

            yield chunk

            # If we got fewer features than requested, we've reached the end
            if len(chunk) < chunk_size:
                if show_progress:
                    print(
                        f"Completed: fetched {total_features:,} total features",
                        file=sys.stderr,
                    )
                break

            offset += chunk_size
