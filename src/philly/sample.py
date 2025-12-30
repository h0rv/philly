"""Sample/preview functionality for Philly datasets.

This module provides utilities for sampling data from various formats without
downloading entire files. It handles CSV streaming, JSON/GeoJSON slicing, and
format conversion.
"""

import csv
from typing import Any

import httpx
import pandas as pd

from philly.urls import normalize_url


async def sample_csv(url: str, n: int) -> list[dict[str, str]]:
    """Sample first N rows from CSV using streaming to avoid full download.

    Args:
        url: URL of the CSV file
        n: Number of rows to sample

    Returns:
        List of dictionaries, one per row (up to N rows)

    Examples:
        >>> sample = await sample_csv("https://example.com/data.csv", 10)
        >>> len(sample) <= 10
        True
    """
    url = normalize_url(url)
    if url.startswith("http://"):
        url = url.replace("http://", "https://")

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=60,
        ) as response:
            _ = response.raise_for_status()

            buffer = ""
            rows: list[dict[str, str]] = []
            headers: list[str] | None = None

            # Stream chunks until we have N rows
            async for chunk in response.aiter_bytes(chunk_size=8192):
                buffer += chunk.decode("utf-8", errors="replace")

                # Split into lines, keeping incomplete line in buffer
                lines = buffer.split("\n")
                buffer = lines[-1]

                for line in lines[:-1]:
                    # Skip empty lines
                    if not line.strip():
                        continue

                    # First non-empty line is headers
                    if headers is None:
                        try:
                            headers = next(csv.reader([line]))
                        except (csv.Error, StopIteration):
                            continue
                        continue

                    # Parse data row
                    try:
                        values = next(csv.reader([line]))
                        # Pad or truncate to match header length
                        if len(values) < len(headers):
                            values.extend([""] * (len(headers) - len(values)))
                        elif len(values) > len(headers):
                            values = values[: len(headers)]
                        row = dict(zip(headers, values))
                        rows.append(row)
                    except (csv.Error, StopIteration):
                        # Skip malformed rows
                        continue

                    # Stop after N rows
                    if len(rows) >= n:
                        return rows

            # Process any remaining data in buffer
            if buffer.strip() and headers:
                try:
                    values = next(csv.reader([buffer]))
                    if len(values) < len(headers):
                        values.extend([""] * (len(headers) - len(values)))
                    elif len(values) > len(headers):
                        values = values[: len(headers)]
                    row = dict(zip(headers, values))
                    rows.append(row)
                except (csv.Error, StopIteration):
                    pass

            return rows


async def sample_json(url: str, n: int) -> list[dict[str, Any]]:
    """Sample first N records from JSON data.

    Handles different JSON structures:
    - List: returns data[:n]
    - Dict with 'data', 'results', 'features', or 'records' keys: returns data[key][:n]
    - Other dict: returns [data]

    Args:
        url: URL of the JSON file
        n: Number of records to sample

    Returns:
        List of dictionaries (up to N records)

    Examples:
        >>> sample = await sample_json("https://example.com/data.json", 10)
        >>> len(sample) <= 10
        True
    """
    url = normalize_url(url)
    if url.startswith("http://"):
        url = url.replace("http://", "https://")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=60)
        _ = response.raise_for_status()

        data = response.json()

        # Handle different JSON structures
        if isinstance(data, list):
            return data[:n]
        elif isinstance(data, dict):
            # Try common keys that contain arrays
            for key in ["data", "results", "features", "records"]:
                if key in data and isinstance(data[key], list):
                    return data[key][:n]
            # Fallback: return the dict wrapped in list
            return [data]
        else:
            return []


async def sample_geojson(url: str, n: int) -> list[dict[str, Any]]:
    """Sample first N features from GeoJSON.

    Args:
        url: URL of the GeoJSON file
        n: Number of features to sample

    Returns:
        List of feature dictionaries (up to N features)

    Examples:
        >>> sample = await sample_geojson("https://example.com/data.geojson", 10)
        >>> len(sample) <= 10
        True
    """
    url = normalize_url(url)
    if url.startswith("http://"):
        url = url.replace("http://", "https://")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=60)
        _ = response.raise_for_status()

        data = response.json()

        # GeoJSON has a 'features' array
        if "features" in data and isinstance(data["features"], list):
            return data["features"][:n]
        else:
            return []


def format_chunk(data: list[dict[str, Any]], format: str) -> Any:
    """Convert sample data to requested format.

    Args:
        data: List of dictionaries
        format: Output format ("records" or "dataframe")

    Returns:
        Data in requested format (list[dict] or pd.DataFrame)

    Examples:
        >>> data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        >>> format_chunk(data, "records")
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        >>> format_chunk(data, "dataframe")
           a  b
        0  1  2
        1  3  4
    """
    if format == "records":
        return data
    elif format == "dataframe":
        return pd.DataFrame(data)
    else:
        # Default to records for unknown formats
        return data


def get_columns_from_sample(sample: list[dict[str, Any]]) -> list[str]:
    """Extract column names from sample data.

    Args:
        sample: List of dictionaries

    Returns:
        List of column names (keys from first record)

    Examples:
        >>> sample = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        >>> get_columns_from_sample(sample)
        ['a', 'b']
        >>> get_columns_from_sample([])
        []
    """
    if not sample:
        return []
    return list(sample[0].keys())


def infer_schema_from_sample(sample: list[dict[str, Any]]) -> dict[str, str]:
    """Infer data types from sample data.

    Converts sample to DataFrame and extracts dtype information.

    Args:
        sample: List of dictionaries

    Returns:
        Dictionary mapping column names to type strings

    Examples:
        >>> sample = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        >>> schema = infer_schema_from_sample(sample)
        >>> "a" in schema and "b" in schema
        True
    """
    if not sample:
        return {}

    df = pd.DataFrame(sample)
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        schema[col] = str(dtype)

    return schema
