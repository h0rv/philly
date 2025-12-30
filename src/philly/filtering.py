"""Server-side filtering support for Philly datasets.

This module provides backend detection and query building for server-side filtering,
limiting, and column selection for data sources that support it (Carto, ArcGIS REST).
"""

import re
from enum import Enum
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


class BackendType(Enum):
    """Supported backend types for datasets."""

    CARTO = "carto"
    ARCGIS = "arcgis"
    STATIC = "static"
    UNKNOWN = "unknown"


def detect_backend(url: str) -> BackendType:
    """Detect which backend type serves a URL.

    Args:
        url: The resource URL to analyze

    Returns:
        BackendType enum indicating the detected backend

    Examples:
        >>> detect_backend("https://phl.carto.com/api/v2/sql?q=SELECT...")
        BackendType.CARTO

        >>> detect_backend("https://services.arcgis.com/.../FeatureServer/0/query")
        BackendType.ARCGIS

        >>> detect_backend("https://example.com/data.csv")
        BackendType.STATIC
    """
    if "carto.com" in url and "/api/v2/sql" in url:
        return BackendType.CARTO

    if "arcgis.com" in url or "FeatureServer" in url:
        return BackendType.ARCGIS

    # Check if it's a direct file download
    static_extensions = [".csv", ".json", ".geojson", ".shp", ".zip"]
    if any(url.endswith(ext) for ext in static_extensions):
        return BackendType.STATIC

    return BackendType.UNKNOWN


def validate_where_clause(where: str) -> str:
    """Basic validation of WHERE clause to prevent SQL injection.

    Args:
        where: The WHERE clause to validate

    Returns:
        The validated WHERE clause (unchanged if valid)

    Raises:
        ValueError: If dangerous keywords are detected

    Examples:
        >>> validate_where_clause("district = '6'")
        "district = '6'"

        >>> validate_where_clause("district = '6'; DROP TABLE users")
        Traceback (most recent call last):
        ...
        ValueError: WHERE clause contains disallowed keyword: DROP
    """
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE"]
    where_upper = where.upper()

    for keyword in dangerous_keywords:
        if keyword in where_upper:
            raise ValueError(f"WHERE clause contains disallowed keyword: {keyword}")

    return where


def _extract_table_name(query: str) -> str:
    """Extract table name from a SELECT query.

    Args:
        query: SQL query string like "SELECT * FROM table_name"

    Returns:
        The extracted table name

    Raises:
        ValueError: If table name cannot be extracted

    Examples:
        >>> _extract_table_name("SELECT * FROM my_table")
        'my_table'

        >>> _extract_table_name("SELECT col1, col2 FROM schema.table")
        'schema.table'
    """
    # Match: SELECT ... FROM table_name (with optional schema)
    # Handle various whitespace and case variations
    match = re.search(r"\bFROM\s+([a-zA-Z0-9_\.]+)", query, re.IGNORECASE)

    if not match:
        raise ValueError(f"Could not extract table name from query: {query}")

    return match.group(1)


def build_carto_query(
    base_url: str,
    where: str | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> str:
    """Build Carto SQL query URL with filtering parameters.

    Args:
        base_url: The base Carto API URL
        where: SQL WHERE clause (without the WHERE keyword)
        columns: List of column names to select
        limit: Maximum number of rows to return
        offset: Number of rows to skip (for pagination)

    Returns:
        Modified URL with the new query parameters

    Raises:
        ValueError: If WHERE clause contains dangerous keywords or table name cannot be extracted

    Examples:
        >>> url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        >>> build_carto_query(url, where="district = '6'", limit=100)
        "https://phl.carto.com/api/v2/sql?q=SELECT+%2A+FROM+crimes+WHERE+district+%3D+%276%27+LIMIT+100&format=csv"
    """
    # Validate WHERE clause if provided
    if where:
        _ = validate_where_clause(where)

    # Parse the URL
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    # Get existing query and extract table name
    existing_query = params.get("q", [""])[0]
    if not existing_query:
        raise ValueError("Carto URL must contain a 'q' parameter with SQL query")

    table_name = _extract_table_name(existing_query)

    # Build new query
    select_cols = ", ".join(columns) if columns else "*"
    query = f"SELECT {select_cols} FROM {table_name}"

    if where:
        query += f" WHERE {where}"

    if limit is not None:
        query += f" LIMIT {limit}"

    if offset is not None:
        query += f" OFFSET {offset}"

    # Update query parameter
    params["q"] = [query]

    # Rebuild URL preserving all other parameters (format, filename, etc.)
    new_query_string = urlencode(params, doseq=True)
    # Replace + with %20 for SQL query compatibility (Carto expects %20)
    new_query_string = new_query_string.replace("+", "%20")
    new_parsed = parsed._replace(query=new_query_string)

    return urlunparse(new_parsed)


def build_arcgis_query(
    base_url: str,
    where: str | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> str:
    """Build ArcGIS REST API query URL with filtering parameters.

    Args:
        base_url: The base ArcGIS FeatureServer URL
        where: SQL WHERE clause (without the WHERE keyword)
        columns: List of column names to select
        limit: Maximum number of features to return
        offset: Number of features to skip (for pagination)

    Returns:
        Modified URL with the new query parameters

    Examples:
        >>> url = "https://services.arcgis.com/.../FeatureServer/0/query?where=1=1&outFields=*&f=geojson"
        >>> build_arcgis_query(url, where="STATUS = 'Active'", limit=50)
        "https://services.arcgis.com/.../FeatureServer/0/query?where=STATUS+%3D+%27Active%27&outFields=%2A&f=geojson&resultRecordCount=50"
    """
    # Parse the URL
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    # Update parameters based on what's provided
    if where is not None:
        params["where"] = [where]

    if columns is not None:
        params["outFields"] = [",".join(columns)]

    if limit is not None:
        params["resultRecordCount"] = [str(limit)]

    if offset is not None:
        params["resultOffset"] = [str(offset)]

    # Rebuild URL
    new_query_string = urlencode(params, doseq=True)
    new_parsed = parsed._replace(query=new_query_string)

    return urlunparse(new_parsed)
