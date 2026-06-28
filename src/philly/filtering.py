"""Server-side filtering support for Philly datasets.

This module provides backend detection and query building for server-side filtering,
limiting, and column selection for data sources that support it (Carto, ArcGIS REST).
"""

import re
from enum import Enum
import json
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


CARTO_RESERVED_WORDS: set[str] = {
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "IS",
    "NULL",
    "LIKE",
    "BETWEEN",
    "ORDER",
    "BY",
    "GROUP",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "AS",
    "ON",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "FULL",
    "CROSS",
    "UNION",
    "ALL",
    "DISTINCT",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "DESC",
    "ASC",
    "TRUE",
    "FALSE",
    "TABLE",
    "CREATE",
    "ALTER",
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "SET",
    "VALUES",
    "INTO",
    "WITH",
    "EXISTS",
    "UNIQUE",
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "REFERENCES",
    "INDEX",
    "VIEW",
    "CAST",
    "CONVERT",
    "COALESCE",
    "NULLIF",
    "ROW",
    "ROWS",
    "TYPE",
    "DATE",
    "TIME",
    "TIMESTAMP",
    "VARCHAR",
    "INTEGER",
    "NUMERIC",
    "FLOAT",
    "DOUBLE",
    "PRECISION",
    "BOOLEAN",
    "CHAR",
    "TEXT",
    "BIGINT",
    "INT",
    "SMALLINT",
    "DECIMAL",
    "REAL",
    "SERIAL",
    "BLOB",
}


def quote_identifier(name: str) -> str:
    """Wrap an identifier in double quotes if it matches a SQL reserved word."""
    return f'"{name}"' if name.upper() in CARTO_RESERVED_WORDS else name


def extract_computed_columns(sql_query: str) -> dict[str, str]:
    """Return {alias: expression} for computed columns in a SELECT query.

    A column is "computed" if it has an AS alias and the expression before AS
    is not a bare identifier (contains function calls, operators, etc.).
    """
    match = re.search(r"SELECT\s+(.*?)\s+FROM\s", sql_query, re.IGNORECASE | re.DOTALL)
    if not match:
        return {}

    col_list_str = match.group(1).strip()

    # Split column list by commas, respecting parentheses depth
    cols: list[str] = []
    current = ""
    depth = 0
    for char in col_list_str:
        if char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            cols.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        cols.append(current.strip())

    result: dict[str, str] = {}
    for col in cols:
        m = re.search(r"\bAS\s+([a-zA-Z_]\w*)\s*$", col, re.IGNORECASE)
        if m:
            alias = m.group(1)
            expr = col[: m.start()].strip()
            if not re.match(r"^[a-zA-Z_]\w*$", expr):
                result[alias] = expr

    return result


def check_computed_columns(where: str, computed_columns: dict[str, str]) -> list[str]:
    """Check if WHERE clause references computed column aliases."""
    warnings: list[str] = []
    for alias, expr in computed_columns.items():
        if re.search(rf"\b{re.escape(alias)}\b", where):
            warnings.append(
                f"WHERE clause references computed column '{alias}' ({expr}). "
                f"Use the expression directly: '{expr}' instead of '{alias}'."
            )
    return warnings


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
        validate_where_clause(where)

    # Parse the URL
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    # Get existing query and extract table name
    existing_query = params.get("q", [""])[0]
    if not existing_query:
        raise ValueError("Carto URL must contain a 'q' parameter with SQL query")

    table_name = _extract_table_name(existing_query)

    # Check if WHERE clause references computed columns
    if where:
        computed = extract_computed_columns(existing_query)
        conflicts = check_computed_columns(where, computed)
        if conflicts:
            raise ValueError("; ".join(conflicts))

    # Build new query
    if columns:
        select_cols = ", ".join(quote_identifier(col.strip()) for col in columns)
    else:
        select_cols = "*"
    query = f"SELECT {select_cols} FROM {quote_identifier(table_name)}"

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


def build_carto_agg_query(
    base_url: str,
    by: str | None = None,
    metric: str = "COUNT(*)",
    where: str | None = None,
    limit: int | None = None,
) -> str:
    """Build a Carto aggregation query URL with GROUP BY."""
    if where:
        validate_where_clause(where)

    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    existing_query = params.get("q", [""])[0]
    if not existing_query:
        raise ValueError("Carto URL must contain a 'q' parameter with SQL query")

    table_name = _extract_table_name(existing_query)

    if by:
        select_expr = f"{by}, {metric}"
        group_clause = f" GROUP BY {by}"
    else:
        select_expr = metric
        group_clause = ""

    query = f"SELECT {select_expr} FROM {quote_identifier(table_name)}"

    if where:
        query += f" WHERE {where}"

    query += group_clause

    if limit is not None:
        query += f" LIMIT {limit}"

    params["q"] = [query]

    new_query_string = urlencode(params, doseq=True)
    new_query_string = new_query_string.replace("+", "%20")
    new_parsed = parsed._replace(query=new_query_string)

    return urlunparse(new_parsed)


def build_carto_distinct_query(
    base_url: str,
    column: str,
    where: str | None = None,
    limit: int | None = None,
) -> str:
    """Build a Carto SELECT DISTINCT query URL.

    Args:
        base_url: The base Carto API URL with a 'q' parameter.
        column: The column to select distinct values from.
        where: SQL WHERE clause (without the WHERE keyword).
        limit: Maximum number of rows to return.

    Returns:
        Modified URL with the DISTINCT query.

    Raises:
        ValueError: If the WHERE clause is invalid or table name cannot be extracted.
    """
    if where:
        validate_where_clause(where)

    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    existing_query = params.get("q", [""])[0]
    if not existing_query:
        raise ValueError("Carto URL must contain a 'q' parameter with SQL query")

    table_name = _extract_table_name(existing_query)

    query = (
        f"SELECT DISTINCT {quote_identifier(column.strip())}"
        f" FROM {quote_identifier(table_name)}"
    )

    if where:
        query += f" WHERE {where}"

    if limit is not None:
        query += f" LIMIT {limit}"

    params["q"] = [query]

    new_query_string = urlencode(params, doseq=True)
    new_query_string = new_query_string.replace("+", "%20")
    new_parsed = parsed._replace(query=new_query_string)

    return urlunparse(new_parsed)


def _parse_metric(metric: str) -> tuple[str, str]:
    """Parse an aggregate expression into (statistic_type, field_name).

    >>> _parse_metric("COUNT(*)")
    ("COUNT", "*")
    >>> _parse_metric("SUM(amount)")
    ("SUM", "amount")
    >>> _parse_metric("count")
    ("COUNT", "*")
    """
    m = re.match(r"(\w+)\s*\((.+?)\)", metric.strip(), re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return metric.strip().upper(), "*"


def build_arcgis_agg_query(
    base_url: str,
    by: str | None = None,
    metric: str | None = None,
    where: str | None = None,
    limit: int | None = None,
) -> str:
    """Build an ArcGIS aggregation query URL.

    Uses groupByFieldsForStatistics and outStatistics parameters.
    """
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    stat_type, stat_field = _parse_metric(metric or "COUNT(*)")
    if by:
        params["groupByFieldsForStatistics"] = [by]
        out_statistics = json.dumps(
            [
                {
                    "statisticType": stat_type,
                    "onStatisticField": by if stat_field == "*" else stat_field,
                    "outStatisticFieldName": "count",
                }
            ]
        )
        params["outStatistics"] = [out_statistics]
    else:
        out_statistics = json.dumps(
            [
                {
                    "statisticType": stat_type,
                    "onStatisticField": stat_field if stat_field != "*" else by or "1",
                    "outStatisticFieldName": "count",
                }
            ]
        )
        params["outStatistics"] = [out_statistics]

    if where:
        params["where"] = [where]

    if limit is not None:
        params["resultRecordCount"] = [str(limit)]

    params["returnGeometry"] = ["false"]

    new_query_string = urlencode(params, doseq=True)
    new_parsed = parsed._replace(query=new_query_string)

    return urlunparse(new_parsed)


def build_arcgis_distinct_query(
    base_url: str,
    column: str,
    where: str | None = None,
    limit: int | None = None,
) -> str:
    """Build an ArcGIS distinct values query URL.

    Uses outFields, returnDistinctValues, and resultRecordCount parameters.

    Args:
        base_url: The base ArcGIS FeatureServer URL.
        column: The column to get distinct values from.
        where: SQL WHERE clause (without the WHERE keyword).
        limit: Maximum number of features to return.

    Returns:
        Modified URL with distinct query parameters.
    """
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    params["outFields"] = [column]
    params["returnDistinctValues"] = ["true"]
    params["returnGeometry"] = ["false"]

    if where:
        params["where"] = [where]

    if limit is not None:
        params["resultRecordCount"] = [str(limit)]

    new_query_string = urlencode(params, doseq=True)
    new_parsed = parsed._replace(query=new_query_string)

    return urlunparse(new_parsed)
