"""Filter discovery and validation utilities for Philly datasets.

This module provides helper functions for discovering filterable columns,
inferring schemas, generating example queries, and validating WHERE clauses.
"""

import re
from typing import Any

import pandas as pd

from philly.sample import get_columns_from_sample


def get_filterable_columns(sample: list[dict[str, Any]]) -> list[str]:
    """Return list of column names from sample data.

    This is a convenience wrapper around get_columns_from_sample.

    Args:
        sample: List of dictionaries representing sampled data

    Returns:
        List of column names (empty list if sample is empty)

    Examples:
        >>> sample = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]
        >>> get_filterable_columns(sample)
        ['col1', 'col2']
        >>> get_filterable_columns([])
        []
    """
    return get_columns_from_sample(sample)


def get_filter_schema(
    sample: list[dict[str, Any]], sample_size: int = 100
) -> dict[str, dict[str, Any]]:
    """Infer schema from sample data with detailed metadata.

    Converts sample to DataFrame and analyzes each column to determine:
    - type: pandas dtype as string
    - nullable: whether any null values exist
    - example: first non-null value
    - null_count: count of null values
    - unique_count: count of unique values

    Args:
        sample: List of dictionaries representing sampled data
        sample_size: Expected sample size (unused, kept for API compatibility)

    Returns:
        Dictionary mapping column names to metadata dicts

    Examples:
        >>> sample = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": None}]
        >>> schema = get_filter_schema(sample)
        >>> schema['a']['type']
        'int64'
        >>> schema['b']['nullable']
        True
        >>> schema['b']['null_count']
        1
    """
    if not sample:
        return {}

    df = pd.DataFrame(sample)
    schema: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        dtype = df[col].dtype
        nullable = df[col].isna().any()
        non_null = df[col].dropna()

        schema[col] = {
            "type": str(dtype),
            "nullable": bool(nullable),
            "example": str(non_null.iloc[0]) if len(non_null) > 0 else None,
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique()),
        }

    return schema


def get_filter_examples(schema: dict[str, dict[str, Any]]) -> list[str]:
    """Generate example WHERE clauses based on column schema.

    Generates different types of examples based on column types:
    - datetime: "col >= '2024-01-01'"
    - string: "col = 'example'" and "col LIKE '%example%'"
    - numeric: "col > value"
    - nullable columns: "col IS NOT NULL"

    Args:
        schema: Schema dictionary from get_filter_schema()

    Returns:
        List of example WHERE clause strings (max 10)

    Examples:
        >>> schema = {
        ...     'date': {'type': 'datetime64', 'nullable': False, 'example': '2024-01-01'},
        ...     'name': {'type': 'object', 'nullable': True, 'example': 'John'},
        ...     'count': {'type': 'int64', 'nullable': False, 'example': '42'}
        ... }
        >>> examples = get_filter_examples(schema)
        >>> any('date >=' in ex for ex in examples)
        True
        >>> any('name =' in ex for ex in examples)
        True
    """
    examples: list[str] = []

    for col, meta in schema.items():
        dtype = meta["type"]
        example_val = meta.get("example")

        if not example_val:
            continue

        # Generate examples based on type
        if "datetime" in dtype.lower():
            examples.append(f"{col} >= '2024-01-01'")
            examples.append(f"{col} BETWEEN '2024-01-01' AND '2024-12-31'")

        elif dtype in ["object", "str"] or "str" in dtype.lower():
            examples.append(f"{col} = '{example_val}'")
            examples.append(f"{col} LIKE '%{example_val}%'")

        elif (
            dtype in ["int64", "float64"]
            or "int" in dtype.lower()
            or "float" in dtype.lower()
        ):
            examples.append(f"{col} > {example_val}")
            examples.append(f"{col} BETWEEN 0 AND 100")

        # Add NULL check for nullable columns
        if meta.get("nullable"):
            examples.append(f"{col} IS NOT NULL")

    # Return first 10 examples to avoid overwhelming output
    return examples[:10]


def validate_filter(where: str, columns: list[str]) -> dict[str, Any]:
    """Validate a WHERE clause against available columns.

    Performs basic validation:
    - Checks for dangerous SQL keywords (DROP, DELETE, etc.)
    - Extracts column references from WHERE clause
    - Filters out SQL keywords
    - Checks if referenced columns exist

    Note: This is a simple validation that uses regex to extract identifiers.
    It may produce false positives for column names within string literals.
    For production use, consider using a proper SQL parser.

    Args:
        where: WHERE clause string to validate
        columns: List of valid column names

    Returns:
        Dict with 'valid' (bool) and 'error' (str or None)
        May also include 'available_columns' if invalid columns found

    Examples:
        >>> validate_filter("col1 = 1", ["col1", "col2"])
        {'valid': True, 'error': None}
        >>> result = validate_filter("invalid_col = 1", ["col1", "col2"])
        >>> result['valid']
        False
        >>> 'Unknown column' in result['error']
        True
        >>> result = validate_filter("DROP TABLE users", ["col1"])
        >>> result['valid']
        False
    """
    # Check for dangerous SQL keywords FIRST (before validating columns)
    dangerous_keywords = [
        "drop",
        "delete",
        "update",
        "insert",
        "truncate",
        "alter",
        "create",
    ]
    where_lower = where.lower()

    for keyword in dangerous_keywords:
        if keyword in where_lower:
            return {"valid": False, "error": f"Dangerous SQL keyword: {keyword}"}

    # Extract column names from WHERE clause
    # Match identifiers: start with letter or underscore, followed by alphanumerics/underscores
    referenced_cols = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", where)

    # SQL keywords to filter out
    sql_keywords = {
        "select",
        "from",
        "where",
        "and",
        "or",
        "not",
        "like",
        "between",
        "in",
        "is",
        "null",
        "true",
        "false",
        "order",
        "by",
        "asc",
        "desc",
        "limit",
        "offset",
        "group",
        "having",
        "distinct",
        "as",
        "on",
        "inner",
        "outer",
        "left",
        "right",
        "join",
        "union",
        "case",
        "when",
        "then",
        "else",
        "end",
    }

    # Filter out SQL keywords
    referenced_cols = [c for c in referenced_cols if c.lower() not in sql_keywords]

    # Check if referenced columns exist
    invalid_cols = [c for c in referenced_cols if c not in columns]

    if invalid_cols:
        return {
            "valid": False,
            "error": f"Unknown column(s): {', '.join(invalid_cols)}",
            "available_columns": columns,
        }

    return {"valid": True, "error": None}
