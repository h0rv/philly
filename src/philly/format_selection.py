"""Format auto-selection utilities for Philly datasets.

This module provides utilities for finding and selecting resources by format,
handling multiple resources with the same format through preference strategies,
and querying available formats.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from philly.models.dataset import Dataset
    from philly.models.resource import Resource


# Default format preference order
# Prefers structured, easy-to-process formats (CSV, JSON, Parquet)
# over binary/proprietary formats (Shapefile, XLSX, TIFF)
DEFAULT_FORMAT_PREFERENCE = [
    "csv",
    "geojson",
    "json",
    "geoparquet",
    "parquet",
    "api",
    "shp",
    "xlsx",
    "tiff",
    "tif",
]


def find_resource_by_format(
    dataset: "Dataset",
    format_str: str,
    prefer: str = "latest",
) -> "Resource | None":
    """Find a resource matching the specified format.

    Args:
        dataset: Dataset to search within
        format_str: Format to find (case-insensitive)
        prefer: Selection strategy when multiple resources match.
                "latest" - Select resource with highest year in name
                "oldest" - Select resource with lowest year in name

    Returns:
        Matching resource or None if no match found
    """
    if not dataset.resources:
        return None

    format_lower = format_str.lower()

    # Find all resources with matching format
    matches = [r for r in dataset.resources if str(r.format).lower() == format_lower]

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    # Multiple matches - apply preference strategy
    return select_by_year(matches, prefer)


def select_by_year(
    resources: list["Resource"],
    prefer: str,
) -> "Resource":
    """Select a resource from a list based on year preference.

    Extracts 4-digit years (20XX) from resource names and sorts by year.
    Resources without years are treated specially:
    - For "latest": assigned year 0 (sorted last)
    - For "oldest": assigned year 9999 (sorted last)

    Args:
        resources: List of resources to select from
        prefer: "latest" for highest year, "oldest" for lowest year

    Returns:
        Selected resource
    """

    def extract_year(name: str) -> int:
        """Extract 4-digit year from resource name."""
        years = re.findall(r"\b(20\d{2})\b", name)

        if not years:
            # No year found - assign sentinel value
            # For "latest", 0 comes last in descending sort
            # For "oldest", 9999 comes last in ascending sort
            return 0 if prefer == "latest" else 9999

        # For "latest", use the highest year in the name
        # For "oldest", use the lowest year in the name
        return int(max(years)) if prefer == "latest" else int(min(years))

    # Sort by year based on preference
    if prefer == "latest":
        # Descending order - highest year first
        sorted_resources = sorted(
            resources, key=lambda r: extract_year(r.name), reverse=True
        )
    else:  # prefer == "oldest"
        # Ascending order - lowest year first
        sorted_resources = sorted(resources, key=lambda r: extract_year(r.name))

    return sorted_resources[0]


def get_formats(dataset: "Dataset") -> list[str]:
    """Get all unique formats available in a dataset.

    Args:
        dataset: Dataset to query

    Returns:
        Sorted list of unique format strings
    """
    if not dataset.resources:
        return []

    # Extract unique formats and convert to strings
    formats = {str(r.format).lower() for r in dataset.resources}

    return sorted(formats)


def has_format(dataset: "Dataset", format_str: str) -> bool:
    """Check if a dataset has a resource with the specified format.

    Args:
        dataset: Dataset to check
        format_str: Format to look for (case-insensitive)

    Returns:
        True if format exists in dataset, False otherwise
    """
    formats = get_formats(dataset)
    return format_str.lower() in formats
