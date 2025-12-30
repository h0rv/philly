"""Metadata and info API for Philly datasets and resources."""

from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from philly.models.dataset import Dataset
from philly.models.resource import Resource


def get_remote_size(url: str, timeout: float = 10.0) -> float | None:
    """Get the size of a remote resource in MB.

    Args:
        url: The URL to check
        timeout: Request timeout in seconds

    Returns:
        Size in MB, or None if unable to determine
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.head(url)
            content_length: str | None = response.headers.get("Content-Length")
            if content_length is not None:
                bytes_size = int(content_length)
                return bytes_size / 1024 / 1024
            return None
    except Exception:
        return None


def get_remote_last_modified(url: str, timeout: float = 10.0) -> datetime | None:
    """Get the last modified date of a remote resource.

    Args:
        url: The URL to check
        timeout: Request timeout in seconds

    Returns:
        Last modified datetime, or None if unable to determine
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.head(url)
            last_modified: str | None = response.headers.get("Last-Modified")
            if last_modified is not None:
                return parsedate_to_datetime(last_modified)
            return None
    except Exception:
        return None


def is_url_available(url: str, timeout: float = 10.0) -> bool:
    """Check if a URL is available (returns 200 OK).

    Args:
        url: The URL to check
        timeout: Request timeout in seconds

    Returns:
        True if URL returns 200, False otherwise
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.head(url)
            return response.status_code == 200
    except Exception:
        return False


def get_resource_info(dataset: Dataset, resource: Resource) -> dict[str, Any]:
    """Get detailed information about a resource.

    Args:
        dataset: The dataset containing the resource
        resource: The resource to get info for

    Returns:
        Dictionary with resource metadata including:
        - dataset: dataset title
        - resource: resource name
        - format: resource format
        - url: resource URL
        - size_mb: size in MB (or None)
        - last_modified: last modified datetime (or None)
        - organization: dataset organization
        - category: dataset category
        - description: dataset description
        - license: dataset license
        - maintainer_email: dataset maintainer email
    """
    url = resource.url or ""

    return {
        "dataset": dataset.title,
        "resource": resource.name,
        "format": str(resource.format),
        "url": url,
        "size_mb": get_remote_size(url) if url else None,
        "last_modified": get_remote_last_modified(url) if url else None,
        "organization": dataset.organization,
        "category": dataset.category,
        "description": dataset.notes,
        "license": dataset.license,
        "maintainer_email": dataset.maintainer_email,
    }


def get_dataset_info(dataset: Dataset) -> dict[str, Any]:
    """Get information about a dataset and its resources.

    Args:
        dataset: The dataset to get info for

    Returns:
        Dictionary with dataset metadata including:
        - title: dataset title
        - organization: dataset organization
        - category: dataset category
        - description: dataset description
        - license: dataset license
        - maintainer_email: dataset maintainer email
        - source: dataset source
        - created: dataset created date
        - num_resources: number of resources
        - resources: list of minimal resource info (name, format, url)
    """
    resources = dataset.resources or []

    return {
        "title": dataset.title,
        "organization": dataset.organization,
        "category": dataset.category,
        "description": dataset.notes,
        "license": dataset.license,
        "maintainer_email": dataset.maintainer_email,
        "source": dataset.source,
        "created": dataset.created,
        "num_resources": len(resources),
        "resources": [
            {
                "name": resource.name,
                "format": str(resource.format),
                "url": resource.url,
            }
            for resource in resources
        ],
    }
