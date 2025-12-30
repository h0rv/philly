"""Update checking for cached datasets."""

import time
from datetime import datetime
from typing import Any

from philly.cache import CacheEntry
from philly.metadata import get_remote_last_modified


def check_single_update(
    cache_entry: CacheEntry, url: str, timeout: float = 10.0
) -> dict[str, Any]:
    """Check if a single cached resource has an update available.

    Args:
        cache_entry: The cache entry to check
        url: The URL of the remote resource
        timeout: Request timeout in seconds

    Returns:
        Dictionary with update information including:
        - dataset: dataset name
        - resource: resource name
        - cached_date: datetime when the resource was cached
        - remote_date: datetime of last modification on remote (or None)
        - has_update: True if remote is newer, False if not, None if unknown
        - cache_age_hours: age of the cache entry in hours
        - size_mb: size of the cached resource in MB
    """
    cached_date = datetime.fromtimestamp(cache_entry.cached_at)
    remote_date = get_remote_last_modified(url, timeout=timeout)

    # Determine if update is available
    has_update: bool | None = None
    if remote_date is not None:
        # Compare timestamps directly to avoid timezone conversion issues
        # Convert both to UTC timestamps for accurate comparison
        cached_timestamp = cache_entry.cached_at
        remote_timestamp = remote_date.timestamp()
        has_update = remote_timestamp > cached_timestamp

    cache_age_hours = (time.time() - cache_entry.cached_at) / 3600
    size_mb = cache_entry.size_bytes / 1024 / 1024

    return {
        "dataset": cache_entry.dataset_name,
        "resource": cache_entry.resource_name,
        "cached_date": cached_date,
        "remote_date": remote_date,
        "has_update": has_update,
        "cache_age_hours": cache_age_hours,
        "size_mb": size_mb,
    }


def check_updates_batch(
    cache_metadata: dict[str, CacheEntry],
    dataset_filter: list[str] | None = None,
    only_outdated: bool = False,
) -> list[dict[str, Any]]:
    """Check for updates on multiple cached resources.

    Args:
        cache_metadata: Dictionary of cache entries keyed by cache_key
        dataset_filter: Optional list of dataset names to filter by
        only_outdated: If True, only return entries with available updates

    Returns:
        List of update information dictionaries
    """
    results: list[dict[str, Any]] = []

    for cache_entry in cache_metadata.values():
        # Apply dataset filter if provided
        if dataset_filter is not None:
            if cache_entry.dataset_name not in dataset_filter:
                continue

        # Check for updates
        try:
            update_info = check_single_update(cache_entry, cache_entry.url)

            # Apply outdated filter if requested
            if only_outdated:
                if update_info["has_update"] is True:
                    results.append(update_info)
            else:
                results.append(update_info)
        except Exception:
            # If we can't check updates for this entry, skip it silently
            # (e.g., network issues, invalid URL, etc.)
            continue

    return results


def get_outdated_entries(cache_metadata: dict[str, CacheEntry]) -> list[CacheEntry]:
    """Get cache entries that have updates available.

    Args:
        cache_metadata: Dictionary of cache entries keyed by cache_key

    Returns:
        List of CacheEntry objects that have updates available
    """
    outdated_info = check_updates_batch(cache_metadata, only_outdated=True)

    # Build a set of (dataset, resource) tuples for quick lookup
    outdated_set = {(info["dataset"], info["resource"]) for info in outdated_info}

    # Return the matching CacheEntry objects
    outdated_entries: list[CacheEntry] = []
    for cache_entry in cache_metadata.values():
        if (cache_entry.dataset_name, cache_entry.resource_name) in outdated_set:
            outdated_entries.append(cache_entry)

    return outdated_entries
