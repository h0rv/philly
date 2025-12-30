"""Tests for the updates module."""

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from philly.cache import CacheEntry
from philly.updates import (
    check_single_update,
    check_updates_batch,
    get_outdated_entries,
)


@pytest.fixture
def sample_cache_entry():
    """Create a sample cache entry for testing."""
    return CacheEntry(
        cache_key="test_dataset_resource_csv",
        dataset_name="test_dataset",
        resource_name="resource",
        format="csv",
        url="https://example.com/data.csv",
        cached_at=time.time() - 86400,  # 1 day ago
        last_accessed=time.time(),
        ttl=604800,  # 7 days
        size_bytes=1024 * 1024,  # 1 MB
        query_params=None,
    )


@pytest.fixture
def cache_metadata(sample_cache_entry):
    """Create sample cache metadata with multiple entries."""
    entry1 = sample_cache_entry
    entry2 = CacheEntry(
        cache_key="another_dataset_resource_json",
        dataset_name="another_dataset",
        resource_name="resource2",
        format="json",
        url="https://example.com/data.json",
        cached_at=time.time() - 172800,  # 2 days ago
        last_accessed=time.time(),
        ttl=604800,
        size_bytes=2 * 1024 * 1024,  # 2 MB
        query_params=None,
    )
    return {
        entry1.cache_key: entry1,
        entry2.cache_key: entry2,
    }


def test_check_single_update_with_newer_remote(sample_cache_entry):
    """Test check_single_update when remote has newer data."""
    # Mock remote date to be newer than cached date
    newer_date = datetime.fromtimestamp(
        sample_cache_entry.cached_at + 3600, tz=timezone.utc
    )

    with patch("philly.updates.get_remote_last_modified", return_value=newer_date):
        result = check_single_update(
            sample_cache_entry, sample_cache_entry.url, timeout=5.0
        )

    assert result["dataset"] == "test_dataset"
    assert result["resource"] == "resource"
    assert result["has_update"] is True
    assert result["remote_date"] == newer_date
    assert isinstance(result["cached_date"], datetime)
    assert result["cache_age_hours"] > 23  # Roughly 24 hours
    assert result["size_mb"] == 1.0


def test_check_single_update_with_older_remote(sample_cache_entry):
    """Test check_single_update when remote is older than cache."""
    # Mock remote date to be older than cached date
    older_date = datetime.fromtimestamp(
        sample_cache_entry.cached_at - 3600, tz=timezone.utc
    )

    with patch("philly.updates.get_remote_last_modified", return_value=older_date):
        result = check_single_update(
            sample_cache_entry, sample_cache_entry.url, timeout=5.0
        )

    assert result["has_update"] is False
    assert result["remote_date"] == older_date


def test_check_single_update_with_no_remote_date(sample_cache_entry):
    """Test check_single_update when server doesn't provide Last-Modified."""
    with patch("philly.updates.get_remote_last_modified", return_value=None):
        result = check_single_update(
            sample_cache_entry, sample_cache_entry.url, timeout=5.0
        )

    assert result["has_update"] is None
    assert result["remote_date"] is None
    assert result["dataset"] == "test_dataset"


def test_check_updates_batch_all(cache_metadata):
    """Test check_updates_batch without filters."""
    newer_date = datetime.now(tz=timezone.utc)

    with patch("philly.updates.get_remote_last_modified", return_value=newer_date):
        results = check_updates_batch(cache_metadata)

    assert len(results) == 2
    assert all(r["has_update"] is True for r in results)
    assert {r["dataset"] for r in results} == {"test_dataset", "another_dataset"}


def test_check_updates_batch_with_dataset_filter(cache_metadata):
    """Test check_updates_batch with dataset filter."""
    newer_date = datetime.now(tz=timezone.utc)

    with patch("philly.updates.get_remote_last_modified", return_value=newer_date):
        results = check_updates_batch(cache_metadata, dataset_filter=["test_dataset"])

    assert len(results) == 1
    assert results[0]["dataset"] == "test_dataset"


def test_check_updates_batch_only_outdated(cache_metadata):
    """Test check_updates_batch with only_outdated flag."""

    def mock_remote_date(url, timeout):
        # Make only one dataset have an update
        if "data.csv" in url:
            # Newer date for first entry
            return datetime.now(tz=timezone.utc)
        else:
            # Older date for second entry
            return datetime.fromtimestamp(0, tz=timezone.utc)

    with patch("philly.updates.get_remote_last_modified", side_effect=mock_remote_date):
        results = check_updates_batch(cache_metadata, only_outdated=True)

    assert len(results) == 1
    assert results[0]["dataset"] == "test_dataset"
    assert results[0]["has_update"] is True


def test_check_updates_batch_handles_exceptions(cache_metadata):
    """Test that check_updates_batch handles exceptions gracefully."""
    with patch(
        "philly.updates.get_remote_last_modified",
        side_effect=Exception("Network error"),
    ):
        results = check_updates_batch(cache_metadata)

    # Should return empty list since all checks failed
    assert len(results) == 0


def test_get_outdated_entries(cache_metadata):
    """Test get_outdated_entries returns correct CacheEntry objects."""

    def mock_remote_date(url, timeout):
        # Make only one dataset have an update
        if "data.csv" in url:
            return datetime.now(tz=timezone.utc)
        else:
            return datetime.fromtimestamp(0, tz=timezone.utc)

    with patch("philly.updates.get_remote_last_modified", side_effect=mock_remote_date):
        outdated = get_outdated_entries(cache_metadata)

    assert len(outdated) == 1
    assert isinstance(outdated[0], CacheEntry)
    assert outdated[0].dataset_name == "test_dataset"
    assert outdated[0].resource_name == "resource"


def test_get_outdated_entries_empty(cache_metadata):
    """Test get_outdated_entries when nothing is outdated."""
    # Mock all remote dates as very old
    old_date = datetime.fromtimestamp(0, tz=timezone.utc)

    with patch("philly.updates.get_remote_last_modified", return_value=old_date):
        outdated = get_outdated_entries(cache_metadata)

    assert len(outdated) == 0
