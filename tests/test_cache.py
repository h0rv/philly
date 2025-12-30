import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from philly import Philly
from philly.cache import CacheEntry, FileCache


class TestFileCache:
    """Test FileCache class"""

    def test_cache_init(self):
        """Test cache initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir, ttl=3600)
            assert cache.cache_dir.exists()
            assert cache.ttl == 3600
            assert cache.metadata_file.exists()

    def test_cache_set_and_get(self):
        """Test setting and getting cache entries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir, ttl=3600)

            cache_key = "test_key"
            test_data = {"test": "data"}
            current_time = time.time()
            entry = CacheEntry(
                cache_key=cache_key,
                dataset_name="Test Dataset",
                resource_name="Test Resource",
                format="json",
                url="https://example.com",
                cached_at=current_time,
                last_accessed=current_time,
                ttl=3600,
                size_bytes=0,
                query_params=None,
            )

            cache.set(cache_key, test_data, entry)
            retrieved = cache.get(cache_key)

            assert retrieved == test_data

    def test_cache_expired(self):
        """Test that expired cache entries are not returned"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir, ttl=1)

            cache_key = "test_key"
            test_data = {"test": "data"}
            current_time = time.time()
            entry = CacheEntry(
                cache_key=cache_key,
                dataset_name="Test Dataset",
                resource_name="Test Resource",
                format="json",
                url="https://example.com",
                cached_at=current_time,
                last_accessed=current_time,
                ttl=1,
                size_bytes=0,
                query_params=None,
            )

            cache.set(cache_key, test_data, entry)

            # Wait for expiration
            time.sleep(2)

            # Should return None for expired entry
            assert cache.get(cache_key) is None

    def test_cache_clear_all(self):
        """Test clearing all cache entries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir, ttl=3600)

            # Add multiple entries
            for i in range(3):
                cache_key = f"test_key_{i}"
                current_time = time.time()
                entry = CacheEntry(
                    cache_key=cache_key,
                    dataset_name=f"Dataset {i}",
                    resource_name="Resource",
                    format="json",
                    url="https://example.com",
                    cached_at=current_time,
                    last_accessed=current_time,
                    ttl=3600,
                    size_bytes=0,
                    query_params=None,
                )
                cache.set(cache_key, {"data": i}, entry)

            # Clear all
            cache.clear()

            # All should be gone
            for i in range(3):
                assert cache.get(f"test_key_{i}") is None

    def test_cache_clear_pattern(self):
        """Test clearing specific dataset cache entries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir, ttl=3600)

            # Add entries for different datasets
            for dataset in ["Dataset A", "Dataset B"]:
                cache_key = f"key_{dataset.replace(' ', '_')}"
                current_time = time.time()
                entry = CacheEntry(
                    cache_key=cache_key,
                    dataset_name=dataset,
                    resource_name="Resource",
                    format="json",
                    url="https://example.com",
                    cached_at=current_time,
                    last_accessed=current_time,
                    ttl=3600,
                    size_bytes=0,
                    query_params=None,
                )
                cache.set(cache_key, {"dataset": dataset}, entry)

            # Clear only Dataset A
            cache.clear(pattern="Dataset A")

            # Dataset A should be gone, Dataset B should remain
            assert cache.get("key_Dataset_A") is None
            assert cache.get("key_Dataset_B") is not None

    def test_cache_info(self):
        """Test cache info statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir, ttl=3600)

            cache_key = "test_key"
            current_time = time.time()
            entry = CacheEntry(
                cache_key=cache_key,
                dataset_name="Dataset",
                resource_name="Resource",
                format="json",
                url="https://example.com",
                cached_at=current_time,
                last_accessed=current_time,
                ttl=3600,
                size_bytes=0,
                query_params=None,
            )
            cache.set(cache_key, {"test": "data"}, entry)

            info = cache.info()
            assert info["entries"] == 1
            assert "total_size_mb" in info
            assert "cache_dir" in info

    def test_corrupted_cache_file(self):
        """Test that corrupted cache files are handled gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir, ttl=3600)

            cache_key = "test_key"
            current_time = time.time()
            entry = CacheEntry(
                cache_key=cache_key,
                dataset_name="Dataset",
                resource_name="Resource",
                format="json",
                url="https://example.com",
                cached_at=current_time,
                last_accessed=current_time,
                ttl=3600,
                size_bytes=0,
                query_params=None,
            )
            cache.set(cache_key, {"test": "data"}, entry)

            # Corrupt the cache file
            data_file = cache.cache_dir / f"{cache_key}.data"
            with open(data_file, "wb") as f:
                f.write(b"corrupted data")

            # Should return None and delete the corrupted entry
            result = cache.get(cache_key)
            assert result is None
            assert cache_key not in cache.metadata

    def test_unpickleable_data(self):
        """Test that unpickleable data is handled gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir, ttl=3600)

            cache_key = "test_key"
            current_time = time.time()
            entry = CacheEntry(
                cache_key=cache_key,
                dataset_name="Dataset",
                resource_name="Resource",
                format="json",
                url="https://example.com",
                cached_at=current_time,
                last_accessed=current_time,
                ttl=3600,
                size_bytes=0,
                query_params=None,
            )

            # Try to cache unpickleable data (lambda function)
            unpickleable_data = {"func": lambda x: x}
            cache.set(cache_key, unpickleable_data, entry)

            # Should not have cached the data
            assert cache_key not in cache.metadata

    def test_lru_eviction(self):
        """Test that LRU eviction works correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache with 1KB limit
            cache = FileCache(cache_dir=tmpdir, ttl=3600, max_size_mb=0.001)

            # Add 3 entries
            for i in range(3):
                cache_key = f"key_{i}"
                # Create larger data (approx 500 bytes each)
                test_data = {"data": "x" * 400, "index": i}
                current_time = time.time()
                entry = CacheEntry(
                    cache_key=cache_key,
                    dataset_name=f"Dataset {i}",
                    resource_name="Resource",
                    format="json",
                    url="https://example.com",
                    cached_at=current_time,
                    last_accessed=current_time,
                    ttl=3600,
                    size_bytes=0,
                    query_params=None,
                )
                cache.set(cache_key, test_data, entry)
                time.sleep(0.01)  # Ensure different timestamps

            # Access key_0 to make it more recently used
            cache.get("key_0")

            # Add another entry, should evict key_1 (least recently used)
            cache_key = "key_3"
            current_time = time.time()
            entry = CacheEntry(
                cache_key=cache_key,
                dataset_name="Dataset 3",
                resource_name="Resource",
                format="json",
                url="https://example.com",
                cached_at=current_time,
                last_accessed=current_time,
                ttl=3600,
                size_bytes=0,
                query_params=None,
            )
            cache.set(cache_key, {"data": "x" * 400, "index": 3}, entry)

            # key_1 should be evicted, key_0 and key_2 should remain
            assert "key_0" in cache.metadata  # Recently accessed
            assert "key_1" not in cache.metadata  # Evicted (LRU)


class TestPhillyCaching:
    """Test Philly class caching functionality"""

    def test_cache_enabled_by_default(self):
        """Test that caching is enabled by default"""
        phl = Philly()
        assert phl.cache_enabled is True
        assert phl._cache is not None

    def test_cache_disabled(self):
        """Test disabling cache"""
        phl = Philly(cache=False)
        assert phl.cache_enabled is False
        assert phl._cache is None

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test that second load uses cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            phl = Philly(cache=True, cache_dir=tmpdir)

            # Mock a dataset and resource
            with patch.object(phl, "_get_dataset") as mock_get_dataset:
                mock_dataset = MagicMock()
                mock_resource = MagicMock()
                mock_resource.url = "https://example.com/data.csv"
                mock_resource.format = "CSV"
                mock_resource.name = "Test Resource"
                mock_dataset.get_resource.return_value = mock_resource
                mock_get_dataset.return_value = mock_dataset

                # Mock the load function
                with patch("philly.philly.load", new_callable=AsyncMock) as mock_load:
                    mock_load.return_value = [{"test": "data"}]

                    # First load - should hit network
                    data1 = await phl.load("Test Dataset", "Test Resource")
                    assert mock_load.call_count == 1

                    # Second load - should use cache
                    data2 = await phl.load("Test Dataset", "Test Resource")
                    assert mock_load.call_count == 1  # Still 1, didn't call again

                    # Both should return same data
                    assert data1 == data2

    @pytest.mark.asyncio
    async def test_cache_bypass(self):
        """Test that cache=False forces network request"""
        with tempfile.TemporaryDirectory() as tmpdir:
            phl = Philly(cache=True, cache_dir=tmpdir)

            with patch.object(phl, "_get_dataset") as mock_get_dataset:
                mock_dataset = MagicMock()
                mock_resource = MagicMock()
                mock_resource.url = "https://example.com/data.csv"
                mock_resource.format = "CSV"
                mock_resource.name = "Test Resource"
                mock_dataset.get_resource.return_value = mock_resource
                mock_get_dataset.return_value = mock_dataset

                with patch("philly.philly.load", new_callable=AsyncMock) as mock_load:
                    mock_load.return_value = [{"test": "data"}]

                    # First load
                    await phl.load("Test Dataset", "Test Resource")
                    assert mock_load.call_count == 1

                    # Second load with use_cache=False
                    await phl.load("Test Dataset", "Test Resource", use_cache=False)
                    assert mock_load.call_count == 2  # Called again

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test cache clearing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            phl = Philly(cache=True, cache_dir=tmpdir)

            with patch.object(phl, "_get_dataset") as mock_get_dataset:
                mock_dataset = MagicMock()
                mock_resource = MagicMock()
                mock_resource.url = "https://example.com/data.csv"
                mock_resource.format = "CSV"
                mock_resource.name = "Test Resource"
                mock_dataset.get_resource.return_value = mock_resource
                mock_get_dataset.return_value = mock_dataset

                with patch("philly.philly.load", new_callable=AsyncMock) as mock_load:
                    mock_load.return_value = [{"test": "data"}]

                    # Load and cache
                    await phl.load("Test Dataset", "Test Resource")
                    assert mock_load.call_count == 1

                    # Clear cache
                    phl.cache_clear()

                    # Should reload from network
                    await phl.load("Test Dataset", "Test Resource")
                    assert mock_load.call_count == 2

    def test_cache_info(self):
        """Test cache info retrieval"""
        with tempfile.TemporaryDirectory() as tmpdir:
            phl = Philly(cache=True, cache_dir=tmpdir)
            info = phl.cache_info()
            assert "entries" in info
            assert "total_size_mb" in info
            assert "cache_dir" in info

    def test_cache_methods_error_when_disabled(self):
        """Test that cache methods raise error when cache is disabled"""
        phl = Philly(cache=False)

        with pytest.raises(RuntimeError):
            phl.cache_clear()

        with pytest.raises(RuntimeError):
            phl.cache_info()

    @pytest.mark.asyncio
    async def test_different_query_params_different_cache(self):
        """Test that different query params create different cache entries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            phl = Philly(cache=True, cache_dir=tmpdir)

            with patch.object(phl, "_get_dataset") as mock_get_dataset:
                mock_dataset = MagicMock()
                mock_resource = MagicMock()
                mock_resource.url = "https://example.com/data.csv"
                mock_resource.format = "CSV"
                mock_resource.name = "Test Resource"
                mock_dataset.get_resource.return_value = mock_resource
                mock_get_dataset.return_value = mock_dataset

                with patch("philly.philly.load", new_callable=AsyncMock) as mock_load:
                    # Mock different responses for different calls
                    mock_load.side_effect = [
                        [{"where": "A"}],
                        [{"where": "B"}],
                    ]

                    # Load with different params
                    data1 = await phl.load("Test Dataset", "Test Resource", where="A")
                    data2 = await phl.load("Test Dataset", "Test Resource", where="B")

                    # Should have made 2 network requests
                    assert mock_load.call_count == 2

                    # Data should be different
                    assert data1 != data2
