import json
import logging
import os
import pickle
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    cache_key: str
    dataset_name: str
    resource_name: str
    format: str
    url: str
    cached_at: float
    last_accessed: float
    ttl: int
    size_bytes: int
    query_params: dict[str, Any] | None

    def is_expired(self) -> bool:
        return time.time() - self.cached_at > self.ttl

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(**data)


class FileCache:
    """Simple file-based cache with TTL and LRU eviction"""

    def __init__(
        self, cache_dir: str, ttl: int, max_size_mb: float | None = None
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.max_size_mb = max_size_mb
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata: dict[str, CacheEntry] = {}
        self._total_size_bytes: int = 0
        self._logger = logging.getLogger(__name__)
        self._load_metadata()
        # Calculate total size from loaded metadata
        self._total_size_bytes = sum(e.size_bytes for e in self.metadata.values())

    def _load_metadata(self) -> None:
        """Load metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    self.metadata = {
                        k: CacheEntry.from_dict(v) for k, v in data.items()
                    }
            except (json.JSONDecodeError, KeyError, TypeError):
                self.metadata = {}
        else:
            # Create empty metadata file
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Atomically save metadata to disk"""
        data = {k: v.to_dict() for k, v in self.metadata.items()}

        # Write to temporary file first
        fd, temp_path = tempfile.mkstemp(
            dir=self.cache_dir, prefix=".metadata-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure written to disk
            # Atomic rename
            os.replace(temp_path, self.metadata_file)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def get(self, cache_key: str) -> Any | None:
        """Get from cache if exists and not expired"""
        entry = self.metadata.get(cache_key)
        if not entry:
            return None

        if entry.is_expired():
            self.delete(cache_key)
            return None

        data_file = self.cache_dir / f"{cache_key}.data"
        if not data_file.exists():
            return None

        try:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            # Corrupted cache file - delete it
            self._logger.warning(f"Corrupted cache file {cache_key}: {e}")
            self.delete(cache_key)
            return None

        # Update last accessed time for LRU
        entry.last_accessed = time.time()
        self._save_metadata()

        return data

    def set(self, cache_key: str, data: Any, entry: CacheEntry) -> None:
        """Store in cache"""
        # Validate data is pickleable before attempting to cache
        try:
            pickle.dumps(data)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            self._logger.warning(f"Data not cacheable: {e}")
            return  # Skip caching, don't fail the operation

        # Track old size if updating existing entry
        old_size = 0
        if cache_key in self.metadata:
            old_size = self.metadata[cache_key].size_bytes

        if self.max_size_mb:
            self._enforce_size_limit()

        data_file = self.cache_dir / f"{cache_key}.data"
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

        entry.size_bytes = data_file.stat().st_size
        self.metadata[cache_key] = entry

        # Update total size
        self._total_size_bytes = self._total_size_bytes - old_size + entry.size_bytes

        self._save_metadata()

    def delete(self, cache_key: str) -> None:
        """Delete cache entry"""
        # Update total size before deleting
        if cache_key in self.metadata:
            self._total_size_bytes -= self.metadata[cache_key].size_bytes

        data_file = self.cache_dir / f"{cache_key}.data"
        data_file.unlink(missing_ok=True)
        self.metadata.pop(cache_key, None)
        self._save_metadata()

    def clear(self, pattern: str | None = None) -> None:
        """Clear cache entries matching pattern"""
        if pattern is None:
            for cache_key in list(self.metadata.keys()):
                self.delete(cache_key)
        else:
            for cache_key, entry in list(self.metadata.items()):
                if entry.dataset_name == pattern:
                    self.delete(cache_key)

    def info(
        self, dataset_name: str | None = None, resource_name: str | None = None
    ) -> dict[str, Any]:
        """Get cache statistics"""
        if dataset_name and resource_name:
            # Get info for specific resource
            matching_entries = [
                e
                for e in self.metadata.values()
                if e.dataset_name == dataset_name and e.resource_name == resource_name
            ]
            if not matching_entries:
                return {
                    "dataset": dataset_name,
                    "resource": resource_name,
                    "cached": False,
                }
            entry = matching_entries[0]
            return {
                "dataset": dataset_name,
                "resource": resource_name,
                "cached": True,
                "cached_at": entry.cached_at,
                "expires_at": entry.cached_at + entry.ttl,
                "expired": entry.is_expired(),
                "size_mb": entry.size_bytes / 1024 / 1024,
                "url": entry.url,
                "format": entry.format,
            }

        # Use cached total size for better performance
        expired = sum(1 for e in self.metadata.values() if e.is_expired())

        return {
            "entries": len(self.metadata),
            "total_size_mb": self._total_size_bytes / 1024 / 1024,
            "expired": expired,
            "cache_dir": str(self.cache_dir),
        }

    def _enforce_size_limit(self) -> None:
        """Enforce max cache size by removing least recently used entries (LRU)"""
        if not self.max_size_mb:
            return

        max_bytes = self.max_size_mb * 1024 * 1024

        if self._total_size_bytes <= max_bytes:
            return

        # Sort by last_accessed (least recently used first) for LRU eviction
        sorted_entries = sorted(self.metadata.items(), key=lambda x: x[1].last_accessed)

        for cache_key, _ in sorted_entries:
            self.delete(cache_key)
            if self._total_size_bytes <= max_bytes:
                break
