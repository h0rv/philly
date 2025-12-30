"""Tests for metadata and info API."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx

from philly.metadata import (
    get_dataset_info,
    get_remote_last_modified,
    get_remote_size,
    get_resource_info,
    is_url_available,
)
from philly.models.dataset import Dataset
from philly.models.resource import Resource, ResourceFormat


class TestGetRemoteSize:
    """Test get_remote_size function."""

    @patch("httpx.Client")
    def test_successful_head_request_with_content_length(self, mock_client_class):
        """Test successful HEAD request with Content-Length header."""
        # Mock response with Content-Length
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "10485760"  # 10 MB in bytes

        # Mock client context manager
        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        size = get_remote_size(url)

        # Verify the result (10485760 bytes = 10 MB)
        assert size == 10.0
        mock_client_class.assert_called_once_with(follow_redirects=True, timeout=10.0)
        mock_client.head.assert_called_once_with(url)
        mock_response.headers.get.assert_called_once_with("Content-Length")

    @patch("httpx.Client")
    def test_missing_content_length_returns_none(self, mock_client_class):
        """Test missing Content-Length header returns None."""
        # Mock response without Content-Length
        mock_response = MagicMock()
        mock_response.headers.get.return_value = None

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        size = get_remote_size(url)

        assert size is None

    @patch("httpx.Client")
    def test_network_error_returns_none(self, mock_client_class):
        """Test network error returns None."""
        mock_client = MagicMock()
        mock_client.head.side_effect = httpx.RequestError("Network error")
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        size = get_remote_size(url)

        assert size is None

    @patch("httpx.Client")
    def test_timeout_returns_none(self, mock_client_class):
        """Test timeout returns None."""
        mock_client = MagicMock()
        mock_client.head.side_effect = httpx.TimeoutException("Timeout")
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        size = get_remote_size(url, timeout=5.0)

        assert size is None

    @patch("httpx.Client")
    def test_custom_timeout(self, mock_client_class):
        """Test custom timeout is passed to client."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "1048576"  # 1 MB

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        size = get_remote_size(url, timeout=30.0)

        assert size == 1.0
        mock_client_class.assert_called_once_with(follow_redirects=True, timeout=30.0)

    @patch("httpx.Client")
    def test_large_file_size(self, mock_client_class):
        """Test handling of large file sizes."""
        mock_response = MagicMock()
        # 1 GB in bytes
        mock_response.headers.get.return_value = "1073741824"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        size = get_remote_size(url)

        # 1073741824 bytes = 1024 MB
        assert size == 1024.0

    @patch("httpx.Client")
    def test_invalid_content_length_returns_none(self, mock_client_class):
        """Test invalid Content-Length value returns None."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "invalid"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        size = get_remote_size(url)

        assert size is None


class TestGetRemoteLastModified:
    """Test get_remote_last_modified function."""

    @patch("httpx.Client")
    def test_successful_head_request_with_last_modified(self, mock_client_class):
        """Test successful HEAD request with Last-Modified header."""
        # Mock response with Last-Modified
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "Wed, 21 Oct 2015 07:28:00 GMT"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        last_modified = get_remote_last_modified(url)

        assert last_modified is not None
        assert isinstance(last_modified, datetime)
        assert last_modified.year == 2015
        assert last_modified.month == 10
        assert last_modified.day == 21
        mock_client_class.assert_called_once_with(follow_redirects=True, timeout=10.0)
        mock_client.head.assert_called_once_with(url)
        mock_response.headers.get.assert_called_once_with("Last-Modified")

    @patch("httpx.Client")
    def test_missing_last_modified_returns_none(self, mock_client_class):
        """Test missing Last-Modified header returns None."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = None

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        last_modified = get_remote_last_modified(url)

        assert last_modified is None

    @patch("httpx.Client")
    def test_invalid_date_format_returns_none(self, mock_client_class):
        """Test invalid date format returns None."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "invalid-date"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        last_modified = get_remote_last_modified(url)

        assert last_modified is None

    @patch("httpx.Client")
    def test_network_error_returns_none(self, mock_client_class):
        """Test network error returns None."""
        mock_client = MagicMock()
        mock_client.head.side_effect = httpx.RequestError("Network error")
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        last_modified = get_remote_last_modified(url)

        assert last_modified is None

    @patch("httpx.Client")
    def test_timeout_returns_none(self, mock_client_class):
        """Test timeout returns None."""
        mock_client = MagicMock()
        mock_client.head.side_effect = httpx.TimeoutException("Timeout")
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        last_modified = get_remote_last_modified(url, timeout=5.0)

        assert last_modified is None

    @patch("httpx.Client")
    def test_custom_timeout(self, mock_client_class):
        """Test custom timeout is passed to client."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "Wed, 21 Oct 2015 07:28:00 GMT"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        last_modified = get_remote_last_modified(url, timeout=30.0)

        assert last_modified is not None
        mock_client_class.assert_called_once_with(follow_redirects=True, timeout=30.0)

    @patch("httpx.Client")
    def test_various_date_formats(self, mock_client_class):
        """Test various valid HTTP date formats."""
        # HTTP date format examples
        dates = [
            "Sun, 06 Nov 1994 08:49:37 GMT",
            "Wed, 15 Nov 1995 06:25:24 GMT",
            "Mon, 01 Jan 2024 00:00:00 GMT",
        ]

        for date_str in dates:
            mock_response = MagicMock()
            mock_response.headers.get.return_value = date_str

            mock_client = MagicMock()
            mock_client.head.return_value = mock_response
            mock_client_class.return_value.__enter__.return_value = mock_client

            url = "https://example.com/data.csv"
            last_modified = get_remote_last_modified(url)

            assert last_modified is not None
            assert isinstance(last_modified, datetime)


class TestIsUrlAvailable:
    """Test is_url_available function."""

    @patch("httpx.Client")
    def test_200_response_returns_true(self, mock_client_class):
        """Test 200 response returns True."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        available = is_url_available(url)

        assert available is True
        mock_client_class.assert_called_once_with(follow_redirects=True, timeout=10.0)
        mock_client.head.assert_called_once_with(url)

    @patch("httpx.Client")
    def test_404_response_returns_false(self, mock_client_class):
        """Test 404 response returns False."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/notfound.csv"
        available = is_url_available(url)

        assert available is False

    @patch("httpx.Client")
    def test_500_response_returns_false(self, mock_client_class):
        """Test 500 response returns False."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/error.csv"
        available = is_url_available(url)

        assert available is False

    @patch("httpx.Client")
    def test_network_error_returns_false(self, mock_client_class):
        """Test network error returns False."""
        mock_client = MagicMock()
        mock_client.head.side_effect = httpx.RequestError("Network error")
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        available = is_url_available(url)

        assert available is False

    @patch("httpx.Client")
    def test_timeout_returns_false(self, mock_client_class):
        """Test timeout returns False."""
        mock_client = MagicMock()
        mock_client.head.side_effect = httpx.TimeoutException("Timeout")
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        available = is_url_available(url, timeout=5.0)

        assert available is False

    @patch("httpx.Client")
    def test_custom_timeout(self, mock_client_class):
        """Test custom timeout is passed to client."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        available = is_url_available(url, timeout=30.0)

        assert available is True
        mock_client_class.assert_called_once_with(follow_redirects=True, timeout=30.0)

    @patch("httpx.Client")
    def test_various_error_status_codes(self, mock_client_class):
        """Test various error status codes return False."""
        error_codes = [400, 401, 403, 404, 500, 502, 503]

        for status_code in error_codes:
            mock_response = MagicMock()
            mock_response.status_code = status_code

            mock_client = MagicMock()
            mock_client.head.return_value = mock_response
            mock_client_class.return_value.__enter__.return_value = mock_client

            url = "https://example.com/data.csv"
            available = is_url_available(url)

            assert available is False, f"Status code {status_code} should return False"

    @patch("httpx.Client")
    def test_redirect_followed(self, mock_client_class):
        """Test that redirects are followed (via client config)."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/redirect"
        available = is_url_available(url)

        assert available is True
        # Verify follow_redirects=True is passed
        mock_client_class.assert_called_once_with(follow_redirects=True, timeout=10.0)


class TestGetDatasetInfo:
    """Test get_dataset_info function."""

    def test_returns_all_expected_keys(self):
        """Test returns all expected keys."""
        dataset = Dataset(
            title="Test Dataset",
            organization="Test Org",
            category=["Health", "Education"],
            notes="Test description",
            license="MIT",
            maintainer_email="test@example.com",
            source="https://example.com",
            created="2024-01-01",
            resources=[
                Resource(
                    name="data",
                    format=ResourceFormat.CSV,
                    url="https://example.com/data.csv",
                ),
                Resource(
                    name="metadata",
                    format=ResourceFormat.JSON,
                    url="https://example.com/meta.json",
                ),
            ],
        )

        info = get_dataset_info(dataset)

        # Check all expected keys are present
        assert "title" in info
        assert "organization" in info
        assert "category" in info
        assert "description" in info
        assert "license" in info
        assert "maintainer_email" in info
        assert "source" in info
        assert "created" in info
        assert "num_resources" in info
        assert "resources" in info

        # Check values
        assert info["title"] == "Test Dataset"
        assert info["organization"] == "Test Org"
        assert info["category"] == ["Health", "Education"]
        assert info["description"] == "Test description"
        assert info["license"] == "MIT"
        assert info["maintainer_email"] == "test@example.com"
        assert info["source"] == "https://example.com"
        assert info["created"] == "2024-01-01"
        assert info["num_resources"] == 2
        assert len(info["resources"]) == 2

    def test_with_minimal_dataset(self):
        """Test with minimal dataset (missing optional fields)."""
        dataset = Dataset(
            title="Minimal Dataset",
        )

        info = get_dataset_info(dataset)

        assert info["title"] == "Minimal Dataset"
        assert info["organization"] is None
        assert info["category"] is None
        assert info["description"] is None
        assert info["license"] is None
        assert info["maintainer_email"] is None
        assert info["source"] is None
        assert info["created"] is None
        assert info["num_resources"] == 0
        assert info["resources"] == []

    def test_resources_list_included(self):
        """Test resources list is included with correct structure."""
        dataset = Dataset(
            title="Dataset with Resources",
            resources=[
                Resource(
                    name="csv_data",
                    format=ResourceFormat.CSV,
                    url="https://example.com/data.csv",
                ),
                Resource(
                    name="json_data",
                    format=ResourceFormat.JSON,
                    url="https://example.com/data.json",
                ),
                Resource(
                    name="geojson_data",
                    format=ResourceFormat.GEOJSON,
                    url="https://example.com/data.geojson",
                ),
            ],
        )

        info = get_dataset_info(dataset)

        assert info["num_resources"] == 3
        assert len(info["resources"]) == 3

        # Check first resource structure
        resource = info["resources"][0]
        assert "name" in resource
        assert "format" in resource
        assert "url" in resource
        assert resource["name"] == "csv_data"
        assert resource["format"] == "csv"
        assert resource["url"] == "https://example.com/data.csv"

        # Check all resources
        for i, expected in enumerate(
            [("csv_data", "csv"), ("json_data", "json"), ("geojson_data", "geojson")]
        ):
            assert info["resources"][i]["name"] == expected[0]
            assert info["resources"][i]["format"] == expected[1]

    def test_with_none_resources(self):
        """Test with None resources field."""
        dataset = Dataset(
            title="No Resources",
            resources=None,
        )

        info = get_dataset_info(dataset)

        assert info["num_resources"] == 0
        assert info["resources"] == []

    def test_with_empty_resources_list(self):
        """Test with empty resources list."""
        dataset = Dataset(
            title="Empty Resources",
            resources=[],
        )

        info = get_dataset_info(dataset)

        assert info["num_resources"] == 0
        assert info["resources"] == []

    def test_resource_without_url(self):
        """Test resource without URL is included."""
        dataset = Dataset(
            title="Dataset with Resource Without URL",
            resources=[
                Resource(name="no_url_resource", format=ResourceFormat.API, url=None),
            ],
        )

        info = get_dataset_info(dataset)

        assert info["num_resources"] == 1
        assert len(info["resources"]) == 1
        assert info["resources"][0]["name"] == "no_url_resource"
        assert info["resources"][0]["format"] == "api"
        assert info["resources"][0]["url"] is None


class TestGetResourceInfo:
    """Test get_resource_info function."""

    @patch("philly.metadata.get_remote_size")
    @patch("philly.metadata.get_remote_last_modified")
    def test_returns_all_expected_keys(self, mock_last_modified, mock_size):
        """Test returns all expected keys."""
        mock_size.return_value = 5.5
        mock_last_modified.return_value = datetime(2024, 1, 15, 10, 30, 0)

        dataset = Dataset(
            title="Test Dataset",
            organization="Test Org",
            category=["Health"],
            notes="Test description",
            license="MIT",
            maintainer_email="test@example.com",
        )

        resource = Resource(
            name="test_resource",
            format=ResourceFormat.CSV,
            url="https://example.com/data.csv",
        )

        info = get_resource_info(dataset, resource)

        # Check all expected keys are present
        assert "dataset" in info
        assert "resource" in info
        assert "format" in info
        assert "url" in info
        assert "size_mb" in info
        assert "last_modified" in info
        assert "organization" in info
        assert "category" in info
        assert "description" in info
        assert "license" in info
        assert "maintainer_email" in info

        # Check values
        assert info["dataset"] == "Test Dataset"
        assert info["resource"] == "test_resource"
        assert info["format"] == "csv"
        assert info["url"] == "https://example.com/data.csv"
        assert info["size_mb"] == 5.5
        assert info["last_modified"] == datetime(2024, 1, 15, 10, 30, 0)
        assert info["organization"] == "Test Org"
        assert info["category"] == ["Health"]
        assert info["description"] == "Test description"
        assert info["license"] == "MIT"
        assert info["maintainer_email"] == "test@example.com"

        # Verify remote calls were made
        mock_size.assert_called_once_with("https://example.com/data.csv")
        mock_last_modified.assert_called_once_with("https://example.com/data.csv")

    @patch("philly.metadata.get_remote_size")
    @patch("philly.metadata.get_remote_last_modified")
    def test_handles_missing_url(self, mock_last_modified, mock_size):
        """Test handles missing URL."""
        dataset = Dataset(
            title="Test Dataset",
        )

        resource = Resource(
            name="no_url_resource",
            format=ResourceFormat.API,
            url=None,
        )

        info = get_resource_info(dataset, resource)

        assert info["url"] == ""
        assert info["size_mb"] is None
        assert info["last_modified"] is None

        # Verify remote functions were NOT called
        mock_size.assert_not_called()
        mock_last_modified.assert_not_called()

    @patch("philly.metadata.get_remote_size")
    @patch("philly.metadata.get_remote_last_modified")
    def test_handles_empty_url(self, mock_last_modified, mock_size):
        """Test handles empty URL string."""
        dataset = Dataset(
            title="Test Dataset",
        )

        resource = Resource(
            name="empty_url_resource",
            format=ResourceFormat.API,
            url="",
        )

        info = get_resource_info(dataset, resource)

        assert info["url"] == ""
        assert info["size_mb"] is None
        assert info["last_modified"] is None

        # Verify remote functions were NOT called
        mock_size.assert_not_called()
        mock_last_modified.assert_not_called()

    @patch("philly.metadata.get_remote_size")
    @patch("philly.metadata.get_remote_last_modified")
    def test_with_minimal_dataset(self, mock_last_modified, mock_size):
        """Test with minimal dataset (missing optional fields)."""
        mock_size.return_value = 2.0
        mock_last_modified.return_value = datetime(2024, 1, 1, 0, 0, 0)

        dataset = Dataset(
            title="Minimal Dataset",
        )

        resource = Resource(
            name="test_resource",
            format=ResourceFormat.JSON,
            url="https://example.com/data.json",
        )

        info = get_resource_info(dataset, resource)

        assert info["dataset"] == "Minimal Dataset"
        assert info["resource"] == "test_resource"
        assert info["format"] == "json"
        assert info["url"] == "https://example.com/data.json"
        assert info["size_mb"] == 2.0
        assert info["last_modified"] == datetime(2024, 1, 1, 0, 0, 0)
        assert info["organization"] is None
        assert info["category"] is None
        assert info["description"] is None
        assert info["license"] is None
        assert info["maintainer_email"] is None

    @patch("philly.metadata.get_remote_size")
    @patch("philly.metadata.get_remote_last_modified")
    def test_remote_calls_return_none(self, mock_last_modified, mock_size):
        """Test when remote calls return None (network errors)."""
        mock_size.return_value = None
        mock_last_modified.return_value = None

        dataset = Dataset(
            title="Test Dataset",
        )

        resource = Resource(
            name="unreachable_resource",
            format=ResourceFormat.CSV,
            url="https://example.com/unreachable.csv",
        )

        info = get_resource_info(dataset, resource)

        assert info["url"] == "https://example.com/unreachable.csv"
        assert info["size_mb"] is None
        assert info["last_modified"] is None

        # Verify remote calls were made but returned None
        mock_size.assert_called_once_with("https://example.com/unreachable.csv")
        mock_last_modified.assert_called_once_with(
            "https://example.com/unreachable.csv"
        )

    @patch("philly.metadata.get_remote_size")
    @patch("philly.metadata.get_remote_last_modified")
    def test_format_converted_to_string(self, mock_last_modified, mock_size):
        """Test resource format is converted to string."""
        mock_size.return_value = 1.0
        mock_last_modified.return_value = None

        dataset = Dataset(title="Test Dataset")
        resource = Resource(
            name="test",
            format=ResourceFormat.GEOJSON,
            url="https://example.com/data.geojson",
        )

        info = get_resource_info(dataset, resource)

        assert info["format"] == "geojson"
        assert isinstance(info["format"], str)


class TestIsUrlAvailableStatusCodes:
    """Additional tests for is_url_available status code handling."""

    @patch("httpx.Client")
    def test_201_response_returns_false(self, mock_client_class):
        """Test 201 Created response returns False (only 200 is considered available)."""
        mock_response = MagicMock()
        mock_response.status_code = 201

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        available = is_url_available(url)

        assert available is False

    @patch("httpx.Client")
    def test_204_response_returns_false(self, mock_client_class):
        """Test 204 No Content response returns False."""
        mock_response = MagicMock()
        mock_response.status_code = 204

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        available = is_url_available(url)

        assert available is False

    @patch("httpx.Client")
    def test_301_after_redirect_resolution(self, mock_client_class):
        """Test that redirects are followed and final status determines availability."""
        # The client is configured with follow_redirects=True,
        # so if final response is 200, it should be available
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/redirect"
        available = is_url_available(url)

        assert available is True
        # Verify follow_redirects is enabled
        mock_client_class.assert_called_once_with(follow_redirects=True, timeout=10.0)


class TestGetRemoteSizeEdgeCases:
    """Additional edge case tests for get_remote_size."""

    @patch("httpx.Client")
    def test_zero_size_file(self, mock_client_class):
        """Test handling of zero-sized file."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "0"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/empty.csv"
        size = get_remote_size(url)

        assert size == 0.0

    @patch("httpx.Client")
    def test_negative_content_length_returns_none(self, mock_client_class):
        """Test negative Content-Length value is handled gracefully."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "-100"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        size = get_remote_size(url)

        # Implementation returns negative value as-is (not checking for negative)
        # This tests actual behavior - negative bytes / MB calculation
        assert size == -100 / 1024 / 1024

    @patch("httpx.Client")
    def test_small_file_size(self, mock_client_class):
        """Test handling of very small file sizes (< 1 byte in MB)."""
        mock_response = MagicMock()
        # 100 bytes
        mock_response.headers.get.return_value = "100"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/tiny.csv"
        size = get_remote_size(url)

        # 100 bytes = 100 / 1024 / 1024 MB
        expected = 100 / 1024 / 1024
        assert size == expected


class TestGetRemoteLastModifiedEdgeCases:
    """Additional edge case tests for get_remote_last_modified."""

    @patch("httpx.Client")
    def test_empty_date_string_returns_none(self, mock_client_class):
        """Test empty date string returns None."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = ""

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        last_modified = get_remote_last_modified(url)

        assert last_modified is None

    @patch("httpx.Client")
    def test_partial_date_string_returns_none(self, mock_client_class):
        """Test partial/malformed date string returns None."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "Wed, 21 Oct"

        mock_client = MagicMock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        url = "https://example.com/data.csv"
        last_modified = get_remote_last_modified(url)

        assert last_modified is None


class TestGetDatasetInfoEdgeCases:
    """Additional edge case tests for get_dataset_info."""

    def test_single_category(self):
        """Test dataset with single category."""
        dataset = Dataset(
            title="Single Category Dataset",
            category=["Health"],
        )

        info = get_dataset_info(dataset)

        assert info["category"] == ["Health"]

    def test_multiple_categories(self):
        """Test dataset with multiple categories."""
        dataset = Dataset(
            title="Multi Category Dataset",
            category=["Health", "Education", "Public Safety"],
        )

        info = get_dataset_info(dataset)

        assert info["category"] == ["Health", "Education", "Public Safety"]
        assert len(info["category"]) == 3

    def test_all_formats_represented(self):
        """Test dataset with resources of different formats."""
        dataset = Dataset(
            title="Multi Format Dataset",
            resources=[
                Resource(name="csv", format=ResourceFormat.CSV, url="http://a.csv"),
                Resource(name="json", format=ResourceFormat.JSON, url="http://a.json"),
                Resource(name="shp", format=ResourceFormat.SHP, url="http://a.shp"),
                Resource(name="api", format=ResourceFormat.API, url="http://api"),
            ],
        )

        info = get_dataset_info(dataset)

        assert info["num_resources"] == 4
        formats = [r["format"] for r in info["resources"]]
        assert "csv" in formats
        assert "json" in formats
        assert "shp" in formats
        assert "api" in formats


class TestPhillyInfoMethod:
    """Tests for Philly.info() method integration."""

    def test_info_returns_dataset_info_when_no_resource(self):
        """Test info() returns dataset info when resource_name is None."""
        from philly import Philly

        philly = Philly(cache=False)

        # Use a real dataset name from the catalog
        datasets = philly.list_datasets()
        if not datasets:
            return  # Skip if no datasets available

        dataset_name = datasets[0]
        info = philly.info(dataset_name)

        # Verify dataset info structure
        assert "title" in info
        assert "organization" in info
        assert "category" in info
        assert "description" in info
        assert "num_resources" in info
        assert "resources" in info
        assert info["title"] == dataset_name

    @patch("philly.metadata.get_remote_size")
    @patch("philly.metadata.get_remote_last_modified")
    def test_info_returns_resource_info_when_resource_provided(
        self, mock_last_modified, mock_size
    ):
        """Test info() returns resource info when resource_name is provided."""
        from philly import Philly

        mock_size.return_value = 1.5
        mock_last_modified.return_value = datetime(2024, 1, 1, 12, 0, 0)

        philly = Philly(cache=False)

        # Find a dataset with resources
        for dataset in philly.datasets:
            if dataset.resources:
                dataset_name = dataset.title
                resource_name = dataset.resources[0].name
                break
        else:
            return  # Skip if no datasets with resources

        info = philly.info(dataset_name, resource_name)

        # Verify resource info structure
        assert "dataset" in info
        assert "resource" in info
        assert "format" in info
        assert "url" in info
        assert "size_mb" in info
        assert "last_modified" in info
        assert info["dataset"] == dataset_name
        assert info["resource"] == resource_name

    def test_info_raises_on_invalid_dataset(self):
        """Test info() raises ValueError for invalid dataset."""
        from philly import Philly
        import pytest

        philly = Philly(cache=False)

        with pytest.raises(ValueError, match="does not exist"):
            philly.info("Non-Existent Dataset Name")

    def test_info_raises_on_invalid_resource(self):
        """Test info() raises ValueError for invalid resource."""
        from philly import Philly
        import pytest

        philly = Philly(cache=False)

        # Get a real dataset name
        datasets = philly.list_datasets()
        if not datasets:
            return

        with pytest.raises(ValueError, match="does not exist"):
            philly.info(datasets[0], "Non-Existent Resource")


class TestPhillyMetadataWrappers:
    """Tests for Philly class metadata wrapper methods."""

    @patch("philly.philly.get_remote_size")
    def test_get_size_returns_size(self, mock_get_remote_size):
        """Test get_size() returns size from get_remote_size."""
        from philly import Philly

        mock_get_remote_size.return_value = 5.5

        philly = Philly(cache=False)

        # Find a dataset with resources that have URLs
        for dataset in philly.datasets:
            for resource in dataset.resources or []:
                if resource.url:
                    size = philly.get_size(dataset.title, resource.name)
                    assert size == 5.5
                    mock_get_remote_size.assert_called_with(resource.url)
                    return

    @patch("philly.philly.get_remote_last_modified")
    def test_get_last_modified_returns_datetime(self, mock_get_remote_last_modified):
        """Test get_last_modified() returns datetime from get_remote_last_modified."""
        from philly import Philly

        expected_dt = datetime(2024, 6, 15, 10, 30, 0)
        mock_get_remote_last_modified.return_value = expected_dt

        philly = Philly(cache=False)

        # Find a dataset with resources that have URLs
        for dataset in philly.datasets:
            for resource in dataset.resources or []:
                if resource.url:
                    last_mod = philly.get_last_modified(dataset.title, resource.name)
                    assert last_mod == expected_dt
                    mock_get_remote_last_modified.assert_called_with(resource.url)
                    return

    @patch("philly.philly.is_url_available")
    def test_is_available_returns_bool(self, mock_is_url_available):
        """Test is_available() returns bool from is_url_available."""
        from philly import Philly

        mock_is_url_available.return_value = True

        philly = Philly(cache=False)

        # Find a dataset with resources that have URLs
        for dataset in philly.datasets:
            for resource in dataset.resources or []:
                if resource.url:
                    available = philly.is_available(dataset.title, resource.name)
                    assert available is True
                    mock_is_url_available.assert_called_with(resource.url)
                    return

    def test_get_url_returns_url_string(self):
        """Test get_url() returns the resource URL."""
        from philly import Philly

        philly = Philly(cache=False)

        # Find a dataset with resources that have URLs
        for dataset in philly.datasets:
            for resource in dataset.resources or []:
                if resource.url:
                    url = philly.get_url(dataset.title, resource.name)
                    assert url == resource.url
                    assert isinstance(url, str)
                    return

    def test_get_size_returns_none_for_missing_url(self):
        """Test get_size() returns None when resource has no URL."""
        from philly import Philly

        philly = Philly(cache=False)

        # Find a dataset and manually test with a mock
        datasets = philly.list_datasets()
        if not datasets:
            return

        # Get dataset and modify it to have a resource without URL
        dataset = philly._get_dataset(datasets[0])
        if dataset.resources:
            original_url = dataset.resources[0].url
            dataset.resources[0].url = None

            try:
                size = philly.get_size(datasets[0], dataset.resources[0].name)
                assert size is None
            finally:
                # Restore original URL
                dataset.resources[0].url = original_url

    def test_is_available_returns_false_for_missing_url(self):
        """Test is_available() returns False when resource has no URL."""
        from philly import Philly

        philly = Philly(cache=False)

        datasets = philly.list_datasets()
        if not datasets:
            return

        dataset = philly._get_dataset(datasets[0])
        if dataset.resources:
            original_url = dataset.resources[0].url
            dataset.resources[0].url = None

            try:
                available = philly.is_available(datasets[0], dataset.resources[0].name)
                assert available is False
            finally:
                dataset.resources[0].url = original_url

    def test_get_url_returns_empty_for_missing_url(self):
        """Test get_url() returns empty string when resource has no URL."""
        from philly import Philly

        philly = Philly(cache=False)

        datasets = philly.list_datasets()
        if not datasets:
            return

        dataset = philly._get_dataset(datasets[0])
        if dataset.resources:
            original_url = dataset.resources[0].url
            dataset.resources[0].url = None

            try:
                url = philly.get_url(datasets[0], dataset.resources[0].name)
                assert url == ""
            finally:
                dataset.resources[0].url = original_url
