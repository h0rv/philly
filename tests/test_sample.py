"""Tests for sample/preview functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
from httpx import Response

from philly import sample


class TestGetColumnsFromSample:
    """Tests for get_columns_from_sample function."""

    def test_valid_sample_data(self):
        """Test with valid sample data returns column names."""
        sample_data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        columns = sample.get_columns_from_sample(sample_data)

        assert columns == ["name", "age"]

    def test_empty_sample(self):
        """Test with empty sample returns empty list."""
        columns = sample.get_columns_from_sample([])

        assert columns == []

    def test_column_order_preserved(self):
        """Test that column order is preserved from first record."""
        # Python 3.7+ dicts maintain insertion order
        sample_data = [{"z": 1, "a": 2, "m": 3}]
        columns = sample.get_columns_from_sample(sample_data)

        assert columns == ["z", "a", "m"]

    def test_single_record(self):
        """Test with single record."""
        sample_data = [{"single": "value"}]
        columns = sample.get_columns_from_sample(sample_data)

        assert columns == ["single"]

    def test_multiple_keys(self):
        """Test with many columns."""
        sample_data = [{"col" + str(i): i for i in range(10)}]
        columns = sample.get_columns_from_sample(sample_data)

        assert len(columns) == 10
        assert all(col.startswith("col") for col in columns)


class TestInferSchemaFromSample:
    """Tests for infer_schema_from_sample function."""

    def test_numeric_types(self):
        """Test inference of numeric types."""
        sample_data = [
            {"int_col": 1, "float_col": 1.5},
            {"int_col": 2, "float_col": 2.5},
        ]
        schema = sample.infer_schema_from_sample(sample_data)

        assert "int_col" in schema
        assert "float_col" in schema
        assert "int" in schema["int_col"]
        assert "float" in schema["float_col"]

    def test_string_types(self):
        """Test inference of string types."""
        sample_data = [{"name": "Alice", "city": "NYC"}, {"name": "Bob", "city": "LA"}]
        schema = sample.infer_schema_from_sample(sample_data)

        assert "name" in schema
        assert "city" in schema
        assert "object" in schema["name"]
        assert "object" in schema["city"]

    def test_mixed_types(self):
        """Test inference with mixed types."""
        sample_data = [
            {"id": 1, "name": "Alice", "score": 95.5, "active": True},
            {"id": 2, "name": "Bob", "score": 87.3, "active": False},
        ]
        schema = sample.infer_schema_from_sample(sample_data)

        assert len(schema) == 4
        assert "int" in schema["id"]
        assert "object" in schema["name"]
        assert "float" in schema["score"]
        assert "bool" in schema["active"]

    def test_nullable_detection(self):
        """Test that nullable columns are detected."""
        sample_data = [{"a": 1, "b": None}, {"a": 2, "b": None}]
        schema = sample.infer_schema_from_sample(sample_data)

        assert "a" in schema
        assert "b" in schema
        # None values result in object type in pandas
        assert "object" in schema["b"] or "float" in schema["b"]

    def test_empty_sample(self):
        """Test with empty sample returns empty dict."""
        schema = sample.infer_schema_from_sample([])

        assert schema == {}

    def test_single_record(self):
        """Test schema inference with single record."""
        sample_data = [{"x": 10, "y": "test"}]
        schema = sample.infer_schema_from_sample(sample_data)

        assert len(schema) == 2
        assert "int" in schema["x"]
        assert "object" in schema["y"]

    def test_all_columns_present(self):
        """Test that all columns are in schema."""
        sample_data = [{"col1": 1, "col2": 2, "col3": 3, "col4": 4}]
        schema = sample.infer_schema_from_sample(sample_data)

        assert len(schema) == 4
        assert all(f"col{i}" in schema for i in range(1, 5))


class TestFormatChunk:
    """Tests for format_chunk function."""

    def test_records_format_returns_list(self):
        """Test that 'records' format returns list of dicts."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = sample.format_chunk(data, "records")

        assert result == data
        assert isinstance(result, list)

    def test_dataframe_format_returns_dataframe(self):
        """Test that 'dataframe' format returns DataFrame."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = sample.format_chunk(data, "dataframe")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 3]
        assert result["b"].tolist() == [2, 4]

    def test_unknown_format_defaults_to_records(self):
        """Test that unknown format defaults to records."""
        data = [{"test": "value"}]
        result = sample.format_chunk(data, "unknown_format")

        assert result == data
        assert isinstance(result, list)

    def test_empty_data_records(self):
        """Test empty data with records format."""
        result = sample.format_chunk([], "records")

        assert result == []

    def test_empty_data_dataframe(self):
        """Test empty data with dataframe format."""
        result = sample.format_chunk([], "dataframe")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_dataframe_preserves_types(self):
        """Test that dataframe format preserves data types."""
        data = [{"int_col": 1, "str_col": "a"}, {"int_col": 2, "str_col": "b"}]
        result = sample.format_chunk(data, "dataframe")

        assert result["int_col"].dtype == "int64"
        assert result["str_col"].dtype == "object"


class TestSampleJson:
    """Tests for sample_json function with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_plain_array_response(self):
        """Test JSON response that is a plain array."""
        mock_data = [{"id": 1}, {"id": 2}, {"id": 3}]

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 2)

            assert len(result) == 2
            assert result == [{"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_dict_with_data_key(self):
        """Test JSON response with 'data' key containing array."""
        mock_data = {"data": [{"id": 1}, {"id": 2}, {"id": 3}], "meta": {}}

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 2)

            assert len(result) == 2
            assert result == [{"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_dict_with_features_key(self):
        """Test GeoJSON-like response with 'features' key."""
        mock_data = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "id": 1}, {"type": "Feature", "id": 2}],
        }

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 1)

            assert len(result) == 1
            assert result[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_dict_with_results_key(self):
        """Test JSON response with 'results' key containing array."""
        mock_data = {"results": [{"name": "A"}, {"name": "B"}], "count": 2}

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 10)

            assert len(result) == 2
            assert result == [{"name": "A"}, {"name": "B"}]

    @pytest.mark.asyncio
    async def test_dict_with_records_key(self):
        """Test JSON response with 'records' key containing array."""
        mock_data = {"records": [{"value": 1}, {"value": 2}, {"value": 3}]}

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 2)

            assert len(result) == 2
            assert result == [{"value": 1}, {"value": 2}]

    @pytest.mark.asyncio
    async def test_single_dict_response(self):
        """Test JSON response that is a single dict (not array)."""
        mock_data = {"status": "ok", "message": "success"}

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 5)

            # Single dict wrapped in list
            assert len(result) == 1
            assert result[0] == mock_data

    @pytest.mark.asyncio
    async def test_empty_array_response(self):
        """Test JSON response that is an empty array."""
        mock_data = []

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 5)

            assert result == []

    @pytest.mark.asyncio
    async def test_url_normalization(self):
        """Test that URLs are normalized properly."""
        mock_data = [{"test": "data"}]

        with patch("philly.sample.normalize_url") as mock_normalize:
            mock_normalize.return_value = "https://normalized.com/data.json"

            with patch("philly.sample.httpx.AsyncClient") as mock_client:
                mock_response = Mock(spec=Response)
                mock_response.json.return_value = mock_data
                mock_response.raise_for_status = Mock()

                mock_get = AsyncMock(return_value=mock_response)
                mock_client.return_value.__aenter__.return_value.get = mock_get

                await sample.sample_json("http://example.com/data.json", 5)

                # Verify normalize_url was called
                mock_normalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_to_https_upgrade(self):
        """Test that http:// URLs are upgraded to https://."""
        mock_data = [{"test": "data"}]

        with patch("philly.sample.normalize_url") as mock_normalize:
            mock_normalize.return_value = "http://example.com/data.json"

            with patch("philly.sample.httpx.AsyncClient") as mock_client:
                mock_response = Mock(spec=Response)
                mock_response.json.return_value = mock_data
                mock_response.raise_for_status = Mock()

                mock_get = AsyncMock(return_value=mock_response)
                mock_client.return_value.__aenter__.return_value.get = mock_get

                await sample.sample_json("http://example.com/data.json", 5)

                # Verify the URL was upgraded to https
                mock_get.assert_called_once()
                called_url = mock_get.call_args[0][0]
                assert called_url.startswith("https://")

    @pytest.mark.asyncio
    async def test_non_list_non_dict_response_returns_empty(self):
        """Test JSON response that is neither list nor dict returns empty list."""
        # JSON can return primitives like strings, numbers, null
        mock_data = "just a string"

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 5)

            assert result == []

    @pytest.mark.asyncio
    async def test_null_response_returns_empty(self):
        """Test JSON response that is null returns empty list."""
        mock_data = None

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 5)

            assert result == []

    @pytest.mark.asyncio
    async def test_numeric_response_returns_empty(self):
        """Test JSON response that is a number returns empty list."""
        mock_data = 42

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_json("https://example.com/data.json", 5)

            assert result == []


class TestSampleGeojson:
    """Tests for sample_geojson function with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_features_extraction(self):
        """Test extraction of features from GeoJSON."""
        mock_data = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"id": 1}},
                {"type": "Feature", "properties": {"id": 2}},
                {"type": "Feature", "properties": {"id": 3}},
            ],
        }

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_geojson("https://example.com/data.geojson", 2)

            assert len(result) == 2
            assert result[0]["properties"]["id"] == 1
            assert result[1]["properties"]["id"] == 2

    @pytest.mark.asyncio
    async def test_empty_features(self):
        """Test GeoJSON with empty features array."""
        mock_data = {"type": "FeatureCollection", "features": []}

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_geojson("https://example.com/data.geojson", 5)

            assert result == []

    @pytest.mark.asyncio
    async def test_n_limit(self):
        """Test that n limit is applied correctly."""
        features = [{"type": "Feature", "id": i} for i in range(10)]
        mock_data = {"type": "FeatureCollection", "features": features}

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_geojson("https://example.com/data.geojson", 3)

            assert len(result) == 3
            assert result[0]["id"] == 0
            assert result[1]["id"] == 1
            assert result[2]["id"] == 2

    @pytest.mark.asyncio
    async def test_missing_features_key(self):
        """Test GeoJSON without 'features' key returns empty list."""
        mock_data = {"type": "FeatureCollection", "other": "data"}

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_geojson("https://example.com/data.geojson", 5)

            assert result == []

    @pytest.mark.asyncio
    async def test_features_not_array(self):
        """Test GeoJSON where 'features' is not an array."""
        mock_data = {"type": "FeatureCollection", "features": "not an array"}

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock(spec=Response)
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await sample.sample_geojson("https://example.com/data.geojson", 5)

            assert result == []

    @pytest.mark.asyncio
    async def test_url_normalization(self):
        """Test that URLs are normalized."""
        mock_data = {"type": "FeatureCollection", "features": []}

        with patch("philly.sample.normalize_url") as mock_normalize:
            mock_normalize.return_value = "https://normalized.com/data.geojson"

            with patch("philly.sample.httpx.AsyncClient") as mock_client:
                mock_response = Mock(spec=Response)
                mock_response.json.return_value = mock_data
                mock_response.raise_for_status = Mock()

                mock_get = AsyncMock(return_value=mock_response)
                mock_client.return_value.__aenter__.return_value.get = mock_get

                await sample.sample_geojson("http://example.com/data.geojson", 5)

                mock_normalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_to_https_upgrade(self):
        """Test that http:// URLs are upgraded to https://."""
        mock_data = {"type": "FeatureCollection", "features": [{"id": 1}]}

        with patch("philly.sample.normalize_url") as mock_normalize:
            mock_normalize.return_value = "http://example.com/data.geojson"

            with patch("philly.sample.httpx.AsyncClient") as mock_client:
                mock_response = Mock(spec=Response)
                mock_response.json.return_value = mock_data
                mock_response.raise_for_status = Mock()

                mock_get = AsyncMock(return_value=mock_response)
                mock_client.return_value.__aenter__.return_value.get = mock_get

                await sample.sample_geojson("http://example.com/data.geojson", 5)

                # Verify the URL was upgraded to https
                mock_get.assert_called_once()
                called_url = mock_get.call_args[0][0]
                assert called_url.startswith("https://")


class TestSampleCsv:
    """Tests for sample_csv function with mocked HTTP streaming."""

    @pytest.mark.asyncio
    async def test_csv_row_with_fewer_columns_than_header(self):
        """Test CSV row with fewer columns is padded with empty strings."""
        csv_content = b"name,age,city\nAlice,30\nBob,25,NYC\n"

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()

            async def mock_aiter_bytes(chunk_size):
                yield csv_content

            mock_response.aiter_bytes = mock_aiter_bytes

            mock_stream = Mock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock()

            mock_client.return_value.__aenter__.return_value.stream = Mock(
                return_value=mock_stream
            )

            result = await sample.sample_csv("https://example.com/data.csv", 2)

            assert len(result) == 2
            # First row should have empty city
            assert result[0] == {"name": "Alice", "age": "30", "city": ""}
            assert result[1] == {"name": "Bob", "age": "25", "city": "NYC"}

    @pytest.mark.asyncio
    async def test_csv_row_with_more_columns_than_header(self):
        """Test CSV row with more columns is truncated to header length."""
        csv_content = b"name,age\nAlice,30,extra_value,another\nBob,25\n"

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()

            async def mock_aiter_bytes(chunk_size):
                yield csv_content

            mock_response.aiter_bytes = mock_aiter_bytes

            mock_stream = Mock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock()

            mock_client.return_value.__aenter__.return_value.stream = Mock(
                return_value=mock_stream
            )

            result = await sample.sample_csv("https://example.com/data.csv", 2)

            assert len(result) == 2
            # First row should be truncated to 2 columns
            assert result[0] == {"name": "Alice", "age": "30"}
            assert result[1] == {"name": "Bob", "age": "25"}

    @pytest.mark.asyncio
    async def test_csv_remaining_buffer_processed(self):
        """Test that remaining data in buffer after stream ends is processed."""
        # CSV without trailing newline - last row is in buffer
        csv_content = b"name,age\nAlice,30\nBob,25"

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()

            async def mock_aiter_bytes(chunk_size):
                yield csv_content

            mock_response.aiter_bytes = mock_aiter_bytes

            mock_stream = Mock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock()

            mock_client.return_value.__aenter__.return_value.stream = Mock(
                return_value=mock_stream
            )

            result = await sample.sample_csv("https://example.com/data.csv", 10)

            assert len(result) == 2
            assert result[0] == {"name": "Alice", "age": "30"}
            assert result[1] == {"name": "Bob", "age": "25"}

    @pytest.mark.asyncio
    async def test_csv_http_to_https_upgrade(self):
        """Test that http:// URLs are upgraded to https://."""
        csv_content = b"a,b\n1,2\n"

        with patch("philly.sample.normalize_url") as mock_normalize:
            mock_normalize.return_value = "http://example.com/data.csv"

            with patch("philly.sample.httpx.AsyncClient") as mock_client:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()

                async def mock_aiter_bytes(chunk_size):
                    yield csv_content

                mock_response.aiter_bytes = mock_aiter_bytes

                mock_stream = Mock()
                mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
                mock_stream.__aexit__ = AsyncMock()

                mock_client.return_value.__aenter__.return_value.stream = Mock(
                    return_value=mock_stream
                )

                await sample.sample_csv("http://example.com/data.csv", 2)

                # Verify stream was called with https URL
                call_args = (
                    mock_client.return_value.__aenter__.return_value.stream.call_args
                )
                # Get URL from positional or keyword args
                if call_args[0] and len(call_args[0]) > 1:
                    called_url = call_args[0][1]
                else:
                    called_url = call_args[1].get("url", "")
                assert called_url.startswith("https://")

    @pytest.mark.asyncio
    async def test_basic_csv_sampling(self):
        """Test basic CSV sampling with complete rows."""
        csv_content = b"name,age\nAlice,30\nBob,25\nCharlie,35\n"

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()

            async def mock_aiter_bytes(chunk_size):
                yield csv_content

            mock_response.aiter_bytes = mock_aiter_bytes

            mock_stream = Mock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock()

            mock_client.return_value.__aenter__.return_value.stream = Mock(
                return_value=mock_stream
            )

            result = await sample.sample_csv("https://example.com/data.csv", 2)

            assert len(result) == 2
            assert result[0] == {"name": "Alice", "age": "30"}
            assert result[1] == {"name": "Bob", "age": "25"}

    @pytest.mark.asyncio
    async def test_csv_with_empty_lines(self):
        """Test CSV with empty lines are skipped."""
        csv_content = b"name,age\n\nAlice,30\n\nBob,25\n"

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()

            async def mock_aiter_bytes(chunk_size):
                yield csv_content

            mock_response.aiter_bytes = mock_aiter_bytes

            mock_stream = Mock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock()

            mock_client.return_value.__aenter__.return_value.stream = Mock(
                return_value=mock_stream
            )

            result = await sample.sample_csv("https://example.com/data.csv", 2)

            assert len(result) == 2
            assert result[0]["name"] == "Alice"
            assert result[1]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_csv_chunked_streaming(self):
        """Test CSV with data split across multiple chunks."""
        chunk1 = b"name,age\nAl"
        chunk2 = b"ice,30\nBob,25\n"

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()

            async def mock_aiter_bytes(chunk_size):
                yield chunk1
                yield chunk2

            mock_response.aiter_bytes = mock_aiter_bytes

            mock_stream = Mock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock()

            mock_client.return_value.__aenter__.return_value.stream = Mock(
                return_value=mock_stream
            )

            result = await sample.sample_csv("https://example.com/data.csv", 2)

            assert len(result) == 2
            assert result[0]["name"] == "Alice"
            assert result[1]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_csv_early_stop_at_n_rows(self):
        """Test that CSV sampling stops after n rows."""
        csv_content = b"a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n"

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()

            async def mock_aiter_bytes(chunk_size):
                yield csv_content

            mock_response.aiter_bytes = mock_aiter_bytes

            mock_stream = Mock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock()

            mock_client.return_value.__aenter__.return_value.stream = Mock(
                return_value=mock_stream
            )

            result = await sample.sample_csv("https://example.com/data.csv", 2)

            assert len(result) == 2
            assert result[0] == {"a": "1", "b": "2"}
            assert result[1] == {"a": "3", "b": "4"}

    @pytest.mark.asyncio
    async def test_csv_with_quoted_values(self):
        """Test CSV with quoted values containing commas."""
        csv_content = b'name,city\n"Smith, John","New York, NY"\nJane,LA\n'

        with patch("philly.sample.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()

            async def mock_aiter_bytes(chunk_size):
                yield csv_content

            mock_response.aiter_bytes = mock_aiter_bytes

            mock_stream = Mock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.__aexit__ = AsyncMock()

            mock_client.return_value.__aenter__.return_value.stream = Mock(
                return_value=mock_stream
            )

            result = await sample.sample_csv("https://example.com/data.csv", 2)

            assert len(result) == 2
            assert result[0]["name"] == "Smith, John"
            assert result[0]["city"] == "New York, NY"

    @pytest.mark.asyncio
    async def test_csv_url_normalization(self):
        """Test that URLs are normalized."""
        csv_content = b"a,b\n1,2\n"

        with patch("philly.sample.normalize_url") as mock_normalize:
            mock_normalize.return_value = "https://normalized.com/data.csv"

            with patch("philly.sample.httpx.AsyncClient") as mock_client:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()

                async def mock_aiter_bytes(chunk_size):
                    yield csv_content

                mock_response.aiter_bytes = mock_aiter_bytes

                mock_stream = Mock()
                mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
                mock_stream.__aexit__ = AsyncMock()

                mock_client.return_value.__aenter__.return_value.stream = Mock(
                    return_value=mock_stream
                )

                await sample.sample_csv("http://example.com/data.csv", 2)

                mock_normalize.assert_called_once()
