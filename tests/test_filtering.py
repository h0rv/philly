"""Tests for the filtering module."""

import pytest

from philly.filtering import (
    BackendType,
    _extract_table_name,
    build_arcgis_query,
    build_carto_query,
    detect_backend,
    validate_where_clause,
)


class TestBackendDetection:
    """Test backend detection functionality."""

    def test_detect_carto(self):
        """Test detection of Carto API URLs."""
        url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM table"
        assert detect_backend(url) == BackendType.CARTO

    def test_detect_arcgis(self):
        """Test detection of ArcGIS REST API URLs."""
        url = "https://services.arcgis.com/.../FeatureServer/0/query"
        assert detect_backend(url) == BackendType.ARCGIS

        url = "https://example.com/FeatureServer/0/query"
        assert detect_backend(url) == BackendType.ARCGIS

    def test_detect_static_csv(self):
        """Test detection of static CSV files."""
        url = "https://example.com/data.csv"
        assert detect_backend(url) == BackendType.STATIC

    def test_detect_static_json(self):
        """Test detection of static JSON files."""
        url = "https://example.com/data.json"
        assert detect_backend(url) == BackendType.STATIC

    def test_detect_static_geojson(self):
        """Test detection of static GeoJSON files."""
        url = "https://example.com/data.geojson"
        assert detect_backend(url) == BackendType.STATIC

    def test_detect_unknown(self):
        """Test detection of unknown backend types."""
        url = "https://example.com/unknown"
        assert detect_backend(url) == BackendType.UNKNOWN


class TestWhereClauseValidation:
    """Test WHERE clause validation."""

    def test_valid_simple_clause(self):
        """Test validation of simple WHERE clause."""
        clause = "district = '6'"
        assert validate_where_clause(clause) == clause

    def test_valid_complex_clause(self):
        """Test validation of complex WHERE clause."""
        clause = "date >= '2024-01-01' AND date < '2024-12-31'"
        assert validate_where_clause(clause) == clause

    def test_valid_in_clause(self):
        """Test validation of IN clause."""
        clause = "status IN ('Active', 'Pending')"
        assert validate_where_clause(clause) == clause

    def test_invalid_drop(self):
        """Test rejection of DROP keyword."""
        clause = "x = 1; DROP TABLE users"
        with pytest.raises(ValueError, match="DROP"):
            validate_where_clause(clause)

    def test_invalid_delete(self):
        """Test rejection of DELETE keyword."""
        clause = "DELETE FROM table"
        with pytest.raises(ValueError, match="DELETE"):
            validate_where_clause(clause)

    def test_invalid_update(self):
        """Test rejection of UPDATE keyword."""
        clause = "UPDATE table SET x = 1"
        with pytest.raises(ValueError, match="UPDATE"):
            validate_where_clause(clause)

    def test_invalid_insert(self):
        """Test rejection of INSERT keyword."""
        clause = "x = 1; INSERT INTO table VALUES (1)"
        with pytest.raises(ValueError, match="INSERT"):
            validate_where_clause(clause)

    def test_invalid_truncate(self):
        """Test rejection of TRUNCATE keyword."""
        clause = "TRUNCATE TABLE users"
        with pytest.raises(ValueError, match="TRUNCATE"):
            validate_where_clause(clause)


class TestTableNameExtraction:
    """Test table name extraction from SQL queries."""

    def test_extract_simple_table(self):
        """Test extraction of simple table name."""
        query = "SELECT * FROM crimes"
        assert _extract_table_name(query) == "crimes"

    def test_extract_schema_table(self):
        """Test extraction of schema.table name."""
        query = "SELECT col1, col2 FROM public.crimes"
        assert _extract_table_name(query) == "public.crimes"

    def test_extract_with_whitespace(self):
        """Test extraction with various whitespace."""
        query = "SELECT * FROM   crimes"
        assert _extract_table_name(query) == "crimes"

    def test_extract_case_insensitive(self):
        """Test extraction is case insensitive."""
        query = "select * from crimes"
        assert _extract_table_name(query) == "crimes"

    def test_extract_missing_from(self):
        """Test extraction fails without FROM."""
        query = "SELECT * crimes"
        with pytest.raises(ValueError, match="Could not extract table name"):
            _extract_table_name(query)


class TestCartoQueryBuilding:
    """Test Carto SQL query building."""

    def test_build_with_where(self):
        """Test building query with WHERE clause."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        result = build_carto_query(base_url, where="district = '6'")

        assert "WHERE" in result
        assert "district" in result
        assert "format=csv" in result

    def test_build_with_columns(self):
        """Test building query with column selection."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        result = build_carto_query(base_url, columns=["date", "location", "type"])

        assert "date" in result
        assert "location" in result
        assert "type" in result
        # Should not have * in the SELECT part
        assert "*" not in result.split("FROM")[0]

    def test_build_with_limit(self):
        """Test building query with LIMIT."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        result = build_carto_query(base_url, limit=100)

        assert "LIMIT" in result or "LIMIT+100" in result

    def test_build_with_offset(self):
        """Test building query with OFFSET."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        result = build_carto_query(base_url, limit=100, offset=50)

        assert "OFFSET" in result or "OFFSET+50" in result

    def test_build_with_all_parameters(self):
        """Test building query with all parameters."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv&filename=crimes"
        result = build_carto_query(
            base_url,
            where="district = '6'",
            columns=["date", "type"],
            limit=500,
            offset=100,
        )

        assert "WHERE" in result
        assert "date" in result
        assert "type" in result
        assert "LIMIT" in result
        assert "OFFSET" in result
        assert "format=csv" in result
        assert "filename=crimes" in result

    def test_build_preserves_format(self):
        """Test that format parameter is preserved."""
        base_url = (
            "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=geojson"
        )
        result = build_carto_query(base_url, where="district = '6'")

        assert "format=geojson" in result

    def test_build_missing_query_param(self):
        """Test error when URL missing 'q' parameter."""
        base_url = "https://phl.carto.com/api/v2/sql?format=csv"
        with pytest.raises(ValueError, match="must contain a 'q' parameter"):
            build_carto_query(base_url, where="district = '6'")

    def test_build_with_invalid_where(self):
        """Test error with invalid WHERE clause."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        with pytest.raises(ValueError, match="DROP"):
            build_carto_query(base_url, where="x = 1; DROP TABLE users")


class TestBackendTypeEnum:
    """Test BackendType enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert BackendType.CARTO.value == "carto"
        assert BackendType.ARCGIS.value == "arcgis"
        assert BackendType.STATIC.value == "static"
        assert BackendType.UNKNOWN.value == "unknown"

    def test_enum_from_value(self):
        """Test creating enum from value string."""
        assert BackendType("carto") == BackendType.CARTO
        assert BackendType("arcgis") == BackendType.ARCGIS

    def test_enum_names(self):
        """Test enum names match expected values."""
        names = [b.name for b in BackendType]
        assert names == ["CARTO", "ARCGIS", "STATIC", "UNKNOWN"]


class TestArcGISQueryBuilding:
    """Test ArcGIS REST API query building."""

    def test_build_with_where(self):
        """Test building query with WHERE clause."""
        base_url = (
            "https://services.arcgis.com/.../query?where=1%3D1&outFields=*&f=geojson"
        )
        result = build_arcgis_query(base_url, where="STATUS = 'Active'")

        assert "where=STATUS" in result
        assert "f=geojson" in result

    def test_build_with_columns(self):
        """Test building query with column selection."""
        base_url = (
            "https://services.arcgis.com/.../query?where=1%3D1&outFields=*&f=geojson"
        )
        result = build_arcgis_query(base_url, columns=["NAME", "STATUS", "DATE"])

        assert "outFields=NAME" in result

    def test_build_with_limit(self):
        """Test building query with LIMIT."""
        base_url = (
            "https://services.arcgis.com/.../query?where=1%3D1&outFields=*&f=geojson"
        )
        result = build_arcgis_query(base_url, limit=100)

        assert "resultRecordCount=100" in result

    def test_build_with_offset(self):
        """Test building query with OFFSET."""
        base_url = (
            "https://services.arcgis.com/.../query?where=1%3D1&outFields=*&f=geojson"
        )
        result = build_arcgis_query(base_url, offset=50)

        assert "resultOffset=50" in result

    def test_build_with_all_parameters(self):
        """Test building query with all parameters."""
        base_url = (
            "https://services.arcgis.com/.../query?where=1%3D1&outFields=*&f=geojson"
        )
        result = build_arcgis_query(
            base_url,
            where="STATUS = 'Active'",
            columns=["NAME", "STATUS"],
            limit=50,
            offset=25,
        )

        assert "where=STATUS" in result
        assert "outFields=" in result
        assert "resultRecordCount=50" in result
        assert "resultOffset=25" in result
        assert "f=geojson" in result

    def test_build_preserves_format(self):
        """Test that format parameter is preserved."""
        base_url = (
            "https://services.arcgis.com/.../query?where=1%3D1&outFields=*&f=json"
        )
        result = build_arcgis_query(base_url, where="STATUS = 'Active'")

        assert "f=json" in result

    def test_build_no_modifications(self):
        """Test building query with no modifications preserves original."""
        base_url = (
            "https://services.arcgis.com/.../query?where=1%3D1&outFields=*&f=geojson"
        )
        result = build_arcgis_query(base_url)
        assert "where=1" in result
        assert "outFields=" in result

    def test_build_url_without_where(self):
        """Test adding where clause to URL that lacks one."""
        base_url = "https://services.arcgis.com/.../query?outFields=*&f=geojson"
        result = build_arcgis_query(base_url, where="status='Active'")
        assert "where=status" in result

    def test_build_empty_columns(self):
        """Test building query with empty columns list."""
        base_url = (
            "https://services.arcgis.com/.../query?where=1%3D1&outFields=*&f=geojson"
        )
        result = build_arcgis_query(base_url, columns=[])
        # Empty columns list should result in empty outFields
        assert "outFields=" in result


class TestBackendDetectionEdgeCases:
    """Test edge cases for backend detection."""

    def test_detect_carto_without_query_params(self):
        """Test Carto detection without query params."""
        url = "https://phl.carto.com/api/v2/sql"
        assert detect_backend(url) == BackendType.CARTO

    def test_detect_carto_different_subdomain(self):
        """Test Carto detection with different subdomain."""
        url = "https://other.carto.com/api/v2/sql?q=SELECT"
        assert detect_backend(url) == BackendType.CARTO

    def test_detect_arcgis_only_domain(self):
        """Test ArcGIS detection via domain only."""
        url = "https://services.arcgis.com/some/path"
        assert detect_backend(url) == BackendType.ARCGIS

    def test_detect_static_shapefile(self):
        """Test detection of shapefile URLs."""
        url = "https://example.com/data.shp"
        assert detect_backend(url) == BackendType.STATIC

    def test_detect_static_zip(self):
        """Test detection of zip file URLs."""
        url = "https://example.com/data.zip"
        assert detect_backend(url) == BackendType.STATIC

    def test_detect_empty_url(self):
        """Test detection with empty URL."""
        assert detect_backend("") == BackendType.UNKNOWN


class TestWhereClauseEdgeCases:
    """Test edge cases for WHERE clause validation."""

    def test_case_insensitive_detection(self):
        """Test that dangerous keywords are detected case-insensitively."""
        with pytest.raises(ValueError, match="DROP"):
            validate_where_clause("drop table users")

        with pytest.raises(ValueError, match="DELETE"):
            validate_where_clause("delete from users")

    def test_valid_between_clause(self):
        """Test BETWEEN clause validation."""
        clause = "date BETWEEN '2024-01-01' AND '2024-12-31'"
        assert validate_where_clause(clause) == clause

    def test_valid_like_clause(self):
        """Test LIKE clause validation."""
        clause = "name LIKE '%test%'"
        assert validate_where_clause(clause) == clause

    def test_valid_is_not_null(self):
        """Test IS NOT NULL clause validation."""
        clause = "id IS NOT NULL"
        assert validate_where_clause(clause) == clause

    def test_valid_comparison_operators(self):
        """Test various comparison operators."""
        clause = "value > 100 AND value < 200"
        assert validate_where_clause(clause) == clause


class TestTableNameExtractionEdgeCases:
    """Test edge cases for table name extraction."""

    def test_extract_alphanumeric_table(self):
        """Test extraction of alphanumeric table name."""
        query = "SELECT * FROM table123"
        assert _extract_table_name(query) == "table123"

    def test_extract_underscore_prefix(self):
        """Test extraction of underscore-prefixed table name."""
        query = "SELECT * FROM _private_table"
        assert _extract_table_name(query) == "_private_table"

    def test_extract_from_complex_query(self):
        """Test extraction from query with WHERE clause."""
        query = "SELECT a, b, c FROM my_table WHERE x = 1"
        assert _extract_table_name(query) == "my_table"

    def test_extract_with_newlines(self):
        """Test extraction with newline whitespace."""
        query = "SELECT * FROM\n\t  spacy_table"
        assert _extract_table_name(query) == "spacy_table"


class TestCartoQueryBuildingEdgeCases:
    """Test edge cases for Carto query building."""

    def test_build_no_modifications(self):
        """Test building with no parameters preserves table."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        result = build_carto_query(base_url)
        assert "crimes" in result

    def test_build_only_limit(self):
        """Test building with only limit parameter."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        result = build_carto_query(base_url, limit=50)
        assert "LIMIT" in result
        assert "50" in result

    def test_build_only_offset(self):
        """Test building with only offset parameter."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        result = build_carto_query(base_url, offset=100)
        assert "OFFSET" in result
        assert "100" in result

    def test_build_empty_columns(self):
        """Test building with empty columns list defaults to *."""
        base_url = "https://phl.carto.com/api/v2/sql?q=SELECT * FROM crimes&format=csv"
        result = build_carto_query(base_url, columns=[])
        # Empty list is falsy, so implementation falls back to *
        assert "%2A" in result or "*" in result  # %2A is URL-encoded *
