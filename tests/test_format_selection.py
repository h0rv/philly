"""Tests for the format_selection module."""

from philly.format_selection import (
    DEFAULT_FORMAT_PREFERENCE,
    find_resource_by_format,
    get_formats,
    has_format,
    select_by_year,
)
from philly.models.dataset import Dataset
from philly.models.resource import Resource, ResourceFormat


class TestFindResourceByFormat:
    """Test find_resource_by_format functionality."""

    def test_finds_resource_by_format(self):
        """Test finds resource by exact format match."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
                Resource(
                    name="Data JSON",
                    format=ResourceFormat.JSON,
                    url="http://example.com/data.json",
                ),
            ],
        )

        result = find_resource_by_format(dataset, "csv")

        assert result is not None
        assert result.name == "Data CSV"
        assert result.format == ResourceFormat.CSV

    def test_finds_resource_case_insensitive(self):
        """Test format matching is case insensitive."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
            ],
        )

        result_lower = find_resource_by_format(dataset, "csv")
        result_upper = find_resource_by_format(dataset, "CSV")
        result_mixed = find_resource_by_format(dataset, "CsV")

        assert result_lower is not None
        assert result_upper is not None
        assert result_mixed is not None
        assert result_lower.name == "Data CSV"
        assert result_upper.name == "Data CSV"
        assert result_mixed.name == "Data CSV"

    def test_returns_none_when_format_not_found(self):
        """Test returns None when no resource matches the format."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
                Resource(
                    name="Data JSON",
                    format=ResourceFormat.JSON,
                    url="http://example.com/data.json",
                ),
            ],
        )

        result = find_resource_by_format(dataset, "xlsx")

        assert result is None

    def test_returns_none_when_no_resources(self):
        """Test returns None when dataset has no resources."""
        dataset = Dataset(title="Test Dataset", resources=None)

        result = find_resource_by_format(dataset, "csv")

        assert result is None

    def test_returns_none_when_empty_resources(self):
        """Test returns None when dataset has empty resources list."""
        dataset = Dataset(title="Test Dataset", resources=[])

        result = find_resource_by_format(dataset, "csv")

        assert result is None

    def test_multiple_resources_same_format_prefer_latest(self):
        """Test with multiple resources of same format, selects by year (latest)."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data 2020 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2020.csv",
                ),
                Resource(
                    name="Data 2022 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2022.csv",
                ),
                Resource(
                    name="Data 2021 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2021.csv",
                ),
            ],
        )

        result = find_resource_by_format(dataset, "csv", prefer="latest")

        assert result is not None
        assert result.name == "Data 2022 CSV"

    def test_multiple_resources_same_format_prefer_oldest(self):
        """Test with multiple resources of same format, selects by year (oldest)."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data 2020 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2020.csv",
                ),
                Resource(
                    name="Data 2022 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2022.csv",
                ),
                Resource(
                    name="Data 2021 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2021.csv",
                ),
            ],
        )

        result = find_resource_by_format(dataset, "csv", prefer="oldest")

        assert result is not None
        assert result.name == "Data 2020 CSV"

    def test_resources_without_years_in_name_prefer_latest(self):
        """Test resources without years when prefer=latest (sorted last)."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Current Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/current.csv",
                ),
                Resource(
                    name="Data 2022 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2022.csv",
                ),
            ],
        )

        result = find_resource_by_format(dataset, "csv", prefer="latest")

        # Should prefer 2022 over no year
        assert result is not None
        assert result.name == "Data 2022 CSV"

    def test_resources_without_years_in_name_prefer_oldest(self):
        """Test resources without years when prefer=oldest (sorted last)."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Current Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/current.csv",
                ),
                Resource(
                    name="Data 2022 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2022.csv",
                ),
            ],
        )

        result = find_resource_by_format(dataset, "csv", prefer="oldest")

        # Should prefer 2022 over no year (no year gets 9999)
        assert result is not None
        assert result.name == "Data 2022 CSV"

    def test_all_resources_without_years(self):
        """Test when all resources lack year information."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Current Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/current.csv",
                ),
                Resource(
                    name="Latest Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/latest.csv",
                ),
            ],
        )

        result = find_resource_by_format(dataset, "csv", prefer="latest")

        # Should return first one when no years present (stable sort)
        assert result is not None
        assert result.name in ["Current Data CSV", "Latest Data CSV"]


class TestSelectByYear:
    """Test select_by_year functionality."""

    def test_extracts_four_digit_years_correctly(self):
        """Test extracts 4-digit years starting with 20."""
        resources = [
            Resource(
                name="Data 2020",
                format=ResourceFormat.CSV,
                url="http://example.com/2020.csv",
            ),
            Resource(
                name="Data 2022",
                format=ResourceFormat.CSV,
                url="http://example.com/2022.csv",
            ),
            Resource(
                name="Data 2021",
                format=ResourceFormat.CSV,
                url="http://example.com/2021.csv",
            ),
        ]

        result_latest = select_by_year(resources, prefer="latest")
        result_oldest = select_by_year(resources, prefer="oldest")

        assert result_latest.name == "Data 2022"
        assert result_oldest.name == "Data 2020"

    def test_prefer_latest_sorts_descending(self):
        """Test prefer='latest' selects highest year."""
        resources = [
            Resource(
                name="Data 2015",
                format=ResourceFormat.CSV,
                url="http://example.com/2015.csv",
            ),
            Resource(
                name="Data 2023",
                format=ResourceFormat.CSV,
                url="http://example.com/2023.csv",
            ),
            Resource(
                name="Data 2019",
                format=ResourceFormat.CSV,
                url="http://example.com/2019.csv",
            ),
        ]

        result = select_by_year(resources, prefer="latest")

        assert result.name == "Data 2023"

    def test_prefer_oldest_sorts_ascending(self):
        """Test prefer='oldest' selects lowest year."""
        resources = [
            Resource(
                name="Data 2015",
                format=ResourceFormat.CSV,
                url="http://example.com/2015.csv",
            ),
            Resource(
                name="Data 2023",
                format=ResourceFormat.CSV,
                url="http://example.com/2023.csv",
            ),
            Resource(
                name="Data 2019",
                format=ResourceFormat.CSV,
                url="http://example.com/2019.csv",
            ),
        ]

        result = select_by_year(resources, prefer="oldest")

        assert result.name == "Data 2015"

    def test_handles_resources_with_no_year(self):
        """Test handles resources without year information."""
        resources = [
            Resource(
                name="Current Data",
                format=ResourceFormat.CSV,
                url="http://example.com/current.csv",
            ),
            Resource(
                name="Data 2022",
                format=ResourceFormat.CSV,
                url="http://example.com/2022.csv",
            ),
        ]

        result_latest = select_by_year(resources, prefer="latest")
        result_oldest = select_by_year(resources, prefer="oldest")

        # Both should prefer the one with a year
        assert result_latest.name == "Data 2022"
        assert result_oldest.name == "Data 2022"

    def test_handles_multiple_years_in_name_prefer_latest(self):
        """Test handles multiple years in name (uses max for latest)."""
        resources = [
            Resource(
                name="Data 2020-2021",
                format=ResourceFormat.CSV,
                url="http://example.com/2020-2021.csv",
            ),
            Resource(
                name="Data 2022-2023",
                format=ResourceFormat.CSV,
                url="http://example.com/2022-2023.csv",
            ),
        ]

        result = select_by_year(resources, prefer="latest")

        # Should use 2023 (max year from second resource)
        assert result.name == "Data 2022-2023"

    def test_handles_multiple_years_in_name_prefer_oldest(self):
        """Test handles multiple years in name (uses min for oldest)."""
        resources = [
            Resource(
                name="Data 2020-2021",
                format=ResourceFormat.CSV,
                url="http://example.com/2020-2021.csv",
            ),
            Resource(
                name="Data 2022-2023",
                format=ResourceFormat.CSV,
                url="http://example.com/2022-2023.csv",
            ),
        ]

        result = select_by_year(resources, prefer="oldest")

        # Should use 2020 (min year from first resource)
        assert result.name == "Data 2020-2021"

    def test_year_in_middle_of_name(self):
        """Test extracts year from middle of resource name."""
        resources = [
            Resource(
                name="Philadelphia 2020 Census Data",
                format=ResourceFormat.CSV,
                url="http://example.com/1.csv",
            ),
            Resource(
                name="Philadelphia 2022 Census Data",
                format=ResourceFormat.CSV,
                url="http://example.com/2.csv",
            ),
        ]

        result = select_by_year(resources, prefer="latest")

        assert result.name == "Philadelphia 2022 Census Data"

    def test_ignores_non_20xx_years(self):
        """Test ignores years that don't start with 20."""
        resources = [
            Resource(
                name="Data 1999",
                format=ResourceFormat.CSV,
                url="http://example.com/1999.csv",
            ),
            Resource(
                name="Data 2022",
                format=ResourceFormat.CSV,
                url="http://example.com/2022.csv",
            ),
        ]

        result = select_by_year(resources, prefer="latest")

        # Should select 2022, 1999 should be treated as no year
        assert result.name == "Data 2022"

    def test_single_resource(self):
        """Test with single resource returns that resource."""
        resources = [
            Resource(
                name="Single Data",
                format=ResourceFormat.CSV,
                url="http://example.com/data.csv",
            ),
        ]

        result_latest = select_by_year(resources, prefer="latest")
        result_oldest = select_by_year(resources, prefer="oldest")

        assert result_latest.name == "Single Data"
        assert result_oldest.name == "Single Data"


class TestGetFormats:
    """Test get_formats functionality."""

    def test_returns_unique_formats(self):
        """Test returns unique formats without duplicates."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data 1 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/1.csv",
                ),
                Resource(
                    name="Data 2 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2.csv",
                ),
                Resource(
                    name="Data JSON",
                    format=ResourceFormat.JSON,
                    url="http://example.com/data.json",
                ),
            ],
        )

        formats = get_formats(dataset)

        assert len(formats) == 2
        assert "csv" in formats
        assert "json" in formats

    def test_returns_sorted_list(self):
        """Test returns formats in sorted order."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data XLSX",
                    format=ResourceFormat.XLSX,
                    url="http://example.com/data.xlsx",
                ),
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
                Resource(
                    name="Data JSON",
                    format=ResourceFormat.JSON,
                    url="http://example.com/data.json",
                ),
            ],
        )

        formats = get_formats(dataset)

        # Should be sorted alphabetically
        assert formats == ["csv", "json", "xlsx"]

    def test_lowercase_normalization(self):
        """Test formats are normalized to lowercase."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
                Resource(
                    name="Data JSON",
                    format=ResourceFormat.JSON,
                    url="http://example.com/data.json",
                ),
            ],
        )

        formats = get_formats(dataset)

        # All formats should be lowercase
        assert all(fmt == fmt.lower() for fmt in formats)
        assert "csv" in formats
        assert "json" in formats

    def test_empty_resources(self):
        """Test returns empty list when dataset has no resources."""
        dataset = Dataset(title="Test Dataset", resources=None)

        formats = get_formats(dataset)

        assert formats == []

    def test_empty_resources_list(self):
        """Test returns empty list when dataset has empty resources list."""
        dataset = Dataset(title="Test Dataset", resources=[])

        formats = get_formats(dataset)

        assert formats == []

    def test_single_format(self):
        """Test with dataset containing only one format."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
            ],
        )

        formats = get_formats(dataset)

        assert formats == ["csv"]

    def test_many_formats(self):
        """Test with dataset containing many different formats."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
                Resource(
                    name="Data JSON",
                    format=ResourceFormat.JSON,
                    url="http://example.com/data.json",
                ),
                Resource(
                    name="Data XLSX",
                    format=ResourceFormat.XLSX,
                    url="http://example.com/data.xlsx",
                ),
                Resource(
                    name="Data SHP",
                    format=ResourceFormat.SHP,
                    url="http://example.com/data.shp",
                ),
                Resource(
                    name="Data GEOJSON",
                    format=ResourceFormat.GEOJSON,
                    url="http://example.com/data.geojson",
                ),
            ],
        )

        formats = get_formats(dataset)

        assert len(formats) == 5
        assert formats == ["csv", "geojson", "json", "shp", "xlsx"]


class TestHasFormat:
    """Test has_format functionality."""

    def test_returns_true_when_format_exists(self):
        """Test returns True when format exists in dataset."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
                Resource(
                    name="Data JSON",
                    format=ResourceFormat.JSON,
                    url="http://example.com/data.json",
                ),
            ],
        )

        assert has_format(dataset, "csv") is True
        assert has_format(dataset, "json") is True

    def test_returns_false_when_format_missing(self):
        """Test returns False when format does not exist in dataset."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
                Resource(
                    name="Data JSON",
                    format=ResourceFormat.JSON,
                    url="http://example.com/data.json",
                ),
            ],
        )

        assert has_format(dataset, "xlsx") is False
        assert has_format(dataset, "shp") is False

    def test_case_insensitive_matching(self):
        """Test format matching is case insensitive."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
            ],
        )

        assert has_format(dataset, "csv") is True
        assert has_format(dataset, "CSV") is True
        assert has_format(dataset, "CsV") is True
        assert has_format(dataset, "CsV") is True

    def test_empty_resources(self):
        """Test returns False when dataset has no resources."""
        dataset = Dataset(title="Test Dataset", resources=None)

        assert has_format(dataset, "csv") is False

    def test_empty_resources_list(self):
        """Test returns False when dataset has empty resources list."""
        dataset = Dataset(title="Test Dataset", resources=[])

        assert has_format(dataset, "csv") is False

    def test_multiple_resources_same_format(self):
        """Test returns True when multiple resources have same format."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data 2020 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2020.csv",
                ),
                Resource(
                    name="Data 2021 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2021.csv",
                ),
                Resource(
                    name="Data 2022 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2022.csv",
                ),
            ],
        )

        assert has_format(dataset, "csv") is True


class TestDefaultFormatPreference:
    """Test DEFAULT_FORMAT_PREFERENCE constant."""

    def test_default_format_preference_is_list(self):
        """Test DEFAULT_FORMAT_PREFERENCE is a list."""
        assert isinstance(DEFAULT_FORMAT_PREFERENCE, list)

    def test_default_format_preference_has_common_formats(self):
        """Test DEFAULT_FORMAT_PREFERENCE includes common formats."""
        assert "csv" in DEFAULT_FORMAT_PREFERENCE
        assert "json" in DEFAULT_FORMAT_PREFERENCE
        assert "geojson" in DEFAULT_FORMAT_PREFERENCE

    def test_default_format_preference_prefers_csv_first(self):
        """Test CSV is first in preference order."""
        assert DEFAULT_FORMAT_PREFERENCE[0] == "csv"

    def test_default_format_preference_all_lowercase(self):
        """Test all formats in DEFAULT_FORMAT_PREFERENCE are lowercase."""
        assert all(fmt == fmt.lower() for fmt in DEFAULT_FORMAT_PREFERENCE)


class TestEdgeCases:
    """Test edge cases for format selection functions."""

    def test_find_resource_empty_format_string(self):
        """Test find_resource_by_format with empty string returns None."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
            ],
        )

        result = find_resource_by_format(dataset, "")

        assert result is None

    def test_has_format_empty_format_string(self):
        """Test has_format with empty string returns False."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
            ],
        )

        assert has_format(dataset, "") is False

    def test_find_resource_with_whitespace(self):
        """Test format with whitespace is not matched."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/data.csv",
                ),
            ],
        )

        # Whitespace should not match (not stripped)
        result = find_resource_by_format(dataset, " csv ")
        assert result is None

        result = find_resource_by_format(dataset, " csv")
        assert result is None

        result = find_resource_by_format(dataset, "csv ")
        assert result is None

    def test_find_resource_default_prefer_is_latest(self):
        """Test find_resource_by_format defaults to prefer='latest'."""
        dataset = Dataset(
            title="Test Dataset",
            resources=[
                Resource(
                    name="Data 2020 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2020.csv",
                ),
                Resource(
                    name="Data 2023 CSV",
                    format=ResourceFormat.CSV,
                    url="http://example.com/2023.csv",
                ),
            ],
        )

        # Without specifying prefer, should default to latest
        result = find_resource_by_format(dataset, "csv")

        assert result is not None
        assert result.name == "Data 2023 CSV"

    def test_select_by_year_with_year_range_in_name(self):
        """Test resources with year ranges like 2009-2012."""
        resources = [
            Resource(
                name="2009-2012 Police Advisory Commission Complaints",
                format=ResourceFormat.CSV,
                url="http://example.com/1.csv",
            ),
            Resource(
                name="2013-2016 Police Advisory Commission Complaints",
                format=ResourceFormat.CSV,
                url="http://example.com/2.csv",
            ),
        ]

        result_latest = select_by_year(resources, prefer="latest")
        result_oldest = select_by_year(resources, prefer="oldest")

        # Latest should use max year (2016 from second resource)
        assert result_latest.name == "2013-2016 Police Advisory Commission Complaints"
        # Oldest should use min year (2009 from first resource)
        assert result_oldest.name == "2009-2012 Police Advisory Commission Complaints"

    def test_select_by_year_boundary_years(self):
        """Test with years at decade boundaries."""
        resources = [
            Resource(
                name="Data 2000",
                format=ResourceFormat.CSV,
                url="http://example.com/2000.csv",
            ),
            Resource(
                name="Data 2099",
                format=ResourceFormat.CSV,
                url="http://example.com/2099.csv",
            ),
        ]

        result_latest = select_by_year(resources, prefer="latest")
        result_oldest = select_by_year(resources, prefer="oldest")

        assert result_latest.name == "Data 2099"
        assert result_oldest.name == "Data 2000"
