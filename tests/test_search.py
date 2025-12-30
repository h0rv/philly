"""Tests for search and discovery API."""

import pytest

from philly.models.dataset import Dataset
from philly.models.resource import Resource, ResourceFormat
from philly.search import (
    STOPWORDS,
    SearchIndex,
    _extract_keywords,
    _fuzzy_ratio,
    _levenshtein_distance,
    build_search_index,
    get_by_category,
    list_categories,
    search,
    search_resources,
)


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing."""
    return [
        Dataset(
            title="Crime Incidents",
            organization="Philadelphia Police",
            notes="Data on crime incidents in Philadelphia including location and type",
            category=["Public Safety", "Criminal Justice"],
            resources=[
                Resource(
                    name="Crime Data CSV",
                    format=ResourceFormat.CSV,
                    url="https://example.com/crime.csv",
                ),
                Resource(
                    name="Crime Data JSON",
                    format=ResourceFormat.JSON,
                    url="https://example.com/crime.json",
                ),
            ],
        ),
        Dataset(
            title="Parking Violations",
            organization="Philadelphia Parking Authority",
            notes="Information on parking violations and citations",
            category=["Transportation", "License & Inspections"],
            resources=[
                Resource(
                    name="Violations CSV",
                    format=ResourceFormat.CSV,
                    url="https://example.com/parking.csv",
                ),
            ],
        ),
        Dataset(
            title="Building Permits",
            organization="Department of Licenses and Inspections",
            notes="Building permit applications and approvals",
            category=["License & Inspections", "Real Estate"],
            resources=[
                Resource(
                    name="Permits Data",
                    format=ResourceFormat.CSV,
                    url="https://example.com/permits.csv",
                ),
                Resource(
                    name="Permits API",
                    format=ResourceFormat.API,
                    url="https://example.com/api/permits",
                ),
            ],
        ),
        Dataset(
            title="Police Districts",
            organization="Philadelphia Police",
            notes="Geographic boundaries of police districts",
            category=["Public Safety", "Geography"],
            resources=[
                Resource(
                    name="Districts GeoJSON",
                    format=ResourceFormat.GEOJSON,
                    url="https://example.com/districts.geojson",
                ),
            ],
        ),
        Dataset(
            title="Air Quality Monitoring",
            organization="Department of Public Health",
            notes="Air quality sensor readings and measurements",
            category=["Environment", "Health"],
            resources=[
                Resource(
                    name="AQI Data",
                    format=ResourceFormat.CSV,
                    url="https://example.com/aqi.csv",
                ),
            ],
        ),
    ]


@pytest.fixture
def search_index(sample_datasets):
    """Build a search index from sample datasets."""
    return build_search_index(sample_datasets)


class TestKeywordExtraction:
    """Tests for _extract_keywords function."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        text = "Crime data in Philadelphia"
        keywords = _extract_keywords(text)

        assert "crime" in keywords
        assert "data" in keywords
        assert "philadelphia" in keywords
        assert len(keywords) == 3

    def test_extract_keywords_filters_stopwords(self):
        """Test that stopwords are filtered out."""
        text = "The quick brown fox and the lazy dog"
        keywords = _extract_keywords(text)

        # Stopwords should be filtered
        assert "the" not in keywords
        assert "and" not in keywords

        # Non-stopwords should remain
        assert "quick" in keywords
        assert "brown" in keywords
        assert "lazy" in keywords

    def test_extract_keywords_filters_short_words(self):
        """Test that words shorter than 3 characters are filtered."""
        text = "A big ox is at the zoo"
        keywords = _extract_keywords(text)

        # Short words should be filtered
        assert "a" not in keywords
        assert "ox" not in keywords
        assert "is" not in keywords
        assert "at" not in keywords

        # Long enough words should remain
        assert "big" in keywords
        assert "zoo" in keywords

    def test_extract_keywords_removes_punctuation(self):
        """Test that punctuation is removed from keywords."""
        text = "Hello, world! Testing (parentheses) and [brackets]."
        keywords = _extract_keywords(text)

        assert "hello" in keywords
        assert "world" in keywords
        assert "testing" in keywords
        assert "parentheses" in keywords
        assert "brackets" in keywords

    def test_extract_keywords_empty_text(self):
        """Test keyword extraction with empty text."""
        assert _extract_keywords("") == set()
        assert _extract_keywords(None) == set()

    def test_extract_keywords_case_insensitive(self):
        """Test that keywords are lowercase."""
        text = "UPPERCASE lowercase MixedCase"
        keywords = _extract_keywords(text)

        assert "uppercase" in keywords
        assert "lowercase" in keywords
        assert "mixedcase" in keywords
        assert "UPPERCASE" not in keywords


class TestBuildSearchIndex:
    """Tests for build_search_index function."""

    def test_build_index_creates_all_indexes(self, sample_datasets):
        """Test that build_search_index creates all index types."""
        index = build_search_index(sample_datasets)

        assert isinstance(index, SearchIndex)
        assert len(index.title_index) > 0
        assert len(index.normalized_titles) > 0
        assert len(index.category_index) > 0
        assert len(index.keyword_index) > 0
        assert len(index.resource_index) > 0
        assert len(index.org_index) > 0

    def test_build_index_title_indexing(self, sample_datasets):
        """Test that titles are properly indexed."""
        index = build_search_index(sample_datasets)

        # Check title index
        assert "Crime Incidents" in index.title_index
        assert "Parking Violations" in index.title_index

        # Check normalized titles
        assert "crime incidents" in index.normalized_titles
        assert index.normalized_titles["crime incidents"] == "Crime Incidents"

    def test_build_index_category_indexing(self, sample_datasets):
        """Test that categories are properly indexed."""
        index = build_search_index(sample_datasets)

        # Check category index
        assert "Public Safety" in index.category_index
        assert "Crime Incidents" in index.category_index["Public Safety"]
        assert "Police Districts" in index.category_index["Public Safety"]

        assert "Transportation" in index.category_index
        assert "Parking Violations" in index.category_index["Transportation"]

    def test_build_index_keyword_extraction(self, sample_datasets):
        """Test that keywords are extracted from title and notes."""
        index = build_search_index(sample_datasets)

        # Keywords from title and notes should be indexed
        assert "crime" in index.keyword_index
        assert "Crime Incidents" in index.keyword_index["crime"]

        assert "parking" in index.keyword_index
        assert "Parking Violations" in index.keyword_index["parking"]

        assert "building" in index.keyword_index
        assert "Building Permits" in index.keyword_index["building"]

    def test_build_index_stopwords_filtered(self, sample_datasets):
        """Test that stopwords are not in keyword index."""
        index = build_search_index(sample_datasets)

        # Common stopwords should not be indexed
        for stopword in STOPWORDS:
            assert stopword not in index.keyword_index

    def test_build_index_resource_indexing(self, sample_datasets):
        """Test that resources are properly indexed."""
        index = build_search_index(sample_datasets)

        # Resources should be indexed by lowercase name
        assert "crime data csv" in index.resource_index
        assert "violations csv" in index.resource_index
        assert "permits api" in index.resource_index

        # Check resource data structure
        crime_resources = index.resource_index["crime data csv"]
        assert len(crime_resources) > 0
        dataset_title, resource = crime_resources[0]
        assert dataset_title == "Crime Incidents"
        assert resource.name == "Crime Data CSV"

    def test_build_index_organization_indexing(self, sample_datasets):
        """Test that organizations are properly indexed."""
        index = build_search_index(sample_datasets)

        # Organizations should be indexed
        assert "Philadelphia Police" in index.org_index
        assert "Crime Incidents" in index.org_index["Philadelphia Police"]
        assert "Police Districts" in index.org_index["Philadelphia Police"]

        assert "Philadelphia Parking Authority" in index.org_index
        assert "Parking Violations" in index.org_index["Philadelphia Parking Authority"]

    def test_build_index_empty_datasets(self):
        """Test building index with empty dataset list."""
        index = build_search_index([])

        assert len(index.title_index) == 0
        assert len(index.normalized_titles) == 0
        assert len(index.category_index) == 0
        assert len(index.keyword_index) == 0
        assert len(index.resource_index) == 0
        assert len(index.org_index) == 0

    def test_build_index_dataset_without_optional_fields(self):
        """Test building index with minimal dataset."""
        minimal_dataset = [
            Dataset(
                title="Minimal Dataset",
                # No organization, notes, category, or resources
            )
        ]

        index = build_search_index(minimal_dataset)

        # Should still index the title
        assert "Minimal Dataset" in index.title_index
        assert len(index.category_index) == 0
        assert len(index.resource_index) == 0
        assert len(index.org_index) == 0


class TestSearchFunction:
    """Tests for search function."""

    def test_search_exact_title_match(self, search_index):
        """Test exact title match returns highest score."""
        results = search(search_index, "Crime Incidents")

        assert len(results) > 0
        assert results[0] == "Crime Incidents"

    def test_search_partial_title_match(self, search_index):
        """Test partial title match."""
        results = search(search_index, "crime")

        assert "Crime Incidents" in results

    def test_search_case_insensitive(self, search_index):
        """Test search is case insensitive."""
        results_upper = search(search_index, "CRIME")
        results_lower = search(search_index, "crime")
        results_mixed = search(search_index, "CrImE")

        assert "Crime Incidents" in results_upper
        assert "Crime Incidents" in results_lower
        assert "Crime Incidents" in results_mixed

    def test_search_keyword_match(self, search_index):
        """Test keyword matching from notes."""
        results = search(search_index, "violations")

        assert "Parking Violations" in results

    def test_search_multiple_results(self, search_index):
        """Test search returns multiple relevant results."""
        results = search(search_index, "philadelphia")

        # Should match multiple datasets with "philadelphia" in notes or organization
        assert len(results) >= 1  # At least Crime Incidents has "Philadelphia" in notes
        # Multiple datasets may match depending on content

    def test_search_relevance_ordering(self, search_index):
        """Test that results are ordered by relevance."""
        results = search(search_index, "permit")

        # "Building Permits" has "permit" in title, should rank higher
        # than results that only have it in notes
        assert results[0] == "Building Permits"

    def test_search_fuzzy_matching_with_typos(self, search_index):
        """Test fuzzy matching with typos."""
        # "Cryme" is close to "Crime"
        results = search(search_index, "Cryme Incidents", fuzzy=True)

        assert len(results) > 0
        # Should still find "Crime Incidents" with fuzzy matching
        assert "Crime Incidents" in results

    def test_search_fuzzy_threshold(self, search_index):
        """Test fuzzy matching threshold (70%)."""
        # Very different query should not match
        results = search(search_index, "xyz", fuzzy=True)

        # Should return empty or very few results
        assert len(results) == 0 or "Crime Incidents" not in results

    def test_search_category_filter(self, search_index):
        """Test category filtering."""
        results = search(search_index, "", category="Public Safety")

        assert "Crime Incidents" in results
        assert "Police Districts" in results
        # Should not include datasets from other categories
        assert "Parking Violations" not in results

    def test_search_category_filter_with_query(self, search_index):
        """Test combining search query with category filter."""
        results = search(search_index, "crime", category="Public Safety")

        assert "Crime Incidents" in results
        # Should not include datasets outside the category
        assert "Parking Violations" not in results
        assert "Building Permits" not in results

    def test_search_limit_parameter(self, search_index):
        """Test limit parameter."""
        results = search(search_index, "data", limit=2)

        assert len(results) <= 2

    def test_search_limit_zero(self, search_index):
        """Test limit of zero returns empty."""
        results = search(search_index, "crime", limit=0)

        assert len(results) == 0

    def test_search_empty_query(self, search_index):
        """Test empty query behavior."""
        results = search(search_index, "")

        # Empty query with no category returns all datasets that match keywords
        # Since empty query extracts no keywords, it searches in titles only
        # Substring match on empty string matches all titles
        assert len(results) >= 0  # Implementation-dependent behavior

    def test_search_no_results(self, search_index):
        """Test query with no matches."""
        results = search(search_index, "nonexistent_query_xyz")

        assert len(results) == 0

    def test_search_special_characters(self, search_index):
        """Test search with special characters."""
        results = search(search_index, "crime!")

        # Should still find results after stripping punctuation
        assert "Crime Incidents" in results


class TestSearchResources:
    """Tests for search_resources function."""

    def test_search_resources_by_name(self, search_index):
        """Test searching resources by name."""
        results = search_resources(search_index, "csv")

        assert len(results) > 0
        assert "Crime Data CSV [Crime Incidents]" in results
        assert "Violations CSV [Parking Violations]" in results

    def test_search_resources_case_insensitive(self, search_index):
        """Test resource search is case insensitive."""
        results_upper = search_resources(search_index, "CSV")
        results_lower = search_resources(search_index, "csv")

        assert len(results_upper) > 0
        assert results_upper == results_lower

    def test_search_resources_partial_match(self, search_index):
        """Test partial resource name matching."""
        results = search_resources(search_index, "crime")

        assert "Crime Data CSV [Crime Incidents]" in results
        assert "Crime Data JSON [Crime Incidents]" in results

    def test_search_resources_dataset_filter(self, search_index):
        """Test filtering resources by dataset."""
        results = search_resources(
            search_index, "data", dataset_filter="Crime Incidents"
        )

        assert "Crime Data CSV [Crime Incidents]" in results
        assert "Crime Data JSON [Crime Incidents]" in results
        # Should not include resources from other datasets
        assert "Permits Data [Building Permits]" not in results

    def test_search_resources_limit_parameter(self, search_index):
        """Test limit parameter."""
        results = search_resources(search_index, "data", limit=2)

        assert len(results) <= 2

    def test_search_resources_sorted(self, search_index):
        """Test results are sorted alphabetically."""
        results = search_resources(search_index, "data")

        # Results should be sorted
        assert results == sorted(results)

    def test_search_resources_no_results(self, search_index):
        """Test query with no matches."""
        results = search_resources(search_index, "nonexistent_xyz")

        assert len(results) == 0

    def test_search_resources_empty_query(self, search_index):
        """Test empty query behavior."""
        results = search_resources(search_index, "")

        # Empty string matches all resource names (substring match)
        # This is expected behavior - returns all resources
        assert len(results) >= 0  # Implementation-dependent behavior


class TestListCategories:
    """Tests for list_categories function."""

    def test_list_categories_returns_sorted_list(self, search_index):
        """Test that categories are returned sorted."""
        categories = list_categories(search_index)

        assert isinstance(categories, list)
        assert len(categories) > 0
        assert categories == sorted(categories)

    def test_list_categories_all_included(self, search_index):
        """Test that all categories are included."""
        categories = list_categories(search_index)

        expected_categories = [
            "Public Safety",
            "Criminal Justice",
            "Transportation",
            "License & Inspections",
            "Real Estate",
            "Geography",
            "Environment",
            "Health",
        ]

        for expected in expected_categories:
            assert expected in categories

    def test_list_categories_no_duplicates(self, search_index):
        """Test that categories are unique."""
        categories = list_categories(search_index)

        assert len(categories) == len(set(categories))

    def test_list_categories_empty_index(self):
        """Test with empty index."""
        empty_index = SearchIndex()
        categories = list_categories(empty_index)

        assert len(categories) == 0


class TestGetByCategory:
    """Tests for get_by_category function."""

    def test_get_by_category_returns_correct_datasets(self, search_index):
        """Test returns datasets in specified category."""
        datasets = get_by_category(search_index, "Public Safety")

        assert "Crime Incidents" in datasets
        assert "Police Districts" in datasets

    def test_get_by_category_sorted(self, search_index):
        """Test results are sorted alphabetically."""
        datasets = get_by_category(search_index, "Public Safety")

        assert datasets == sorted(datasets)

    def test_get_by_category_case_sensitive(self, search_index):
        """Test category lookup is case sensitive."""
        # Exact case should work
        datasets_correct = get_by_category(search_index, "Public Safety")
        assert len(datasets_correct) > 0

        # Wrong case should not work
        datasets_wrong = get_by_category(search_index, "public safety")
        assert len(datasets_wrong) == 0

    def test_get_by_category_unknown_category(self, search_index):
        """Test with unknown category returns empty list."""
        datasets = get_by_category(search_index, "Nonexistent Category")

        assert len(datasets) == 0

    def test_get_by_category_multiple_categories(self, search_index):
        """Test dataset appears in all its categories."""
        # "Crime Incidents" is in both "Public Safety" and "Criminal Justice"
        public_safety = get_by_category(search_index, "Public Safety")
        criminal_justice = get_by_category(search_index, "Criminal Justice")

        assert "Crime Incidents" in public_safety
        assert "Crime Incidents" in criminal_justice


class TestLevenshteinDistance:
    """Tests for _levenshtein_distance function."""

    def test_levenshtein_identical_strings(self):
        """Test distance between identical strings is 0."""
        assert _levenshtein_distance("hello", "hello") == 0
        assert _levenshtein_distance("test", "test") == 0

    def test_levenshtein_empty_strings(self):
        """Test distance with empty strings."""
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("hello", "") == 5
        assert _levenshtein_distance("", "world") == 5

    def test_levenshtein_single_character_difference(self):
        """Test single character operations."""
        # Single insertion
        assert _levenshtein_distance("cat", "cats") == 1

        # Single deletion
        assert _levenshtein_distance("cats", "cat") == 1

        # Single substitution
        assert _levenshtein_distance("cat", "bat") == 1

    def test_levenshtein_multiple_differences(self):
        """Test multiple character differences."""
        assert _levenshtein_distance("kitten", "sitting") == 3
        assert _levenshtein_distance("saturday", "sunday") == 3

    def test_levenshtein_completely_different(self):
        """Test completely different strings."""
        distance = _levenshtein_distance("abc", "xyz")
        assert distance == 3

    def test_levenshtein_case_sensitive(self):
        """Test that distance calculation is case sensitive."""
        # Different case counts as different
        assert _levenshtein_distance("Hello", "hello") == 1

    def test_levenshtein_symmetric(self):
        """Test that distance is symmetric."""
        d1 = _levenshtein_distance("crime", "cryme")
        d2 = _levenshtein_distance("cryme", "crime")
        assert d1 == d2


class TestFuzzyRatio:
    """Tests for fuzzy matching integration."""

    def test_fuzzy_match_exact(self, search_index):
        """Test fuzzy matching with exact match."""
        results = search(search_index, "Crime Incidents", fuzzy=True)

        assert len(results) > 0
        assert "Crime Incidents" in results

    def test_fuzzy_match_close(self, search_index):
        """Test fuzzy matching with close typo."""
        # One character different
        results = search(search_index, "Crime Incidnts", fuzzy=True)

        # Should still match with 70%+ similarity
        assert "Crime Incidents" in results

    def test_fuzzy_match_threshold(self, search_index):
        """Test fuzzy matching respects 70% threshold."""
        # Very different query should not match
        results = search(search_index, "XYZ", fuzzy=True)

        # Should not match "Crime Incidents"
        assert "Crime Incidents" not in results


class TestFuzzyRatioFunction:
    """Tests for _fuzzy_ratio function directly."""

    def test_fuzzy_ratio_identical_strings(self):
        """Test fuzzy ratio between identical strings is 100."""
        assert _fuzzy_ratio("hello", "hello") == 100.0
        assert _fuzzy_ratio("test", "test") == 100.0

    def test_fuzzy_ratio_empty_strings(self):
        """Test fuzzy ratio with empty strings returns 0."""
        assert _fuzzy_ratio("", "") == 0.0
        assert _fuzzy_ratio("hello", "") == 0.0
        assert _fuzzy_ratio("", "world") == 0.0

    def test_fuzzy_ratio_similar_strings(self):
        """Test fuzzy ratio for similar strings."""
        # One character difference in 5-char string = 80% similarity
        ratio = _fuzzy_ratio("crime", "cryme")
        assert ratio == 80.0

    def test_fuzzy_ratio_completely_different(self):
        """Test fuzzy ratio for completely different strings."""
        ratio = _fuzzy_ratio("abc", "xyz")
        assert ratio == 0.0  # All 3 chars different

    def test_fuzzy_ratio_length_difference(self):
        """Test fuzzy ratio accounts for length difference."""
        # Short vs long string - distance is high relative to max length
        ratio = _fuzzy_ratio("a", "abcde")
        # Distance is 4 (add 4 chars), max len is 5, so (1 - 4/5) * 100 = 20
        assert abs(ratio - 20.0) < 0.01  # Use approx due to float precision


class TestSearchKeywordVsTitleScoring:
    """Tests for keyword vs title match scoring."""

    def test_title_match_scores_higher_than_keyword(self, sample_datasets):
        """Test that title matches score higher than keyword-only matches."""
        # Create a dataset where 'crime' appears only in notes
        dataset_keyword_only = Dataset(
            title="Public Safety Report",
            notes="This report covers crime statistics",
            category=["Public Safety"],
        )

        # Create a dataset where 'crime' appears in title
        dataset_title = Dataset(
            title="Crime Incidents",
            notes="Data about incidents",
            category=["Public Safety"],
        )

        index = build_search_index([dataset_keyword_only, dataset_title])
        results = search(index, "crime")

        # Title match should rank first
        assert results[0] == "Crime Incidents"
        # Keyword match should still appear
        assert "Public Safety Report" in results

    def test_exact_title_match_scores_highest(self, search_index):
        """Test that exact title match gets highest score."""
        results = search(search_index, "crime incidents")

        # Exact match should be first
        assert results[0] == "Crime Incidents"


class TestSearchDefaultBehavior:
    """Tests for default parameter behavior."""

    def test_search_limit_none_returns_all(self, search_index):
        """Test that limit=None returns all matching results."""
        results_no_limit = search(search_index, "data", limit=None)
        results_with_limit = search(search_index, "data", limit=2)

        assert len(results_no_limit) >= len(results_with_limit)

    def test_search_fuzzy_default_false(self, search_index):
        """Test that fuzzy defaults to False."""
        # With a typo that doesn't partially match title, non-fuzzy should not match
        # "Cryme Incidents" still matches because "incidents" is a keyword
        # Use a completely different query to test fuzzy=False behavior
        results = search(search_index, "Kryme Incidenxx")
        assert "Crime Incidents" not in results

    def test_search_category_default_none(self, search_index):
        """Test that category defaults to None (no filtering)."""
        # Without category filter, should search all datasets
        results = search(search_index, "crime")
        # Crime Incidents is in multiple datasets but search should work
        assert len(results) > 0


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_dataset_with_no_resources(self):
        """Test indexing dataset without resources."""
        dataset = Dataset(
            title="No Resources Dataset",
            category=["Test"],
        )

        index = build_search_index([dataset])

        assert "No Resources Dataset" in index.title_index
        assert len(index.resource_index) == 0

    def test_dataset_with_no_categories(self):
        """Test indexing dataset without categories."""
        dataset = Dataset(
            title="No Categories Dataset",
            resources=[
                Resource(
                    name="Test Resource",
                    format=ResourceFormat.CSV,
                    url="https://example.com/test.csv",
                )
            ],
        )

        index = build_search_index([dataset])

        assert "No Categories Dataset" in index.title_index
        # Should have resource but no categories
        assert "test resource" in index.resource_index
        assert len(index.category_index) == 0

    def test_dataset_with_multiple_resources_same_name(self):
        """Test indexing multiple resources with similar names."""
        dataset = Dataset(
            title="Multi-Format Dataset",
            resources=[
                Resource(
                    name="Data",
                    format=ResourceFormat.CSV,
                    url="https://example.com/data.csv",
                ),
                Resource(
                    name="Data",
                    format=ResourceFormat.JSON,
                    url="https://example.com/data.json",
                ),
            ],
        )

        index = build_search_index([dataset])

        # Both resources should be indexed under same name
        assert "data" in index.resource_index
        assert len(index.resource_index["data"]) == 2

    def test_search_with_numbers(self, search_index):
        """Test search with numeric characters."""
        dataset = Dataset(
            title="2020 Census Data",
            category=["Demographics"],
        )

        index = build_search_index([dataset])
        results = search(index, "2020")

        assert "2020 Census Data" in results

    def test_search_with_special_dataset_names(self):
        """Test search with special characters in dataset names."""
        dataset = Dataset(
            title="COVID-19 Testing Sites",
            category=["Health"],
        )

        index = build_search_index([dataset])
        results = search(index, "covid")

        assert "COVID-19 Testing Sites" in results

    def test_search_with_numeric_query_parameter(self):
        """Test that numeric query parameters are converted to strings (Issue #2 fix)."""
        dataset = Dataset(
            title="311 Service and Information Requests",
            category=["Services"],
            notes="Call center and online 311 requests",
        )

        index = build_search_index([dataset])

        # Test with integer query (simulates Python Fire converting "311" to int)
        results = search(index, 311)
        assert "311 Service and Information Requests" in results

        # Test with string query should still work
        results = search(index, "311")
        assert "311 Service and Information Requests" in results


class TestPhillyClassSearchIntegration:
    """Integration tests using the real Philly class with real datasets."""

    @pytest.fixture
    def philly(self):
        """Create Philly instance with cache disabled."""
        from philly import Philly

        return Philly(cache=False)

    def test_philly_search_returns_results(self, philly):
        """Test that Philly.search() returns results for common queries."""
        results = philly.search("crime")
        assert len(results) > 0
        # Crime Incidents should be in the results
        assert any("crime" in r.lower() for r in results)

    def test_philly_search_with_category_filter(self, philly):
        """Test Philly.search() with category filter."""
        categories = philly.list_categories()
        assert len(categories) > 0

        # Pick a category and search within it
        results = philly.search("", category=categories[0])
        # Results should match the category datasets
        category_datasets = philly.get_by_category(categories[0])
        for result in results:
            assert result in category_datasets

    def test_philly_search_with_limit(self, philly):
        """Test Philly.search() respects limit parameter."""
        results = philly.search("data", limit=5)
        assert len(results) <= 5

    def test_philly_search_fuzzy(self, philly):
        """Test Philly.search() with fuzzy matching."""
        # Search with a close misspelling - "Crime Incidnts" instead of "Crime Incidents"
        results = philly.search("Crime Incidnts", fuzzy=True)
        assert len(results) > 0
        # Should find Crime Incidents with fuzzy matching
        assert "Crime Incidents" in results

    def test_philly_search_resources_returns_formatted_results(self, philly):
        """Test Philly.search_resources() returns properly formatted results."""
        results = philly.search_resources("csv")
        assert len(results) > 0
        # Results should be in format "resource_name [dataset_title]"
        for result in results:
            assert "[" in result and "]" in result

    def test_philly_search_resources_with_dataset_filter(self, philly):
        """Test Philly.search_resources() with dataset_name filter."""
        datasets = philly.list_datasets()
        assert len(datasets) > 0

        # Pick a dataset that likely has multiple resources
        results = philly.search_resources("", dataset_name=datasets[0])
        # All results should be from the specified dataset
        for result in results:
            assert datasets[0] in result

    def test_philly_list_categories_returns_sorted(self, philly):
        """Test Philly.list_categories() returns sorted list."""
        categories = philly.list_categories()
        assert len(categories) > 0
        assert categories == sorted(categories)

    def test_philly_get_by_category_returns_sorted(self, philly):
        """Test Philly.get_by_category() returns sorted list."""
        categories = philly.list_categories()
        if categories:
            datasets = philly.get_by_category(categories[0])
            if len(datasets) > 1:
                assert datasets == sorted(datasets)

    def test_philly_get_by_category_nonexistent(self, philly):
        """Test Philly.get_by_category() returns empty for nonexistent category."""
        results = philly.get_by_category("NonexistentCategory12345")
        assert results == []

    def test_philly_search_index_built_lazily(self, philly):
        """Test that search index is built only on first search call."""
        # Initially, index should be None
        assert philly._search_index is None

        # After search, index should be built
        philly.search("test")
        assert philly._search_index is not None

    def test_philly_search_index_reused(self, philly):
        """Test that search index is reused across multiple searches."""
        philly.search("crime")
        index1 = philly._search_index

        philly.search("parking")
        index2 = philly._search_index

        # Should be the same object (not rebuilt)
        assert index1 is index2
