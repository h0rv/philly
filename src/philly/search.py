"""Search and discovery API for Philly datasets."""

from dataclasses import dataclass, field

from philly.models.dataset import Dataset
from philly.models.resource import Resource

# Stopwords for keyword extraction
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "is",
    "it",
    "this",
    "that",
}


@dataclass
class SearchIndex:
    """Search index for efficient dataset and resource lookup."""

    title_index: dict[str, Dataset] = field(default_factory=dict)
    normalized_titles: dict[str, str] = field(default_factory=dict)
    category_index: dict[str, list[str]] = field(default_factory=dict)
    keyword_index: dict[str, list[str]] = field(default_factory=dict)
    resource_index: dict[str, list[tuple[str, Resource]]] = field(default_factory=dict)
    org_index: dict[str, list[str]] = field(default_factory=dict)


def _extract_keywords(text: str | None) -> set[str]:
    """Extract keywords from text by filtering stopwords and short words."""
    if not text:
        return set()

    words = str(text).lower().split()
    keywords: set[str] = set()

    for word in words:
        # Remove punctuation from both ends
        word = word.strip(".,;:!?()[]{}\"'")
        # Keep words that are at least 3 chars and not stopwords
        if len(word) >= 3 and word not in STOPWORDS:
            keywords.add(word)

    return keywords


def build_search_index(datasets: list[Dataset]) -> SearchIndex:
    """Build a search index from a list of datasets.

    Args:
        datasets: List of Dataset objects to index

    Returns:
        SearchIndex with populated indices
    """
    index = SearchIndex()

    for dataset in datasets:
        title = dataset.title
        normalized_title = title.lower()

        # Index by title
        index.title_index[title] = dataset
        index.normalized_titles[normalized_title] = title

        # Index by category
        if dataset.category:
            for cat in dataset.category:
                if cat not in index.category_index:
                    index.category_index[cat] = []
                index.category_index[cat].append(title)

        # Extract and index keywords from title and notes
        keywords = _extract_keywords(title)
        if dataset.notes:
            keywords.update(_extract_keywords(dataset.notes))

        for keyword in keywords:
            if keyword not in index.keyword_index:
                index.keyword_index[keyword] = []
            index.keyword_index[keyword].append(title)

        # Index resources by name
        if dataset.resources:
            for resource in dataset.resources:
                resource_name = resource.name.lower()
                if resource_name not in index.resource_index:
                    index.resource_index[resource_name] = []
                index.resource_index[resource_name].append((title, resource))

        # Index by organization
        if dataset.organization:
            org = dataset.organization
            if org not in index.org_index:
                index.org_index[org] = []
            index.org_index[org].append(title)

    return index


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _fuzzy_ratio(s1: str, s2: str) -> float:
    """Calculate fuzzy match ratio between two strings (0-100)."""
    if not s1 or not s2:
        return 0.0

    distance = _levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 100.0

    ratio = (1 - distance / max_len) * 100
    return ratio


def search(
    index: SearchIndex,
    query: str,
    category: str | None = None,
    fuzzy: bool = False,
    limit: int | None = None,
) -> list[str]:
    """Search for datasets by query.

    Args:
        index: SearchIndex to search
        query: Search query string
        category: Optional category filter
        fuzzy: Whether to use fuzzy matching (default: False)
        limit: Maximum number of results to return

    Returns:
        List of dataset titles sorted by relevance
    """
    query_lower = str(query).lower()
    results: dict[str, int] = {}  # title -> relevance score

    # Search in titles (substring or fuzzy match)
    for normalized_title, original_title in index.normalized_titles.items():
        if fuzzy:
            # Fuzzy match with 70% threshold
            ratio = _fuzzy_ratio(query_lower, normalized_title)
            if ratio >= 70:
                # Higher score for better matches
                results[original_title] = int(ratio * 10)  # Scale up for priority
        else:
            # Substring match
            if query_lower in normalized_title:
                # Exact match gets higher score
                if query_lower == normalized_title:
                    results[original_title] = 1000
                else:
                    results[original_title] = 500

    # Search in keywords
    query_keywords = _extract_keywords(query)
    for keyword in query_keywords:
        if keyword in index.keyword_index:
            for title in index.keyword_index[keyword]:
                if title not in results:
                    results[title] = 0
                results[title] += 100  # Keyword match is good but less than title match

    # Filter by category if provided
    if category:
        category_datasets = set(index.category_index.get(category, []))
        results = {
            title: score
            for title, score in results.items()
            if title in category_datasets
        }

    # Sort by relevance score (descending)
    sorted_results = sorted(results.keys(), key=lambda t: results[t], reverse=True)

    # Apply limit if specified
    if limit is not None:
        sorted_results = sorted_results[:limit]

    return sorted_results


def search_resources(
    index: SearchIndex,
    query: str,
    dataset_filter: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Search for resources by name.

    Args:
        index: SearchIndex to search
        query: Search query string for resource name
        dataset_filter: Optional dataset title filter
        limit: Maximum number of results to return

    Returns:
        List of strings in format "resource_name [dataset_title]"
    """
    query_lower = str(query).lower()
    results: list[str] = []

    # Search resource names
    for resource_name, resource_list in index.resource_index.items():
        if query_lower in resource_name:
            for dataset_title, resource in resource_list:
                # Apply dataset filter if provided
                if dataset_filter and dataset_title != dataset_filter:
                    continue

                result_str = f"{resource.name} [{dataset_title}]"
                results.append(result_str)

    # Sort by resource name for consistency
    results.sort()

    # Apply limit if specified
    if limit is not None:
        results = results[:limit]

    return results


def list_categories(index: SearchIndex) -> list[str]:
    """Get a sorted list of all categories.

    Args:
        index: SearchIndex to query

    Returns:
        Sorted list of category names
    """
    return sorted(index.category_index.keys())


def get_by_category(index: SearchIndex, category: str) -> list[str]:
    """Get all datasets in a specific category.

    Args:
        index: SearchIndex to query
        category: Category name to filter by

    Returns:
        List of dataset titles in the category (sorted)
    """
    titles = index.category_index.get(category, [])
    return sorted(titles)
