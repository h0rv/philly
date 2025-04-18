from os import listdir
from os.path import abspath, dirname, join
from typing import Any, Callable

from cheesesnake.models.dataset import Dataset
from cheesesnake.models.search import MatchType, SearchResult


class Cheesesnake:
    def __init__(self):
        self._module_dir = dirname(abspath(__file__))
        self.datasets = sorted(self._load_datasets(), key=lambda x: x.title)

        # Basic dataset lookups
        self.titles = sorted([dataset.title for dataset in self.datasets])
        self.title_dataset_map = {dataset.title: dataset for dataset in self.datasets}

        # Precompute search indexes
        self.search_indexes = {
            MatchType.TITLE: {},
            MatchType.NOTES: {},
            MatchType.ORGANIZATION: {},
            MatchType.CATEGORY: {},
        }

        for dataset in self.datasets:
            # Index title
            self._index_field(MatchType.TITLE, dataset.title, dataset)

            # Index notes
            if dataset.notes:
                self._index_field(MatchType.NOTES, dataset.notes, dataset)

            # Index organization
            if dataset.organization:
                self._index_field(MatchType.ORGANIZATION, dataset.organization, dataset)

            # Index categories
            if dataset.category:
                for category in dataset.category:
                    self._index_field(MatchType.CATEGORY, category, dataset)

    def filter(self, field: str, filter_fn: Callable) -> list[Dataset]:
        """Filter datasets using a functional approach with path resolution."""
        path_parts = field.split('.')
        
        def apply_filter(dataset):
            values = self._extract_values(dataset, path_parts)
            return any(filter_fn(v) for v in values if v is not None)
            
        return list(filter(apply_filter, self.datasets))
    
    def _extract_values(self, obj: Any, path_parts: list[str]) -> list[Any]:
        """Extract all values matching a path pattern."""
        if not path_parts:
            return [obj]
            
        part, *rest = path_parts
        
        if part == '*':
            if not isinstance(obj, list):
                return []
            values = []
            for item in obj:
                values.extend(self._extract_values(item, rest))
            return values
        else:
            value = getattr(obj, part, None)
            if value is None:
                return []
            return self._extract_values(value, rest)

    def search(self, query: str) -> list[SearchResult]:
        query = query.lower()
        results: dict[int, SearchResult] = {}

        # Score weights for different match types
        match_scores = {
            MatchType.TITLE: 5.0,
            MatchType.NOTES: 1.0,
            MatchType.ORGANIZATION: 2.0,
            MatchType.CATEGORY: 2.0,
        }

        # Search in all indexed fields
        for match_type, index in self.search_indexes.items():
            for indexed_value, datasets in index.items():
                if query in indexed_value:
                    # Exact match gets higher score
                    score_multiplier = 1.5 if indexed_value == query else 1.0
                    score = match_scores[match_type] * score_multiplier

                    for dataset in datasets:
                        dataset_id = id(dataset)
                        if dataset_id not in results:
                            results[dataset_id] = SearchResult(dataset=dataset)
                        results[dataset_id].add_match(match_type, score)

        # Sort by score (descending) then by title
        return sorted(results.values(), key=lambda x: (-x.score, x.dataset.title))

    @property
    def _datasets_dir(self):
        return join(self._module_dir, "datasets")

    def _load_datasets(self):
        return [
            Dataset.from_file(join(self._datasets_dir, file))
            for file in listdir(self._datasets_dir)
            if file.endswith(".yaml")
        ]

    def _index_field(self, match_type: MatchType, value: str, dataset: Dataset) -> None:
        """Index a field value for searching."""
        value_lower = value.lower()
        if value_lower not in self.search_indexes[match_type]:
            self.search_indexes[match_type][value_lower] = []
        self.search_indexes[match_type][value_lower].append(dataset)
