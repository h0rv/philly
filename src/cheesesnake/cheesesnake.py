from os import listdir
from os.path import abspath, dirname, join
from typing import Callable

import duckdb
import pandas as pd

from cheesesnake.models import Dataset, Resource, ResourceFormat


class Cheesesnake:
    def __init__(self) -> None:
        self._module_dir: str = dirname(abspath(__file__))
        self.datasets: list[Dataset] = sorted(
            self._load_datasets(), key=lambda x: x.title
        )

        # Basic dataset lookups
        self.titles = sorted([dataset.title for dataset in self.datasets])
        self.title_dataset_map = {dataset.title: dataset for dataset in self.datasets}

        self.datasets_df = pd.DataFrame(
            [dataset.model_dump() for dataset in self.datasets]
        )

    def query_datasets(self, query: str) -> pd.DataFrame:
        duckdb.register("datasets", self.datasets_df)
        try:
            return duckdb.query(query).to_df()
        finally:
            duckdb.unregister("datasets")

    def get_resources(self, formats: list[ResourceFormat]) -> list[Resource]:
        return [
            resource
            for dataset in self.datasets
            for resource in dataset.resources or []
            if resource.format in formats
        ]

    def filter(self, field: str, filter_fn: Callable) -> list[Dataset]:
        """Filter datasets using a functional approach with path resolution."""
        path_parts = field.split(".")

        def apply_filter(dataset):
            values = self._extract_values(dataset, path_parts)
            return any(filter_fn(v) for v in values if v is not None)

        return list(filter(apply_filter, self.datasets))

    def _extract_values(self, obj: object, path_parts: list[str]) -> list[object]:
        """Extract all values matching a path pattern."""
        if not path_parts:
            return [obj]

        part, *rest = path_parts

        if part == "*":
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

    def search_by_title(self, query: str) -> list[Resource]:
        query = query.lower()
        results: list[Resource] = []

        for dataset in self.datasets:
            if query in dataset.title.lower():
                results.extend(dataset.resources)

        return results

    def search_by_notes(self, query: str) -> list[Resource]:
        query = query.lower()
        results: list[Resource] = []

        for dataset in self.datasets:
            if query in dataset.notes.lower():
                results.extend(dataset.resources)

        return results

    @property
    def _datasets_dir(self):
        return join(self._module_dir, "datasets")

    def _load_datasets(self):
        return [
            Dataset.from_file(join(self._datasets_dir, file))
            for file in listdir(self._datasets_dir)
            if file.endswith(".yaml")
        ]
