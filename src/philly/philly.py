import asyncio
import logging
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from philly.models import Dataset
from philly.loaders import load


class Philly:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

        self._module_dir: Path = Path(__file__).parent.resolve()
        self._datasets_dir: Path = self._module_dir / "datasets"

        self.datasets: list[Dataset] = sorted(
            [
                Dataset.from_file(str(self._datasets_dir / file))
                for file in self._datasets_dir.glob("*.yaml")
            ],
            key=lambda x: x.title,
        )

        self._datasets_map: dict[str, Dataset] = {
            dataset.title: dataset for dataset in self.datasets
        }

    def _get_dataset(self, dataset_name: str) -> Dataset:
        dataset = self._datasets_map.get(dataset_name)

        if not dataset:
            raise ValueError(f"dataset '{dataset_name}' does not exist")

        return dataset

    def list_datasets(self) -> list[str]:
        return [d.title for d in self.datasets]

    def list_resources(self, dataset_name: str, names_only: bool = False) -> list[str]:
        dataset = self._get_dataset(dataset_name)

        resources = dataset.resources or []

        if names_only:
            return "\n".join([r.name for r in resources])

        return "".join([str(r) for r in resources])

    def list_all_resources(self) -> list[str]:
        resources = [
            f"{resource.name} [{dataset.title}]"
            for dataset in self.datasets
            for resource in (dataset.resources or [])
        ]

        return "\n".join(resources)

    async def load(
        self,
        dataset_name: str,
        resource_name: str,
        format: str | None = None,
        ignore_load_errors: bool = False,
    ) -> object | None:
        dataset = self._get_dataset(dataset_name)

        resource = dataset.get_resource(resource_name, format=format)

        if not resource.url:
            return None

        try:
            data = await load(resource, ignore_load_errors)
        except Exception as e:
            if ignore_load_errors:
                raise e
            self._logger.warning(
                f"Resource {resource.name} could not be loaded (error: {e}). Skipping."
            )

        if data is None:
            return None

        return data

    async def load_all(self, show_progress: bool = False) -> list[object]:
        tasks = []
        for dataset in self.datasets:
            for resource in dataset.resources or []:
                tasks.append(load(resource, ignore_load_errors=False))

        gather_fn = tqdm_asyncio.gather if show_progress else asyncio.gather

        return await gather_fn(*tasks)
