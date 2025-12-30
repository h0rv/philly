import asyncio
import logging
from pathlib import Path

import httpx
from tqdm import tqdm

from philly.models import Dataset, Resource
from philly.loaders import load, use_http_client


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

    def list_resources(self, dataset_name: str, names_only: bool = False) -> str:
        dataset = self._get_dataset(dataset_name)

        resources = dataset.resources or []

        if names_only:
            return "\n".join([r.name for r in resources])

        return "".join([str(r) for r in resources])

    def list_all_resources(self) -> str:
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

        data: object | None = None
        try:
            data = await load(resource, ignore_errors=ignore_load_errors)
        except Exception as e:
            if not ignore_load_errors:
                raise e
            self._logger.warning(
                f"Resource {resource.name} could not be loaded (error: {e}). Skipping."
            )
            return None

        if data is None:
            return None

        return data

    async def load_all(
        self,
        show_progress: bool = False,
        ignore_load_errors: bool = True,
        max_concurrency: int | None = 20,
    ) -> list[object | None]:
        resources = [
            resource
            for dataset in self.datasets
            for resource in (dataset.resources or [])
        ]
        if not resources:
            return []

        concurrency = (
            max_concurrency
            if max_concurrency and max_concurrency > 0
            else len(resources)
        )
        results: list[object | None] = [None] * len(resources)
        semaphore = asyncio.Semaphore(concurrency)

        progress = tqdm(total=len(resources)) if show_progress else None

        async def _load_one(index: int, resource: Resource) -> None:
            async with semaphore:
                if ignore_load_errors:
                    try:
                        results[index] = await load(resource, ignore_errors=True)
                    except Exception as e:
                        self._logger.warning(
                            f"Resource load failed for {resource}: {e}. Skipping."
                        )
                        results[index] = None
                else:
                    results[index] = await load(resource, ignore_errors=False)
            if progress is not None:
                progress.update(1)

        async with httpx.AsyncClient() as client:
            async with use_http_client(client):
                async with asyncio.TaskGroup() as tg:
                    for index, resource in enumerate(resources):
                        tg.create_task(_load_one(index, resource))

        if progress is not None:
            progress.close()

        return results
