import asyncio
import logging
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from cheesesnake.models import Dataset, Resource
from cheesesnake.services import ResourcesLoader


class Cheesesnake:
    """
    Cheesesnake provides methods for loading, extracting, and querying OpenDataPhilly datasets and resources.

    Args:
        datasets_view_name (str): Name of the DuckDB view for datasets. Defaults to "datasets".

    Example:
        cs = Cheesesnake()
        df = cs.query("SELECT * FROM datasets")
    """

    duckdb_extensions = ["spatial"]

    def __init__(
        self,
        db_dir: str | Path = ".",
        db_name: str = "cheesesnake.db",
        datasets_table_name: str = "datasets",
    ) -> None:
        self._module_dir: Path = Path(__file__).parent.resolve()
        self._datasets_dir: Path = self._module_dir / "datasets"

        self.logger = logging.getLogger(__name__)

        self.datasets: list[Dataset] = sorted(
            [
                Dataset.from_file(str(self._datasets_dir / file))
                for file in self._datasets_dir.glob("*.yaml")
            ],
            key=lambda x: x.title,
        )
        self.datasets_df = pd.DataFrame(
            [dataset.model_dump() for dataset in self.datasets]
        )

        self.resources_loader = ResourcesLoader()

        self.duckdb_path = Path(db_dir) / db_name
        if not self.duckdb_path.exists():
            self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        self.duckdb: duckdb.DuckDBPyConnection = duckdb.connect(str(self.duckdb_path))

        for extension in self.duckdb_extensions:
            self.duckdb.install_extension(extension)
            self.duckdb.load_extension(extension)

        self.query = self.duckdb.query

        self.add_table(
            datasets_table_name,
            self.datasets_df,
            overwrite=True,
        )

    def to_dataframe_safe(self, resource: object) -> pd.DataFrame | None:
        # workaround for https://github.com/duckdb/duckdb-spatial/issues/311
        if isinstance(resource, gpd.GeoDataFrame) and isinstance(
            resource.geometry, gpd.GeoSeries
        ):
            resource["geometry"] = resource["geometry"].to_wkt()

        try:
            return pd.DataFrame(resource)
        except Exception as e:
            # Check if resource is a Resource object with a name attribute
            resource_name = getattr(resource, "name", "unknown")
            self.logger.debug(
                f"[WARNING] Resource {resource_name} is not a valid DataFrame (error: {e}). Skipping."
            )
            return None

    def tables(self) -> list[str]:
        return self.duckdb.execute("SHOW TABLES").fetchall()

    def add_table(self, name: str, data: object, overwrite: bool = False) -> None:
        if overwrite:
            result = self.duckdb.execute(f'DROP TABLE IF EXISTS "{name}"')
            result.fetchall()  # Process the result
            self.duckdb.commit()

        df = (
            self.to_dataframe_safe(data) if not isinstance(data, pd.DataFrame) else data
        )

        if df is None or df.empty or len(df.columns) == 0:
            return

        self.duckdb.from_df(df).create(name)

    async def load(
        self,
        resource: Resource,
        ignore_load_errors: bool = True,
    ) -> object | None:
        if not self.resources_loader.is_dataframable(resource):
            return None

        if not resource.url:
            return None

        try:
            data = await self.resources_loader.load(resource, ignore_load_errors)
        except Exception as e:
            if ignore_load_errors:
                raise e
            self.logger.warning(
                f"Resource {resource.name} could not be loaded (error: {e}). Skipping."
            )

        if data is None:
            return None

        if not isinstance(data, pd.DataFrame):
            # Run dataframe conversion in a thread to avoid blocking
            data = self.to_dataframe_safe(data)
            if data is None:
                return None

        return data

    async def load_into_table(
        self,
        resource: Resource,
        table_name: str,
        ignore_load_errors: bool = True,
    ) -> object | None:
        data = await self.load(resource, ignore_load_errors)
        if data is None:
            return None

        self.add_table(table_name, data, overwrite=True)

        return data

    async def load_all(self, show_progress: bool = False) -> list[object]:
        tasks = []
        for dataset in self.datasets:
            for resource in dataset.resources or []:
                tasks.append(self.load(resource, ignore_load_errors=False))

        gather_fn = tqdm_asyncio.gather if show_progress else asyncio.gather

        return await gather_fn(*tasks)

    async def load_all_into_tables(self, show_progress: bool = False) -> list[object]:
        tasks = [
            self.load_into_table(
                resource=resource,
                table_name=f"{dataset.title}_{resource.name}_{resource.format}",
                ignore_load_errors=False,
            )
            for dataset in self.datasets
            for resource in dataset.resources or []
        ]

        gather_fn = tqdm_asyncio.gather if show_progress else asyncio.gather

        return await gather_fn(*tasks)
