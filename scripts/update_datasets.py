import asyncio
import os
import re

import yaml

from philly.models import Dataset
from philly.services import GitHub


async def get_all_datasets():
    async for file in GitHub.get_all_files_contents(
        repo="opendataphilly/opendataphilly-jkan",
        path="_datasets",
    ):
        # split the document by its ---

        files = file.split("---")

        for file in files:
            file = file.strip()

            # fix typo
            if "hhttps://" in file:
                file = file.replace("hhttps://", "https://")

            if file:
                try:
                    yield Dataset.from_yaml(file)
                except Exception as e:
                    print(f"Error parsing dataset: {e}")
                    print("Skipping...")


async def main():
    os.makedirs("datasets", exist_ok=True)

    async for dataset in get_all_datasets():
        # Replace any character that is not alphanumeric, dash, or underscore with underscore
        clean_title = re.sub(r"[^\w\-]", "_", dataset.title)
        with open(
            f"src/philly/datasets/{clean_title}.yaml",
            "w",
            encoding="utf-8",
        ) as f:
            print(f"Writing {dataset.title} to {f.name}")
            yaml.dump(dataset.model_dump(), f)


if __name__ == "__main__":
    asyncio.run(main())
