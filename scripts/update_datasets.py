import re
import os
import asyncio

import yaml

from cheesesnake.services.github import get_all_files_contents
from cheesesnake.models import Dataset


async def get_all_datasets():
    async for file in get_all_files_contents(
        repo="opendataphilly/opendataphilly-jkan",
        path="_datasets",
    ):
        # split the document by its ---

        files = file.split("---")

        for file in files:
            file = file.strip()
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
            f"src/cheesesnake/datasets/{clean_title}.yaml",
            "w",
            encoding="utf-8",
        ) as f:
            print(f"Writing {dataset.title} to {f.name}")
            yaml.dump(dataset.model_dump(), f)


if __name__ == "__main__":
    asyncio.run(main())
