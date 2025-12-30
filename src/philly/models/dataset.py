import json

import yaml
from pydantic import BaseModel

from philly.models.resource import Resource


class Dataset(BaseModel):
    title: str
    organization: str | None = None
    notes: str | None = None
    area_of_interest: str | None = None
    created: str | None = None
    license: str | None = None
    maintainer: str | None = None
    maintainer_email: str | None = None
    maintainer_link: str | None = None
    maintainer_phone: str | None = None
    opendataphilly_rating: str | None = None
    source: str | None = None
    time_period: str | int | None = None
    usage: str | None = None
    category: list[str] | None = None
    resources: list[Resource] | None = None

    def get_resource(
        self,
        resource_name: str,
        format: str | None = None,
    ) -> Resource:
        r = [
            r
            for r in self.resources
            if r.name == resource_name and (format is None or r.format == format)
        ]

        if len(r) == 0:
            raise ValueError(
                f"resource '{resource_name}' does not exist for dataset '{self.title}'"
            )

        if len(r) > 1:
            raise ValueError(
                "resource name '{resource_name}' is ambiguous, requires format to be specified"
            )

        return r[0]

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        # Handle comma-separated formats in resources
        if "resources" in data and data["resources"]:
            resources = []
            for resource in data["resources"]:
                if (
                    "format" in resource
                    and isinstance(resource["format"], str)
                    and "," in resource["format"]
                ):
                    formats = [fmt.strip() for fmt in resource["format"].split(",")]
                    for fmt in formats:
                        # Create a copy of the resource for each format
                        new_resource = resource.copy()
                        new_resource["format"] = fmt
                        resources.append(new_resource)
                else:
                    resources.append(resource)
            data["resources"] = resources

        return cls(**data)

    @classmethod
    def from_yaml(cls, data: str, retry: bool = True) -> "Dataset":
        data = data.replace("\t", " ")
        try:
            return cls.from_dict(yaml.safe_load(data))
        except yaml.YAMLError:
            if retry and data.strip().startswith("-"):
                # Extract first line content after dash as title
                lines = data.split("\n")
                title = lines[0].split("-", 1)[1].strip()
                rest = "\n".join(lines[1:])
                fixed_data = f"title: {title}\n{rest}"
                return cls.from_yaml(fixed_data, retry=False)
            raise

    @classmethod
    def from_json(cls, data: str) -> "Dataset":
        return cls.from_dict(json.loads(data))

    @classmethod
    def from_file(cls, file: str) -> "Dataset":
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        if file.endswith(".yaml"):
            return cls.from_yaml(content)
        elif file.endswith(".json"):
            return cls.from_json(content)
        else:
            raise ValueError(f"Unsupported file type: {file}")
