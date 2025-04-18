import json
from enum import Enum

import yaml

from pydantic import BaseModel, field_validator


class YamlEnum(str, Enum):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for dumper in (yaml.SafeDumper, yaml.Dumper):
            dumper.add_multi_representer(
                cls, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v.value)
            )


class Format(YamlEnum):
    API = "api"
    CSV = "csv"
    JSON = "json"
    GEOJSON = "geojson"
    HTML = "html"
    KML = "kml"
    KMZ = "kmz"
    SHP = "shp"
    XML = "xml"
    JPEG = "jpeg"
    TIFF = "tiff"
    TIF = "tif"
    ECW = "ecw"
    XLSX = "xlsx"
    XSLX = "xslx"
    TEXT = "txt"
    APP = "app"
    APPLICATION = "application"
    RSS = "rss"
    GDB = "gdb"
    GEOSERVICE = "geoservice"
    PDF = "pdf"
    LAS = "las"
    ZIP = "zip"
    IMG = "img"
    PNG = "png"
    PNG_24 = "png_24"
    GTFS = "gtfs"
    GEOPARQUET = "geoparquet"
    GTFS_RT = "gtfs_rt"

    def __str__(self) -> str:
        return self.value


class Resource(BaseModel):
    name: str
    format: Format
    url: str | None = None

    @field_validator("format", mode="before")
    @classmethod
    def lowercase_format(cls, value: any) -> any:
        if not isinstance(value, str):
            raise ValueError("format must be a string")

        normalized = value.replace(" ", "_").replace("-", "_").lower()

        try:
            return Format(normalized)
        except ValueError:
            valid_formats = [f.value for f in Format]

            for fmt in valid_formats:
                fmt_no_underscores = fmt.replace("_", "")
                normalized_no_underscores = normalized.replace("_", "")

                if fmt_no_underscores == normalized_no_underscores:
                    return Format(fmt)

            raise ValueError(
                f"'{value}' is not a valid format. Valid formats are: {', '.join(valid_formats)}"
            )


class Dataset(BaseModel):
    title: str
    organization: str | None = None
    notes: str | None = None
    area_of_interest: str | None = None
    created: str | None = None
    category: list[str] | None = None
    license: str | None = None
    maintainer: str | None = None
    maintainer_email: str | None = None
    maintainer_link: str | None = None
    maintainer_phone: str | None = None
    opendataphilly_rating: str | None = None
    source: str | None = None
    time_period: str | int | None = None
    usage: str | None = None
    resources: list[Resource] | None = None

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
