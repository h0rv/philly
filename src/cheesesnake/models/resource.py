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


class ResourceFormat(YamlEnum):
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
    format: ResourceFormat
    url: str | None = None

    @field_validator("format", mode="before")
    @classmethod
    def lowercase_format(cls, value: any) -> any:
        if not isinstance(value, str):
            raise ValueError("format must be a string")

        value = str(value).strip()

        normalized = value.replace(" ", "_").replace("-", "_").lower()

        try:
            return ResourceFormat(normalized)
        except ValueError as e:
            valid_formats = [f.value for f in ResourceFormat]

            for fmt in valid_formats:
                fmt_no_underscores = fmt.replace("_", "")
                normalized_no_underscores = normalized.replace("_", "")

                if fmt_no_underscores == normalized_no_underscores:
                    return ResourceFormat(fmt)

            raise ValueError(
                f"'{value}' is not a valid format. Valid formats are: {', '.join(valid_formats)}"
            ) from e
