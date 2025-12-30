from pydantic import BaseModel, field_validator

from philly.models.yaml_enum import YamlEnum


class ResourceFormat(YamlEnum):
    API = "api"
    APP = "app"
    APPLICATION = "application"
    CSV = "csv"
    ECW = "ecw"
    GDB = "gdb"
    GEOJSON = "geojson"
    GEOPACKAGE = "geopackage"
    GEOPARQUET = "geoparquet"
    GEOSERVICE = "geoservice"
    GTFS = "gtfs"
    GTFS_RT = "gtfs_rt"
    HTML = "html"
    IMG = "img"
    JPEG = "jpeg"
    JSON = "json"
    KML = "kml"
    KMZ = "kmz"
    LAS = "las"
    PDF = "pdf"
    PNG = "png"
    PNG_24 = "png_24"
    RSS = "rss"
    SHP = "shp"
    TEXT = "txt"
    TIF = "tif"
    TIFF = "tiff"
    XLSX = "xlsx"
    XML = "xml"
    XSLX = "xslx"
    ZIP = "zip"

    def __str__(self) -> str:
        return self.value


class Resource(BaseModel):
    name: str
    format: ResourceFormat
    url: str | None = None

    @classmethod
    @field_validator("format", mode="before")
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

    @classmethod
    def from_dict(cls, data: dict) -> "Resource":
        return cls(**data)

    def __str__(self) -> str:
        return f"""\
{self.name}
    {self.format}
    {self.url}
"""
