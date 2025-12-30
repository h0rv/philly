import csv
import json
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from contextlib import asynccontextmanager
from contextvars import ContextVar
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Awaitable, Callable

import geojson
import geopandas as gpd
import httpx
import pandas as pd
from google.transit import gtfs_realtime_pb2 as gtfs_rt

from philly.models import Resource, ResourceFormat
from philly.services import GitHub
from philly.urls import normalize_url


_http_client: ContextVar[httpx.AsyncClient | None] = ContextVar(
    "philly_http_client", default=None
)


@asynccontextmanager
async def use_http_client(client: httpx.AsyncClient):
    token = _http_client.set(client)
    try:
        yield
    finally:
        _http_client.reset(token)


async def _get_content(url: str | None) -> bytes:
    if not url:
        raise ValueError("Resource URL is not set")

    url = normalize_url(url)

    if url.startswith("http"):
        url = url.replace("http://", "https://")

    if url.startswith("https://github.com/"):
        url = GitHub.convert_app_url_to_content_url(url)

    client = _http_client.get()
    if client is None:
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=60,
            ) as response,
        ):
            response.raise_for_status()
            content = await response.aread()
            return content

    async with client.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        content = await response.aread()
        return content


async def load_csv(resource: Resource) -> list[dict[str, str]]:
    content = await _get_content(resource.url)
    csv_data = StringIO(content.decode("utf-8", errors="replace"))
    try:
        reader = csv.DictReader(csv_data)
        return list(reader)
    except csv.Error as e:
        if "new-line character seen in unquoted field" in str(e):
            # Try with universal newline mode
            csv_data = StringIO(content.decode("utf-8", errors="replace"), newline=None)
            reader = csv.DictReader(csv_data)
            return list(reader)
        raise


async def load_json(resource: Resource) -> dict[str, object]:
    content = await _get_content(resource.url)
    return json.loads(content.decode("utf-8", errors="replace"))


async def load_geojson(resource: Resource) -> dict[str, object]:
    content = await _get_content(resource.url)
    return geojson.loads(content.decode("utf-8", errors="replace"))


async def load_geopackage(resource: Resource) -> gpd.GeoDataFrame:
    content = await _get_content(resource.url)
    with BytesIO(content) as f:
        gdf = gpd.read_file(f)
    return gdf


async def load_xml(resource: Resource) -> ET.Element | None:
    content = await _get_content(resource.url)
    content_str = content.decode("utf-8", errors="replace")

    # Try to pre-process XML content to fix common issues
    if content_str.lstrip().startswith("<?xml"):
        xml_decl_end = content_str.find("?>")
        if xml_decl_end > 0:
            # Check for issues in XML declaration
            xml_decl = content_str[: xml_decl_end + 2]
            if "encoding=" in xml_decl and not (
                'encoding="utf-8"' in xml_decl or "encoding='utf-8'" in xml_decl
            ):
                # Standardize encoding to utf-8
                content_str = (
                    content_str[: xml_decl_end + 2]
                    .replace('encoding="ISO-8859-1"', 'encoding="utf-8"')
                    .replace("encoding='ISO-8859-1'", "encoding='utf-8'")
                    + content_str[xml_decl_end + 2 :]
                )

    content_str = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", content_str)

    return ET.fromstring(content_str.encode("utf-8"))


async def load_excel(resource: Resource) -> dict[str, pd.DataFrame]:
    content = await _get_content(resource.url)
    excel_file = BytesIO(content)
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    return excel_data


async def load_pdf(resource: Resource) -> bytes:
    return await _get_content(resource.url)


async def load_zip(resource: Resource) -> dict[str, bytes]:
    content = await _get_content(resource.url)
    zip_file = BytesIO(content)

    result = {}
    with zipfile.ZipFile(zip_file) as z:
        for filename in z.namelist():
            result[filename] = z.read(filename)

    return result


async def load_image(resource: Resource) -> bytes:
    return await _get_content(resource.url)


async def load_text(resource: Resource) -> str:
    content = await _get_content(resource.url)
    return content.decode("utf-8", errors="replace")


async def load_gtfs(resource: Resource) -> str:
    content = await _get_content(resource.url)
    return str(content)


async def load_gtfs_rt(resource: Resource) -> Any:
    feed_cls: Any = getattr(gtfs_rt, "FeedMessage")
    feed = feed_cls()
    content = await _get_content(resource.url)
    feed.ParseFromString(content)
    return feed


async def load_geoparquet(resource: Resource) -> gpd.GeoDataFrame:
    content = await _get_content(resource.url)
    buffer: Any = BytesIO(content)
    return gpd.read_parquet(buffer)


async def load_api(resource: Resource) -> dict[str, Any]:
    """Load data from an API endpoint asynchronously"""
    content = await _get_content(resource.url)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_content": content}


async def load_app(resource: Resource) -> bytes:
    """Load application binary data asynchronously"""
    return await _get_content(resource.url)


async def load_ecw(resource: Resource) -> bytes:
    """Load Enhanced Compression Wavelet (ECW) format asynchronously"""
    return await _get_content(resource.url)


async def load_gdb(resource: Resource) -> gpd.GeoDataFrame | NotImplementedError:
    """Load Esri File Geodatabase asynchronously"""
    raise NotImplementedError("GDB format loading requires GDAL/OGR support")


async def load_geoservice(resource: Resource) -> dict[str, Any]:
    """Load data from a geo service endpoint asynchronously"""
    content = await _get_content(resource.url)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_content": content}


async def load_html(resource: Resource) -> str:
    """Load HTML content as text asynchronously"""
    content = await _get_content(resource.url)
    return content.decode("utf-8", errors="replace")


async def load_kml(resource: Resource) -> ET.Element:
    """Load KML (Keyhole Markup Language) file asynchronously"""
    content = await _get_content(resource.url)
    return ET.fromstring(content)


async def load_kmz(resource: Resource) -> ET.Element:
    """Load KMZ (compressed KML) file asynchronously"""
    content = await _get_content(resource.url)
    kmz_file = BytesIO(content)

    with zipfile.ZipFile(kmz_file) as z:
        kml_files = [f for f in z.namelist() if f.endswith(".kml")]
        if not kml_files:
            raise ValueError("No KML file found in KMZ archive")

        kml_content = z.read(kml_files[0])
        return ET.fromstring(kml_content)


async def load_las(resource: Resource) -> bytes:
    """Load LAS (LiDAR point cloud) file asynchronously"""
    return await _get_content(resource.url)


async def load_rss(resource: Resource) -> ET.Element:
    """Load RSS feed as XML asynchronously"""
    content = await _get_content(resource.url)
    return ET.fromstring(content)


async def load_shp(resource: Resource) -> gpd.GeoDataFrame | NotImplementedError:
    """Load Shapefile as GeoDataFrame asynchronously"""
    if not resource.url:
        raise ValueError("Resource URL is not set")

    normalized_url = normalize_url(resource.url)

    if normalized_url.endswith(".zip"):
        content = await _get_content(resource.url)
        try:
            return _read_shp_zip_bytes(content)
        except Exception as e:
            raise ValueError(f"Could not read shapefile from zip: {str(e)}") from e

    content = await _get_content(resource.url)
    if content[:4] == b"PK\x03\x04":
        try:
            return _read_shp_zip_bytes(content)
        except Exception as e:
            raise ValueError(f"Could not read shapefile from zip bytes: {str(e)}") from e

    if normalized_url.startswith("http"):
        raise ValueError("Shapefile URL did not return a zip archive")

    try:
        return gpd.read_file(normalized_url)
    except Exception as e:
        raise ValueError(f"Could not read shapefile from path: {str(e)}") from e


async def load_tiff(resource: Resource) -> bytes:
    """Load TIFF/GeoTIFF image file asynchronously"""
    return await _get_content(resource.url)


async def load_raw(resource: Resource) -> bytes:
    """Load raw bytes asynchronously for unsupported formats."""
    return await _get_content(resource.url)


def _read_shp_zip_bytes(content: bytes) -> gpd.GeoDataFrame:
    with NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)
    try:
        return gpd.read_file(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


loaders: dict[ResourceFormat, Callable[[Resource], Awaitable[object | None]]] = {
    ResourceFormat.API: load_api,
    ResourceFormat.APP: load_app,
    ResourceFormat.APPLICATION: load_app,
    ResourceFormat.CSV: load_csv,
    ResourceFormat.ECW: load_ecw,
    ResourceFormat.GDB: load_gdb,
    ResourceFormat.GEOJSON: load_geojson,
    ResourceFormat.GEOPARQUET: load_geoparquet,
    ResourceFormat.GEOSERVICE: load_geoservice,
    ResourceFormat.GEOPACKAGE: load_geopackage,
    ResourceFormat.GTFS: load_gtfs,
    ResourceFormat.GTFS_RT: load_gtfs_rt,
    ResourceFormat.HTML: load_html,
    ResourceFormat.IMG: load_image,
    ResourceFormat.JPEG: load_image,
    ResourceFormat.JSON: load_json,
    ResourceFormat.KML: load_kml,
    ResourceFormat.KMZ: load_kmz,
    ResourceFormat.LAS: load_las,
    ResourceFormat.PDF: load_pdf,
    ResourceFormat.PNG: load_image,
    ResourceFormat.PNG_24: load_image,
    ResourceFormat.RSS: load_rss,
    ResourceFormat.SHP: load_shp,
    ResourceFormat.TEXT: load_text,
    ResourceFormat.TIF: load_tiff,
    ResourceFormat.TIFF: load_tiff,
    ResourceFormat.XLSX: load_excel,
    ResourceFormat.XML: load_xml,
    ResourceFormat.XSLX: load_excel,
    ResourceFormat.ZIP: load_zip,
}


async def load(
    resource: Resource,
    ignore_errors: bool = True,
) -> object | None:
    if resource.url is None:
        if ignore_errors:
            logging.warning(
                f"Resource {resource.name} could not be loaded: resource URL is not set"
            )
            return None
        raise ValueError("Cannot load resource: resource URL is not set")

    loader = loaders.get(resource.format)
    if loader is None:
        if ignore_errors:
            logging.warning(
                f"Unsupported format {resource.format} for {resource.name}. Loading raw bytes."
            )
            return await load_raw(resource)
        raise ValueError(f"Unsupported format: {resource.format}")

    try:
        data = await loader(resource)
    except (Exception, NotADirectoryError) as e:
        if ignore_errors:
            if isinstance(e, (httpx.HTTPStatusError, httpx.RequestError)):
                logging.warning(
                    f"Error loading resource {resource.name}: {e}. Skipping..."
                )
                return None
            logging.warning(
                f"Error loading resource {resource.name}: {e}. Falling back to raw bytes."
            )
            try:
                return await load_raw(resource)
            except Exception as raw_error:
                logging.warning(
                    f"Error loading raw resource {resource.name}: {raw_error}. Skipping..."
                )
                return None
        raise e

    return data
