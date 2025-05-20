import csv
import json
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Callable

import geojson
import geopandas as gpd
import httpx
import pandas as pd
from google.transit import gtfs_realtime_pb2 as gtfs_rt

from cheesesnake.models import Resource, ResourceFormat
from cheesesnake.services.github import GitHub


@dataclass
class ResourceLoader:
    format: ResourceFormat
    load_func: Callable[[Resource], object | None]
    dataframable: bool = False

    def __call__(self, resource: Resource) -> object | None:
        return self.load_func(resource)


class ResourcesLoader:
    def __init__(self) -> None:
        self.loaders: dict[ResourceFormat, ResourceLoader] = {
            ResourceFormat.API: ResourceLoader(
                format=ResourceFormat.API,
                load_func=self.load_api,
                dataframable=True,
            ),
            ResourceFormat.APP: ResourceLoader(
                format=ResourceFormat.APP,
                load_func=self.load_app,
                dataframable=True,
            ),
            ResourceFormat.APPLICATION: ResourceLoader(
                format=ResourceFormat.APPLICATION,
                load_func=self.load_app,
                dataframable=True,
            ),
            ResourceFormat.CSV: ResourceLoader(
                format=ResourceFormat.CSV,
                load_func=self.load_csv,
                dataframable=True,
            ),
            ResourceFormat.ECW: ResourceLoader(
                format=ResourceFormat.ECW,
                load_func=self.load_ecw,
                dataframable=False,
            ),
            ResourceFormat.GDB: ResourceLoader(
                format=ResourceFormat.GDB,
                load_func=self.load_gdb,
                dataframable=True,
            ),
            ResourceFormat.GEOJSON: ResourceLoader(
                format=ResourceFormat.GEOJSON,
                load_func=self.load_geojson,
                dataframable=True,
            ),
            ResourceFormat.GEOPARQUET: ResourceLoader(
                format=ResourceFormat.GEOPARQUET,
                load_func=self.load_geoparquet,
                dataframable=True,
            ),
            ResourceFormat.GEOSERVICE: ResourceLoader(
                format=ResourceFormat.GEOSERVICE,
                load_func=self.load_geoservice,
                dataframable=True,
            ),
            ResourceFormat.GEOPACKAGE: ResourceLoader(
                format=ResourceFormat.GEOPACKAGE,
                load_func=self.load_geopackage,
                dataframable=True,
            ),
            ResourceFormat.GTFS: ResourceLoader(
                format=ResourceFormat.GTFS,
                load_func=self.load_gtfs,
                dataframable=True,
            ),
            ResourceFormat.GTFS_RT: ResourceLoader(
                format=ResourceFormat.GTFS_RT,
                load_func=self.load_gtfs_rt,
                dataframable=True,
            ),
            ResourceFormat.HTML: ResourceLoader(
                format=ResourceFormat.HTML,
                load_func=self.load_html,
                dataframable=True,
            ),
            ResourceFormat.IMG: ResourceLoader(
                format=ResourceFormat.IMG,
                load_func=self.load_image,
                dataframable=False,
            ),
            ResourceFormat.JPEG: ResourceLoader(
                format=ResourceFormat.JPEG,
                load_func=self.load_image,
                dataframable=False,
            ),
            ResourceFormat.JSON: ResourceLoader(
                format=ResourceFormat.JSON,
                load_func=self.load_json,
                dataframable=True,
            ),
            ResourceFormat.KML: ResourceLoader(
                format=ResourceFormat.KML,
                load_func=self.load_kml,
                dataframable=True,
            ),
            ResourceFormat.KMZ: ResourceLoader(
                format=ResourceFormat.KMZ,
                load_func=self.load_kmz,
                dataframable=True,
            ),
            ResourceFormat.LAS: ResourceLoader(
                format=ResourceFormat.LAS,
                load_func=self.load_las,
                dataframable=False,
            ),
            ResourceFormat.PDF: ResourceLoader(
                format=ResourceFormat.PDF,
                load_func=self.load_pdf,
                dataframable=False,
            ),
            ResourceFormat.PNG: ResourceLoader(
                format=ResourceFormat.PNG,
                load_func=self.load_image,
                dataframable=False,
            ),
            ResourceFormat.PNG_24: ResourceLoader(
                format=ResourceFormat.PNG_24,
                load_func=self.load_image,
                dataframable=False,
            ),
            ResourceFormat.RSS: ResourceLoader(
                format=ResourceFormat.RSS,
                load_func=self.load_rss,
                dataframable=True,
            ),
            ResourceFormat.SHP: ResourceLoader(
                format=ResourceFormat.SHP,
                load_func=self.load_shp,
                dataframable=False,
            ),
            ResourceFormat.TEXT: ResourceLoader(
                format=ResourceFormat.TEXT,
                load_func=self.load_text,
                dataframable=True,
            ),
            ResourceFormat.TIF: ResourceLoader(
                format=ResourceFormat.TIF,
                load_func=self.load_tiff,
                dataframable=False,
            ),
            ResourceFormat.TIFF: ResourceLoader(
                format=ResourceFormat.TIFF,
                load_func=self.load_tiff,
                dataframable=False,
            ),
            ResourceFormat.XLSX: ResourceLoader(
                format=ResourceFormat.XLSX,
                load_func=self.load_excel,
                dataframable=True,
            ),
            ResourceFormat.XML: ResourceLoader(
                format=ResourceFormat.XML,
                load_func=self.load_xml,
                dataframable=True,
            ),
            ResourceFormat.XSLX: ResourceLoader(
                format=ResourceFormat.XSLX,
                load_func=self.load_excel,
                dataframable=True,
            ),
            ResourceFormat.ZIP: ResourceLoader(
                format=ResourceFormat.ZIP,
                load_func=self.load_zip,
                dataframable=True,
            ),
        }

    async def load(
        self,
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

        loader = self.loaders.get(resource.format)
        if loader is None:
            raise ValueError(f"Unsupported format: {resource.format}")

        try:
            data = await loader(resource)
        except (Exception, NotADirectoryError) as e:
            if ignore_errors:
                logging.warning(
                    f"Error loading resource {resource.name}: {e}. Skipping..."
                )
                return None
            raise e

        return data

    def is_dataframable(self, resource: Resource) -> bool:
        return self.loaders.get(resource.format).dataframable

    async def _get_content(self, url: str) -> bytes:
        if not url:
            raise ValueError("Resource URL is not set")

        if url.startswith("http"):
            url = url.replace("http://", "https://")

        if url.startswith("https://github.com/"):
            url = GitHub.convert_app_url_to_content_url(url)

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

    async def load_csv(self, resource: Resource) -> list[dict[str, object]]:
        content = await self._get_content(resource.url)
        csv_data = StringIO(content.decode("utf-8", errors="replace"))
        try:
            reader = csv.DictReader(csv_data)
            return list(reader)
        except csv.Error as e:
            if "new-line character seen in unquoted field" in str(e):
                # Try with universal newline mode
                csv_data = StringIO(
                    content.decode("utf-8", errors="replace"), newline=None
                )
                reader = csv.DictReader(csv_data)
                return list(reader)
            raise

    async def load_json(self, resource: Resource) -> dict[str, object]:
        content = await self._get_content(resource.url)
        return json.loads(content.decode("utf-8", errors="replace"))

    async def load_geojson(self, resource: Resource) -> dict[str, object]:
        content = await self._get_content(resource.url)
        return geojson.loads(content.decode("utf-8", errors="replace"))

    async def load_geopackage(self, resource: Resource) -> gpd.GeoDataFrame:
        content = await self._get_content(resource.url)
        with BytesIO(content) as f:
            gdf = gpd.read_file(f)
        return gdf

    async def load_xml(self, resource: Resource) -> ET.Element | None:
        content = await self._get_content(resource.url)
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

    async def load_excel(self, resource: Resource) -> dict[str, pd.DataFrame]:
        content = await self._get_content(resource.url)
        excel_file = BytesIO(content)
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        return excel_data

    async def load_pdf(self, resource: Resource) -> bytes:
        return await self._get_content(resource.url)

    async def load_zip(self, resource: Resource) -> dict[str, bytes]:
        content = await self._get_content(resource.url)
        zip_file = BytesIO(content)

        result = {}
        with zipfile.ZipFile(zip_file) as z:
            for filename in z.namelist():
                result[filename] = z.read(filename)

        return result

    async def load_image(self, resource: Resource) -> bytes:
        return await self._get_content(resource.url)

    async def load_text(self, resource: Resource) -> str:
        content = await self._get_content(resource.url)
        return content.decode("utf-8", errors="replace")

    async def load_gtfs(self, resource: Resource) -> str:
        content = await self._get_content(resource.url)
        return str(content)

    async def load_gtfs_rt(self, resource: Resource) -> gtfs_rt.FeedMessage:
        feed = gtfs_rt.FeedMessage()
        content = await self._get_content(resource.url)
        feed.ParseFromString(content)
        return feed

    async def load_geoparquet(self, resource: Resource) -> gpd.GeoDataFrame:
        content = await self._get_content(resource.url)
        with BytesIO(content) as f:
            gdf = gpd.read_parquet(f)
        return gdf

    async def load_api(self, resource: Resource) -> dict[str, any]:
        """Load data from an API endpoint asynchronously"""
        content = await self._get_content(resource.url)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_content": content}

    async def load_app(self, resource: Resource) -> bytes:
        """Load application binary data asynchronously"""
        return await self._get_content(resource.url)

    async def load_ecw(self, resource: Resource) -> bytes:
        """Load Enhanced Compression Wavelet (ECW) format asynchronously"""
        return await self._get_content(resource.url)

    async def load_gdb(
        self, resource: Resource
    ) -> gpd.GeoDataFrame | NotImplementedError:
        """Load Esri File Geodatabase asynchronously"""
        raise NotImplementedError("GDB format loading requires GDAL/OGR support")

    async def load_geoservice(self, resource: Resource) -> dict[str, any]:
        """Load data from a geo service endpoint asynchronously"""
        content = await self._get_content(resource.url)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_content": content}

    async def load_html(self, resource: Resource) -> str:
        """Load HTML content as text asynchronously"""
        content = await self._get_content(resource.url)
        return content.decode("utf-8", errors="replace")

    async def load_kml(self, resource: Resource) -> ET.Element:
        """Load KML (Keyhole Markup Language) file asynchronously"""
        content = await self._get_content(resource.url)
        return ET.fromstring(content)

    async def load_kmz(self, resource: Resource) -> ET.Element:
        """Load KMZ (compressed KML) file asynchronously"""
        content = await self._get_content(resource.url)
        kmz_file = BytesIO(content)

        with zipfile.ZipFile(kmz_file) as z:
            kml_files = [f for f in z.namelist() if f.endswith(".kml")]
            if not kml_files:
                raise ValueError("No KML file found in KMZ archive")

            kml_content = z.read(kml_files[0])
            return ET.fromstring(kml_content)

    async def load_las(self, resource: Resource) -> bytes:
        """Load LAS (LiDAR point cloud) file asynchronously"""
        return await self._get_content(resource.url)

    async def load_rss(self, resource: Resource) -> ET.Element:
        """Load RSS feed as XML asynchronously"""
        content = await self._get_content(resource.url)
        return ET.fromstring(content)

    async def load_shp(
        self, resource: Resource
    ) -> gpd.GeoDataFrame | NotImplementedError:
        """Load Shapefile as GeoDataFrame asynchronously"""
        if resource.url and resource.url.endswith(".zip"):
            content = await self._get_content(resource.url)
            zip_file = BytesIO(content)

            try:
                return gpd.read_file(zip_file)
            except Exception as e:
                raise ValueError(f"Could not read shapefile from zip: {str(e)}")
        else:
            raise NotImplementedError(
                "Shapefile loading from individual .shp file is not supported. Provide a zipped shapefile."
            )

    async def load_tiff(self, resource: Resource) -> bytes:
        """Load TIFF/GeoTIFF image file asynchronously"""
        return await self._get_content(resource.url)
