import csv
import json
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO, StringIO

import httpx
import pandas as pd

from cheesesnake.models.resource import Resource, ResourceFormat


class ResourceLoader:
    def __init__(self, resource: Resource):
        self.resource = resource

    def load(self) -> any:
        match self.resource.format:
            case ResourceFormat.CSV:
                return self.load_csv()
            case ResourceFormat.JSON:
                return self.load_json()
            case ResourceFormat.GEOJSON:
                return self.load_geojson()
            case ResourceFormat.XML:
                return self.load_xml()
            case ResourceFormat.XLSX | ResourceFormat.XSLX:
                return self.load_excel()
            case ResourceFormat.PDF:
                return self.load_pdf()
            case ResourceFormat.ZIP:
                return self.load_zip()
            case (
                ResourceFormat.PNG
                | ResourceFormat.PNG_24
                | ResourceFormat.JPEG
                | ResourceFormat.TIFF
                | ResourceFormat.TIF
                | ResourceFormat.IMG
            ):
                return self.load_image()
            case ResourceFormat.TEXT:
                return self.load_text()
            case ResourceFormat.GTFS | ResourceFormat.GTFS_RT:
                return self.load_gtfs()
            case _:
                raise ValueError(f"Unsupported format: {self.resource.format}")

    def _get_content(self) -> bytes | str:
        if not self.resource.url:
            raise ValueError("Resource URL is not set")

        with httpx.stream("GET", self.resource.url) as response:
            response.raise_for_status()
            return response.content

    def load_csv(self) -> list[dict[str, str]]:
        content = self._get_content()
        csv_data = StringIO(content.decode("utf-8"))
        reader = csv.DictReader(csv_data)
        return list(reader)

    def load_json(self) -> dict[str, any]:
        content = self._get_content()
        return json.loads(content)

    def load_geojson(self) -> dict[str, any]:
        return self.load_json()

    def load_xml(self) -> ET.Element:
        content = self._get_content()
        return ET.fromstring(content)

    def load_excel(self) -> dict[str, pd.DataFrame]:
        content = self._get_content()
        excel_file = BytesIO(content)
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        return excel_data

    def load_pdf(self) -> bytes:
        # Basic loader that just returns the PDF content
        # For more advanced PDF processing, PyPDF2 or similar would be needed
        return self._get_content()

    def load_zip(self) -> dict[str, bytes]:
        content = self._get_content()
        zip_file = BytesIO(content)

        result = {}
        with zipfile.ZipFile(zip_file) as z:
            for filename in z.namelist():
                result[filename] = z.read(filename)

        return result

    def load_image(self) -> bytes:
        # Basic image loader that returns the raw bytes
        # For image processing, Pillow would be needed
        return self._get_content()

    def load_text(self) -> str:
        content = self._get_content()
        return content.decode("utf-8")

    def load_gtfs(self) -> str:
        content = self._get_content()
        return str(content)
