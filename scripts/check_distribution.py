"""Check built wheel version, CLI entry point and packaged catalog provenance."""

import json
from pathlib import Path
import sys
import tomllib
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
with ZipFile(directory / f"philly-{version}-py3-none-any.whl") as wheel:
    assert (
        f"Version: {version}\n"
        in wheel.read(f"philly-{version}.dist-info/METADATA").decode()
    )
    assert (
        "phl = philly.__main__:main"
        in wheel.read(f"philly-{version}.dist-info/entry_points.txt").decode()
    )
    for notice in ["LICENSE", "THIRD_PARTY_NOTICES", "assets/NOTICE"]:
        assert (
            wheel.read(f"philly-{version}.dist-info/licenses/{notice}")
            == (ROOT / notice).read_bytes()
        )
    expected = {
        p.name: p.read_bytes() for p in (ROOT / "src/philly/datasets").glob("*.yaml")
    }
    packaged = {
        Path(name).name: wheel.read(name)
        for name in wheel.namelist()
        if name.startswith("philly/datasets/") and name.endswith(".yaml")
    }
    assert packaged == expected, "Wheel catalog differs from source"
    manifest = wheel.read("philly/catalog-source.json")
    assert manifest == (ROOT / "src/philly/catalog-source.json").read_bytes()
    assert set(json.loads(manifest)["records"]) <= expected.keys()
print(f"Wheel {version}: CLI, dataset bytes and source provenance verified")
