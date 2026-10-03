"""The Worker catalog is a faithful, deterministic view of the Python YAML source."""

import hashlib
import json
from pathlib import Path

import yaml

from philly.filtering import BackendType, detect_backend

ROOT = Path(__file__).resolve().parents[1]


def test_worker_catalog_preserves_ids_resources_and_license():
    catalog = json.loads((ROOT / "mcp/src/generated/catalog.json").read_text())
    paths = sorted((ROOT / "src/philly/datasets").glob("*.yaml"))
    assert len(catalog) == len(paths)
    for dataset, path in zip(catalog, paths, strict=True):
        source = yaml.safe_load(path.read_text())
        assert dataset["id"] == path.stem
        assert dataset["title"] == source["title"]
        assert dataset["license"] == source["license"]
        for resource, original in zip(
            dataset["resources"], source.get("resources", []), strict=True
        ):
            assert resource["url"] == (original["url"] or "")
            identity = f"{original['url'] or ''}\n{original.get('name') or ''}\n{original.get('format') or 'unknown'}"
            assert resource["id"] == hashlib.sha256(identity.encode()).hexdigest()[:16]
            if resource["query"]:
                assert detect_backend(resource["url"]) == BackendType.CARTO


def test_shared_carto_fixture_has_typed_row_contract():
    fixture = json.loads((ROOT / "mcp/test/fixtures/carto.json").read_text())
    assert fixture["fields"]["cartodb_id"]["type"] == "number"
    assert all(isinstance(row["cartodb_id"], int) for row in fixture["rows"])
    assert all(set(row) <= set(fixture["fields"]) for row in fixture["rows"])
