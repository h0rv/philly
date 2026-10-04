"""Refreshes validate before writing and retain public dataset identifiers."""

import json
from pathlib import Path
import runpy

import pytest
import yaml

module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/update_datasets.py"))
prepare_refresh = module["prepare_refresh"]
refresh = module["refresh"]


def record(source, name, title="Example", **values):
    directory = source / "_datasets"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(
        "---\n"
        + yaml.safe_dump({"title": title, **values})
        + "---\nBody is not YAML.\n"
    )


def test_refresh_preserves_document_content_and_legacy_ids(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "datasets"
    destination.mkdir()
    (destination / "Legacy.yaml").write_text("title: Legacy\n")
    record(
        source,
        "example.md",
        notes="before---after",
        resources=[{"name": "CSV", "url": "https://example.org/data", "format": "csv"}],
    )
    writes, report = prepare_refresh(source, destination, "abc")
    assert yaml.safe_load(writes["Example.yaml"])["notes"] == "before---after"
    assert report["retained_legacy_files"] == ["Legacy.yaml"]
    assert report["records"]["Example.yaml"]["path"] == "_datasets/example.md"
    assert len(report["records"]["Example.yaml"]["sha256"]) == 64
    assert (destination / "Legacy.yaml").exists()


def test_invalid_and_colliding_records_never_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "datasets"
    destination.mkdir()
    record(source, "valid.md", "Valid")
    record(source, "a.md", "Collision")
    record(source, "b.md", "Collision")
    record(source, "invalid.md", None)
    old = destination / "Collision.yaml"
    old.write_text("title: Collision\nnotes: retained\n")
    globals_ = refresh.__globals__
    monkeypatch.setitem(
        globals_,
        "git",
        lambda _, *args: "https://github.com/opendataphilly/opendataphilly-jkan.git"
        if args[0] == "remote"
        else "abc"
        if args[0] == "rev-parse"
        else "",
    )
    manifest = tmp_path / "source.json"
    with pytest.raises(ValueError, match="No files written"):
        refresh(source, destination, manifest)
    assert not (destination / "Valid.yaml").exists()
    assert not manifest.exists()
    report = refresh(source, destination, manifest, allow_partial=True)
    assert len(report["issues"]) == 2
    assert old.read_text() == "title: Collision\nnotes: retained\n"
    assert json.loads(manifest.read_text())["revision"] == "abc"
    assert (destination / "Valid.yaml").exists()


def test_empty_source_fails_without_writes(tmp_path):
    with pytest.raises(ValueError, match="No upstream"):
        prepare_refresh(tmp_path, tmp_path, "abc")


def test_cli_version_comes_from_installed_distribution():
    from importlib.metadata import version
    from philly.__main__ import __version__

    assert __version__ == version("philly")


def test_resource_rename_preserves_public_id_across_refreshes(tmp_path):
    import hashlib

    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "datasets"
    destination.mkdir()
    old = {"name": "2018", "url": "https://example.org/data", "format": "csv"}
    (destination / "Example.yaml").write_text(
        yaml.safe_dump({"title": "Example", "resources": [old]})
    )
    expected = hashlib.sha256(b"https://example.org/data\n2018\ncsv").hexdigest()[:16]
    for name in ["2025", "2026", "2026"]:
        record(source, "example.md", resources=[{**old, "name": name}])
        writes, _ = prepare_refresh(source, destination, "abc")
        assert yaml.safe_load(writes["Example.yaml"])["resources"][0]["id"] == expected
        (destination / "Example.yaml").write_text(writes["Example.yaml"])


def test_ambiguous_resource_rename_is_reported_and_retained(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "datasets"
    destination.mkdir()
    resource = {"url": "https://example.org/data", "format": "csv"}
    (destination / "Example.yaml").write_text(
        yaml.safe_dump(
            {
                "title": "Example",
                "resources": [{**resource, "name": "A"}, {**resource, "name": "B"}],
            }
        )
    )
    record(source, "example.md", resources=[{**resource, "name": "C"}])
    writes, report = prepare_refresh(source, destination, "abc")
    assert "Example.yaml" not in writes
    assert "Ambiguous resource rename" in report["issues"][0]["error"]
    assert report["retained_legacy_files"] == ["Example.yaml"]
