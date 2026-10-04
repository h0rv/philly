"""Refresh catalog metadata from a pinned Git checkout, without deleting legacy IDs."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import yaml

from philly.models import Dataset

REPO = os.environ.get("REPO", "opendataphilly/opendataphilly-jkan")
BRANCH = os.environ.get("BRANCH", "main")
ROOT = Path(__file__).resolve().parents[1]


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True
    ).strip()


def prepare_refresh(
    source: Path, destination: Path, revision: str
) -> tuple[dict, dict]:
    """Parse every record before writes; collisions never silently overwrite an ID."""
    candidates = defaultdict(list)
    issues = []
    files = sorted((source / "_datasets").glob("*.md"))
    if not files:
        raise ValueError("No upstream dataset files found")
    for path in files:
        text = path.read_text(encoding="utf-8")
        try:
            parts = re.split(r"(?m)^---\s*$", text, maxsplit=2)
            frontmatter = parts[1] if len(parts) > 1 and not parts[0].strip() else text
            dataset = Dataset.from_yaml(frontmatter.replace("hhttps://", "https://"))
            name = re.sub(r"[^\w\-]", "_", dataset.title) + ".yaml"
            if not dataset.title.strip():
                raise ValueError("Empty title")
            candidates[name].append(
                (path, dataset, hashlib.sha256(text.encode()).hexdigest())
            )
        except (ValueError, TypeError, yaml.YAMLError) as error:
            issues.append({"path": path.name, "error": str(error)})
    writes = {}
    records = {}
    for name, entries in sorted(candidates.items()):
        if len(entries) != 1:
            issues.append(
                {
                    "paths": [p.name for p, _, _ in entries],
                    "error": f"Duplicate dataset ID: {name}",
                }
            )
            continue
        path, dataset, digest = entries[0]
        data = dataset.model_dump()
        existing = destination / name
        ambiguous = False
        if existing.exists():
            old_resources = yaml.safe_load(existing.read_text()).get("resources") or []
            for resource in data.get("resources") or []:
                matches = [
                    old
                    for old in old_resources
                    if old.get("url") == resource.get("url")
                    and old.get("format") == resource.get("format")
                ]
                exact = [
                    old for old in matches if old.get("name") == resource.get("name")
                ]
                matches = exact or matches
                if len(matches) > 1 and not exact:
                    issues.append(
                        {
                            "path": path.name,
                            "error": f"Ambiguous resource rename: {resource.get('name')}",
                        }
                    )
                    ambiguous = True
                    break
                if len(matches) == 1:
                    old = matches[0]
                    if old.get("id") or old.get("name") != resource.get("name"):
                        identity = f"{old.get('url') or ''}\n{old.get('name') or ''}\n{old.get('format') or 'unknown'}"
                        resource["id"] = (
                            old.get("id")
                            or hashlib.sha256(identity.encode()).hexdigest()[:16]
                        )
        if ambiguous:
            continue
        content = yaml.safe_dump(data, allow_unicode=False)
        # Avoid formatting-only churn in already equivalent records.
        existing = destination / name
        if existing.exists() and Dataset.from_file(str(existing)) == dataset:
            content = existing.read_text(encoding="utf-8")
        writes[name] = content
        records[name] = {"path": f"_datasets/{path.name}", "sha256": digest}
    report = {
        "repository": REPO,
        "revision": revision,
        "records": records,
        "issues": issues,
        "retained_legacy_files": sorted(
            p.name for p in destination.glob("*.yaml") if p.name not in writes
        ),
    }
    return writes, report


def refresh(
    source: Path, destination: Path, manifest: Path, *, allow_partial: bool = False
) -> dict:
    origin = git(source, "remote", "get-url", "origin")
    if origin.removesuffix(".git") != f"https://github.com/{REPO}":
        raise ValueError("Source checkout origin does not match REPO")
    revision = git(source, "rev-parse", "HEAD")
    if git(source, "status", "--porcelain", "--", "_datasets"):
        raise ValueError("Upstream dataset checkout has uncommitted changes")
    writes, report = prepare_refresh(source, destination, revision)
    if report["issues"] and not allow_partial:
        raise ValueError(
            json.dumps(report["issues"], indent=2)
            + "\nNo files written. Review issues before using --allow-partial."
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in writes.items():
        (destination / name).write_text(content, encoding="utf-8")
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Existing clean Git checkout (offline refresh)"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Refresh valid records and record every issue; retain ambiguous/legacy records",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="philly-catalog-") as temporary:
        source = args.source
        if source is None:
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", REPO):
                raise ValueError("REPO must be a GitHub owner/repository")
            source = Path(temporary) / "upstream"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    "--branch",
                    BRANCH,
                    "--",
                    f"https://github.com/{REPO}.git",
                    str(source),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(source), "sparse-checkout", "set", "_datasets"],
                check=True,
            )
        report = refresh(
            source,
            ROOT / "src/philly/datasets",
            ROOT / "src/philly/catalog-source.json",
            allow_partial=args.allow_partial,
        )
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "records"},
                indent=2,
            )
        )
        print(f"Refreshed {len(report['records'])} records from {report['revision']}")


if __name__ == "__main__":
    main()
