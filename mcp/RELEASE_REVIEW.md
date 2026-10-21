# Philly 1.2.0 integration review — 2026-10-21

Integrated PR #7 (`fa2086a`) and PR #8 (`3ee53ed`) with main `b1f2d57` on a separate integration branch, preserving all existing commits. README conflicts retain both public-client diagnosis and client composition guidance. ArcGIS and scoped provider requests now use the production fix's Workers-compatible manual redirect mode with non-2xx rejection. Source permissions, endpoints and credentials are unchanged.

Independent bounded Luna read-only review found no actionable release-blocking correctness/security defects in the integrated adapters, catalog routing or Python SDK/CLI. This was an engineering review, not a security scan.

Local validation passed:

- 516 Python tests; Ruff format/lint and ty checks.
- 51 MCP tests; TypeScript, Prettier and generated-catalog drift checks.
- Locked Python environment sync; 1.2.0 wheel/sdist build and wheel version/entry point/catalog/provenance verification. `phl --version` reports 1.2.0.
- Static site build through the installed Astro CLI (Bun unavailable here), and all 10 desktop/mobile Chromium tests.
- Worker dry-run bundle; real native Workerd fetch with offline outbound fixtures for CARTO, ArcGIS and all four scoped providers across current/legacy protocols. Redirects are rejected after exactly one outbound request; MusicBrainz immediate duplicate queries fail locally with BUSY.
- Actual Python SDK and CLI through local MCP Worker transport into provider fixtures; local Node microbenchmark.

No live provider checks were attempted: previously recorded CONNECT 403 denials remain respected. These fixtures do not establish upstream availability, real schemas/data, provider coverage, production latency or billing.

Git fetch/remote-head inspection succeeded. `gh pr view 7`, `gh pr view 8` and `gh release list` returned GitHub API Forbidden. No alternate API route was attempted. Exact-head GitHub CI, PR state/base updates, merges, release publication, PyPI workflow and published artifacts remain unverified and blocked until authorized API access is restored. No release tag or deployment was created.

Prepared release target is 1.2.0; existing publish workflow requires an unprefixed tag matching pyproject.toml. Merge PR7 before PR8 after exact-head CI; if merge strategy rewrites ancestry, refresh PR8 against the merged main and recheck that exact head before merging. Publish from merged main only, then verify release-build and PyPI publish results.
