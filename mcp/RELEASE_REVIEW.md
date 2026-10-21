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

Git fetch, push and remote-head inspection succeeded. The local `gh pr view` and `gh release list` calls printed `Post "https://api.github.com/graphql": Forbidden`; no numeric status, response body or permission/scope explanation was captured. This is a CLI GraphQL-path failure of unknown cause, not evidence that all GitHub access is blocked. Those denied calls were not retried.

The parent verified exact-head green CI with the connected GitHub tools, merged PR #7 as `1f2a1200e51c2f35950b28fd62a7dab095928688`, and retargeted PR #8 to main. The integration branch was subsequently merged with that main commit without conflicts or implementation changes. The parent owns final exact-head CI verification, merge and release publication through the supported connector. This worker has not merged PR #8, created a release tag, published a release or deployed a Worker.

Prepared release target is 1.2.0; the existing publish workflow requires the unprefixed tag matching pyproject.toml. Publish from merged main only, then verify release-build and PyPI publish results. Remote provider capabilities still require the matching Worker deployment, which is separate from the Python release.
