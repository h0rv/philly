# Philly 1.2.0

This release adds bounded remote data access from Python and the CLI, and expands the MCP server beyond CARTO resources.

- `RemotePhilly` and `phl remote` share the six existing MCP tools, source IDs, selected columns, paging metadata and structured errors. Existing local Python loaders remain available.
- Approved ArcGIS FeatureServer layers support authoritative schema discovery, typed filters, stable ordering and bounded query pages. Aggregation remains unsupported for ArcGIS.
- Four scoped provider datasets add LoC Philadelphia engineering search, USGS latest Schuylkill observations, MusicBrainz Philadelphia-associated places and NOAA Philadelphia hourly tide predictions today UTC. They support selected columns and native paging; filters, custom ordering and aggregation return explicit errors. MusicBrainz coordinates and partial operating dates retain their provenance and uncertainty.
- All MCP tools advertise output schemas and preserve matching JSON text and structured content. Responses include source identity and explicit completeness reasons. No server-side code execution tool is added.
- New adapters retain the production Workers fetch fix: native fetch preserves its receiver, redirects are returned manually and rejected without following Location. Requests remain restricted to approved catalog endpoints with row, byte, time and cancellation bounds.

## Install and connect

After publication, install with `uv add "philly>=1.2.0"` or `uv tool install "philly>=1.2.0"`. Run `phl remote --help`; see the README for SDK and CLI examples. The default endpoint is `https://philly.horv.co/mcp`. Remote adapter availability depends on the deployed Worker version; package installation does not deploy it.

## Scope and validation limits

Provider permissions and licenses remain source-specific. LoC rights vary by item; NOAA predictions are not observed water levels; USGS latest observations are not a historical series; MusicBrainz area associations are not verified city boundaries or current venue operation.

Live ArcGIS and provider checks were previously rejected by the execution environment CONNECT proxy with HTTP 403. They have not been retried through alternate routes. Offline fixtures validate contracts and failure behavior, not current provider schemas, availability, coverage or real data.

The release tag must be `1.2.0`, matching `pyproject.toml`. Publishing a GitHub release invokes the existing distribution build/provenance checks and PyPI trusted-publishing workflow. No infrastructure or credentials are added.

---

Earlier release notes below describe validation/deployment status at their preparation time.

# 1.1.0

- Refresh catalog metadata from OpenDataPhilly commit `72c443beb6d83d7f9d3cd8b9d4cd0a1d59f33d12`: update Litter Index and add Dumpster, Unsafe Property Complaints by Hex Bins, and PIT Zones metadata. Record rejected/ambiguous upstream entries and retain legacy records; no live resource availability claim.
- Validate refreshes before writes, preserve dataset/resource identifiers, package source provenance, and check generated catalog consistency in CI.
- Add bounded read-only MCP discovery, schema, queries and aggregation with explicit provenance. Search resource years, sort recent rows and rank aggregate groups. Current and legacy MCP clients plus stdio are fixture-tested; live CARTO and Cloudflare deployment remain unverified.
- Simplify the install/connect site, share one SVG logo, remove drifting catalog counts, and add concise CLI and accessible Claude Code, Codex and Pi MCP setup tabs.
- Credit OpenDataPhilly as the catalog source, distinguish individual data publishers and terms, and include upstream notices in release distributions.
- Report the installed Python package version from distribution metadata, support `phl --version`, and make the local HTTP build prerequisites explicit.

The Python library and `phl` CLI are the PyPI distribution. The TypeScript MCP server remains a source-checkout installation. GitHub Pages hosts the static website; the remote MCP URL is a placeholder for a separately deployed service.
