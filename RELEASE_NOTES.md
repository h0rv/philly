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
