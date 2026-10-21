# Scoped providers and Python transport review

Stack base: PR #7 at `fa2086a45729f6928da3a25f875507056b3d154b` (exact-head Python and MCP/site CI passed). This follow-up uses `feat/bounded-provider-catalog`, preserving PR #7 and all prior branches/artifacts.

Four distinct source identities are separate from their adapters: LoC Philadelphia engineering search, USGS v1 latest Schuylkill observations at USGS-01474500, MusicBrainz Philadelphia-associated places, and NOAA today's UTC hourly predictions at station 8545240. Existing YAML-derived OpenDataPhilly IDs and catalog bytes are unchanged. Provider IDs are namespaced. Provider schemas explicitly identify normalized documented projections, not live metadata.

Providers support selected columns and bounded native paging. Filters, custom order and aggregation fail before networking; no approximate post-page filtering is hidden behind query semantics. NOAA retains meters/MLLW/UTC and prediction identity; USGS retains timestamps/parameter/quality codes. LoC rights vary by item; no blanket license is asserted. MusicBrainz projects core fields only. Every request starts from a fixed reviewed endpoint and scope. Returned URLs (including USGS next links) are never followed. Shared byte/row/time/cancellation gates remain enforced. MusicBrainz has an instance-local one-second admission cooldown, not a distributed limiter; no retries are added.

Python `RemotePhilly` and `phl remote` consume the same MCP tools with bounded request/decoded response sizes and a total timeout. HTTPS or loopback HTTP only; no redirects, credentials, alternate upstream fallback, or automatic paging. Both JSON and one finite legacy SSE event are supported. A real local Worker transport smoke check caught the need for SSE support; the committed check exercises SDK and CLI with provider fixtures and runs in CI.

Independent bounded Luna review found stale backend/capability text in the usage resource and unsupported-resource error; both were corrected. The final review also requested waiting for child-process `close` rather than `exit` before parsing smoke stdout; this was corrected to avoid a pipe race. No other high-confidence regression/security issue was found. This is an engineering review, not a Codex Security scan.

Local checks: 50 MCP tests, 516 Python tests, format/catalog/type/lint checks, Python distributions, Worker dry-run and offline workerd, local benchmarks, and actual Python SDK/CLI-to-Worker provider fixture smoke passed. The earlier full site build and 10 browser tests also passed; no site/exploration source edits were made.

All four live provider probes were rejected by the environment CONNECT proxy with HTTP 403. No denial was bypassed; fixture success is not proof of live schema/availability or real data. Official API documentation was consulted (links in README). The public endpoint does not acquire these draft capabilities until a separately coordinated deployment. No merge/deploy, new credentials or paid resources were used.

## MusicBrainz connection fields (2026-10-18)

The existing place search now retains optional coordinates and lifespan fields documented by MusicBrainz, with a canonical citation URL. Decimal coordinate strings are validated/range-checked before numeric conversion; missing values become null rather than zero. Partial date precision and false/null status are retained, with source paths in schema type descriptions. Tests use the official Chipping Norton documentation example for contract shape plus explicitly synthetic null/malformed cases; neither establishes live Philadelphia coverage. The fixed area search and network permissions are unchanged, and no denied endpoint was retried. `MUSICBRAINZ.md` documents a bounded-page address candidate join with location/date context, not an automatic identity assertion or recordings relationship.

Bounded independent Luna review of the MusicBrainz extension found no actionable defect. Updated current/legacy MCP contract tests and the actual Python SDK/CLI-to-Worker smoke verify coordinate/date/status values from the documented fixture; no live upstream request is substituted for fixture evidence.
