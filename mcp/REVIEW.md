# Engineering review of PR #2

Reviewed baseline `5308ed921f81fd81f4718cd461c6a89a3e4837a8` through implementation commit `0c8d1423a6ca67310e2fdc8541fbe7aa7384fc75`, then reviewed and regression-tested the follow-up changes in this document's commit. This was a fresh source-level correctness/security review of the complete changed-file inventory, not a completed Codex Security scan: that skill's preflight tools and referenced setup resource were unavailable in this host. No live data fetch or deployment was part of the review.

## Issues corrected

- **HTTP batch work amplification:** the SDK's legacy fallback accepted arrays of tool calls within one HTTP admission slot. A two-call reproduction returned HTTP 200 with two results, contradicting the one-operation/two-subrequest budget. HTTP batches now return a JSON-RPC invalid-request error before SDK dispatch. Single-message current and legacy clients still pass. A regression asserts zero upstream calls for a data-query batch.
- **Text-only client compatibility:** successful tools returned all data only in `structuredContent`; their text block contained a placeholder. Tools now return identical serialized JSON in a text block, following MCP's backward-compatibility guidance. The result cap is reduced from 96 to 60 KiB to leave room for both copies and the HTTP protocol envelope. Tests compare parsed text to structured output and exercise escape-heavy payloads.
- **Empty Origin accepted:** an explicitly empty Origin was treated like an absent Origin. Only an absent header now bypasses the browser-origin comparison; empty, `null`, multiple and mismatched values fail. Originless native clients continue to work. This closes a validation discrepancy; it is not an authentication boundary for native clients, which can omit Origin.
- **Aggregate ordering alias collision:** grouping a source column named `value` ordered by the aggregate's output alias rather than the grouping key. Ordering now names the explicit `group_value` output alias.
- **Schema/content-type validation gaps:** array/empty field containers could be described as authoritative schema, and substring media-type checking accepted unrelated MIME types containing `application/json`. Schema now requires nonempty object metadata and zero rows from `LIMIT 0`; MIME matching is exact and case-insensitive. The schema retrieval timestamp is taken after retrieval. Response cancellation rejects are handled.
- **Atlas service-worker scope:** moving the page to `/explorations/city-atlas/` while registering a script beneath `/explorations/philly-timelapse/` left the new page outside its tile-cache scope. The original worker script is now served under the new route; browser tests register it and verify the scope without fetching map services.

## Boundaries checked

The generated catalog derives stable dataset/resource IDs from checked-in YAML. Every approved source retains its complete generator-validated SQL, including year predicates, inside a subquery. Source SQL accepts only a narrow single-table grammar with known optional coordinates/date comparisons. User requests select catalog IDs; they cannot supply URLs or source SQL. Outgoing requests always use HTTPS `phl.carto.com/api/v2/sql`, reject redirects, and never forward client credentials.

Selected/filter/group/order identifiers are restricted to ASCII identifier characters and checked against backend-owned fields before quoting. Operations and comparison operators are enums. String values are single-quote escaped and backslashes/control characters are restricted. Numeric values must be finite. No injection or arbitrary-host path was found within this reviewed input contract. Unsupported resources fail before fetching.

Upstream bytes are counted while reading, before JSON parsing, including absent/false Content-Length cases. The schema and data calls share one eight-second deadline. Rows, columns, filters, offsets and output are capped; admission gates limit concurrent HTTP/data operations per isolate. Batch rejection restores the documented per-request subrequest bound. A body over the result budget fails explicitly; it is not silently presented as complete data. Source URLs/licenses/timestamps/filter/truncation metadata are retained.

Current and legacy protocol clients and both stdio eras are exercised offline. The site uses escaped Astro content and system fonts. The original Atlas code differs only in its route-specific worker registration and home link. The package lock resolves only to `registry.npmjs.org`; `npm audit --omit=dev` reported zero known runtime dependency vulnerabilities on 2026-10-03. This is advisory coverage, not proof that dependencies are vulnerability-free.

## Limits of the result

No live CARTO responses have been checked; approval for `phl.carto.com:443` is still pending. Fixtures prove construction and failure behavior, not backend availability, current schema or real counts. No public endpoint exists. Local benchmarks and workerd success do not prove deployed Free-plan CPU/memory/traffic fit. Isolate-local concurrency is not distributed rate limiting; public traffic can still consume the account's daily quota. Existing Pages workflows/domain are untouched.

Cloudflare credential presence was inspected without reading values: no recognized credential/account environment variables, default Wrangler auth files, or project credential files were found. No Cloudflare API authentication was attempted. A human-approved device OAuth sign-in and explicit Free-account selection are still needed; see `DEPLOYMENT.md`.

Reference: [MCP structured-content compatibility guidance](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#structured-content).
