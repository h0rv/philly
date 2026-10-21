# ArcGIS query recovery review

Base: `26954c5deec8361a41ee5a99ec78484c29875fc0`. Recovery uses a distinct branch; `feat/bounded-data-adapters` remains untouched. No saved remote implementation or open PR was present at recovery. No discovery artifact or exploration source was edited.

## Scope and boundary

The existing six MCP tools now advertise full Zod output contracts on SDK 2.3.0. Contracts include structured errors because the SDK v1 client validates structured content even for `isError` results. Text JSON and structured content remain identical. Preview remains a query alias. Source identity is independent of the transport adapter; existing dataset/resource IDs and CARTO default selection are preserved.

ArcGIS support accepts catalog resources only, HTTPS `services.arcgis.com`, tenant `fLeGjb7u4uXqeF9q`, and numbered FeatureServer layers. The GeotechnicalTest service root has a single explicit layer-0 mapping. Arbitrary client URLs, nontrivial source predicates, unknown URL parameters, path traversal, credentials and redirects are rejected. Typed filters validate fields and escape literals. No raw SQL input, server eval, new credentials, deployment or paid service was added.

Metadata and rows share the existing admission gate, eight-second deadline, cancellation signal, two-request budget, decoded byte caps and output caps. ArcGIS queries require upstream ordering/pagination capabilities. Object ID breaks ordering ties. Geometry is not requested or returned. Pagination advances by actual rows returned, respects transfer limits, and conservatively reports unknown completeness when a full server-capped page has no transfer flag. Aggregation remains explicitly unsupported for ArcGIS.

## Independent review

Two bounded read-only Luna passes reviewed the complete implementation, including new files. No high-confidence regression/security defect was found. Review requested explicit evidence for server-capped paging and SDK error schema compatibility; both are covered in tests. This is an engineering review, not a Codex Security scan.

## Local verification (2026-10-17)

- TypeScript, catalog generation drift check, formatting: passed.
- MCP tests: 32 passed, including ArcGIS fixtures, cancellation, oversize/malformed results, field/filter/order policy, transfer/offset limits, real SDK v2 and v1 calls, output contracts and text parity.
- Python: 509 tests passed; `poe all` passed; wheel/sdist and source provenance validation passed.
- Website production build and 10 desktop/mobile browser tests passed; no website source edits.
- Worker dry-run packaging and offline workerd current/legacy clients passed.
- Local Node microbenchmark passed; results are not production billing/latency evidence.

Live ArcGIS metadata check failed at the environment's CONNECT proxy with HTTP 403. No denial was bypassed. Fixtures establish construction and failure behavior, not live schema, upstream availability or real rows. Current official SDK v2 and Cloudflare Code Mode/Workers guidance was consulted; client-side composition uses the bounded typed tools without adding server execution.

Remote exact-head CI must be checked on the PR before merge. No merge/deploy is authorized in this recovery task.
