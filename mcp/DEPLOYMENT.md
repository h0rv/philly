# Deployment

Website: **https://philly.horv.co/** · Remote MCP: **https://philly.horv.co/mcp**

Cloudflare deployment was confirmed on 2026-10-15 for main commit `a939489531095f08f79098d44a4b7086724ace39`: build `ae2fc263-d135-4ab5-8651-c587f584ddf9`, Worker version `b7bfc786-2610-4a53-9f58-6e208ad71cd4` at 100%. The custom domain is attached, with an active wildcard certificate; the dedicated certificate was pending at confirmation. Automatic builds from main are enabled.

## Configuration

One stateless `philly-mcp` Worker serves the existing Astro site and `/mcp`. Static assets are served first; `/mcp` and `/mcp/*` run the Worker first. `PUBLIC_ORIGIN` is `https://philly.horv.co`. The production environment selects the existing Cloudflare account. Local development retains its loopback origin.

The deployment uses Workers Free, 23 static assets and the existing authorized build credential. No new credentials, paid resources, Durable Objects, D1, KV, R2, scheduled jobs or session storage were added. Observability storage, workers.dev and preview URLs remain disabled. The custom domain is managed separately from Wrangler. The existing GitHub Pages workflow remains in place; the unrelated `isophilly` Worker is unchanged.

## Build and deploy

Use Node 24+ and Bun 1.3.11. From the repository root:

```sh
cd website
bun install --frozen-lockfile
bun run build
cd ../mcp
npm ci
npm run build -- --env production
```

The last command only packages a dry run. With the existing authorized Cloudflare build credential, deploy from `mcp`:

```sh
npx --no-install wrangler deploy --env production
```

The Worker entrypoint is `src/worker.ts`; the `ASSETS` binding uses `../website/dist`. The reviewed bundle was 407 KiB gzip. Do not enable paid services or change credential permissions for this deployment.

## Verification status

Local validation passed: 509 Python tests, 25 MCP tests, 10 browser tests, lint/type/catalog checks, Astro build, production dry-run packaging, and current/legacy MCP clients against workerd. PR and main CI passed for the deployment commit.

Deployment confirmation is not a live protocol test. This environment's proxy denied the direct HTTPS check with CONNECT 403; no bypass was attempted. Public TLS, current/legacy MCP discovery, origin rejection, and bounded live CARTO schema/preview/filter/count checks remain unverified here. Production CPU, memory and traffic headroom have not been measured. No authenticated agent model call is claimed.

The public service has isolate-local concurrency limits, not a distributed rate limit. Abuse can exhaust the account's Free daily request allowance. Keep the account on Free; do not upgrade to address quota exhaustion.

Rollback uses a previously verified Worker version through the existing deployment connection. Check domain and asset behavior after rollback; the Python package is independent.

References: [Workers limits](https://developers.cloudflare.com/workers/platform/limits/), [static asset billing](https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/), [asset routing](https://developers.cloudflare.com/workers/static-assets/routing/advanced/).

## Runtime incident (2026-10-18)

An authorized independent public-client check from the existing GitHub Actions runner reached `https://philly.horv.co/mcp` at 16:39:35–16:39:36 UTC. The local execution environment's prior CONNECT 403 remains a separate proxy restriction; it was not retried or rerouted. [Diagnostic run](https://github.com/h0rv/philly/actions/runs/39145986082/job/117729808185), diagnostic commit `faf15c418612dec6eb5f32fb913133aa1b7b84a9`, exercised production with the official SDK 2.3.0 legacy Streamable HTTP client.

Public initialization returned server `philly` version `0.1.0`; tools/list returned all six tools; `search_datasets({query:"crime",limit:10})` and `describe_dataset({datasetId:"Crime_Incidents",limit:10})` succeeded. `get_schema({datasetId:"Crime_Incidents",resourceId:"a8ff3f73800114fc"})` then returned `isError:true` with `Illegal invocation: function called with incorrect this reference`. The check stopped after seven HTTP requests. No count or row query ran and no person/address data was retrieved. GET 405 during optional SSE establishment is expected for this stateless endpoint, not this failure's cause.

Read-only control-plane inspection confirmed the enabled custom domain, `PUBLIC_ORIGIN=https://philly.horv.co`, and 100% deployment of version `937f71c3-a782-417b-acef-4df0ed41cbf4` created 2026-10-15 16:09:51 UTC. These settings and green prior CI did not establish data-tool runtime health.

Two runtime compatibility defects were reproduced with the built Worker under Workerd:

1. `DataService` stored native global fetch and invoked it as an instance method, giving it the wrong receiver. The default now wraps `globalThis.fetch(url, init)` while injected fetchers remain unchanged.
2. After fixing that receiver, Workerd rejected `redirect:"error"` before any upstream request. The request now uses `manual`, while the existing non-2xx/redirected-response checks reject every redirect without following Location. [Workers request documentation](https://developers.cloudflare.com/workers/runtime-apis/request/#parameters) confirms this supported mode.

The strengthened Workerd smoke fails on the original receiver bug, exposes the unsupported redirect mode after the first fix alone, and passes schema/count/ID-only-query under both protocol modes after both fixes. Only outbound responses are fixtures; native Worker fetch and the production default service are exercised. A 302 to an unapproved host is rejected with exactly one outbound call. A separate unit regression enforces the global receiver. Independent Luna review found no boundary regression.

The production fix is separate from PRs #7/#8 and introduces no credentials, dependency changes, services or security-permission changes. It does not deploy by itself. After the targeted fix is merged and the existing authorized main build deploys it, dispatch the manual public-client diagnostic to verify real upstream behavior. Until that succeeds, public schema/count/query remediation remains unverified. The workflow must exist on the default branch before manual dispatch is available. No automatic retries or new logging storage are enabled.
