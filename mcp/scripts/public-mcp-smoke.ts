// Bounded public-client diagnostic. No direct upstream requests or retries.
import assert from "node:assert/strict";
import { readBounded, LIMITS } from "../src/bounds.ts";
import {
  Client,
  StreamableHTTPClientTransport,
} from "@modelcontextprotocol/client";
if (process.env.PHL_PUBLIC_TEST !== "1")
  throw new Error(
    "Set PHL_PUBLIC_TEST=1 only in an authorized public-client test environment.",
  );
const endpoint = "https://philly.horv.co/mcp";
let denied = false;
let stopped = false;
let requests = 0;
const startedAt = new Date().toISOString();
const client = new Client(
  { name: "philly-public-diagnostic", version: "1" },
  { versionNegotiation: { mode: "legacy" } },
);
const transport = new StreamableHTTPClientTransport(new URL(endpoint), {
  fetch: async (url, init) => {
    if (stopped) throw new Error("Previous failure; stopping this route.");
    const target = new URL(String(url));
    assert.equal(target.href, endpoint);
    assert.ok(++requests <= 12, "HTTP request budget exceeded");
    const signal = AbortSignal.any([
      ...(init?.signal ? [init.signal] : []),
      AbortSignal.timeout(12000),
    ]);
    const response = await fetch(url, {
      ...init,
      redirect: "error",
      signal,
    });
    console.log(
      JSON.stringify({
        event: "http",
        at: new Date().toISOString(),
        method: init?.method ?? "GET",
        status: response.status,
        contentType: response.headers.get("content-type"),
      }),
    );
    if (response.status === 401 || response.status === 403) {
      denied = true;
      stopped = true;
      void response.body?.cancel();
      throw new Error(
        `Public MCP access denied: HTTP ${response.status}; no retry.`,
      );
    }
    if (!response.ok && response.status !== 405) {
      stopped = true;
      void response.body?.cancel();
      throw new Error(
        `Public MCP HTTP ${response.status}; stopping without retry.`,
      );
    }
    const body = await readBounded(response.body, LIMITS.responseBytes, signal);
    return new Response(body || null, {
      status: response.status,
      headers: response.headers,
    });
  },
});
async function call(name: string, args: Record<string, unknown>) {
  console.log(
    JSON.stringify({
      event: "tool",
      at: new Date().toISOString(),
      name,
      arguments: args,
    }),
  );
  const result = await client.callTool({ name, arguments: args });
  if (result.isError) {
    stopped = true;
    throw new Error(
      `MCP tool ${name} failed: ${JSON.stringify(result.structuredContent ?? result.content).slice(0, 400)}`,
    );
  }
  const value =
    result.structuredContent ??
    JSON.parse(
      (result.content as { type: string; text: string }[]).find(
        (c) => c.type === "text",
      )!.text,
    );
  return value as Record<string, unknown>;
}
try {
  console.log(
    JSON.stringify({
      event: "start",
      endpoint,
      startedAt,
      client: "SDK 2.3.0 legacy initialize / stateless Streamable HTTP",
    }),
  );
  await client.connect(transport);
  console.log(
    JSON.stringify({
      event: "initialized",
      server: client.getServerVersion(),
      at: new Date().toISOString(),
    }),
  );
  const tools = (await client.listTools()).tools;
  console.log(
    JSON.stringify({
      event: "tools/list",
      names: tools.map((t) => t.name),
      at: new Date().toISOString(),
    }),
  );
  for (const name of [
    "search_datasets",
    "describe_dataset",
    "get_schema",
    "query_dataset",
    "aggregate_dataset",
  ])
    assert.ok(
      tools.some((t) => t.name === name),
      name,
    );
  const found = await call("search_datasets", { query: "crime", limit: 10 });
  assert.ok(
    (found.datasets as { id: string }[]).some(
      (d) => d.id === "Crime_Incidents",
    ),
  );
  const description = await call("describe_dataset", {
    datasetId: "Crime_Incidents",
    limit: 10,
  });
  const resource = (
    description.resources as { id: string; name: string; backend: string }[]
  ).find((r) => r.backend === "carto");
  assert.ok(resource);
  const selection = { datasetId: "Crime_Incidents", resourceId: resource.id };
  console.log(
    JSON.stringify({
      event: "resource",
      resource,
      at: new Date().toISOString(),
    }),
  );
  const schema = await call("get_schema", selection);
  assert.ok(Object.hasOwn(schema.fields as object, "cartodb_id"));
  console.log(
    JSON.stringify({
      event: "schema",
      fields: schema.fields,
      retrievedAt: schema.retrievedAt,
      sourceUrl: schema.sourceUrl,
    }),
  );
  const count = await call("aggregate_dataset", {
    ...selection,
    operation: "count",
  });
  console.log(JSON.stringify({ event: "count", result: count }));
  const page = await call("query_dataset", {
    ...selection,
    columns: ["cartodb_id"],
    limit: 1,
  });
  const rows = page.rows as Record<string, unknown>[];
  assert.ok(rows.length <= 1);
  assert.ok(
    rows.every((row) => Object.keys(row).every((k) => k === "cartodb_id")),
  );
  console.log(
    JSON.stringify({
      event: "query",
      rowCount: rows.length,
      columns: page.columns,
      retrievedAt: page.retrievedAt,
      truncated: page.truncated,
      sourceUrl: page.sourceUrl,
    }),
  );
  console.log(
    JSON.stringify({
      event: "passed",
      startedAt,
      finishedAt: new Date().toISOString(),
      requests,
    }),
  );
} catch (error) {
  console.error(
    JSON.stringify({
      event: "failed",
      startedAt,
      finishedAt: new Date().toISOString(),
      requests,
      denied,
      message: error instanceof Error ? error.message : String(error),
    }),
  );
  process.exitCode = 1;
} finally {
  await client.close();
}
