import assert from "node:assert/strict";
import { Miniflare, convertV4MiniflareOptions } from "miniflare";
import {
  Client,
  StreamableHTTPClientTransport,
} from "@modelcontextprotocol/client";
let returnRedirect = false;
let upstreamCalls = 0;
const mf = new Miniflare(
  convertV4MiniflareOptions({
    host: "127.0.0.1",
    port: 8788,
    modules: true,
    scriptPath: "dist/worker.js",
    compatibilityDate: "2026-10-03",
    bindings: { PUBLIC_ORIGIN: "http://127.0.0.1:8788" },
    // Exercise workerd's real global fetch with the production DataService
    // constructor; only the outbound network response is replaced offline.
    outboundService: async (request) => {
      upstreamCalls++;
      const url = new URL(request.url);
      assert.equal(url.origin, "https://phl.carto.com");
      assert.equal(url.pathname, "/api/v2/sql");
      if (returnRedirect)
        return new Response(null, {
          status: 302,
          headers: { Location: "https://unapproved.invalid/" },
        });
      const sql = url.searchParams.get("q")!;
      return Response.json({
        fields: { cartodb_id: { type: "number" } },
        rows: sql.endsWith("LIMIT 0")
          ? []
          : sql.includes("COUNT(*)")
            ? [{ value: 2 }]
            : [{ cartodb_id: 1 }, { cartodb_id: 2 }],
      });
    },
  }),
);
try {
  await mf.ready;
  for (const mode of ["legacy", { pin: "2026-07-28" }] as const) {
    const client = new Client(
      { name: "workerd-test", version: "1" },
      { versionNegotiation: { mode } },
    );
    await client.connect(
      new StreamableHTTPClientTransport(new URL("http://127.0.0.1:8788/mcp")),
    );
    assert.equal((await client.listTools()).tools.length, 6);
    const result = await client.callTool({
      name: "search_datasets",
      arguments: { query: "crime" },
    });
    assert.ok(JSON.stringify(result).includes("Crime_Incidents"));
    const schema = await client.callTool({
      name: "get_schema",
      arguments: { datasetId: "Crime_Incidents" },
    });
    assert.ok(!schema.isError, JSON.stringify(schema));
    assert.ok(
      (schema.structuredContent as { fields: Record<string, unknown> }).fields
        .cartodb_id,
    );
    const count = await client.callTool({
      name: "aggregate_dataset",
      arguments: { datasetId: "Crime_Incidents", operation: "count" },
    });
    assert.ok(!count.isError, JSON.stringify(count));
    assert.equal(
      (count.structuredContent as { rows: { value: number }[] }).rows[0].value,
      2,
    );
    const page = await client.callTool({
      name: "query_dataset",
      arguments: {
        datasetId: "Crime_Incidents",
        columns: ["cartodb_id"],
        limit: 1,
      },
    });
    assert.ok(!page.isError, JSON.stringify(page));
    assert.deepEqual(
      (page.structuredContent as { rows: Record<string, unknown>[] }).rows,
      [{ cartodb_id: 1 }],
    );
    returnRedirect = true;
    const beforeRedirect = upstreamCalls;
    const rejected = await client.callTool({
      name: "get_schema",
      arguments: { datasetId: "Crime_Incidents" },
    });
    assert.equal(rejected.isError, true);
    assert.match(JSON.stringify(rejected), /Upstream unavailable/);
    assert.equal(
      upstreamCalls,
      beforeRedirect + 1,
      "Redirect must never be followed",
    );
    returnRedirect = false;
    await client.close();
  }
  assert.equal((await fetch("http://127.0.0.1:8788/mcp/health")).status, 200);
  console.log(
    "workerd: current + legacy MCP clients, search, schema, count, id-only query, no-redirect policy and health passed (offline outbound fixture).",
  );
} finally {
  await mf.dispose();
}
