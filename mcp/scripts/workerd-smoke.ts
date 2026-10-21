import assert from "node:assert/strict";
import { setTimeout } from "node:timers/promises";
import { readFileSync } from "node:fs";
const musicbrainzFixture = JSON.parse(
  readFileSync(
    new URL("../test/fixtures/musicbrainz-place-search.json", import.meta.url),
    "utf8",
  ),
);
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
      assert.ok(
        [
          "https://phl.carto.com",
          "https://services.arcgis.com",
          "https://www.loc.gov",
          "https://api.waterdata.usgs.gov",
          "https://musicbrainz.org",
          "https://api.tidesandcurrents.noaa.gov",
        ].includes(url.origin),
        `Unexpected outbound origin: ${url.origin}`,
      );
      if (returnRedirect)
        return new Response(null, {
          status: 302,
          headers: { Location: "https://unapproved.invalid/" },
        });
      if (url.hostname === "services.arcgis.com") {
        assert.match(
          url.pathname,
          /\/fLeGjb7u4uXqeF9q\/ArcGIS\/rest\/services\/GeotechnicalTest\/FeatureServer\/0(?:\/query)?$/i,
        );
        return Response.json(
          url.pathname.endsWith("/query")
            ? {
                features: [{ attributes: { OBJECTID: 1 } }],
                exceededTransferLimit: false,
              }
            : {
                fields: [{ name: "OBJECTID", type: "esriFieldTypeOID" }],
                objectIdField: "OBJECTID",
                maxRecordCount: 1000,
                advancedQueryCapabilities: {
                  supportsPagination: true,
                  supportsOrderBy: true,
                },
              },
        );
      }
      if (url.hostname === "www.loc.gov")
        return Response.json({
          results: [{ id: "fixture", title: "Bridge" }],
          pagination: { total: 1 },
        });
      if (url.hostname === "api.waterdata.usgs.gov")
        return Response.json({
          features: [
            {
              id: "fixture",
              properties: {
                monitoring_location_id: "USGS-01474500",
                parameter_code: "00060",
                time: "2026-10-21T00:00:00Z",
                value: "1",
              },
            },
          ],
          links: [],
        });
      if (url.hostname === "musicbrainz.org")
        return Response.json(musicbrainzFixture);
      if (url.hostname === "api.tidesandcurrents.noaa.gov")
        return Response.json({
          predictions: [{ t: "2026-10-21 00:00", v: "1" }],
        });
      assert.equal(url.pathname, "/api/v2/sql");
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
    for (const datasetId of [
      "PWD_Geotechnical_Tests",
      "loc:philadelphia_engineering",
      "usgs:schuylkill_latest",
      "musicbrainz:philadelphia_places",
      "noaa:philadelphia_tides_today",
    ]) {
      // Observe the real instance-local admission cooldown between fixture calls.
      if (datasetId.startsWith("musicbrainz:")) await setTimeout(1100);
      const args = { datasetId, limit: 1 };
      const projected = await client.callTool({
        name: "query_dataset",
        arguments: args,
      });
      assert.ok(
        !projected.isError,
        `${datasetId}: ${JSON.stringify(projected)}`,
      );
      assert.equal(
        (projected.structuredContent as { rows: unknown[] }).rows.length,
        1,
      );
      if (datasetId.startsWith("musicbrainz:")) {
        const beforeBusy = upstreamCalls;
        const busy = await client.callTool({
          name: "query_dataset",
          arguments: args,
        });
        assert.equal(busy.isError, true);
        assert.match(JSON.stringify(busy), /BUSY/);
        assert.equal(upstreamCalls, beforeBusy);
        await setTimeout(1100);
      }
      returnRedirect = true;
      const before = upstreamCalls;
      const denied = await client.callTool({
        name: "query_dataset",
        arguments: args,
      });
      assert.equal(denied.isError, true);
      assert.match(JSON.stringify(denied), /Upstream unavailable/);
      assert.equal(
        upstreamCalls,
        before + 1,
        "Adapter redirect must not be followed",
      );
      returnRedirect = false;
    }
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
    "workerd: current + legacy MCP clients, search, schema, count, ArcGIS + four provider queries, id-only query, no-redirect policy and health passed (offline outbound fixture).",
  );
} finally {
  await mf.dispose();
}
