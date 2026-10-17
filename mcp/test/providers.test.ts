import { test } from "node:test";
import assert from "node:assert/strict";
import { DataService, querySchema, aggregateSchema } from "../src/data.ts";
import { providerCatalog } from "../src/provider-catalog.ts";
import { search, describe, lookup, catalog } from "../src/catalog.ts";
import { LIMITS } from "../src/bounds.ts";
import { outputSchemas } from "../src/contracts.ts";
const fixtures: Record<string, unknown> = {
  loc: {
    results: [
      {
        id: "https://www.loc.gov/item/test/",
        title: "Bridge",
        date: "1900",
        description: ["Engineering"],
      },
    ],
    pagination: { total: 2 },
  },
  musicbrainz: {
    count: 2,
    offset: 0,
    places: [
      {
        id: "930b08ad-c750-4780-b0d8-2ca45b1e685b",
        name: "Studio",
        type: "Studio",
        area: { name: "Philadelphia" },
      },
    ],
  },
  usgs: {
    features: [
      {
        id: "river-1",
        properties: {
          monitoring_location_id: "USGS-01474500",
          parameter_code: "00060",
          time: "2026-10-17T12:00:00Z",
          value: "12.3",
          unit_of_measure: "ft^3/s",
          approval_status: ["Provisional"],
        },
      },
    ],
    links: [{ rel: "next", href: "http://localhost/never-follow" }],
  },
  noaa: {
    predictions: [
      { t: "2026-10-17 00:00", v: "0.4" },
      { t: "2026-10-17 01:00", v: "0.7" },
    ],
  },
};
for (const dataset of providerCatalog) {
  test(`${dataset.sourceId} fixed catalog identity, schema, bounded projection and native page`, async () => {
    const calls: URL[] = [];
    const service = new DataService(async (url, init) => {
      calls.push(new URL(url));
      assert.equal(init?.redirect, "error");
      assert.ok(init?.signal);
      return Response.json(fixtures[dataset.sourceId!]);
    });
    assert.equal(lookup(dataset.id).dataset.sourceId, dataset.sourceId);
    assert.equal(
      search(dataset.title, 0, 20).datasets[0].sourceId,
      dataset.sourceId,
    );
    const description = describe(dataset.id, 0, 10);
    outputSchemas.describe_dataset.parse(description);
    assert.equal(description.resources[0].backend, dataset.sourceId);
    assert.deepEqual(description.resources[0].capabilities, [
      "schema",
      "preview",
      "query",
    ]);
    const schema = await service.schema({ datasetId: dataset.id });
    outputSchemas.get_schema.parse(schema);
    assert.match(schema.schemaSource, /adapter projection/);
    assert.equal(calls.length, 0);
    const result = await service.query(
      querySchema.parse({
        datasetId: dataset.id,
        limit: 1,
        columns: [Object.keys(schema.fields)[0]],
      }),
    );
    outputSchemas.query_dataset.parse(result);
    assert.equal(result.rows.length, 1);
    assert.equal(result.nextOffset, 1);
    assert.equal(result.truncated, true);
    assert.equal(result.sourceId, dataset.sourceId);
    assert.equal(result.resourceId, dataset.resources[0].id);
    assert.deepEqual(Object.keys(result.rows[0]), result.columns);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].origin, new URL(dataset.resources[0].url).origin);
    assert.equal(calls[0].pathname, new URL(dataset.resources[0].url).pathname);
    if (dataset.sourceId === "loc") {
      assert.equal(calls[0].searchParams.get("sp"), "1");
      assert.equal(calls[0].searchParams.get("q"), "philadelphia engineering");
    }
    if (dataset.sourceId === "musicbrainz") {
      assert.equal(calls[0].searchParams.get("query"), 'area:"Philadelphia"');
      await assert.rejects(
        service.query(querySchema.parse({ datasetId: dataset.id })),
        /Busy/,
      );
    }
    if (dataset.sourceId === "usgs")
      assert.equal(
        calls[0].searchParams.get("monitoring_location_id"),
        "USGS-01474500",
      );
    if (dataset.sourceId === "noaa") {
      assert.equal(calls[0].searchParams.get("station"), "8545240");
      assert.equal(calls[0].searchParams.get("datum"), "MLLW");
      assert.equal(calls[0].searchParams.get("units"), "metric");
    }
  });
  test(`${dataset.sourceId} unsupported operations and invalid columns fail before networking`, async () => {
    let count = 0;
    const service = new DataService(async () => {
      count++;
      return Response.json({});
    });
    for (const args of [
      { filters: [{ column: "title", op: "eq", value: "x" }] },
      { orderBy: "title" },
      { orderDirection: "desc" },
      { columns: ["missing"] },
    ])
      await assert.rejects(
        service.query(querySchema.parse({ datasetId: dataset.id, ...args })),
        /Unsupported|Unknown column/,
      );
    await assert.rejects(
      service.aggregate(aggregateSchema.parse({ datasetId: dataset.id })),
      /Unsupported operation/,
    );
    assert.equal(count, 0);
  });
  test(`${dataset.sourceId} rejects malformed, upstream errors, oversize, redirect and canceled data`, async () => {
    for (const response of [
      Response.json({}),
      Response.json({ error: { message: "private" } }),
      new Response(null, {
        status: 302,
        headers: { location: "http://localhost" },
      }),
      new Response("private", {
        headers: { "content-type": "application/json" },
      }),
      new Response("x".repeat(LIMITS.upstreamBytes + 1), {
        headers: { "content-type": "application/json" },
      }),
    ])
      await assert.rejects(
        new DataService(async () => response).query(
          querySchema.parse({ datasetId: dataset.id }),
        ),
      );
    const controller = new AbortController();
    controller.abort();
    let count = 0;
    await assert.rejects(
      new DataService(async () => {
        count++;
        return Response.json({});
      }).query(querySchema.parse({ datasetId: dataset.id }), controller.signal),
    );
    assert.equal(count, 0);
  });
}
test("provider IDs do not collide and existing CARTO defaults are retained", () => {
  assert.equal(new Set(catalog.map((d) => d.id)).size, catalog.length);
  assert.ok(lookup("Crime_Incidents").resource.query);
});
test("LoC pagination keeps the native page size; NOAA retains units/datum and pages a complete daily window", async () => {
  const loc = providerCatalog[0];
  let calls = 0;
  const service = new DataService(async () => {
    calls++;
    return Response.json(fixtures.loc);
  });
  await assert.rejects(
    service.query(
      querySchema.parse({ datasetId: loc.id, limit: 3, offset: 1 }),
    ),
    /multiple/,
  );
  assert.equal(calls, 0);
  const noaa = providerCatalog[3];
  const page = await new DataService(async () =>
    Response.json(fixtures.noaa),
  ).query(querySchema.parse({ datasetId: noaa.id, limit: 1, offset: 1 }));
  assert.equal(page.nextOffset, null);
  assert.equal(page.truncated, false);
  assert.equal(page.rows[0].height_meters, "0.7");
  assert.equal(page.rows[0].datum, "MLLW");
  assert.equal(page.rows[0].time_zone, "UTC");
});

test("provider results cross actual MCP v2 and legacy tool contracts", async () => {
  const { Client, StreamableHTTPClientTransport } =
    await import("@modelcontextprotocol/client");
  const { Client: LegacyClient } =
    await import("@modelcontextprotocol/sdk/client/index.js");
  const { StreamableHTTPClientTransport: LegacyTransport } =
    await import("@modelcontextprotocol/sdk/client/streamableHttp.js");
  const { createWorker } = await import("../src/worker.ts");
  const origin = "http://127.0.0.1:8787";
  for (const modern of [true, false]) {
    const worker = createWorker(
      new DataService(async (url) => {
        const source = providerCatalog.find(
          (d) => new URL(d.resources[0].url).origin === new URL(url).origin,
        )!.sourceId!;
        return Response.json(fixtures[source]);
      }),
    );
    const fetcher: typeof fetch = async (url, init) =>
      worker.fetch(new Request(url, init), { PUBLIC_ORIGIN: origin });
    const client = modern
      ? new Client(
          { name: "providers", version: "1" },
          { versionNegotiation: { mode: { pin: "2026-07-28" } } },
        )
      : new LegacyClient({ name: "providers", version: "1" });
    await client.connect(
      modern
        ? new StreamableHTTPClientTransport(new URL(origin + "/mcp"), {
            fetch: fetcher,
          })
        : new LegacyTransport(new URL(origin + "/mcp"), { fetch: fetcher }),
    );
    try {
      await client.listTools();
      for (const d of providerCatalog) {
        for (const name of [
          "describe_dataset",
          "get_schema",
          "query_dataset",
        ] as const) {
          const result = await client.callTool({
            name,
            arguments: { datasetId: d.id, limit: 1 },
          });
          // Schema tool has no limit argument; strict SDK input validation is expected.
          if (name === "get_schema") {
            assert.equal(result.isError, true);
            continue;
          }
          assert.ok(!result.isError, JSON.stringify(result));
          outputSchemas[name].parse(result.structuredContent);
          const content = result.content as { text: string }[];
          assert.deepEqual(
            JSON.parse(content[0].text),
            result.structuredContent,
          );
        }
        const schema = await client.callTool({
          name: "get_schema",
          arguments: { datasetId: d.id },
        });
        assert.ok(!schema.isError);
        outputSchemas.get_schema.parse(schema.structuredContent);
      }
    } finally {
      await client.close();
    }
  }
});
