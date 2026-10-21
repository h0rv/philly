import { test } from "node:test";
import assert from "node:assert/strict";
import { DataService, querySchema, aggregateSchema } from "../src/data.ts";
import { describe } from "../src/catalog.ts";
import { arcgisLayer } from "../src/sources.ts";
import { LIMITS } from "../src/bounds.ts";
const datasetId = "PWD_Geotechnical_Tests";
export const arcMetadata = {
  fields: [
    { name: "OBJECTID", type: "esriFieldTypeOID" },
    { name: "name", type: "esriFieldTypeString" },
    { name: "depth", type: "esriFieldTypeDouble" },
  ],
  objectIdField: "OBJECTID",
  maxRecordCount: 1000,
  advancedQueryCapabilities: {
    supportsPagination: true,
    supportsOrderBy: true,
  },
};
const features = [1, 2, 3].map((OBJECTID) => ({
  attributes: { OBJECTID, name: "bore", depth: 10, unselected: "omit" },
}));
function fixture(meta = arcMetadata, page: unknown = { features }) {
  const calls: URL[] = [];
  const service = new DataService(async (url, init) => {
    const u = new URL(url);
    calls.push(u);
    assert.equal(u.origin, "https://services.arcgis.com");
    assert.equal(init?.redirect, "manual");
    assert.ok(init?.signal);
    return Response.json(u.pathname.endsWith("/query") ? page : meta);
  });
  return { service, calls };
}
test("ArcGIS catalog identity, explicit root-to-layer mapping and capabilities", async () => {
  const d = describe(datasetId, 0, 20);
  const resource = d.resources.find((r) => r.backend === "arcgis")!;
  assert.ok(resource);
  assert.equal(resource.sourceId, "opendataphilly");
  assert.deepEqual(resource.capabilities, ["schema", "preview", "query"]);
  assert.ok(arcgisLayer(resource.url)?.endsWith("/FeatureServer/0"));
  const { service, calls } = fixture();
  const schema = await service.schema({ datasetId });
  assert.equal(schema.fields.OBJECTID.type, "esriFieldTypeOID");
  assert.match(schema.schemaSource, /not sample/);
  assert.equal(calls.length, 1);
  await assert.rejects(
    service.aggregate(aggregateSchema.parse({ datasetId })),
    /Unsupported operation/,
  );
  assert.equal(calls.length, 1);
});
test("ArcGIS selected columns, filters, deterministic ties, lookahead and next page", async () => {
  const { service, calls } = fixture();
  const input = querySchema.parse({
    datasetId,
    limit: 2,
    columns: ["OBJECTID", "name"],
    orderBy: "depth",
    orderDirection: "desc",
    filters: [{ column: "name", op: "eq", value: `a' OR 1=1-- "name" =` }],
  });
  const page = await service.query(input);
  const params = calls[1].searchParams;
  assert.equal(params.get("where"), `name = 'a'' OR 1=1-- "name" ='`);
  assert.equal(params.get("outFields"), "OBJECTID,name");
  assert.equal(params.get("orderByFields"), "depth DESC,OBJECTID ASC");
  assert.equal(params.get("returnGeometry"), "false");
  assert.equal(params.get("resultRecordCount"), "3");
  assert.deepEqual(page.rows, [
    { OBJECTID: 1, name: "bore" },
    { OBJECTID: 2, name: "bore" },
  ]);
  assert.equal(page.nextOffset, 2);
  assert.equal(page.truncated, true);
  assert.deepEqual(page.completenessReasons, ["row_limit"]);
  await service.query({ ...input, offset: page.nextOffset! });
  assert.equal(calls[3].searchParams.get("resultOffset"), "2");
});
test("ArcGIS transfer limits advance by returned rows and report no progress/offset caps", async () => {
  const { service } = fixture(
    { ...arcMetadata, maxRecordCount: 1 },
    { features: features.slice(0, 1), exceededTransferLimit: true },
  );
  const page = await service.query(querySchema.parse({ datasetId, limit: 20 }));
  assert.equal(page.rows.length, 1);
  assert.equal(page.nextOffset, 1);
  assert.ok(page.completenessReasons.includes("upstream_transfer_limit"));
  const unknown = await fixture(
    { ...arcMetadata, maxRecordCount: 1 },
    { features: features.slice(0, 1) },
  ).service.query(querySchema.parse({ datasetId }));
  assert.equal(unknown.nextOffset, 1);
  assert.ok(
    unknown.completenessReasons.includes("upstream_completeness_unknown"),
  );
  const capped = await service.query(
    querySchema.parse({ datasetId, offset: LIMITS.offset }),
  );
  assert.equal(capped.nextOffset, null);
  assert.ok(capped.completenessReasons.includes("offset_limit"));
  const empty = await fixture(arcMetadata, {
    features: [],
    exceededTransferLimit: true,
  }).service.query(querySchema.parse({ datasetId }));
  assert.equal(empty.nextOffset, null);
  assert.ok(empty.completenessReasons.includes("upstream_no_progress"));
  const complete = await fixture(arcMetadata, {
    features: features.slice(0, 1),
  }).service.query(querySchema.parse({ datasetId }));
  assert.equal(complete.truncated, false);
  assert.equal(complete.completeness, "complete");
});
test("ArcGIS resource policy rejects arbitrary hosts, redirects, paths and scoped filters", () => {
  const base =
    "https://services.arcgis.com/fLeGjb7u4uXqeF9q/ArcGIS/rest/services/GeotechnicalTest/FeatureServer/0";
  assert.ok(arcgisLayer(base));
  for (const value of [
    base.replace("services.arcgis.com", "evil.test"),
    base.replace("fLeGjb7u4uXqeF9q", "other"),
    base + "?where=depth%3E10",
    base + "?token=secret",
    base + "#x",
    base.replace("https://", "http://"),
    base.replace("services.arcgis.com", "user@services.arcgis.com"),
    base + "/../1",
    base + "/query?outFields=OBJECTID",
  ])
    assert.equal(arcgisLayer(value), null, value);
});
test("ArcGIS rejects unsupported paging, malformed metadata, errors and oversized responses", async () => {
  for (const meta of [
    { ...arcMetadata, objectIdField: "missing" },
    { ...arcMetadata, fields: [...arcMetadata.fields, arcMetadata.fields[0]] },
    { error: { message: "private" } },
    { ...arcMetadata, fields: [] },
  ]) {
    await assert.rejects(
      fixture(meta as typeof arcMetadata).service.schema({ datasetId }),
    );
  }
  await assert.rejects(
    fixture({
      ...arcMetadata,
      advancedQueryCapabilities: {
        supportsPagination: false,
        supportsOrderBy: true,
      },
    }).service.query(querySchema.parse({ datasetId })),
    /Unsupported operation/,
  );
  for (const page of [
    { error: { message: "private" } },
    { features: Array(102).fill(features[0]) },
    { features: [{ attributes: [] }] },
    { features: [], exceededTransferLimit: "yes" },
  ])
    await assert.rejects(
      fixture(arcMetadata, page).service.query(
        querySchema.parse({ datasetId }),
      ),
    );
  for (const response of [
    new Response(null, {
      status: 302,
      headers: { location: "http://localhost" },
    }),
    new Response("{", { headers: { "content-type": "application/json" } }),
    new Response("x".repeat(LIMITS.upstreamBytes + 1), {
      headers: { "content-type": "application/json" },
    }),
  ])
    await assert.rejects(
      new DataService(async () => response).schema({ datasetId }),
    );
  const f = fixture();
  await assert.rejects(
    f.service.query(
      querySchema.parse({
        datasetId,
        filters: [{ column: "missing", op: "eq", value: 1 }],
      }),
    ),
    /Unknown column/,
  );
  assert.equal(f.calls.length, 1);
});
test("ArcGIS cancellation reaches network and pre-aborted calls never fetch", async () => {
  const controller = new AbortController();
  let calls = 0;
  const service = new DataService(async (_, init) => {
    calls++;
    return new Promise((_, reject) =>
      init?.signal?.addEventListener(
        "abort",
        () => reject(init.signal?.reason),
        { once: true },
      ),
    );
  });
  const pending = service.schema({ datasetId }, controller.signal);
  controller.abort();
  await assert.rejects(pending);
  assert.equal(calls, 1);
  await assert.rejects(service.schema({ datasetId }, controller.signal));
  assert.equal(calls, 1);
});
