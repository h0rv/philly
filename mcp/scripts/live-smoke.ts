import assert from "node:assert/strict";
import { DataService, querySchema, aggregateSchema } from "../src/data.ts";
if (process.env.PHL_LIVE_TEST !== "1")
  throw new Error(
    "Requires explicit live-network approval. Then run PHL_LIVE_TEST=1 npm run test:live.",
  );
const service = new DataService();
const datasetId = "Crime_Incidents";
const schema = await service.schema({ datasetId });
assert.ok(schema.fields.cartodb_id);
const preview = await service.query(
  querySchema.parse({
    datasetId,
    limit: 5,
    columns: ["cartodb_id", "text_general_code"],
  }),
);
assert.ok(preview.rows.length <= 5);
const filtered = await service.query(
  querySchema.parse({
    datasetId,
    limit: 5,
    columns: ["cartodb_id", "text_general_code"],
    filters: [{ column: "cartodb_id", op: "gt", value: 0 }],
  }),
);
const count = await service.aggregate(
  aggregateSchema.parse({
    datasetId,
    filters: [{ column: "cartodb_id", op: "gt", value: 0 }],
  }),
);
assert.equal(typeof count.rows[0]?.value, "number");
console.log(
  JSON.stringify(
    {
      status: "live upstream checks passed",
      resourceId: preview.resourceId,
      sourceUrl: preview.sourceUrl,
      retrievedAt: preview.retrievedAt,
      fields: Object.keys(schema.fields).length,
      previewRows: preview.rows.length,
      filteredRows: filtered.rows.length,
      count: count.rows[0]?.value,
    },
    null,
    2,
  ),
);
