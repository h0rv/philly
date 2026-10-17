import { z } from "zod";
import { filterSchema } from "./data.ts";
const nullableString = z.string().nullable();
const nextOffset = z.number().int().nonnegative().nullable();
const rows = z.array(z.record(z.string(), z.unknown()));
const provenance = {
  sourceId: z.string(),
  datasetId: z.string(),
  resourceId: z.string(),
  resourceName: z.string(),
  sourceUrl: z.string(),
  license: nullableString,
  retrievedAt: z.string(),
  appliedFilters: z.array(filterSchema),
  truncated: z.boolean(),
  nextOffset,
  completeness: z.enum(["complete", "partial"]),
  completenessReasons: z.array(z.string()),
};
const successSchemas = {
  search_datasets: z.object({
    total: z.number().int(),
    offset: z.number().int(),
    nextOffset,
    datasets: z.array(
      z.object({
        id: z.string(),
        sourceId: z.string(),
        title: z.string(),
        categories: z.array(z.string()),
        queryable: z.boolean(),
        license: nullableString,
      }),
    ),
  }),
  describe_dataset: z.object({
    id: z.string(),
    sourceId: z.string(),
    title: z.string(),
    description: z.string(),
    organization: nullableString,
    license: nullableString,
    categories: z.array(z.string()),
    totalResources: z.number().int(),
    nextOffset,
    resources: z.array(
      z.object({
        id: z.string(),
        sourceId: z.string(),
        name: z.string(),
        format: z.string(),
        url: z.string(),
        backend: z.enum([
          "carto",
          "arcgis",
          "loc",
          "usgs",
          "musicbrainz",
          "noaa",
          "discovery-only",
        ]),
        capabilities: z.array(z.string()),
        unsupportedReason: nullableString,
      }),
    ),
  }),
  get_schema: z.object({
    ...provenance,
    schemaSource: z.string(),
    fields: z.record(z.string(), z.object({ type: z.string() })),
  }),
  query_dataset: z.object({
    ...provenance,
    columns: z.array(z.string()),
    orderBy: nullableString,
    orderDirection: z.enum(["asc", "desc"]).nullable(),
    rows,
    offset: z.number().int(),
    paginationNote: z.string(),
    columnsTruncated: z.boolean(),
  }),
  aggregate_dataset: z.object({
    ...provenance,
    operation: z.enum(["count", "sum", "avg", "min", "max"]),
    column: nullableString,
    groupBy: nullableString,
    orderBy: z.enum(["group", "value"]).nullable(),
    orderDirection: z.enum(["asc", "desc"]).nullable(),
    rows,
    scope: z.string(),
  }),
};
const errorSchema = z.object({
  error: z.object({ code: z.string(), message: z.string() }),
});
// SDK v1 validates structured errors too. Advertise both branches without
// changing the successful result envelope used by existing clients.
export const outputSchemas = {
  search_datasets: z.union([successSchemas.search_datasets, errorSchema]),
  describe_dataset: z.union([successSchemas.describe_dataset, errorSchema]),
  get_schema: z.union([successSchemas.get_schema, errorSchema]),
  query_dataset: z.union([successSchemas.query_dataset, errorSchema]),
  aggregate_dataset: z.union([successSchemas.aggregate_dataset, errorSchema]),
};

export function errorCode(error: unknown, signal?: AbortSignal): string {
  if (signal?.aborted) return "CANCELLED";
  const message = error instanceof Error ? error.message : "";
  if (/timeout/i.test(message)) return "UPSTREAM_TIMEOUT";
  if (/Busy/.test(message)) return "BUSY";
  if (/Unknown dataset/.test(message)) return "DATASET_NOT_FOUND";
  if (/Unsupported operation/.test(message)) return "UNSUPPORTED_OPERATION";
  if (/Unsupported/.test(message)) return "UNSUPPORTED_RESOURCE";
  if (/limit exceeded|oversized/i.test(message)) return "LIMIT_EXCEEDED";
  if (/column|Comparisons|Pagination|Numeric|supported columns/i.test(message))
    return "INVALID_QUERY";
  return "UPSTREAM_ERROR";
}
