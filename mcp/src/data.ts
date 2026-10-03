import { z } from "zod";
import { lookup } from "./catalog.ts";
import { LIMITS, readBounded } from "./bounds.ts";
export const identifier = z
  .string()
  .max(63)
  .regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/);
export const selection = {
  datasetId: z.string().min(1).max(200),
  resourceId: z
    .string()
    .regex(/^[a-f0-9]{16}$/)
    .optional(),
};
export const filterSchema = z.strictObject({
  column: identifier,
  op: z.enum(["eq", "ne", "lt", "lte", "gt", "gte", "is_null", "not_null"]),
  value: z
    .union([
      z
        .string()
        .max(200)
        .regex(/^[^\x00-\x1f\\]*$/),
      z.number().finite(),
      z.boolean(),
      z.null(),
    ])
    .optional(),
});
export const filtersSchema = z
  .array(filterSchema)
  .max(LIMITS.filters)
  .default([]);
export const querySchema = z.strictObject({
  ...selection,
  filters: filtersSchema,
  columns: z.array(identifier).min(1).max(LIMITS.columns).optional(),
  limit: z.number().int().min(1).max(LIMITS.rows).default(20),
  offset: z.number().int().min(0).max(LIMITS.offset).default(0),
  orderBy: identifier.optional(),
});
export const aggregateSchema = z.strictObject({
  ...selection,
  filters: filtersSchema,
  operation: z.enum(["count", "sum", "avg", "min", "max"]).default("count"),
  column: identifier.optional(),
  groupBy: identifier.optional(),
  limit: z.number().int().min(1).max(50).default(20),
});
export type Fetcher = (
  url: string | URL,
  init?: RequestInit,
) => Promise<Response>;
type Fields = Record<string, { type: string }>;
const quote = (s: string) => `"${identifier.parse(s)}"`;
export function where(filters: z.infer<typeof filtersSchema>, fields: Fields) {
  return filters.length
    ? " WHERE " +
        filters
          .map((f) => {
            checkColumn(f.column, fields);
            if (f.op === "is_null" || f.op === "not_null")
              return `${quote(f.column)} IS ${f.op === "not_null" ? "NOT " : ""}NULL`;
            if (f.value === undefined || f.value === null)
              throw new Error(
                "Comparisons require a value; use is_null / not_null for null.",
              );
            const literal =
              typeof f.value === "string"
                ? `'${f.value.replaceAll("'", "''")}'`
                : String(f.value);
            const op = {
              eq: "=",
              ne: "<>",
              lt: "<",
              lte: "<=",
              gt: ">",
              gte: ">=",
            }[f.op];
            return `${quote(f.column)} ${op} ${literal}`;
          })
          .join(" AND ")
    : "";
}
function checkColumn(name: string, fields: Fields) {
  identifier.parse(name);
  if (!Object.hasOwn(fields, name))
    throw new Error(`Unknown column: ${name}. Use get_schema.`);
}
export class DataService {
  private active = 0;
  private fetcher: Fetcher;
  constructor(fetcher: Fetcher = fetch) {
    this.fetcher = fetcher;
  }
  private async run<T>(
    fn: (signal: AbortSignal) => Promise<T>,
    cancellation?: AbortSignal,
  ): Promise<T> {
    if (this.active >= LIMITS.concurrency)
      throw new Error("Busy: retry later.");
    this.active++;
    const controller = new AbortController();
    const timer = setTimeout(
      () => controller.abort(new Error("Upstream timeout.")),
      LIMITS.timeoutMs,
    );
    const signal = cancellation
      ? AbortSignal.any([controller.signal, cancellation])
      : controller.signal;
    try {
      return await fn(signal);
    } finally {
      clearTimeout(timer);
      this.active--;
    }
  }
  private async request(sql: string, signal: AbortSignal) {
    // Fixed origin and path, never a client-provided URL. No redirects, credentials, retries or fanout.
    const url = new URL("https://phl.carto.com/api/v2/sql");
    url.searchParams.set("q", sql);
    url.searchParams.set("format", "json");
    const response = await this.fetcher(url, {
      signal,
      redirect: "error",
      headers: { Accept: "application/json" },
    });
    if (
      !response.ok ||
      response.redirected ||
      !response.headers.get("content-type")?.includes("application/json")
    ) {
      void response.body?.cancel();
      throw new Error("Upstream unavailable or returned a non-JSON response.");
    }
    if (Number(response.headers.get("content-length")) > LIMITS.upstreamBytes) {
      void response.body?.cancel();
      throw new Error("Upstream byte limit exceeded.");
    }
    const text = await readBounded(response.body, LIMITS.upstreamBytes, signal);
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error("Upstream returned malformed JSON.");
    }
    if (
      !data ||
      data.error ||
      !Array.isArray(data.rows) ||
      data.rows.length > LIMITS.rows + 1 ||
      data.rows.some(
        (row: unknown) => !row || typeof row !== "object" || Array.isArray(row),
      )
    )
      throw new Error("Invalid or oversized upstream result.");
    return data as { rows: Record<string, unknown>[]; fields?: Fields };
  }
  private async fields(query: string, signal: AbortSignal): Promise<Fields> {
    const data = await this.request(
      `SELECT * FROM (${query}) AS source LIMIT 0`,
      signal,
    );
    if (
      !data.fields ||
      Object.keys(data.fields).length > 200 ||
      Object.values(data.fields).some((f) => typeof f?.type !== "string")
    )
      throw new Error("Upstream schema unavailable.");
    return data.fields;
  }
  async schema(
    input: z.infer<z.ZodObject<typeof selection>>,
    cancellation?: AbortSignal,
  ) {
    const { dataset, resource } = lookup(input.datasetId, input.resourceId);
    return this.run(
      async (signal) => ({
        datasetId: dataset.id,
        resourceId: resource.id,
        sourceUrl: resource.url,
        license: dataset.license,
        retrievedAt: new Date().toISOString(),
        appliedFilters: [],
        truncated: false,
        nextOffset: null,
        schemaSource: "CARTO query metadata (LIMIT 0), not sample inference",
        fields: await this.fields(resource.query!, signal),
      }),
      cancellation,
    );
  }
  async query(input: z.infer<typeof querySchema>, cancellation?: AbortSignal) {
    const { dataset, resource } = lookup(input.datasetId, input.resourceId);
    return this.run(async (signal) => {
      const fields = await this.fields(resource.query!, signal);
      const columns =
        input.columns ??
        Object.keys(fields)
          .filter(
            (c) =>
              /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(c) &&
              !["the_geom", "the_geom_webmercator"].includes(c),
          )
          .slice(0, LIMITS.columns);
      if (!columns.length) throw new Error("No supported columns.");
      columns.forEach((c) => checkColumn(c, fields));
      const order =
        input.orderBy ??
        (Object.hasOwn(fields, "cartodb_id") ? "cartodb_id" : undefined);
      if (order) checkColumn(order, fields);
      if (input.offset && !order)
        throw new Error("Pagination requires orderBy. Prefer a unique key.");
      const sql = `SELECT ${columns.map(quote).join(",")} FROM (${resource.query}) AS source${where(input.filters, fields)}${order ? ` ORDER BY ${quote(order)}` : ""} LIMIT ${input.limit + 1} OFFSET ${input.offset}`;
      const data = await this.request(sql, signal);
      const truncated = data.rows.length > input.limit;
      return {
        datasetId: dataset.id,
        resourceId: resource.id,
        sourceUrl: resource.url,
        license: dataset.license,
        retrievedAt: new Date().toISOString(),
        appliedFilters: input.filters,
        columns,
        orderBy: order ?? null,
        rows: data.rows.slice(0, input.limit),
        truncated,
        offset: input.offset,
        nextOffset:
          truncated && order && input.offset + input.limit <= LIMITS.offset
            ? input.offset + input.limit
            : null,
        paginationNote:
          "Live data may change between pages; use a unique orderBy. Offset capped at 10000.",
        columnsTruncated:
          !input.columns && columns.length < Object.keys(fields).length,
      };
    }, cancellation);
  }
  async aggregate(
    input: z.infer<typeof aggregateSchema>,
    cancellation?: AbortSignal,
  ) {
    const { dataset, resource } = lookup(input.datasetId, input.resourceId);
    return this.run(async (signal) => {
      const fields = await this.fields(resource.query!, signal);
      if (input.column) checkColumn(input.column, fields);
      if (input.groupBy) checkColumn(input.groupBy, fields);
      if (
        input.operation !== "count" &&
        (!input.column ||
          ![
            "number",
            "integer",
            "numeric",
            "double precision",
            "float",
            "int4",
            "int8",
          ].includes(fields[input.column].type))
      )
        throw new Error(
          "Numeric aggregation requires a numeric schema column.",
        );
      const expression = `${input.operation.toUpperCase()}(${input.column ? quote(input.column) : "*"}) AS value`;
      const group = input.groupBy ? quote(input.groupBy) : null;
      const sql = `SELECT ${group ? `${group} AS group_value,` : ""}${expression} FROM (${resource.query}) AS source${where(input.filters, fields)}${group ? ` GROUP BY ${group} ORDER BY ${group}` : ""} LIMIT ${input.limit + 1}`;
      const data = await this.request(sql, signal);
      return {
        datasetId: dataset.id,
        resourceId: resource.id,
        sourceUrl: resource.url,
        license: dataset.license,
        retrievedAt: new Date().toISOString(),
        appliedFilters: input.filters,
        operation: input.operation,
        column: input.column ?? null,
        groupBy: input.groupBy ?? null,
        rows: data.rows.slice(0, input.limit),
        truncated: data.rows.length > input.limit,
        nextOffset: null,
        scope: "Full matching source resource, not a preview sample.",
      };
    }, cancellation);
  }
}
