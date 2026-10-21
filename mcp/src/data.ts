import { ProviderAdapter } from "./providers.ts";
import { providerByDataset } from "./provider-catalog.ts";
import { z } from "zod";
import { ArcgisAdapter } from "./arcgis.ts";
import { arcgisLayer, SOURCE_ID } from "./sources.ts";
import { lookup } from "./catalog.ts";
import { LIMITS, readBounded } from "./bounds.ts";
import {
  identifier,
  selection,
  filtersSchema,
  querySchema,
  aggregateSchema,
  where,
  checkColumn,
  type Fields,
} from "./query.ts";
export {
  identifier,
  selection,
  filterSchema,
  filtersSchema,
  querySchema,
  aggregateSchema,
  where,
} from "./query.ts";
export type Fetcher = (
  url: string | URL,
  init?: RequestInit,
) => Promise<Response>;
const quote = (s: string) => `"${identifier.parse(s)}"`;
export class DataService {
  private active = 0;
  private providers: ProviderAdapter;
  private fetcher: Fetcher;
  // Workers native fetch requires the global receiver. Never invoke a stored
  // native function as this.fetcher(), which supplies the DataService receiver.
  constructor(fetcher: Fetcher = (url, init) => globalThis.fetch(url, init)) {
    this.fetcher = fetcher;
    this.providers = new ProviderAdapter(fetcher);
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
      signal.throwIfAborted();
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
      // Workers supports manual/follow, not redirect:error. Reject the
      // returned 3xx below before reading a body; never follow Location.
      redirect: "manual",
      headers: { Accept: "application/json" },
    });
    if (
      !response.ok ||
      response.redirected ||
      response.headers
        .get("content-type")
        ?.split(";", 1)[0]
        .trim()
        .toLowerCase() !== "application/json"
    ) {
      void response.body?.cancel().catch(() => {});
      throw new Error("Upstream unavailable or returned a non-JSON response.");
    }
    if (Number(response.headers.get("content-length")) > LIMITS.upstreamBytes) {
      void response.body?.cancel().catch(() => {});
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
      typeof data.fields !== "object" ||
      Array.isArray(data.fields) ||
      data.rows.length !== 0 ||
      Object.keys(data.fields).length === 0 ||
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
    return this.run(async (signal) => {
      if (providerByDataset.has(dataset.id))
        return this.providers.schema(dataset, resource);
      if (arcgisLayer(resource.url))
        return new ArcgisAdapter(this.fetcher).schema(
          dataset,
          resource,
          input,
          signal,
        );
      const fields = await this.fields(resource.query!, signal);
      return {
        sourceId: SOURCE_ID,
        datasetId: dataset.id,
        resourceId: resource.id,
        resourceName: resource.name,
        sourceUrl: resource.url,
        license: dataset.license,
        retrievedAt: new Date().toISOString(),
        appliedFilters: [],
        completeness: "complete",
        completenessReasons: [],
        truncated: false,
        nextOffset: null,
        schemaSource: "CARTO query metadata (LIMIT 0), not sample inference",
        fields,
      };
    }, cancellation);
  }
  async query(input: z.infer<typeof querySchema>, cancellation?: AbortSignal) {
    const { dataset, resource } = lookup(input.datasetId, input.resourceId);
    return this.run(async (signal) => {
      if (providerByDataset.has(dataset.id))
        return this.providers.query(dataset, resource, input, signal);
      if (arcgisLayer(resource.url))
        return new ArcgisAdapter(this.fetcher).query(
          dataset,
          resource,
          input,
          signal,
        );
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
      const sql = `SELECT ${columns.map(quote).join(",")} FROM (${resource.query}) AS source${where(input.filters, fields)}${order ? ` ORDER BY ${quote(order)} ${input.orderDirection.toUpperCase()}` : ""} LIMIT ${input.limit + 1} OFFSET ${input.offset}`;
      const data = await this.request(sql, signal);
      const truncated = data.rows.length > input.limit;
      return {
        sourceId: SOURCE_ID,
        datasetId: dataset.id,
        resourceId: resource.id,
        resourceName: resource.name,
        sourceUrl: resource.url,
        license: dataset.license,
        retrievedAt: new Date().toISOString(),
        appliedFilters: input.filters,
        columns,
        orderBy: order ?? null,
        orderDirection: order ? input.orderDirection : null,
        rows: data.rows.slice(0, input.limit),
        completeness:
          truncated ||
          input.offset > 0 ||
          (!input.columns && columns.length < Object.keys(fields).length)
            ? "partial"
            : "complete",
        completenessReasons: [
          ...(truncated ? ["row_limit"] : []),
          ...(input.offset ? ["offset"] : []),
          ...(truncated && !order ? ["no_stable_order"] : []),
          ...(truncated && input.offset + input.limit > LIMITS.offset
            ? ["offset_limit"]
            : []),
          ...(!input.columns && columns.length < Object.keys(fields).length
            ? ["columns_omitted"]
            : []),
        ],
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
      if (providerByDataset.has(dataset.id))
        throw new Error(
          "Unsupported operation: provider aggregation is not supported.",
        );
      if (arcgisLayer(resource.url))
        throw new Error(
          "Unsupported operation: ArcGIS aggregation is not supported.",
        );
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
      const groupOrder =
        input.orderBy === "value"
          ? `value ${input.orderDirection.toUpperCase()}, group_value ASC`
          : `group_value ${input.orderDirection.toUpperCase()}`;
      const sql = `SELECT ${group ? `${group} AS group_value,` : ""}${expression} FROM (${resource.query}) AS source${where(input.filters, fields)}${group ? ` GROUP BY ${group} ORDER BY ${groupOrder}` : ""} LIMIT ${input.limit + 1}`;
      const data = await this.request(sql, signal);
      return {
        sourceId: SOURCE_ID,
        datasetId: dataset.id,
        resourceId: resource.id,
        resourceName: resource.name,
        sourceUrl: resource.url,
        license: dataset.license,
        retrievedAt: new Date().toISOString(),
        appliedFilters: input.filters,
        operation: input.operation,
        column: input.column ?? null,
        groupBy: input.groupBy ?? null,
        orderBy: group ? input.orderBy : null,
        orderDirection: group ? input.orderDirection : null,
        rows: data.rows.slice(0, input.limit),
        completeness: data.rows.length > input.limit ? "partial" : "complete",
        completenessReasons:
          data.rows.length > input.limit ? ["group_limit"] : [],
        truncated: data.rows.length > input.limit,
        nextOffset: null,
        scope: "Full matching source resource, not a preview sample.",
      };
    }, cancellation);
  }
}
