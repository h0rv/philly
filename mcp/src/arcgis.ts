import { z } from "zod";
import type { Dataset, Resource } from "./catalog.ts";
import { identifier, where, querySchema } from "./query.ts";
import type { Fetcher } from "./data.ts";
import { LIMITS, readBounded } from "./bounds.ts";
import { arcgisLayer, SOURCE_ID } from "./sources.ts";

const field = z.object({ name: identifier, type: z.string().max(100) });
const metadata = z.object({
  fields: z.array(field).min(1).max(200),
  objectIdField: identifier,
  maxRecordCount: z.number().int().positive(),
  advancedQueryCapabilities: z.object({
    supportsPagination: z.boolean().optional(),
    supportsOrderBy: z.boolean().optional(),
  }),
});
export class ArcgisAdapter {
  private fetcher: Fetcher;
  constructor(fetcher: Fetcher) {
    this.fetcher = fetcher;
  }
  private async request(
    layer: string,
    params: Record<string, string>,
    signal: AbortSignal,
    query = false,
  ): Promise<unknown> {
    const approved = arcgisLayer(layer);
    if (!approved) throw new Error("Unsupported ArcGIS layer.");
    const url = new URL(approved + (query ? "/query" : ""));
    url.search = new URLSearchParams({ ...params, f: "json" }).toString();
    signal.throwIfAborted();
    const response = await this.fetcher(url, {
      signal,
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
    if (!data || typeof data !== "object" || Array.isArray(data) || data.error)
      throw new Error("Invalid upstream result.");
    return data;
  }
  private async fields(resource: Resource, signal: AbortSignal) {
    const parsed = metadata.safeParse(
      await this.request(arcgisLayer(resource.url)!, {}, signal),
    );
    if (!parsed.success) throw new Error("Upstream schema unavailable.");
    const meta = parsed.data;
    const fields = Object.fromEntries(
      meta.fields.map((f) => [f.name, { type: f.type }]),
    );
    if (
      Object.keys(fields).length !== meta.fields.length ||
      fields[meta.objectIdField]?.type !== "esriFieldTypeOID"
    )
      throw new Error("Upstream schema unavailable.");
    return { meta, fields };
  }
  private attribution(dataset: Dataset, resource: Resource) {
    return {
      sourceId: SOURCE_ID,
      datasetId: dataset.id,
      resourceId: resource.id,
      resourceName: resource.name,
      sourceUrl: resource.url,
      license: dataset.license,
      retrievedAt: new Date().toISOString(),
    };
  }
  async schema(
    dataset: Dataset,
    resource: Resource,
    _input: unknown,
    signal: AbortSignal,
  ) {
    const { fields } = await this.fields(resource, signal);
    return {
      ...this.attribution(dataset, resource),
      fields,
      appliedFilters: [],
      truncated: false,
      nextOffset: null,
      completeness: "complete",
      completenessReasons: [],
      schemaSource: "ArcGIS layer field metadata, not sample inference",
    };
  }
  async query(
    dataset: Dataset,
    resource: Resource,
    input: z.infer<typeof querySchema>,
    signal: AbortSignal,
  ) {
    const { meta, fields } = await this.fields(resource, signal);
    if (
      !meta.advancedQueryCapabilities.supportsPagination ||
      !meta.advancedQueryCapabilities.supportsOrderBy
    )
      throw new Error(
        "Unsupported operation: layer must support pagination and ordering.",
      );
    const columns =
      input.columns ??
      Object.keys(fields)
        .filter((c) => fields[c].type !== "esriFieldTypeGeometry")
        .slice(0, LIMITS.columns);
    const order = input.orderBy ?? meta.objectIdField;
    for (const c of [...columns, order])
      if (
        !Object.hasOwn(fields, c) ||
        fields[c].type === "esriFieldTypeGeometry"
      )
        throw new Error(`Unknown column: ${c}. Use get_schema.`);
    if (!columns.length) throw new Error("No supported columns.");
    // Reuse the structured filter grammar; ArcGIS standardized SQL accepts field
    // names without quotes. Identifiers and literals have already been validated.
    const predicate =
      where(input.filters, fields, (s) => identifier.parse(s)).replace(
        /^ WHERE /,
        "",
      ) || "1=1";
    const requested = Math.min(input.limit + 1, meta.maxRecordCount);
    const orderFields = `${order} ${input.orderDirection.toUpperCase()}${order !== meta.objectIdField ? `,${meta.objectIdField} ASC` : ""}`;
    const raw = await this.request(
      arcgisLayer(resource.url)!,
      {
        where: predicate,
        outFields: columns.join(","),
        returnGeometry: "false",
        resultRecordCount: String(requested),
        resultOffset: String(input.offset),
        orderByFields: orderFields,
        sqlFormat: "standard",
      },
      signal,
      true,
    );
    const result = z
      .object({
        features: z
          .array(z.object({ attributes: z.record(z.string(), z.unknown()) }))
          .max(requested),
        exceededTransferLimit: z.boolean().optional(),
      })
      .safeParse(raw);
    if (!result.success)
      throw new Error("Invalid or oversized upstream result.");
    // A server limit can be smaller than the requested page. Advance by rows
    // actually returned, never skip records to reach the requested page size.
    const rows = result.data.features
      .slice(0, input.limit)
      .map((f) =>
        Object.fromEntries(
          columns
            .filter((c) => Object.hasOwn(f.attributes, c))
            .map((c) => [c, f.attributes[c]]),
        ),
      );
    const transferUnknown =
      result.data.exceededTransferLimit === undefined &&
      result.data.features.length === requested &&
      requested <= input.limit;
    const truncated =
      result.data.features.length > input.limit ||
      result.data.exceededTransferLimit === true ||
      transferUnknown;
    const next = input.offset + rows.length;
    const columnsTruncated =
      !input.columns && columns.length < Object.keys(fields).length;
    const reasons = [
      ...(result.data.features.length > input.limit ? ["row_limit"] : []),
      ...(result.data.exceededTransferLimit ? ["upstream_transfer_limit"] : []),
      ...(transferUnknown ? ["upstream_completeness_unknown"] : []),
      ...(input.offset ? ["offset"] : []),
      ...(columnsTruncated ? ["columns_omitted"] : []),
      ...(truncated && next > LIMITS.offset ? ["offset_limit"] : []),
      ...(truncated && !rows.length ? ["upstream_no_progress"] : []),
    ];
    return {
      ...this.attribution(dataset, resource),
      appliedFilters: input.filters,
      columns,
      rows,
      orderBy: order,
      orderDirection: input.orderDirection,
      truncated,
      offset: input.offset,
      nextOffset:
        truncated && rows.length > 0 && next <= LIMITS.offset ? next : null,
      completeness: reasons.length ? "partial" : "complete",
      completenessReasons: reasons,
      columnsTruncated: Boolean(columnsTruncated),
      paginationNote:
        "Ordered by requested column then object ID; live data may change between pages. Offset capped at 10000.",
    };
  }
}
