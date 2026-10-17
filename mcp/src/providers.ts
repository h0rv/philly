import { z } from "zod";
import type { Dataset, Resource } from "./catalog.ts";
import type { Fetcher } from "./data.ts";
import { querySchema } from "./query.ts";
import { LIMITS, readBounded } from "./bounds.ts";
import { providerByDataset } from "./provider-catalog.ts";

const text = z.string().max(20_000);
const optionalText = text.nullish();
const locItem = z.object({
  id: text,
  title: text,
  date: optionalText,
  description: z.array(text).optional(),
});
const place = z.object({
  id: z.uuid(),
  name: text,
  type: optionalText,
  address: optionalText,
  area: z.object({ name: text }).optional(),
});
const observation = z.object({
  id: text,
  properties: z.object({
    monitoring_location_id: text,
    parameter_code: text,
    time: text,
    value: z.union([text, z.number(), z.null()]),
    unit_of_measure: optionalText,
    approval_status: z.union([text, z.array(text), z.null()]).optional(),
  }),
});
const tide = z.object({ t: text, v: text });
const columnsBySource: Record<string, string[]> = {
  loc: ["id", "title", "date", "description"],
  usgs: [
    "id",
    "monitoring_location_id",
    "parameter_code",
    "time",
    "value",
    "unit_of_measure",
    "approval_status",
  ],
  musicbrainz: ["id", "name", "type", "address", "area"],
  noaa: ["time", "height_meters", "station", "datum", "time_zone", "product"],
};
export class ProviderAdapter {
  private fetcher: Fetcher;
  private nextMusicBrainzAt = 0;
  constructor(fetcher: Fetcher) {
    this.fetcher = fetcher;
  }
  private approved(dataset: Dataset, resource: Resource) {
    const approved = providerByDataset.get(dataset.id);
    if (
      !approved ||
      approved.sourceId !== dataset.sourceId ||
      approved.resources[0].id !== resource.id ||
      approved.resources[0].url !== resource.url
    )
      throw new Error("Unsupported provider resource.");
    return dataset.sourceId!;
  }
  private attribution(dataset: Dataset, resource: Resource) {
    return {
      sourceId: dataset.sourceId!,
      datasetId: dataset.id,
      resourceId: resource.id,
      resourceName: resource.name,
      sourceUrl: resource.url,
      license: dataset.license,
      retrievedAt: new Date().toISOString(),
    };
  }
  schema(dataset: Dataset, resource: Resource) {
    const source = this.approved(dataset, resource);
    return {
      ...this.attribution(dataset, resource),
      fields: Object.fromEntries(
        columnsBySource[source].map((c) => [
          c,
          { type: "string or null (normalized provider projection)" },
        ]),
      ),
      schemaSource:
        "Versioned adapter projection from documented provider JSON; not live field metadata or sample inference",
      appliedFilters: [],
      truncated: false,
      nextOffset: null,
      completeness: "complete",
      completenessReasons: [],
    };
  }
  private async request(url: URL, signal: AbortSignal) {
    signal.throwIfAborted();
    const response = await this.fetcher(url, {
      signal,
      redirect: "error",
      headers: {
        Accept: "application/json, application/geo+json",
        "User-Agent": "PhillyOpenData/0.1 (https://github.com/h0rv/philly)",
      },
    });
    const mediaType = response.headers
      .get("content-type")
      ?.split(";", 1)[0]
      .trim()
      .toLowerCase();
    if (
      !response.ok ||
      response.redirected ||
      !["application/json", "application/geo+json"].includes(mediaType ?? "")
    ) {
      void response.body?.cancel().catch(() => {});
      throw new Error("Upstream unavailable or returned a non-JSON response.");
    }
    if (Number(response.headers.get("content-length")) > LIMITS.upstreamBytes) {
      void response.body?.cancel().catch(() => {});
      throw new Error("Upstream byte limit exceeded.");
    }
    const body = await readBounded(response.body, LIMITS.upstreamBytes, signal);
    let data;
    try {
      data = JSON.parse(body);
    } catch {
      throw new Error("Upstream returned malformed JSON.");
    }
    if (!data || typeof data !== "object" || Array.isArray(data) || data.error)
      throw new Error("Invalid upstream result.");
    return data;
  }
  async query(
    dataset: Dataset,
    resource: Resource,
    input: z.infer<typeof querySchema>,
    signal: AbortSignal,
  ) {
    const source = this.approved(dataset, resource);
    if (input.filters.length || input.orderBy || input.orderDirection !== "asc")
      throw new Error(
        "Unsupported operation: this provider supports selected columns and native paging only; no filters or custom ordering.",
      );
    const columns = input.columns ?? columnsBySource[source];
    for (const column of columns)
      if (!columnsBySource[source].includes(column))
        throw new Error(`Unknown column: ${column}. Use get_schema.`);
    if (source === "loc" && input.offset % input.limit !== 0)
      throw new Error(
        "Unsupported operation: LoC offset must be a multiple of limit; preserve page size between pages.",
      );
    if (source === "musicbrainz") {
      if (Date.now() < this.nextMusicBrainzAt)
        throw new Error(
          "Busy: MusicBrainz permits one request per second; retry later.",
        );
      this.nextMusicBrainzAt = Date.now() + 1000;
    }
    // Start with the immutable, reviewed catalog endpoint; no input or response
    // URL is ever resolved or followed. Only bounded numeric paging is added.
    const url = new URL(resource.url);
    if (source === "loc") {
      url.searchParams.set("fo", "json");
      url.searchParams.set("at", "results,pagination");
      url.searchParams.set("c", String(input.limit));
      url.searchParams.set("sp", String(input.offset / input.limit + 1));
    }
    if (source === "musicbrainz") {
      url.searchParams.set("limit", String(input.limit));
      url.searchParams.set("offset", String(input.offset));
    }
    if (source === "usgs") {
      url.searchParams.set("f", "json");
      url.searchParams.set("limit", String(input.limit + 1));
      url.searchParams.set("offset", String(input.offset));
    }
    const raw = await this.request(url, signal);
    let rows: Record<string, unknown>[];
    let more: boolean;
    try {
      if (source === "loc") {
        const data = z
          .object({
            results: z.array(locItem).max(input.limit),
            pagination: z.object({ total: z.number().int().nonnegative() }),
          })
          .parse(raw);
        rows = data.results.map((r) => ({
          id: r.id,
          title: r.title,
          date: r.date ?? null,
          description: r.description?.join("\n") ?? null,
        }));
        more = input.offset + rows.length < data.pagination.total;
        if (more && rows.length > 0 && rows.length !== input.limit)
          throw new Error();
      } else if (source === "musicbrainz") {
        const data = z
          .object({
            places: z.array(place).max(input.limit),
            count: z.number().int().nonnegative(),
            offset: z.number().int().nonnegative(),
          })
          .parse(raw);
        if (data.offset !== input.offset) throw new Error();
        rows = data.places.map((r) => ({
          id: r.id,
          name: r.name,
          type: r.type ?? null,
          address: r.address ?? null,
          area: r.area?.name ?? null,
        }));
        more = input.offset + rows.length < data.count;
      } else if (source === "usgs") {
        const data = z
          .object({
            features: z.array(observation).max(input.limit + 1),
            links: z.array(z.object({ rel: text })).max(100),
          })
          .parse(raw);
        if (
          data.features.some(
            (f) => f.properties.monitoring_location_id !== "USGS-01474500",
          )
        )
          throw new Error();
        rows = data.features.map((f) => ({
          id: f.id,
          ...f.properties,
          value:
            f.properties.value === null ? null : String(f.properties.value),
          unit_of_measure: f.properties.unit_of_measure ?? null,
          approval_status: Array.isArray(f.properties.approval_status)
            ? f.properties.approval_status.join(",")
            : (f.properties.approval_status ?? null),
        }));
        more =
          rows.length > input.limit || data.links.some((l) => l.rel === "next");
      } else {
        const data = z
          .object({ predictions: z.array(tide).max(25) })
          .parse(raw);
        rows = data.predictions
          .map((r) => ({
            time: r.t,
            height_meters: r.v,
            station: "8545240",
            datum: "MLLW",
            time_zone: "UTC",
            product: "hourly_tide_prediction",
          }))
          .slice(input.offset);
        more = rows.length > input.limit;
      }
    } catch {
      throw new Error("Invalid or oversized upstream result.");
    }
    rows = rows.slice(0, input.limit);
    const next = input.offset + rows.length;
    const reasons = [
      ...(more ? ["row_limit"] : []),
      ...(input.offset ? ["offset"] : []),
      ...(more && next > LIMITS.offset ? ["offset_limit"] : []),
      ...(more && rows.length === 0 ? ["upstream_no_progress"] : []),
    ];
    return {
      ...this.attribution(dataset, resource),
      rows: rows.map((r) =>
        Object.fromEntries(columns.map((c) => [c, r[c] ?? null])),
      ),
      columns,
      appliedFilters: [],
      orderBy: null,
      orderDirection: null,
      offset: input.offset,
      truncated: more,
      nextOffset:
        more && rows.length > 0 && next <= LIMITS.offset ? next : null,
      completeness: reasons.length ? "partial" : "complete",
      completenessReasons: reasons,
      columnsTruncated: false,
      paginationNote:
        source === "noaa"
          ? "Fixed today UTC hourly predictions; pages can shift at UTC midnight. Meters relative to MLLW, not observed levels."
          : "Native provider order; live search/index changes may shift pages. LoC requires offsets divisible by the page limit. No response continuation URL is followed.",
    };
  }
}
