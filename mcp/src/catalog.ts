import { providerCatalog, providerByDataset } from "./provider-catalog.ts";
import raw from "./generated/catalog.json" with { type: "json" };
import { arcgisLayer, SOURCE_ID } from "./sources.ts";
export interface Resource {
  id: string;
  name: string;
  format: string;
  url: string;
  query: string | null;
}
export interface Dataset {
  sourceId?: string;
  id: string;
  title: string;
  description: string;
  organization: string | null;
  license: string | null;
  categories: string[];
  resources: Resource[];
}
export const openDataCatalog = raw as Dataset[];
export const catalog = [...openDataCatalog, ...providerCatalog];
export const datasets = new Map(catalog.map((d) => [d.id, d]));
const index = catalog.map((d) => ({
  d,
  text: `${d.title} ${d.description} ${d.categories.join(" ")} ${d.resources.map((r) => r.name).join(" ")}`.toLowerCase(),
}));
export function search(query: string, offset: number, limit: number) {
  const words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const found = index.filter(({ text }) =>
    words.every((w) => text.includes(w)),
  );
  return {
    total: found.length,
    offset,
    nextOffset: offset + limit < found.length ? offset + limit : null,
    datasets: found.slice(offset, offset + limit).map(({ d }) => ({
      id: d.id,
      title: d.title,
      categories: d.categories,
      sourceId: d.sourceId ?? SOURCE_ID,
      queryable: d.resources.some(
        (r) =>
          r.query !== null || arcgisLayer(r.url) || providerByDataset.has(d.id),
      ),
      license: d.license,
    })),
  };
}
export function lookup(datasetId: string, resourceId?: string) {
  const dataset = datasets.get(datasetId);
  if (!dataset) throw new Error("Unknown dataset ID. Use search_datasets.");
  const resource = resourceId
    ? dataset.resources.find((r) => r.id === resourceId)
    : providerByDataset.has(dataset.id)
      ? dataset.resources[0]
      : (dataset.resources.find((r) => r.query) ??
        dataset.resources.find((r) => arcgisLayer(r.url)));
  if (
    !resource ||
    (!resource.query &&
      !arcgisLayer(resource.url) &&
      !providerByDataset.has(dataset.id))
  )
    throw new Error(
      "Unsupported resource: use describe_dataset for approved CARTO, Philadelphia ArcGIS, and scoped LoC/USGS/MusicBrainz/NOAA capabilities or source links.",
    );
  return { dataset, resource };
}
export function describe(datasetId: string, offset: number, limit: number) {
  const d = datasets.get(datasetId);
  if (!d) throw new Error("Unknown dataset ID.");
  return {
    ...d,
    sourceId: d.sourceId ?? SOURCE_ID,
    resources: d.resources
      .slice(offset, offset + limit)
      .map(({ query, ...r }) => ({
        ...r,
        sourceId: d.sourceId ?? SOURCE_ID,
        backend: providerByDataset.has(d.id)
          ? d.sourceId!
          : query
            ? "carto"
            : arcgisLayer(r.url)
              ? "arcgis"
              : "discovery-only",
        capabilities: providerByDataset.has(d.id)
          ? ["schema", "preview", "query"]
          : query
            ? ["schema", "preview", "query", "count", "aggregate"]
            : arcgisLayer(r.url)
              ? ["schema", "preview", "query"]
              : [],
        unsupportedReason:
          query || arcgisLayer(r.url) || providerByDataset.has(d.id)
            ? null
            : "No bounded adapter approved for this resource.",
      })),
    totalResources: d.resources.length,
    nextOffset: offset + limit < d.resources.length ? offset + limit : null,
  };
}
