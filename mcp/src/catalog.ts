import raw from "./generated/catalog.json" with { type: "json" };
export interface Resource {
  id: string;
  name: string;
  format: string;
  url: string;
  query: string | null;
}
export interface Dataset {
  id: string;
  title: string;
  description: string;
  organization: string | null;
  license: string | null;
  categories: string[];
  resources: Resource[];
}
export const catalog = raw as Dataset[];
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
      queryable: d.resources.some((r) => r.query !== null),
      license: d.license,
    })),
  };
}
export function lookup(datasetId: string, resourceId?: string) {
  const dataset = datasets.get(datasetId);
  if (!dataset) throw new Error("Unknown dataset ID. Use search_datasets.");
  const resource = resourceId
    ? dataset.resources.find((r) => r.id === resourceId)
    : dataset.resources.find((r) => r.query);
  if (!resource?.query)
    throw new Error(
      "Unsupported resource: only approved CARTO queries support remote operations. Use describe_dataset for source links or the Python CLI locally.",
    );
  return { dataset, resource };
}
export function describe(datasetId: string, offset: number, limit: number) {
  const d = datasets.get(datasetId);
  if (!d) throw new Error("Unknown dataset ID.");
  return {
    ...d,
    resources: d.resources
      .slice(offset, offset + limit)
      .map(({ query, ...r }) => ({
        ...r,
        backend: query ? "carto" : "discovery-only",
        capabilities: query
          ? ["schema", "preview", "query", "count", "aggregate"]
          : [],
        unsupportedReason: query
          ? null
          : "No bounded adapter approved for this resource.",
      })),
    totalResources: d.resources.length,
    nextOffset: offset + limit < d.resources.length ? offset + limit : null,
  };
}
