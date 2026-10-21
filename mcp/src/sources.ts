// Catalog identity is independent of the query transport. Only these reviewed
// origins and the Philadelphia tenant's numbered FeatureServer layers qualify.
export const SOURCE_ID = "opendataphilly";
export function arcgisLayer(value: string): string | null {
  try {
    if (/%|\\|(?:^|\/)\.\.?(?:\/|$)/.test(value.split("?", 1)[0])) return null;
    const url = new URL(value);
    if (
      url.origin !== "https://services.arcgis.com" ||
      url.username ||
      url.password ||
      url.hash
    )
      return null;
    if (
      url.pathname ===
        "/fLeGjb7u4uXqeF9q/arcgis/rest/services/GeotechnicalTest/FeatureServer" &&
      !url.search
    )
      url.pathname += "/0";
    if (
      !/^\/fLeGjb7u4uXqeF9q\/arcgis\/rest\/services\/[A-Za-z0-9_]+\/FeatureServer\/\d+(?:\/query)?$/i.test(
        url.pathname,
      )
    )
      return null;
    for (const [key, value] of url.searchParams) {
      if (!(
        (key === "where" && value === "1=1") ||
        (key === "outFields" && value === "*") ||
        (key === "f" && ["json", "pjson", "geojson"].includes(value))
      ))
        return null;
    }
    return url.origin + url.pathname.replace(/\/query$/i, "");
  } catch {
    return null;
  }
}
