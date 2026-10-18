import type { Menu } from "./model.ts";

export interface Filters {
  readonly search: string;
  readonly decade: string;
  readonly from: number | null;
  readonly to: number | null;
  readonly includeUndated: boolean;
  readonly area: string;
  readonly cuisine: string;
  readonly collection: string;
}

export const defaults: Filters = {
  search: "",
  decade: "all",
  from: null,
  to: null,
  includeUndated: true,
  area: "all",
  cuisine: "all",
  collection: "all",
};

export function filterMenus(menus: readonly Menu[], filters: Filters): Menu[] {
  const query = filters.search.trim().toLocaleLowerCase("en-US");
  return menus.filter((menu) => {
    const year = menu.date.year;
    if (filters.from !== null && filters.to !== null && filters.from > filters.to) return false;
    if (year === null) {
      if (!filters.includeUndated || (filters.decade !== "all" && filters.decade !== "undated"))
        return false;
    } else {
      if (filters.decade === "undated") return false;
      if (filters.decade !== "all" && Math.floor(year / 10) * 10 !== Number(filters.decade))
        return false;
      if (filters.from !== null && year < filters.from) return false;
      if (filters.to !== null && year > filters.to) return false;
    }
    if (
      filters.area === "unlocated"
        ? menu.location !== null
        : filters.area !== "all" && menu.location?.area !== filters.area
    )
      return false;
    if (filters.collection !== "all" && menu.provider !== filters.collection) return false;
    if (
      filters.cuisine === "unspecified"
        ? menu.cuisines.length !== 0
        : filters.cuisine !== "all" && !menu.cuisines.includes(filters.cuisine)
    )
      return false;
    return (
      query === "" ||
      [
        menu.name,
        menu.location?.address ?? "",
        ...menu.dishes.map((d) => d.name),
        ...menu.pages.map((p) => p.transcript),
      ].some((text) => text.toLocaleLowerCase("en-US").includes(query))
    );
  });
}

export function priceLabel(
  price: number | null,
  priceHigh: number | null,
  currency: string,
): string {
  if (price === null) return "Price unrecorded";
  const format = (value: number): string =>
    `${currency === "Dollars" || currency === "USD" ? "$" : ""}${value.toFixed(2)}`;
  const suffix =
    currency === "Dollars" || currency === "USD"
      ? ""
      : ` ${currency === "" ? "(currency unrecorded)" : currency}`;
  return `${format(price)}${priceHigh === null || priceHigh === price ? "" : `–${format(priceHigh)}`}${suffix}`;
}
