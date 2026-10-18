import L from "leaflet";
import { parseArchive } from "./model.ts";
import type { Menu, MenuArchive } from "./model.ts";
import { defaults, filterMenus, priceLabel } from "./filters.ts";
import type { Filters } from "./filters.ts";

function element(id: string): HTMLElement {
  const result = document.getElementById(id);
  if (result === null) throw new Error(`Missing HTML element: ${id}`);
  return result;
}
function input(id: string): HTMLInputElement {
  const result = element(id);
  if (!(result instanceof HTMLInputElement)) throw new Error(`Expected input: ${id}`);
  return result;
}
function select(id: string): HTMLSelectElement {
  const result = element(id);
  if (!(result instanceof HTMLSelectElement)) throw new Error(`Expected select: ${id}`);
  return result;
}
function node<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text = "",
  className = "",
): HTMLElementTagNameMap[K] {
  const result = document.createElement(tag);
  result.textContent = text;
  result.className = className;
  return result;
}
function link(text: string, url: string, className = ""): HTMLAnchorElement {
  const result = node("a", text, className);
  result.href = url;
  return result;
}
function button(text: string, action: () => void, className = ""): HTMLButtonElement {
  const result = node("button", text, className);
  result.type = "button";
  result.addEventListener("click", action);
  return result;
}
function addOptions(target: HTMLSelectElement, values: readonly string[]): void {
  for (const value of values) {
    const option = node("option", value);
    option.value = value;
    target.append(option);
  }
}
function readYear(id: string): number | null {
  const field = input(id);
  if (field.value === "" || !field.validity.valid) return null;
  return field.valueAsNumber;
}
function currentFilters(): Filters {
  return {
    search: input("search").value,
    decade: select("decade").value,
    area: select("area").value,
    cuisine: select("cuisine").value,
    collection: select("collection").value,
    from: readYear("from"),
    to: readYear("to"),
    includeUndated: input("undated").checked,
  };
}
function metadata(label: string, value: string, target: HTMLDListElement): void {
  target.append(node("dt", label), node("dd", value));
}
function section(title: string, ...children: Node[]): HTMLElement {
  const result = node("section", "", "detail-section");
  result.append(node("h3", title), ...children);
  return result;
}
function disclosure(title: string, ...children: Node[]): HTMLDetailsElement {
  const result = node("details");
  result.append(node("summary", title), ...children);
  return result;
}

function initialize(archive: MenuArchive): void {
  const menus = archive.menus;
  if (archive.snapshots.some((snapshot) => snapshot.id === "parent-audit-notes")) {
    const notice = node(
      "p",
      "Preview collection: two catalog records from supplied research notes. Transcriptions and verified map locations are awaiting source files; no pins or prices have been guessed.",
      "notice",
    );
    element("filters").before(notice);
  }
  const map = L.map("map", {
    scrollWheelZoom: false,
    zoomControl: true,
    minZoom: 11,
    maxZoom: 18,
    maxBounds: [
      [39.84, -75.34],
      [40.17, -74.9],
    ],
    attributionControl: true,
  });
  const home: L.LatLngExpression = [39.95, -75.158];
  map.setView(home, 13);
  const locator = L.imageOverlay(
    "locator.svg",
    [
      [39.9, -75.225],
      [40.012, -75.095],
    ],
    { interactive: false },
  ).addTo(map);
  map.attributionControl.setPrefix(link("Leaflet", "https://leafletjs.com").outerHTML);
  const pins = L.layerGroup().addTo(map);
  let selected: string | null = null;
  let shown = [...menus];
  let tiles: L.TileLayer | null = null;
  let tileErrors = 0;

  const decades = [
    ...new Set(
      menus.flatMap((m) => (m.date.year === null ? [] : [Math.floor(m.date.year / 10) * 10])),
    ),
  ].toSorted((a, b) => a - b);
  for (const decade of decades) {
    const option = node("option", `${String(decade)}s`);
    option.value = String(decade);
    select("decade").append(option);
  }
  addOptions(
    select("area"),
    [...new Set(menus.flatMap((m) => (m.location === null ? [] : [m.location.area])))].toSorted(),
  );
  addOptions(select("cuisine"), [...new Set(menus.flatMap((m) => [...m.cuisines]))].toSorted());

  function showList(): void {
    const previous = selected;
    selected = null;
    element("list-view").hidden = false;
    element("detail-view").hidden = true;
    if (location.hash.startsWith("#menu="))
      history.replaceState(null, "", `${location.pathname}${location.search}`);
    renderPins();
    const row = [
      ...element("menu-list").querySelectorAll<HTMLButtonElement>("button[data-menu-id]"),
    ].find((b) => b.dataset["menuId"] === previous);
    if (row !== undefined) row.focus({ preventScroll: true });
    else element("results").focus({ preventScroll: true });
  }

  function showMenu(menu: Menu, moveMap = true): void {
    selected = menu.id;
    element("list-view").hidden = true;
    const detail = element("detail-view");
    detail.hidden = false;
    detail.replaceChildren(button("← Back to the menu file", showList, "back-button"));
    const cover = node("div", "", "detail-cover");
    const title = node("h2", menu.name);
    title.tabIndex = -1;
    cover.append(
      node("p", `${menu.provider} · ${menu.recordId}`, "eyebrow"),
      title,
      node("span", "", "cover-rule"),
      node("p", menu.date.label),
      node("p", menu.location?.address ?? "Philadelphia · exact place unresolved"),
    );
    detail.append(cover);
    const actions = node("div", "", "detail-actions");
    actions.append(link(`View at ${menu.provider} >`, menu.url));
    const firstPage = menu.pages[0];
    if (firstPage !== undefined)
      actions.append(link("Open original scan >", firstPage.url, "secondary"));
    detail.append(actions);
    detail.append(
      node(
        "p",
        "Menu lettering above is an original index card, not a reproduction of the scan.",
        "small",
      ),
    );
    const dating = node("dl");
    metadata("Date basis", menu.date.basis, dating);
    metadata("Recorded precision", menu.date.precision, dating);
    metadata("Original catalog date", menu.date.catalogValue || "Unrecorded", dating);
    metadata(
      "Location",
      menu.location === null
        ? menu.locationNote
        : `${menu.location.address} · ${menu.location.area}`,
      dating,
    );
    if (menu.location !== null) {
      metadata("Pin basis", menu.location.basis, dating);
      dating.append(node("dt", "Location evidence"), node("dd"));
      dating.lastElementChild?.append(link("Review the source >", menu.location.evidence));
    }
    metadata("Cuisine basis", menu.cuisineBasis, dating);
    const dateSection = section("On the record", dating);
    for (const note of menu.notes) dateSection.append(node("p", note));
    detail.append(dateSection);

    const nearbyMenus = shown.filter(
      (other) =>
        other.id !== menu.id &&
        menu.location !== null &&
        other.location?.lat === menu.location.lat &&
        other.location.lng === menu.location.lng,
    );
    if (nearbyMenus.length > 0) {
      const otherLinks = node("div", "", "page-links");
      for (const other of nearbyMenus)
        otherLinks.append(
          button(`${other.name} · ${other.date.label}`, () => {
            showMenu(other, false);
          }),
        );
      detail.append(section("Other menus at this point", otherLinks));
    }
    if (menu.dishes.length > 0) {
      const dishes = section("A taste of the menu");
      const table = node("table", "", "dish-list");
      table.append(
        node(
          "caption",
          `${String(menu.dishes.length)} transcribed item occurrences · ${menu.currency || "currency unrecorded"} · original amounts`,
        ),
      );
      const head = node("thead");
      const headings = node("tr");
      headings.append(node("th", "Dish / item"), node("th", "Historical price"));
      head.append(headings);
      table.append(head);
      const body = node("tbody");
      function appendDishes(all: boolean): void {
        body.replaceChildren();
        for (const dish of all ? menu.dishes : menu.dishes.slice(0, 16)) {
          const row = node("tr");
          const name = node("td", dish.name);
          const page = menu.pages.find((p) => p.id === dish.pageId);
          if (page !== undefined)
            name.append(document.createTextNode(" "), link(`> ${page.label}`, page.url, "small"));
          row.append(name, node("td", priceLabel(dish.price, dish.priceHigh, menu.currency)));
          body.append(row);
        }
      }
      appendDishes(false);
      table.append(body);
      dishes.append(table);
      if (menu.dishes.length > 16)
        dishes.append(
          button(
            `Read all ${String(menu.dishes.length)} items`,
            () => {
              appendDishes(true);
              dishes.querySelector("button")?.remove();
            },
            "all-dishes",
          ),
        );
      detail.append(dishes);
    } else {
      detail.append(
        section(
          "Inside the menu",
          node(
            "p",
            "No item-level transcription is included for this source record. The library may provide page transcripts; open the original to explore.",
          ),
        ),
      );
    }
    if (menu.pages.length > 0) {
      const pages = node("div", "", "page-links");
      for (const page of menu.pages) pages.append(link(page.label, page.url));
      const scans = section("Turn the pages", pages);
      for (const page of menu.pages.filter((p) => p.transcript !== ""))
        scans.append(
          disclosure(
            `${page.label} · existing transcript`,
            node("p", page.transcript, "transcript"),
          ),
        );
      detail.append(scans);
    }
    const provenance = node("dl");
    metadata(
      "Holding library",
      menu.provider === "NYPL" ? "New York Public Library" : "The Culinary Institute of America",
      provenance,
    );
    metadata("Collection", menu.collection, provenance);
    metadata("Record ID", menu.recordId, provenance);
    metadata("Rights / reuse", menu.rights, provenance);
    metadata("Snapshot", menu.snapshotId, provenance);
    const snapshot = archive.snapshots.find((s) => s.id === menu.snapshotId);
    if (snapshot !== undefined) {
      metadata("Retrieved", snapshot.retrievedAt, provenance);
      metadata("Snapshot SHA-256", snapshot.sha256, provenance);
      metadata("Snapshot description", snapshot.description, provenance);
    }
    const catalog = node("dl");
    for (const [key, value] of Object.entries(menu.catalog)) metadata(key, value, catalog);
    detail.append(
      section("Paper trail", provenance, disclosure("Original catalog fields", catalog)),
    );
    history.replaceState(null, "", `#menu=${encodeURIComponent(menu.id)}`);
    detail.scrollTop = 0;
    renderPins();
    if (moveMap && menu.location !== null)
      map.setView([menu.location.lat, menu.location.lng], 14, { animate: false });
    if (matchMedia("(max-width: 680px)").matches)
      element("results").scrollIntoView({ block: "start", behavior: "instant" });
    title.focus({ preventScroll: true });
  }

  function renderPins(): void {
    pins.clearLayers();
    const groups = new Map<string, Menu[]>();
    for (const menu of shown) {
      if (menu.location === null) continue;
      const key = `${String(menu.location.lat)},${String(menu.location.lng)}`;
      const group = groups.get(key) ?? [];
      group.push(menu);
      groups.set(key, group);
    }
    for (const group of groups.values()) {
      const menu = group[0];
      if (menu === undefined || menu.location === null) continue;
      const icon = L.divIcon({
        html: node("span", group.length > 1 ? String(group.length) : "•"),
        className: `menu-pin${group.some((m) => m.id === selected) ? " selected" : ""}`,
        iconSize: [27, 27],
        iconAnchor: [13, 13],
      });
      const marker = L.marker([menu.location.lat, menu.location.lng], {
        icon,
        keyboard: true,
        title: `${menu.name} · ${String(group.length)} menu record${group.length === 1 ? "" : "s"}`,
        alt: `${menu.name}, ${menu.date.label}. Open menu details.`,
      });
      marker.addTo(pins).on("click", () => {
        showMenu(menu);
      });
      // Leaflet's `alt` option only labels image markers, not our div icons.
      marker
        .getElement()
        ?.setAttribute(
          "aria-label",
          `${menu.name} · ${String(group.length)} menu record${group.length === 1 ? "" : "s"}`,
        );
    }
    const located = shown.filter((m) => m.location !== null).length;
    element("pin-count").textContent =
      `${String(groups.size)} point${groups.size === 1 ? "" : "s"} · ${String(located)} record${located === 1 ? "" : "s"}`;
  }

  function renderList(): void {
    const filters = currentFilters();
    shown = filterMenus(menus, filters);
    const list = element("menu-list");
    list.replaceChildren();
    element("result-count").textContent = `${String(shown.length)} / ${String(menus.length)} menus`;
    if (shown.length === 0) {
      const empty = node("div", "", "empty");
      empty.append(
        node("h3", "Nothing on this page."),
        node(
          "p",
          filters.from !== null && filters.to !== null && filters.from > filters.to
            ? "The start year comes after the end year. Adjust the date range to keep browsing."
            : "No source records match these filters. Try a wider decade, another area, or include undated menus.",
        ),
        button("Reset all filters", reset),
      );
      list.append(empty);
    }
    for (const menu of shown) {
      const row = button(
        "",
        () => {
          showMenu(menu);
        },
        "menu-row",
      );
      row.dataset["menuId"] = menu.id;
      const yearText = menu.date.year === null ? "No date" : String(menu.date.year);
      row.append(
        node("span", yearText, menu.date.year === null ? "menu-year unknown" : "menu-year"),
      );
      const copy = node("span");
      const meta = node("span", "", "menu-meta");
      meta.append(
        node("span", menu.provider, "badge"),
        document.createTextNode(menu.location?.area ?? "Location unresolved"),
      );
      copy.append(node("span", menu.name, "menu-title"), meta);
      row.append(copy, node("span", ">", "menu-arrow"));
      row.setAttribute(
        "aria-label",
        `${menu.name}, ${menu.date.label}, ${menu.provider}, ${menu.location?.area ?? "location unresolved"}`,
      );
      list.append(row);
    }
    if (selected !== null && !shown.some((m) => m.id === selected)) showList();
    renderPins();
  }
  function reset(): void {
    input("search").value = defaults.search;
    for (const id of ["decade", "area", "cuisine", "collection"]) select(id).value = "all";
    input("from").value = "";
    input("to").value = "";
    input("undated").checked = true;
    showList();
    renderList();
  }
  element("filters").addEventListener("submit", (e) => {
    e.preventDefault();
  });
  element("filters").addEventListener("input", renderList);
  element("filters").addEventListener("reset", (e) => {
    e.preventDefault();
    reset();
  });
  element("map-reset").addEventListener("click", () => {
    map.setView(home, 13, { animate: false });
  });
  element("street-tiles").addEventListener("click", () => {
    if (tiles !== null) {
      tiles.remove();
      tiles = null;
      locator.addTo(map);
      element("street-tiles").textContent = "Load street map";
      element("street-tiles").setAttribute("aria-pressed", "false");
      element("tile-status").textContent = "";
      return;
    }
    tileErrors = 0;
    tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors · ODbL</a>',
      crossOrigin: true,
    });
    tiles.on("tileerror", () => {
      tileErrors += 1;
      if (tileErrors === 1)
        element("tile-status").textContent =
          "Street tiles are unavailable. The original locator and menu pins remain usable.";
    });
    tiles.addTo(map);
    element("street-tiles").textContent = "Use original locator";
    element("street-tiles").setAttribute("aria-pressed", "true");
    element("tile-status").textContent =
      "Modern street tiles from OpenStreetMap; menu dates remain historical.";
  });
  const provenance = element("provenance");
  for (const snapshot of archive.snapshots) {
    const p = node("p");
    p.append(
      link(snapshot.id, snapshot.url),
      document.createTextNode(` · ${snapshot.retrievedAt}\n${snapshot.description}`),
      node("br"),
      node("span", snapshot.sha256, "hash"),
    );
    provenance.append(p);
  }
  renderList();
  const requested = location.hash.startsWith("#menu=")
    ? decodeURIComponent(location.hash.slice(6))
    : null;
  const initial = menus.find((m) => m.id === requested);
  if (initial !== undefined) showMenu(initial, true);
}

async function start(): Promise<void> {
  const response = await fetch("menus.json");
  if (!response.ok) throw new Error(`Menu data request failed (HTTP ${String(response.status)})`);
  const value: unknown = await response.json();
  initialize(parseArchive(value));
}
start().catch((error: unknown) => {
  element("result-count").textContent = "Archive unavailable";
  const notice = node("div", "", "empty load-error");
  notice.setAttribute("role", "alert");
  notice.append(
    node("h3", "The archive couldn't open."),
    node(
      "p",
      "The menu data is missing or could not be validated. Reload this page, or follow the source links below.",
    ),
    link("NYPL menus >", "https://menus.nypl.org/"),
    node("p", error instanceof Error ? error.message : "Unknown load error", "small"),
  );
  element("menu-list").replaceChildren(notice);
  element("pin-count").textContent = "Data unavailable";
});
