const PHILLY_CENTER = [-75.1652, 39.9526];
const DEFAULT_PADDING = { top: 140, left: 480, right: 80, bottom: 80 };
const THEME_STORAGE_KEY = "block-changelog-theme";

const initialTheme = getInitialTheme();
document.documentElement.dataset.theme = initialTheme;

const MAP_THEME_STYLES = {
  light: {
    neighborhoodFill: "#14b8a6",
    neighborhoodOutline: "#627487",
    blockFill: "#14b8a6",
    blockOutline: "#7b8ea1",
    hoverFill: "#2563eb",
    hoverLine: "#2563eb",
    selectedFill: "#0f9488",
    selectedLine: "#0b766d",
  },
  dark: {
    neighborhoodFill: "#4fd1c5",
    neighborhoodOutline: "#51606d",
    blockFill: "#4fd1c5",
    blockOutline: "#355062",
    hoverFill: "#93c5fd",
    hoverLine: "#93c5fd",
    selectedFill: "#4fd1c5",
    selectedLine: "#67e8f9",
  },
};

const state = {
  activeRange: 30,
  activeFilter: "all",
  viewMode: "neighborhood",
  theme: initialTheme,
  blocks: null,
  neighborhoods: null,
  neighborhoodIndex: null,
  blockIndex: new Map(),
  manifest: null,
  asOfDate: new Date(),
  selectedFeature: null,
  selectedEntityType: null,
  selectedEvents: null,
  tractCache: new Map(),
  neighborhoodCache: new Map(),
  addressIndex: null,
  pendingSelectionKey: null,
};

const els = {
  title: document.getElementById("panel-title"),
  subtitle: document.getElementById("panel-subtitle"),
  statusPill: document.getElementById("status-pill"),
  stat30: document.getElementById("stat-30d"),
  stat90: document.getElementById("stat-90d"),
  stat365: document.getElementById("stat-365d"),
  feed: document.getElementById("feed"),
  empty: document.getElementById("empty-state"),
  emptyTitle: document.getElementById("empty-title"),
  emptyCopy: document.getElementById("empty-copy"),
  feedSummary: document.getElementById("feed-summary"),
  searchForm: document.getElementById("search-form"),
  addressInput: document.getElementById("address-input"),
  themeToggle: document.getElementById("theme-toggle"),
  modeToggle: document.getElementById("mode-toggle"),
  rangeToggle: document.getElementById("range-toggle"),
  filters: document.getElementById("filters"),
  drillButton: document.getElementById("drill-button"),
};

const map = new maplibregl.Map({
  container: "map",
  center: PHILLY_CENTER,
  zoom: 11.6,
  pitch: 0,
  bearing: 0,
  style: {
    version: 8,
    glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    sources: {
      cartoDark: {
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
          "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
          "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      },
      cartoLight: {
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
          "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
          "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      },
    },
    layers: [
      {
        id: "carto-light-layer",
        type: "raster",
        source: "cartoLight",
        layout: { visibility: initialTheme === "light" ? "visible" : "none" },
      },
      {
        id: "carto-dark-layer",
        type: "raster",
        source: "cartoDark",
        layout: { visibility: initialTheme === "dark" ? "visible" : "none" },
      },
    ],
  },
});

const hoverPopup = new maplibregl.Popup({
  closeButton: false,
  closeOnClick: false,
  className: "hover-popup",
  maxWidth: "260px",
});

map.addControl(
  new maplibregl.NavigationControl({ visualizePitch: true }),
  "top-right",
);

map.on("load", () => {
  wireControls();
  applyTheme(state.theme, { persist: false, updateMap: true });
  void initializeApp();
});

async function initializeApp() {
  setLoadingState("Loading map data...");

  try {
    const [blocksResponse, neighborhoodsResponse, manifestResponse] =
      await Promise.all([
        fetch("./data/blocks.geojson"),
        fetch("./data/neighborhoods.geojson"),
        fetch("./data/manifest.json").catch(() => null),
      ]);

    if (!blocksResponse.ok) {
      throw new Error(
        `Failed to load blocks.geojson (${blocksResponse.status})`,
      );
    }
    if (!neighborhoodsResponse.ok) {
      throw new Error(
        `Failed to load neighborhoods.geojson (${neighborhoodsResponse.status})`,
      );
    }

    state.blocks = await blocksResponse.json();
    state.neighborhoods = await neighborhoodsResponse.json();
    state.blockIndex = buildBlockIndex(state.blocks.features || []);

    if (manifestResponse?.ok) {
      state.manifest = await manifestResponse.json();
      if (state.manifest?.as_of_date) {
        state.asOfDate = new Date(state.manifest.as_of_date);
      }
    }

    addGeometryLayers();
    wireMapInteractions();

    const initialQuery = els.addressInput.value.trim();
    const initialMatch = initialQuery
      ? await findFeatureByQueryAsync(initialQuery)
      : null;
    const initialFeature = initialMatch?.feature || findDefaultFeature();

    if (initialFeature) {
      if ((initialMatch?.type || state.viewMode) === "neighborhood") {
        await selectNeighborhood(initialFeature, { flyTo: true, duration: 0 });
      } else {
        await selectBlock(initialFeature, { flyTo: true, duration: 0 });
      }
    } else {
      renderNoSelection();
    }
  } catch (error) {
    console.error(error);
    els.title.textContent = "Could not load the map";
    els.subtitle.textContent = "Try rebuilding the data and refresh the page.";
    els.statusPill.textContent = "Unavailable";
    els.feedSummary.textContent = "Map data failed to load.";
    els.feed.innerHTML = "";
    els.emptyTitle.textContent = "Something went wrong.";
    els.emptyCopy.textContent = "The app could not load its local data files.";
    els.empty.classList.remove("hidden");
  }
}

function addGeometryLayers() {
  map.addSource("neighborhoods", {
    type: "geojson",
    data: state.neighborhoods,
    generateId: true,
  });

  map.addSource("blocks", {
    type: "geojson",
    data: state.blocks,
    generateId: true,
  });

  map.addSource("hover-shape", {
    type: "geojson",
    data: emptyFeatureCollection(),
  });

  map.addSource("selected-shape", {
    type: "geojson",
    data: emptyFeatureCollection(),
  });

  map.addLayer({
    id: "neighborhoods-fill",
    type: "fill",
    source: "neighborhoods",
    paint: {
      "fill-color": MAP_THEME_STYLES[state.theme].neighborhoodFill,
      "fill-opacity": [
        "interpolate",
        ["linear"],
        ["coalesce", ["get", "events_365d"], 0],
        0,
        0,
        1,
        0.03,
        50,
        0.08,
        200,
        0.14,
        500,
        0.2,
        1200,
        0.26,
      ],
    },
  });

  map.addLayer({
    id: "neighborhoods-outline",
    type: "line",
    source: "neighborhoods",
    paint: {
      "line-color": MAP_THEME_STYLES[state.theme].neighborhoodOutline,
      "line-width": ["interpolate", ["linear"], ["zoom"], 9, 0.6, 14, 1.2],
      "line-opacity": 0.72,
    },
  });

  map.addLayer({
    id: "neighborhoods-hit",
    type: "fill",
    source: "neighborhoods",
    paint: {
      "fill-color": "#ffffff",
      "fill-opacity": 0,
    },
  });

  map.addLayer({
    id: "blocks-activity-fill",
    type: "fill",
    source: "blocks",
    layout: { visibility: "none" },
    filter: [">", ["coalesce", ["get", "events_365d"], 0], 0],
    paint: {
      "fill-color": MAP_THEME_STYLES[state.theme].blockFill,
      "fill-opacity": [
        "interpolate",
        ["linear"],
        ["coalesce", ["get", "events_365d"], 0],
        0,
        0,
        1,
        0.02,
        10,
        0.045,
        50,
        0.08,
        150,
        0.14,
        400,
        0.22,
      ],
    },
  });

  map.addLayer({
    id: "blocks-outline",
    type: "line",
    source: "blocks",
    layout: { visibility: "none" },
    paint: {
      "line-color": MAP_THEME_STYLES[state.theme].blockOutline,
      "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.25, 15, 0.9],
      "line-opacity": ["interpolate", ["linear"], ["zoom"], 10, 0.12, 15, 0.28],
    },
  });

  map.addLayer({
    id: "blocks-hit",
    type: "fill",
    source: "blocks",
    layout: { visibility: "none" },
    paint: {
      "fill-color": "#ffffff",
      "fill-opacity": 0,
    },
  });

  map.addLayer({
    id: "hover-shape-fill",
    type: "fill",
    source: "hover-shape",
    paint: {
      "fill-color": MAP_THEME_STYLES[state.theme].hoverFill,
      "fill-opacity": 0.12,
    },
  });

  map.addLayer({
    id: "hover-shape-line",
    type: "line",
    source: "hover-shape",
    paint: {
      "line-color": MAP_THEME_STYLES[state.theme].hoverLine,
      "line-width": 2,
      "line-opacity": 0.9,
    },
  });

  map.addLayer({
    id: "selected-shape-fill",
    type: "fill",
    source: "selected-shape",
    paint: {
      "fill-color": MAP_THEME_STYLES[state.theme].selectedFill,
      "fill-opacity": 0.18,
    },
  });

  map.addLayer({
    id: "selected-shape-line",
    type: "line",
    source: "selected-shape",
    paint: {
      "line-color": MAP_THEME_STYLES[state.theme].selectedLine,
      "line-width": 2.5,
      "line-opacity": 0.98,
    },
  });

  updateViewMode();
}

function wireMapInteractions() {
  for (const layerId of ["neighborhoods-hit", "blocks-hit"]) {
    map.on("mousemove", layerId, (event) => {
      const feature = event.features?.[0];
      if (!feature) return;

      map.getCanvas().style.cursor = "pointer";

      if (isCurrentlySelected(feature)) {
        updateGeoJsonSource("hover-shape", null);
      } else {
        updateGeoJsonSource("hover-shape", toPlainFeature(feature));
      }

      hoverPopup
        .setLngLat(event.lngLat)
        .setHTML(`<div>${escapeHtml(getFeatureLabel(feature))}</div>`)
        .addTo(map);
    });

    map.on("mouseleave", layerId, () => {
      map.getCanvas().style.cursor = "";
      updateGeoJsonSource("hover-shape", null);
      hoverPopup.remove();
    });

    map.on("click", layerId, (event) => {
      const feature = event.features?.[0];
      if (!feature) return;
      hoverPopup.remove();
      if (layerId === "neighborhoods-hit") {
        void selectNeighborhood(toPlainFeature(feature), { flyTo: true });
      } else {
        void selectBlock(toPlainFeature(feature), { flyTo: true });
      }
    });
  }
}

function wireControls() {
  els.themeToggle.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-theme]");
    if (!button) return;
    applyTheme(button.dataset.theme, { persist: true, updateMap: true });
  });

  els.modeToggle.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;

    state.viewMode = button.dataset.mode;
    syncToggleState(els.modeToggle, "mode", state.viewMode);
    updateViewMode();
  });

  els.rangeToggle.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-range]");
    if (!button) return;

    state.activeRange = Number(button.dataset.range);
    syncToggleState(els.rangeToggle, "range", String(state.activeRange));
    renderPanel();
  });

  els.filters.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;

    state.activeFilter = button.dataset.filter;
    syncToggleState(els.filters, "filter", state.activeFilter);
    renderPanel();
  });

  els.drillButton.addEventListener("click", async () => {
    if (state.selectedEntityType === "neighborhood") {
      const block = findBlockForSelectedNeighborhood();
      if (!block) return;
      state.viewMode = "block";
      syncToggleState(els.modeToggle, "mode", state.viewMode);
      updateViewMode();
      await selectBlock(block, { flyTo: true });
      return;
    }

    if (state.selectedEntityType === "block") {
      const neighborhood = findNeighborhoodForSelectedBlock();
      if (!neighborhood) return;
      state.viewMode = "neighborhood";
      syncToggleState(els.modeToggle, "mode", state.viewMode);
      updateViewMode();
      await selectNeighborhood(neighborhood, { flyTo: true });
    }
  });

  els.searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = els.addressInput.value.trim();
    if (!query || !state.blocks) return;

    const result = await findFeatureByQueryAsync(query);
    if (!result) {
      els.feed.innerHTML = "";
      els.empty.classList.remove("hidden");
      els.title.textContent = "Nothing matched";
      els.subtitle.textContent = `Try a neighborhood name or a recent address. You searched for: ${query}`;
      els.statusPill.textContent = "Search";
      els.feedSummary.textContent = "Search by neighborhood name or address.";
      els.stat30.textContent = "—";
      els.stat90.textContent = "—";
      els.stat365.textContent = "—";
      els.emptyTitle.textContent = "No match found.";
      els.emptyCopy.textContent =
        "Try a simpler neighborhood name or a more exact address.";
      hideDrillButton();
      return;
    }

    if (result.type === "neighborhood") {
      state.viewMode = "neighborhood";
      syncToggleState(els.modeToggle, "mode", state.viewMode);
      updateViewMode();
      await selectNeighborhood(result.feature, { flyTo: true });
    } else {
      state.viewMode = "block";
      syncToggleState(els.modeToggle, "mode", state.viewMode);
      updateViewMode();
      await selectBlock(result.feature, { flyTo: true });
    }
  });

  syncToggleState(els.themeToggle, "theme", state.theme);
  syncToggleState(els.modeToggle, "mode", state.viewMode);
  syncToggleState(els.rangeToggle, "range", String(state.activeRange));
  syncToggleState(els.filters, "filter", state.activeFilter);
}

async function selectBlock(feature, { flyTo = true, duration = 900 } = {}) {
  const plainFeature = toPlainFeature(feature);
  const blockId = plainFeature.properties?.block_id;
  if (!blockId) return;

  state.selectedFeature = plainFeature;
  state.selectedEntityType = "block";
  state.pendingSelectionKey = `block:${blockId}`;
  state.selectedEvents = null;

  updateGeoJsonSource("selected-shape", plainFeature);
  updateGeoJsonSource("hover-shape", null);
  renderPanel();

  if (flyTo) {
    fitToFeature(plainFeature, duration);
  }

  const tractId = plainFeature.properties?.tract_id;
  const tractEvents = tractId ? await loadTractEvents(tractId) : [];

  if (state.pendingSelectionKey !== `block:${blockId}`) {
    return;
  }

  state.selectedEvents = tractEvents
    .filter((event) => event.block_id === blockId)
    .sort((a, b) => String(b.event_date).localeCompare(String(a.event_date)));

  renderPanel();
}

async function selectNeighborhood(
  feature,
  { flyTo = true, duration = 900 } = {},
) {
  const plainFeature = toPlainFeature(feature);
  const neighborhoodId = plainFeature.properties?.neighborhood_id;
  if (!neighborhoodId) return;

  state.selectedFeature = plainFeature;
  state.selectedEntityType = "neighborhood";
  state.pendingSelectionKey = `neighborhood:${neighborhoodId}`;
  state.selectedEvents = null;

  updateGeoJsonSource("selected-shape", plainFeature);
  updateGeoJsonSource("hover-shape", null);
  renderPanel();

  if (flyTo) {
    fitToFeature(plainFeature, duration);
  }

  const neighborhoodEvents = await loadNeighborhoodEvents(neighborhoodId);
  if (state.pendingSelectionKey !== `neighborhood:${neighborhoodId}`) {
    return;
  }

  state.selectedEvents = neighborhoodEvents;
  renderPanel();
}

async function loadTractEvents(tractId) {
  if (state.tractCache.has(tractId)) {
    return state.tractCache.get(tractId);
  }

  const response = await fetch(`./data/tracts/${tractId}.json`);
  if (!response.ok) {
    state.tractCache.set(tractId, []);
    return [];
  }

  const rows = await response.json();
  const events = rows.map((row) => ({
    block_id: row.b,
    event_type: row.t,
    event_date: row.d,
    title: row.ti,
    description: row.de,
    source_record_id: row.r,
  }));
  state.tractCache.set(tractId, events);
  return events;
}

async function loadNeighborhoodEvents(neighborhoodId) {
  if (state.neighborhoodCache.has(neighborhoodId)) {
    return state.neighborhoodCache.get(neighborhoodId);
  }

  const neighborhoodIndex = await loadNeighborhoodIndex();
  const entry = neighborhoodIndex?.[neighborhoodId];
  if (!entry) {
    state.neighborhoodCache.set(neighborhoodId, []);
    return [];
  }

  const tractIds = Array.isArray(entry.tract_ids) ? entry.tract_ids : [];
  const blockIds = new Set(
    Array.isArray(entry.block_ids) ? entry.block_ids : [],
  );
  const tractEvents = await Promise.all(
    tractIds.map((tractId) => loadTractEvents(tractId)),
  );
  const events = tractEvents
    .flat()
    .filter((event) => blockIds.has(event.block_id))
    .sort((a, b) => String(b.event_date).localeCompare(String(a.event_date)));

  state.neighborhoodCache.set(neighborhoodId, events);
  return events;
}

function renderPanel() {
  const feature = state.selectedFeature;
  if (!feature) {
    renderNoSelection();
    return;
  }

  const props = feature.properties || {};
  const isNeighborhood = state.selectedEntityType === "neighborhood";
  const label = isNeighborhood
    ? toTitleCase(props.neighborhood_name || "Neighborhood")
    : formatBlockLabel(props.display_name, props.block_id);
  const neighborhoodName = toTitleCase(props.neighborhood_name || "");

  els.title.textContent = label;
  els.subtitle.textContent = isNeighborhood
    ? `${Number(props.block_count || 0).toLocaleString()} blocks in this area`
    : neighborhoodName
      ? `Block view in ${neighborhoodName}`
      : "Block view";
  els.statusPill.textContent = isNeighborhood ? "Neighborhood" : "Block";
  els.stat30.textContent = Number(props.events_30d || 0).toLocaleString();
  els.stat90.textContent = Number(props.events_90d || 0).toLocaleString();
  els.stat365.textContent = Number(props.events_365d || 0).toLocaleString();

  updateDrillButton();

  if (state.selectedEvents === null) {
    setLoadingState(`Loading recent activity for ${label}...`);
    return;
  }

  const filtered = getFilteredEvents(state.selectedEvents);
  els.feedSummary.textContent = buildFeedSummary(
    filtered.length,
    state.activeRange,
    state.activeFilter,
  );
  els.feed.innerHTML = "";

  if (!filtered.length) {
    els.emptyTitle.textContent = isNeighborhood
      ? "Quiet neighborhood."
      : "Quiet block.";
    els.emptyCopy.textContent = `No ${filterLabelText(state.activeFilter)} in the last ${rangeLabelShort(state.activeRange)}.`;
    els.empty.classList.remove("hidden");
    return;
  }

  els.empty.classList.add("hidden");

  for (const event of filtered) {
    const row = document.createElement("article");
    row.className = "feed-item";

    const locationLabel = isNeighborhood ? blockLabelForEvent(event) : "";
    const locationHtml = locationLabel
      ? `<div class="feed-location">${escapeHtml(locationLabel)}</div>`
      : "";

    row.innerHTML = `
      <div class="feed-date">${formatDate(event.event_date)}</div>
      <div>
        <div class="feed-meta">
          <span class="badge ${event.event_type}">${escapeHtml(eventTypeLabel(event.event_type))}</span>
        </div>
        <div class="feed-title">${escapeHtml(event.title || defaultEventTitle(event.event_type))}</div>
        <div class="feed-desc">${escapeHtml(event.description || "Reported city event")}</div>
        ${locationHtml}
      </div>
    `;
    els.feed.appendChild(row);
  }
}

function renderNoSelection() {
  const isNeighborhood = state.viewMode === "neighborhood";
  els.title.textContent = isNeighborhood
    ? "Start with a neighborhood"
    : "Pick a block";
  els.subtitle.textContent = isNeighborhood
    ? "Click a neighborhood on the map or search by name."
    : "Click a block on the map or search by address.";
  els.statusPill.textContent = isNeighborhood
    ? "Neighborhood view"
    : "Block view";
  els.stat30.textContent = "—";
  els.stat90.textContent = "—";
  els.stat365.textContent = "—";
  els.feedSummary.textContent = isNeighborhood
    ? "Neighborhood view is the easiest way to browse the city."
    : "Block view helps you zoom in on one specific area.";
  els.feed.innerHTML = "";
  els.emptyTitle.textContent = "Select a place to begin.";
  els.emptyCopy.textContent = isNeighborhood
    ? "Neighborhoods give you a broader picture before you zoom in."
    : "Use block view when you want a closer look at one area.";
  els.empty.classList.remove("hidden");
  hideDrillButton();
}

function setLoadingState(message) {
  els.feedSummary.textContent = message;
  els.feed.innerHTML = "";
  els.emptyTitle.textContent = "Loading…";
  els.emptyCopy.textContent = "Please wait a moment.";
  els.empty.classList.remove("hidden");
}

function getFilteredEvents(events) {
  const cutoff = new Date(state.asOfDate);
  cutoff.setUTCDate(cutoff.getUTCDate() - state.activeRange);

  return events
    .filter((event) => {
      const eventDate = new Date(event.event_date);
      return !Number.isNaN(eventDate.valueOf()) && eventDate >= cutoff;
    })
    .filter(
      (event) =>
        state.activeFilter === "all" || event.event_type === state.activeFilter,
    )
    .sort((a, b) => String(b.event_date).localeCompare(String(a.event_date)));
}

function findDefaultFeature(type = state.viewMode) {
  const collection =
    type === "neighborhood" ? state.neighborhoods : state.blocks;
  if (!collection?.features?.length) return null;

  const withEvents = collection.features.filter(
    (feature) => Number(feature.properties?.events_365d || 0) > 0,
  );

  return (
    rankFeatures(withEvents.length ? withEvents : collection.features)[0] ||
    null
  );
}

function findNeighborhoodFeature(query) {
  if (!query || !state.neighborhoods?.features?.length) return null;
  const normalized = normalizeQuery(query);

  const exactMatches = state.neighborhoods.features.filter(
    (feature) =>
      normalizeQuery(feature.properties?.neighborhood_name || "") ===
      normalized,
  );
  if (exactMatches.length) {
    return rankFeatures(exactMatches)[0];
  }

  const partialMatches = state.neighborhoods.features.filter((feature) =>
    normalizeQuery(feature.properties?.neighborhood_name || "").includes(
      normalized,
    ),
  );
  if (partialMatches.length) {
    return rankFeatures(partialMatches)[0];
  }

  return null;
}

function findBlockFeature(query) {
  if (!query || !state.blocks?.features?.length) return null;

  const normalized = normalizeQuery(query);
  const compact = normalized.replaceAll(" ", "");
  const features = state.blocks.features;

  if (/^\d{15}$/.test(compact)) {
    return (
      features.find((feature) => feature.properties?.block_id === compact) ||
      null
    );
  }

  const exactMatches = features.filter(
    (feature) =>
      normalizeQuery(feature.properties?.display_name || "") === normalized,
  );
  if (exactMatches.length) {
    return rankFeatures(exactMatches)[0];
  }

  const partialMatches = features.filter((feature) => {
    const haystack = normalizeQuery(
      `${feature.properties?.display_name || ""} ${feature.properties?.block_id || ""}`,
    );
    return haystack.includes(normalized);
  });

  if (partialMatches.length) {
    return rankFeatures(partialMatches)[0];
  }

  return null;
}

async function findFeatureByQueryAsync(query) {
  const neighborhoodFeature = findNeighborhoodFeature(query);
  const blockFeature = findBlockFeature(query);

  if (state.viewMode === "neighborhood" && neighborhoodFeature) {
    return { type: "neighborhood", feature: neighborhoodFeature };
  }
  if (state.viewMode === "block" && blockFeature) {
    return { type: "block", feature: blockFeature };
  }
  if (neighborhoodFeature) {
    return { type: "neighborhood", feature: neighborhoodFeature };
  }
  if (blockFeature) {
    return { type: "block", feature: blockFeature };
  }

  const addressIndex = await loadAddressIndex();
  const blockId = addressIndex?.[normalizeQuery(query)];
  if (!blockId) {
    return null;
  }

  const feature =
    state.blocks?.features?.find(
      (item) => item.properties?.block_id === blockId,
    ) || null;
  return feature ? { type: "block", feature } : null;
}

async function loadAddressIndex() {
  if (state.addressIndex) {
    return state.addressIndex;
  }

  const addressIndexPath =
    state.manifest?.address_index || "address_index.json";
  const response = await fetch(`./data/${addressIndexPath}`);
  if (!response.ok) {
    state.addressIndex = {};
    return state.addressIndex;
  }

  state.addressIndex = await response.json();
  return state.addressIndex;
}

async function loadNeighborhoodIndex() {
  if (state.neighborhoodIndex) {
    return state.neighborhoodIndex;
  }

  const neighborhoodIndexPath =
    state.manifest?.neighborhood_index || "neighborhood_index.json";
  const response = await fetch(`./data/${neighborhoodIndexPath}`);
  if (!response.ok) {
    state.neighborhoodIndex = {};
    return state.neighborhoodIndex;
  }

  state.neighborhoodIndex = await response.json();
  return state.neighborhoodIndex;
}

function rankFeatures(features) {
  return features.slice().sort((a, b) => {
    const eventDiff =
      Number(b.properties?.events_365d || 0) -
      Number(a.properties?.events_365d || 0);
    if (eventDiff !== 0) return eventDiff;

    const labelA = getFeatureLabel(a);
    const labelB = getFeatureLabel(b);
    return labelA.localeCompare(labelB);
  });
}

function normalizeQuery(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .replace(/\s+/g, " ");
}

function updateViewMode() {
  const isNeighborhood = state.viewMode === "neighborhood";
  const visibility = isNeighborhood ? "visible" : "none";
  const inverseVisibility = isNeighborhood ? "none" : "visible";

  for (const layerId of [
    "neighborhoods-fill",
    "neighborhoods-outline",
    "neighborhoods-hit",
  ]) {
    map.setLayoutProperty(layerId, "visibility", visibility);
  }
  for (const layerId of [
    "blocks-activity-fill",
    "blocks-outline",
    "blocks-hit",
  ]) {
    map.setLayoutProperty(layerId, "visibility", inverseVisibility);
  }

  updateGeoJsonSource("hover-shape", null);
  hoverPopup.remove();

  if (!state.selectedFeature || !state.selectedEntityType) {
    renderNoSelection();
    return;
  }

  const selectedMatchesView =
    (isNeighborhood && state.selectedEntityType === "neighborhood") ||
    (!isNeighborhood && state.selectedEntityType === "block");

  if (selectedMatchesView) {
    updateGeoJsonSource("selected-shape", state.selectedFeature);
    renderPanel();
    return;
  }

  const relatedFeature =
    state.viewMode === "neighborhood"
      ? findNeighborhoodForSelectedBlock()
      : findBlockForSelectedNeighborhood();
  const fallback = relatedFeature || findDefaultFeature(state.viewMode);
  if (!fallback) {
    updateGeoJsonSource("selected-shape", null);
    renderNoSelection();
    return;
  }

  if (state.viewMode === "neighborhood") {
    void selectNeighborhood(fallback, { flyTo: true });
  } else {
    void selectBlock(fallback, { flyTo: true });
  }
}

function findNeighborhoodForSelectedBlock() {
  if (state.selectedEntityType !== "block") return null;
  const neighborhoodId = state.selectedFeature?.properties?.neighborhood_id;
  if (!neighborhoodId) return null;
  return (
    state.neighborhoods?.features?.find(
      (feature) => feature.properties?.neighborhood_id === neighborhoodId,
    ) || null
  );
}

function findBlockForSelectedNeighborhood() {
  if (state.selectedEntityType !== "neighborhood") return null;
  const neighborhoodId = state.selectedFeature?.properties?.neighborhood_id;
  if (!neighborhoodId) return null;

  const matches = (state.blocks?.features || []).filter(
    (feature) => feature.properties?.neighborhood_id === neighborhoodId,
  );
  return matches.length ? rankFeatures(matches)[0] : null;
}

function isCurrentlySelected(feature) {
  if (!state.selectedFeature || !state.selectedEntityType) {
    return false;
  }

  if (state.selectedEntityType === "block") {
    return (
      feature.properties?.block_id ===
      state.selectedFeature.properties?.block_id
    );
  }

  return (
    feature.properties?.neighborhood_id ===
    state.selectedFeature.properties?.neighborhood_id
  );
}

function updateGeoJsonSource(sourceId, feature) {
  const source = map.getSource(sourceId);
  if (!source) return;
  source.setData(
    feature ? featureCollection([feature]) : emptyFeatureCollection(),
  );
}

function fitToFeature(feature, duration = 900) {
  const bounds = getFeatureBounds(feature);
  if (!bounds) return;
  map.fitBounds(bounds, {
    padding:
      window.innerWidth < 640
        ? { top: 100, left: 24, right: 24, bottom: 320 }
        : DEFAULT_PADDING,
    duration,
  });
}

function getFeatureBounds(feature) {
  if (!feature?.geometry) return null;
  const coordinates = flattenCoordinates(feature.geometry.coordinates);
  if (!coordinates.length) return null;

  const bounds = new maplibregl.LngLatBounds(coordinates[0], coordinates[0]);
  for (const coordinate of coordinates) {
    bounds.extend(coordinate);
  }
  return bounds;
}

function flattenCoordinates(coordinates, output = []) {
  if (!Array.isArray(coordinates)) return output;

  if (
    coordinates.length >= 2 &&
    typeof coordinates[0] === "number" &&
    typeof coordinates[1] === "number"
  ) {
    output.push(coordinates);
    return output;
  }

  for (const child of coordinates) {
    flattenCoordinates(child, output);
  }
  return output;
}

function toPlainFeature(feature) {
  return {
    type: "Feature",
    properties: { ...(feature.properties || {}) },
    geometry: feature.geometry,
  };
}

function featureCollection(features) {
  return {
    type: "FeatureCollection",
    features,
  };
}

function emptyFeatureCollection() {
  return featureCollection([]);
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "—";

  const currentYear = state.asOfDate.getUTCFullYear();
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    ...(date.getUTCFullYear() !== currentYear ? { year: "numeric" } : {}),
  });
}

function formatBlockLabel(displayName, blockId) {
  const cleaned = String(displayName || "").trim();
  if (cleaned) {
    return toTitleCase(cleaned);
  }
  return blockId ? `Block ${blockId}` : "Selected block";
}

function buildBlockIndex(features) {
  const index = new Map();
  for (const feature of features) {
    const props = feature.properties || {};
    if (!props.block_id) continue;
    index.set(String(props.block_id), {
      displayName: formatBlockLabel(props.display_name, props.block_id),
      neighborhoodName: toTitleCase(props.neighborhood_name || ""),
    });
  }
  return index;
}

function blockLabelForEvent(event) {
  const blockInfo = state.blockIndex.get(String(event.block_id || ""));
  if (!blockInfo) return "";
  return blockInfo.displayName;
}

function getFeatureLabel(feature) {
  const props = feature?.properties || {};
  if (props.neighborhood_name) {
    return toTitleCase(props.neighborhood_name);
  }
  return formatBlockLabel(props.display_name, props.block_id);
}

function toTitleCase(value) {
  return String(value)
    .toLowerCase()
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function eventTypeLabel(eventType) {
  if (eventType === "service_request") return "311 request";
  if (eventType === "crime_incident") return "Crime";
  return "Event";
}

function defaultEventTitle(eventType) {
  if (eventType === "service_request") return "311 request";
  if (eventType === "crime_incident") return "Crime incident";
  return "City event";
}

function filterLabelText(filter) {
  if (filter === "all") return "reported events";
  if (filter === "service_request") return "311 requests";
  if (filter === "crime_incident") return "crime incidents";
  return "events";
}

function rangeLabelShort(range) {
  if (range === 30) return "30 days";
  if (range === 90) return "90 days";
  if (range === 365) return "1 year";
  return `${range} days`;
}

function buildFeedSummary(count, range, filter) {
  const countLabel = `${count.toLocaleString()} ${count === 1 ? "event" : "events"}`;
  if (filter === "all") {
    return `Showing ${countLabel} from the last ${rangeLabelShort(range)}.`;
  }
  return `Showing ${countLabel} for ${filterLabelText(filter)} from the last ${rangeLabelShort(range)}.`;
}

function updateDrillButton() {
  if (state.selectedEntityType === "neighborhood") {
    const target = findBlockForSelectedNeighborhood();
    if (!target) {
      hideDrillButton();
      return;
    }
    els.drillButton.textContent = "Switch to block view for this area";
    els.drillButton.classList.remove("hidden");
    return;
  }

  if (state.selectedEntityType === "block") {
    const target = findNeighborhoodForSelectedBlock();
    if (!target) {
      hideDrillButton();
      return;
    }
    els.drillButton.textContent = "Back to the neighborhood view";
    els.drillButton.classList.remove("hidden");
    return;
  }

  hideDrillButton();
}

function hideDrillButton() {
  els.drillButton.classList.add("hidden");
  els.drillButton.textContent = "";
}

function syncToggleState(container, key, activeValue) {
  for (const chip of container.querySelectorAll(`button[data-${key}]`)) {
    chip.classList.toggle("active", chip.dataset[key] === activeValue);
  }
}

function applyTheme(theme, { persist = true, updateMap = true } = {}) {
  const nextTheme = theme === "light" ? "light" : "dark";
  state.theme = nextTheme;
  document.documentElement.dataset.theme = nextTheme;
  syncToggleState(els.themeToggle, "theme", nextTheme);

  if (persist) {
    localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
  }

  if (!updateMap || !map.isStyleLoaded()) {
    return;
  }

  map.setLayoutProperty(
    "carto-light-layer",
    "visibility",
    nextTheme === "light" ? "visible" : "none",
  );
  map.setLayoutProperty(
    "carto-dark-layer",
    "visibility",
    nextTheme === "dark" ? "visible" : "none",
  );

  const mapTheme = MAP_THEME_STYLES[nextTheme];
  if (map.getLayer("neighborhoods-fill")) {
    map.setPaintProperty(
      "neighborhoods-fill",
      "fill-color",
      mapTheme.neighborhoodFill,
    );
    map.setPaintProperty(
      "neighborhoods-outline",
      "line-color",
      mapTheme.neighborhoodOutline,
    );
    map.setPaintProperty(
      "blocks-activity-fill",
      "fill-color",
      mapTheme.blockFill,
    );
    map.setPaintProperty("blocks-outline", "line-color", mapTheme.blockOutline);
    map.setPaintProperty("hover-shape-fill", "fill-color", mapTheme.hoverFill);
    map.setPaintProperty("hover-shape-line", "line-color", mapTheme.hoverLine);
    map.setPaintProperty(
      "selected-shape-fill",
      "fill-color",
      mapTheme.selectedFill,
    );
    map.setPaintProperty(
      "selected-shape-line",
      "line-color",
      mapTheme.selectedLine,
    );
  }
}

function getInitialTheme() {
  const saved = localStorage.getItem(THEME_STORAGE_KEY);
  if (saved === "light" || saved === "dark") {
    return saved;
  }

  return window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
