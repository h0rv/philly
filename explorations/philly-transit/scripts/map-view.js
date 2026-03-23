import { MAP_BOOT_TIMEOUT_MS, MAP_STYLES } from "./config.js";
import { state, elements, markUiDirty } from "./state.js";
import {
  routeDisplayColor,
  selectionHighlightColor,
  withAlpha,
} from "./utils.js";

export function addBuildings() {
  if (!state.map?.getStyle()?.layers) return;
  const labelLayer = state.map
    .getStyle()
    .layers.find((layer) => layer.type === "symbol")?.id;
  if (state.map.getLayer("philly-3d-buildings")) return;

  state.map.addLayer(
    {
      id: "philly-3d-buildings",
      type: "fill-extrusion",
      source: "carto",
      "source-layer": "building",
      minzoom: 13,
      paint: {
        "fill-extrusion-color": state.theme === "light" ? "#d8e1f0" : "#101828",
        "fill-extrusion-height": [
          "coalesce",
          ["get", "render_height"],
          ["get", "height"],
          12,
        ],
        "fill-extrusion-base": [
          "coalesce",
          ["get", "render_min_height"],
          ["get", "min_height"],
          0,
        ],
        "fill-extrusion-opacity": state.theme === "light" ? 0.62 : 0.72,
      },
    },
    labelLayer,
  );
}

export function createMap(showClientError, onMapReady) {
  const bootTimeout = window.setTimeout(() => {
    if (!state.overlay) {
      showClientError(
        "Map renderer timed out while booting. This usually means WebGL is unavailable or the base style could not load.",
      );
    }
  }, MAP_BOOT_TIMEOUT_MS);

  state.map = new maplibregl.Map({
    container: "map",
    style: MAP_STYLES[state.theme],
    center: [state.data.camera.longitude, state.data.camera.latitude],
    zoom: state.data.camera.zoom,
    pitch: state.data.camera.pitch,
    bearing: state.data.camera.bearing,
    attributionControl: false,
  });

  state.map.addControl(
    new maplibregl.NavigationControl({ visualizePitch: true }),
    "top-right",
  );
  state.map.addControl(new maplibregl.AttributionControl({ compact: true }));

  state.map.on("error", (event) => {
    if (event?.error?.message) showClientError(event.error.message);
  });

  state.map.on("styledata", () => {
    if (state.map?.isStyleLoaded()) {
      try {
        addBuildings();
      } catch {
        // ignore transient style rebuild issues
      }
    }
  });

  state.map.on("load", () => {
    window.clearTimeout(bootTimeout);
    try {
      addBuildings();
      if (!deck.MapboxOverlay) {
        throw new Error(
          "deck.MapboxOverlay is unavailable in the loaded deck.gl bundle.",
        );
      }
      state.overlay = new deck.MapboxOverlay({ interleaved: true, layers: [] });
      state.map.addControl(state.overlay);
      state.map.on("click", () => {
        if (state.pendingSelectionClear) {
          state.pendingSelectionClear = false;
          return;
        }
        state.selectedVehicleId = null;
        markUiDirty();
      });
      onMapReady();
    } catch (error) {
      showClientError(error?.message || String(error));
    } finally {
      elements.loading.style.display = "none";
    }
  });
}

export function buildLayers(visibleVehicles, selectedVehicle) {
  const routeLines = [];
  const railRoutes = [];
  const metroGhostRoutes = [];
  const selectedShapeLines = [];
  const trails = [];
  const railTrails = [];
  const polygons = [];
  const railPolygons = [];
  const estimatedMarkers = [];
  const selectedHalo = [];

  const selectedShapeId = selectedVehicle?.shapeId;

  for (const shape of Object.values(state.shapeMeta)) {
    if (!state.activeModes.has(shape.mode)) continue;
    const route = state.data.routes[shape.routeId];
    if (!route) continue;

    const row = {
      path: shape.coords,
      color: routeDisplayColor({ color: route.color, mode: shape.mode }),
      width:
        shape.mode === "regional_rail" ? 5.5 : shape.mode === "metro" ? 4.5 : 3,
    };

    const coverage = state.data.coverage?.metro?.[shape.routeId];
    const hasUnlocatedMetroTrips =
      shape.mode === "metro" &&
      coverage &&
      coverage.activeTrips > coverage.locatedTrips;

    if (shape.shapeId === selectedShapeId) {
      selectedShapeLines.push(row);
    } else if (hasUnlocatedMetroTrips) {
      metroGhostRoutes.push(row);
    } else if (shape.mode === "regional_rail") {
      railRoutes.push(row);
    } else {
      routeLines.push(row);
    }
  }

  for (const vehicle of visibleVehicles) {
    const displayColor = routeDisplayColor(vehicle);
    const estimateColor =
      vehicle.positionState === "stale"
        ? [148, 163, 184]
        : state.theme === "light"
          ? [202, 138, 4]
          : [251, 191, 36];
    const bodyAlpha =
      vehicle.positionState === "stale"
        ? state.theme === "light"
          ? 70
          : 110
        : vehicle.positionState === "extrapolated"
          ? state.theme === "light"
            ? 145
            : 185
          : vehicle.positionState === "stop_inferred"
            ? state.theme === "light"
              ? 130
              : 170
            : state.theme === "light"
              ? 205
              : 230;
    const lineAlpha =
      vehicle.positionState === "stale"
        ? 70
        : vehicle.positionState === "extrapolated"
          ? state.theme === "light"
            ? 120
            : 180
          : state.theme === "light"
            ? 110
            : 150;
    const trailColor = vehicle.selected
      ? selectionHighlightColor()
      : vehicle.positionState === "live"
        ? withAlpha(displayColor, vehicle.mode === "regional_rail" ? 205 : 190)
        : withAlpha(
            estimateColor,
            vehicle.positionState === "stale" ? 120 : 220,
          );
    const polygonRow = {
      polygon: vehicle.polygon,
      color: displayColor,
      fillAlpha: bodyAlpha,
      lineColor: vehicle.selected
        ? [255, 255, 255, 255]
        : withAlpha(displayColor, lineAlpha),
      elevation: vehicle.displayHeightM,
      vehicle,
    };

    if (vehicle.positionState !== "live" && state.theme !== "light") {
      estimatedMarkers.push({
        position: vehicle.coord,
        color: withAlpha(
          estimateColor,
          vehicle.positionState === "stale" ? 90 : 150,
        ),
        radius:
          vehicle.mode === "regional_rail"
            ? 120
            : vehicle.mode === "metro"
              ? 90
              : 55,
      });
    }

    if (vehicle.mode === "regional_rail") {
      railPolygons.push(polygonRow);
      railTrails.push({
        path: vehicle.trail,
        color: trailColor,
        width: vehicle.selected ? 7 : 5,
      });
    } else {
      trails.push({
        path: vehicle.trail,
        color: trailColor,
        width: vehicle.selected ? 5 : 3.5,
      });
      polygons.push(polygonRow);
    }

    if (vehicle.selected) {
      selectedHalo.push({
        position: vehicle.coord,
        color: selectionHighlightColor(),
        radius: vehicle.mode === "regional_rail" ? 170 : 90,
      });
    }
  }

  return [
    new deck.PathLayer({
      id: "routes",
      data: routeLines,
      getPath: (d) => d.path,
      getColor: () =>
        state.theme === "light" ? [148, 163, 184, 85] : [123, 139, 163, 92],
      getWidth: (d) => d.width,
      widthUnits: "pixels",
      rounded: true,
      parameters: { depthTest: false },
    }),
    new deck.PathLayer({
      id: "metro-ghost-routes",
      data: metroGhostRoutes,
      getPath: (d) => d.path,
      getColor: () =>
        state.theme === "light" ? [180, 83, 9, 72] : [245, 158, 11, 95],
      getWidth: (d) => d.width + (state.theme === "light" ? 0.5 : 1),
      widthUnits: "pixels",
      rounded: true,
      parameters: { depthTest: false },
    }),
    new deck.PathLayer({
      id: "rail-routes",
      data: railRoutes,
      getPath: (d) => d.path,
      getColor: () =>
        state.theme === "light" ? [100, 116, 139, 110] : [148, 163, 184, 150],
      getWidth: (d) => d.width + 2.5,
      widthUnits: "pixels",
      rounded: true,
      parameters: { depthTest: false },
    }),
    new deck.PathLayer({
      id: "selected-shape",
      data: selectedShapeLines,
      getPath: (d) => d.path,
      getColor: () => selectionHighlightColor(),
      getWidth: (d) => d.width + 2.5,
      widthUnits: "pixels",
      rounded: true,
    }),
    new deck.ScatterplotLayer({
      id: "estimated-markers",
      data: estimatedMarkers,
      getPosition: (d) => d.position,
      getRadius: (d) => d.radius,
      radiusUnits: "meters",
      stroked: true,
      filled: false,
      getLineColor: (d) => d.color,
      lineWidthUnits: "pixels",
      getLineWidth: 2,
      parameters: { depthTest: false },
    }),
    new deck.PathLayer({
      id: "trails",
      data: trails,
      getPath: (d) => d.path,
      getColor: (d) => d.color,
      getWidth: (d) => d.width,
      widthUnits: "pixels",
      rounded: true,
    }),
    new deck.PathLayer({
      id: "rail-trails",
      data: railTrails,
      getPath: (d) => d.path,
      getColor: (d) => d.color,
      getWidth: (d) => d.width,
      widthUnits: "pixels",
      rounded: true,
    }),
    new deck.ScatterplotLayer({
      id: "selected-halo",
      data: selectedHalo,
      getPosition: (d) => d.position,
      getRadius: (d) => d.radius,
      radiusUnits: "meters",
      stroked: true,
      filled: false,
      getLineColor: (d) => d.color,
      lineWidthUnits: "pixels",
      getLineWidth: 3,
      parameters: { depthTest: false },
    }),
    new deck.PolygonLayer({
      id: "vehicles",
      data: polygons,
      extruded: true,
      pickable: true,
      stroked: true,
      wireframe: false,
      getPolygon: (d) => d.polygon,
      getFillColor: (d) => withAlpha(d.color, d.fillAlpha),
      getLineColor: (d) => d.lineColor,
      getLineWidth: (d) => (d.vehicle.selected ? 2 : 1),
      lineWidthUnits: "pixels",
      getElevation: (d) => d.elevation,
      material: {
        ambient: 0.45,
        diffuse: 0.6,
        shininess: 12,
        specularColor: [255, 255, 255],
      },
      onHover: (info) => {
        state.map.getCanvas().style.cursor = info.object ? "pointer" : "";
      },
      onClick: (info) => {
        if (!info.object) return;
        state.pendingSelectionClear = true;
        state.selectedVehicleId = info.object.vehicle.id;
        markUiDirty();
      },
    }),
    new deck.PolygonLayer({
      id: "rail-vehicles",
      data: railPolygons,
      extruded: true,
      pickable: true,
      stroked: true,
      wireframe: false,
      getPolygon: (d) => d.polygon,
      getFillColor: (d) => withAlpha(d.color, d.fillAlpha),
      getLineColor: (d) => d.lineColor,
      getLineWidth: (d) => (d.vehicle.selected ? 2.5 : 1.2),
      lineWidthUnits: "pixels",
      getElevation: (d) => d.elevation,
      material: {
        ambient: 0.55,
        diffuse: 0.7,
        shininess: 20,
        specularColor: [255, 255, 255],
      },
      onHover: (info) => {
        state.map.getCanvas().style.cursor = info.object ? "pointer" : "";
      },
      onClick: (info) => {
        if (!info.object) return;
        state.pendingSelectionClear = true;
        state.selectedVehicleId = info.object.vehicle.id;
        markUiDirty();
      },
    }),
  ];
}
