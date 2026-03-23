import {
  ALL_MODES,
  HUD_REFRESH_MS,
  REPLAY_WINDOW_SECONDS,
  TRAIL_SECONDS,
  VISUAL_SCALE,
} from "./config.js";
import { createMap, buildLayers } from "./map-view.js";
import { state, elements, cacheDom, markUiDirty } from "./state.js";
import { applyTheme, toggleTheme } from "./theme.js";
import {
  bearingBetween,
  formatDate,
  getPositionState,
  naturalCompare,
  routeDisplayColor,
  segmentDistance,
  wrapProgress,
} from "./utils.js";
import {
  buildStats,
  getRailVehicles,
  isRailOnlyMode,
  renderMetroCoverage,
  renderModes,
  renderNotes,
  renderRailNavigator,
  renderSelection,
  renderSpeedButtons,
  syncControlState,
  updateSummary,
} from "./ui.js";

function showClientError(message) {
  console.error(message);
  if (elements.loading) elements.loading.style.display = "none";
  if (elements.error) {
    elements.error.style.display = "flex";
    const card = elements.error.querySelector(".card");
    if (card) {
      card.innerHTML = `
        <h2 style="margin:0 0 10px;">Transit view hit an error</h2>
        <div class="card-copy" style="margin-bottom: 14px;">${message}</div>
        <div class="card-copy">
          Rebuild the fixture with <code>uv run python explorations/philly-transit/build_data.py</code> and reload.
        </div>
      `;
    }
  }
}

function preprocessShapes() {
  for (const [shapeId, shape] of Object.entries(state.data.shapes)) {
    const coords = shape.coords;
    const cumulative = [0];
    const segments = [];
    let total = 0;

    for (let i = 0; i < coords.length - 1; i++) {
      const start = coords[i];
      const end = coords[i + 1];
      const length = segmentDistance(start, end);
      total += length;
      cumulative.push(total);
      segments.push({
        start,
        end,
        length,
        bearing: bearingBetween(start, end),
      });
    }

    state.shapeMeta[shapeId] = {
      ...shape,
      shapeId,
      coords,
      cumulative,
      segments,
      lengthM: total || shape.lengthM || 0,
    };
  }
}

function interpolateAlongShape(shape, progressM) {
  if (!shape || !shape.segments.length) {
    const coord = shape?.coords?.[0] || [-75.1652, 39.9526];
    return { coord, bearing: 0 };
  }

  const progress = wrapProgress(progressM, shape.lengthM, shape.loop);
  for (let i = 0; i < shape.segments.length; i++) {
    const startDistance = shape.cumulative[i];
    const endDistance = shape.cumulative[i + 1];
    if (progress <= endDistance || i === shape.segments.length - 1) {
      const segment = shape.segments[i];
      const t = Math.max(
        0,
        Math.min(
          1,
          (progress - startDistance) / Math.max(segment.length, 0.0001),
        ),
      );
      return {
        coord: [
          segment.start[0] + (segment.end[0] - segment.start[0]) * t,
          segment.start[1] + (segment.end[1] - segment.start[1]) * t,
        ],
        bearing: segment.bearing,
      };
    }
  }

  return {
    coord: shape.coords.at(-1),
    bearing: shape.segments.at(-1)?.bearing ?? 0,
  };
}

function makeVehiclePolygon(coord, bearingDeg, lengthM, widthM, mode) {
  const lat = (coord[1] * Math.PI) / 180;
  const metersPerDegLat = 111320;
  const metersPerDegLng = 111320 * Math.cos(lat);
  const theta = (bearingDeg * Math.PI) / 180;
  const halfLength = lengthM / 2;
  const halfWidth = widthM / 2;
  const points =
    mode === "regional_rail"
      ? [
          [halfLength + 4, 0],
          [halfLength * 0.86, halfWidth],
          [-halfLength, halfWidth],
          [-halfLength, -halfWidth],
          [halfLength * 0.86, -halfWidth],
          [halfLength + 4, 0],
        ]
      : [
          [halfLength + 2, 0],
          [halfLength * 0.25, halfWidth],
          [-halfLength, halfWidth],
          [-halfLength, -halfWidth],
          [halfLength * 0.25, -halfWidth],
          [halfLength + 2, 0],
        ];

  return points.map(([forward, right]) => {
    const east = forward * Math.sin(theta) + right * Math.cos(theta);
    const north = forward * Math.cos(theta) - right * Math.sin(theta);
    return [
      coord[0] + east / metersPerDegLng,
      coord[1] + north / metersPerDegLat,
    ];
  });
}

function getSimTimestamp(nowMs) {
  if (state.live) {
    return state.data.generatedAt + (nowMs - state.liveAnchorMs) / 1000;
  }
  return (
    state.data.generatedAt - (REPLAY_WINDOW_SECONDS - state.timelineSeconds)
  );
}

function getAnimatedVehicleById(id) {
  return state.currentAllAnimated.find((vehicle) => vehicle.id === id) || null;
}

function ensureModeVisible(mode) {
  if (!state.activeModes.has(mode)) {
    state.activeModes.add(mode);
    renderModes();
  }
}

function focusRoute(routeId) {
  if (!state.map) return;
  const shapes = Object.values(state.shapeMeta).filter(
    (shape) => shape.routeId === routeId,
  );
  if (!shapes.length) return;
  const bounds = new maplibregl.LngLatBounds();
  shapes.forEach((shape) =>
    shape.coords.forEach((coord) => bounds.extend(coord)),
  );
  state.map.fitBounds(bounds, {
    padding: { top: 80, right: 260, bottom: 80, left: 260 },
    duration: 900,
    pitch: state.map.getPitch(),
    bearing: state.map.getBearing(),
  });
}

function setAllModes() {
  state.activeModes = new Set(ALL_MODES);
  renderModes();
  markUiDirty();
}

function setRailOnlyMode() {
  state.activeModes = new Set(["regional_rail"]);
  renderModes();
  fitToRailVehicles();
  markUiDirty();
}

function selectVehicle(vehicleId, { focus = false } = {}) {
  const rawVehicle = state.data.vehicles.find(
    (vehicle) => vehicle.id === vehicleId,
  );
  if (!rawVehicle) return;
  ensureModeVisible(rawVehicle.mode);
  state.selectedVehicleId = vehicleId;
  if (focus) focusVehicle(vehicleId);
  markUiDirty();
}

function focusVehicle(vehicleId) {
  const vehicle = getAnimatedVehicleById(vehicleId);
  if (!vehicle || !state.map) return;
  state.map.flyTo({
    center: vehicle.coord,
    zoom: vehicle.mode === "regional_rail" ? 13.2 : 14.2,
    pitch: Math.max(state.map.getPitch(), 55),
    bearing: state.map.getBearing(),
    duration: 1100,
    essential: true,
  });
}

function selectRailByOffset(offset) {
  const rails = getRailVehicles();
  if (!rails.length) return;
  let index = rails.findIndex(
    (vehicle) => vehicle.id === state.selectedVehicleId,
  );
  if (index === -1) index = offset > 0 ? -1 : 0;
  index = (index + offset + rails.length) % rails.length;
  selectVehicle(rails[index].id, { focus: true });
}

function fitToRailVehicles() {
  if (!state.map) return;
  const rails = getRailVehicles();
  if (!rails.length) return;
  const bounds = new maplibregl.LngLatBounds();
  rails.forEach((vehicle) => bounds.extend(vehicle.coord));
  state.map.fitBounds(bounds, {
    padding: { top: 80, right: 360, bottom: 80, left: 360 },
    duration: 900,
    pitch: state.map.getPitch(),
    bearing: state.map.getBearing(),
  });
}

function enterReplayMode() {
  state.live = false;
  state.playing = false;
  syncControlState();
  markUiDirty();
}

function setLiveMode() {
  state.live = true;
  state.playing = false;
  state.timelineSeconds = REPLAY_WINDOW_SECONDS;
  state.liveAnchorMs = performance.now();
  syncControlState();
  markUiDirty();
}

function computeAnimatedVehicles(simTimestamp) {
  const allVehicles = [];
  const visibleVehicles = [];

  for (const vehicle of state.data.vehicles) {
    const shape = state.shapeMeta[vehicle.shapeId];
    if (!shape) continue;

    const elapsed = simTimestamp - vehicle.timestamp;
    const progressM = wrapProgress(
      vehicle.progressM + vehicle.speedMps * elapsed,
      shape.lengthM,
      shape.loop,
    );
    const current = interpolateAlongShape(shape, progressM);
    const scale = VISUAL_SCALE[vehicle.mode] || VISUAL_SCALE.bus;
    const trail = interpolateAlongShape(
      shape,
      progressM - vehicle.speedMps * TRAIL_SECONDS * scale.trail,
    );
    const bearing = current.bearing || vehicle.bearing || 0;
    const displayLengthM = vehicle.lengthM * scale.length;
    const displayWidthM = vehicle.widthM * scale.width;
    const displayHeightM = vehicle.heightM * scale.height;

    const animated = {
      ...vehicle,
      coord: current.coord,
      derivedBearing: bearing,
      trail: [trail.coord, current.coord],
      polygon: makeVehiclePolygon(
        current.coord,
        bearing,
        displayLengthM,
        displayWidthM,
        vehicle.mode,
      ),
      displayLengthM,
      displayWidthM,
      displayHeightM,
      haloRadiusM: scale.halo,
      secondsSinceReport: Math.max(0, simTimestamp - vehicle.timestamp),
      positionState: getPositionState(vehicle, simTimestamp),
      selected: vehicle.id === state.selectedVehicleId,
      color: routeDisplayColor(vehicle),
    };

    allVehicles.push(animated);
    if (state.activeModes.has(vehicle.mode)) visibleVehicles.push(animated);
  }

  state.currentAllAnimated = allVehicles.sort((a, b) =>
    naturalCompare(a.id, b.id),
  );
  state.currentVisibleAnimated = visibleVehicles;
  return { allVehicles, visibleVehicles };
}

function updateHud(simTimestamp, visibleVehicles, allVehicles, nowMs) {
  buildStats(visibleVehicles, allVehicles);
  updateSummary(simTimestamp, visibleVehicles, allVehicles);
  syncControlState();
  renderRailNavigator((trainId) => selectVehicle(trainId, { focus: true }));
  renderSelection(
    simTimestamp,
    getAnimatedVehicleById(state.selectedVehicleId),
    {
      focus: () => focusVehicle(state.selectedVehicleId),
      nextTrain: () => selectRailByOffset(1),
      goLive: () => setLiveMode(),
    },
  );
  state.lastHudUpdateMs = nowMs;
  state.uiDirty = false;
}

function tick(nowMs) {
  if (!state.lastFrameMs) state.lastFrameMs = nowMs;
  const deltaSeconds = (nowMs - state.lastFrameMs) / 1000;
  state.lastFrameMs = nowMs;

  if (!state.live && state.playing) {
    state.timelineSeconds = Math.min(
      REPLAY_WINDOW_SECONDS,
      state.timelineSeconds + deltaSeconds * state.playbackRate,
    );
    if (state.timelineSeconds >= REPLAY_WINDOW_SECONDS) state.playing = false;
    markUiDirty();
  }

  if (state.overlay) {
    const simTimestamp = getSimTimestamp(nowMs);
    const { allVehicles, visibleVehicles } =
      computeAnimatedVehicles(simTimestamp);
    state.overlay.setProps({
      layers: buildLayers(
        visibleVehicles,
        getAnimatedVehicleById(state.selectedVehicleId),
      ),
    });
    if (state.uiDirty || nowMs - state.lastHudUpdateMs > HUD_REFRESH_MS) {
      updateHud(simTimestamp, visibleVehicles, allVehicles, nowMs);
    }
  }

  requestAnimationFrame(tick);
}

function syncChromeButtons() {
  elements.toggleLeftBtn?.classList.toggle("active", state.showLeftPanel);
  elements.toggleRightBtn?.classList.toggle("active", state.showRightPanel);
  elements.focusModeBtn?.classList.toggle("active", state.focusMode);
  if (elements.leftColumn)
    elements.leftColumn.classList.toggle("hidden-panel", !state.showLeftPanel);
  if (elements.rightColumn)
    elements.rightColumn.classList.toggle(
      "hidden-panel",
      !state.showRightPanel,
    );
  if (elements.viewToolbar)
    elements.viewToolbar.classList.toggle("hidden-panel", state.focusMode);
  if (elements.exitFocusBtn)
    elements.exitFocusBtn.classList.toggle("hidden-panel", !state.focusMode);
}

function setFocusMode(enabled) {
  state.focusMode = enabled;
  state.showLeftPanel = !enabled;
  state.showRightPanel = !enabled;
  syncChromeButtons();
}

function wireControls() {
  elements.liveBtn.addEventListener("click", () => setLiveMode());
  elements.replayBtn.addEventListener("click", () => enterReplayMode());
  elements.playBtn.addEventListener("click", () => {
    state.playing = !state.playing;
    markUiDirty();
  });
  elements.themeBtn.addEventListener("click", () => toggleTheme());
  elements.toggleLeftBtn?.addEventListener("click", () => {
    state.showLeftPanel = !state.showLeftPanel;
    state.focusMode = false;
    syncChromeButtons();
  });
  elements.toggleRightBtn?.addEventListener("click", () => {
    state.showRightPanel = !state.showRightPanel;
    state.focusMode = false;
    syncChromeButtons();
  });
  elements.focusModeBtn?.addEventListener("click", () => {
    setFocusMode(!state.focusMode);
  });
  elements.exitFocusBtn?.addEventListener("click", () => {
    setFocusMode(false);
  });

  elements.timeline.addEventListener("input", (event) => {
    state.live = false;
    state.playing = false;
    state.timelineSeconds = Number(event.target.value);
    markUiDirty();
  });

  elements.prevTrainBtn.addEventListener("click", () => selectRailByOffset(-1));
  elements.nextTrainBtn.addEventListener("click", () => selectRailByOffset(1));
  elements.focusTrainBtn.addEventListener("click", () => {
    if (state.selectedVehicleId) focusVehicle(state.selectedVehicleId);
  });
  elements.railOnlyBtn.addEventListener("click", () => {
    if (isRailOnlyMode()) setAllModes();
    else setRailOnlyMode();
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "[") {
      event.preventDefault();
      selectRailByOffset(-1);
    }
    if (event.key === "]") {
      event.preventDefault();
      selectRailByOffset(1);
    }
  });
}

function exposeApi() {
  window.PhillyTransit = {
    listTrains() {
      return getRailVehicles().map((vehicle) => ({
        id: vehicle.id,
        line: vehicle.routeShortName,
        route: vehicle.routeLongName,
        destination: vehicle.destination,
        currentStop: vehicle.currentStop,
        nextStop: vehicle.nextStop,
        delayMinutes: vehicle.delayMinutes,
        positionState: vehicle.positionState,
        coordinates: vehicle.coord,
      }));
    },
    selectTrain(id, options = {}) {
      selectVehicle(String(id), { focus: options.focus !== false });
    },
    selectNextTrain() {
      selectRailByOffset(1);
    },
    selectPrevTrain() {
      selectRailByOffset(-1);
    },
    focusSelected() {
      if (state.selectedVehicleId) focusVehicle(state.selectedVehicleId);
    },
    spotlightTrains() {
      setRailOnlyMode();
    },
    showAllModes() {
      setAllModes();
    },
    openReplay() {
      enterReplayMode();
    },
    goLive() {
      setLiveMode();
    },
  };
}

async function init() {
  cacheDom();
  window.addEventListener("error", (event) =>
    showClientError(event.message || "Unknown client error"),
  );
  window.addEventListener("unhandledrejection", (event) => {
    showClientError(
      event.reason?.message ||
        String(event.reason || "Unhandled promise rejection"),
    );
  });

  wireControls();
  renderSpeedButtons();
  const themeParam = new URLSearchParams(window.location.search).get("theme");
  applyTheme(themeParam === "light" ? "light" : "dark", { updateMap: false });

  try {
    const response = await fetch("./data/transit.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
  } catch (error) {
    showClientError(error?.message || String(error));
    return;
  }

  preprocessShapes();
  renderModes();
  renderMetroCoverage((routeId) => {
    ensureModeVisible("metro");
    focusRoute(routeId);
  });
  renderNotes();
  syncChromeButtons();
  elements.rangeStart.textContent = formatDate(
    state.data.generatedAt - REPLAY_WINDOW_SECONDS,
  );
  elements.rangeEnd.textContent = "Now";
  syncControlState();
  exposeApi();

  createMap(showClientError, () => {
    markUiDirty();
  });

  setLiveMode();
  requestAnimationFrame(tick);
}

init();
