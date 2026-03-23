import { ALL_MODES, MODE_STYLE, REPLAY_WINDOW_SECONDS } from "./config.js";

const MODE_DESCRIPTIONS = {
  bus: "Road vehicles",
  trolley: "Street-running rail",
  metro: "Subway / rapid transit",
  regional_rail: "Commuter rail",
};

function routePillStyle(color, textColor) {
  if (state.theme === "light") {
    return `background: rgba(${color.join(",")}, 0.12); color: rgb(${color.join(",")}); border: 1px solid rgba(${color.join(",")}, 0.24);`;
  }
  return `background: rgb(${color.join(",")}); color: rgb(${textColor.join(",")});`;
}
import { elements, markUiDirty, state } from "./state.js";
import {
  formatClock,
  formatCoordinates,
  formatDate,
  formatDateTime,
  formatDelay,
  formatNumber,
  formatSpeed,
  naturalCompare,
  positionStateDescription,
  positionStateLabel,
  routeDisplayColor,
  statusLabel,
} from "./utils.js";

export function syncControlState() {
  elements.liveBtn.classList.toggle("active", state.live);
  elements.replayBtn.classList.toggle("active", !state.live);
  elements.replayShell.classList.toggle("hidden", state.live);
  elements.playBtn.textContent = state.playing ? "Pause" : "Play";
  elements.timeline.value = String(Math.round(state.timelineSeconds));
  elements.modeBadge.textContent = state.live
    ? "Live"
    : `Replay · ${state.playbackRate}×`;
  elements.modeBadge.className = `badge ${state.live ? "live" : "replay"}`;
  elements.railOnlyBtn.textContent = isRailOnlyMode()
    ? "Show all"
    : "Only rail";
  elements.focusTrainBtn.disabled = !state.selectedVehicleId;
}

export function isRailOnlyMode() {
  return state.activeModes.size === 1 && state.activeModes.has("regional_rail");
}

export function buildStats() {}

export function renderModes() {
  elements.modes.innerHTML = "";
  ALL_MODES.forEach((mode) => {
    const style = MODE_STYLE[mode];
    const button = document.createElement("button");
    button.className = "mode-chip";
    if (!state.activeModes.has(mode)) button.classList.add("off");
    button.innerHTML = `
      <span class="mode-dot" style="background: rgb(${style.color.join(",")});"></span>
      <span class="mode-copy">
        <span class="mode-name">${style.label}</span>
        <span class="mode-count">${MODE_DESCRIPTIONS[mode]} · ${formatNumber(state.data.modes[mode]?.count || 0)}</span>
      </span>
    `;
    button.addEventListener("click", () => {
      if (state.activeModes.has(mode)) state.activeModes.delete(mode);
      else state.activeModes.add(mode);
      if (state.activeModes.size === 0) state.activeModes.add(mode);
      renderModes();
      markUiDirty();
    });
    elements.modes.appendChild(button);
  });
}

export function renderSpeedButtons() {
  elements.speedRow.innerHTML = "";
  [1, 5, 15, 60, 120].forEach((rate) => {
    const button = document.createElement("button");
    button.className = "subbutton";
    if (state.playbackRate === rate) button.classList.add("active");
    button.textContent = `${rate}×`;
    button.addEventListener("click", () => {
      state.playbackRate = rate;
      renderSpeedButtons();
      syncControlState();
      markUiDirty();
    });
    elements.speedRow.appendChild(button);
  });
}

export function renderNotes() {}

export function renderMetroCoverage(onSelectRoute) {
  if (!elements.metroCoverage) return;
  const coverage = Object.values(state.data.coverage?.metro || {});
  elements.metroCoverage.innerHTML = coverage
    .map((route) => {
      const statusClass =
        route.locatedTrips === 0
          ? "offline"
          : route.missingTrips > 0
            ? "partial"
            : "live";
      const subtitle =
        route.routeId === "L1"
          ? "Market-Frankford"
          : route.routeId.startsWith("B")
            ? "Broad Street"
            : route.routeLongName
                .replace(" Line Local", "")
                .replace(" All Stops", "");
      const statusText =
        statusClass === "offline"
          ? "No public GPS"
          : statusClass === "partial"
            ? "Some GPS"
            : "GPS live";
      return `
        <button class="coverage-card ${statusClass}" data-route-id="${route.routeId}" aria-label="${route.routeShortName}: ${statusText}">
          <div class="coverage-top">
            <div class="route-pill coverage-pill">${route.routeShortName}</div>
          </div>
          <div class="coverage-name">${subtitle}</div>
          <div class="coverage-status ${statusClass}">${statusText}</div>
        </button>
      `;
    })
    .join("");

  elements.metroCoverage
    .querySelectorAll("[data-route-id]")
    .forEach((button) => {
      button.addEventListener("click", () => {
        onSelectRoute?.(button.dataset.routeId);
      });
    });
}

export function getRailVehicles(animatedVehicles = state.currentAllAnimated) {
  return animatedVehicles
    .filter((vehicle) => vehicle.mode === "regional_rail")
    .sort((a, b) => {
      if (a.id === state.selectedVehicleId) return -1;
      if (b.id === state.selectedVehicleId) return 1;
      return (
        naturalCompare(a.routeShortName, b.routeShortName) ||
        naturalCompare(a.destination, b.destination) ||
        naturalCompare(a.label, b.label)
      );
    });
}

function railStatusCopy(vehicle) {
  if (!vehicle) {
    return {
      title: "No train selected",
      copy: "Pick a train from the list or click one on the map.",
    };
  }
  return {
    title: vehicle.routeLongName,
    copy: `${vehicle.destination ? `To ${vehicle.destination}` : vehicle.routeLongName} · ${vehicle.nextStop || vehicle.currentStop || "Next stop unknown"}`,
  };
}

export function renderRailNavigator(onSelectTrain) {
  const rails = getRailVehicles();
  elements.railCount.textContent = `${formatNumber(rails.length)} live regional rail trains`;

  const selectedRail =
    rails.find((vehicle) => vehicle.id === state.selectedVehicleId) ||
    rails[0] ||
    null;
  const status = railStatusCopy(selectedRail);
  elements.railStatus.innerHTML = `
    <div class="rail-status-title">${status.title}</div>
    <div class="rail-copy">${status.copy}</div>
  `;

  if (!rails.length) {
    elements.railList.innerHTML =
      '<div class="empty-copy">No live trains in the current snapshot.</div>';
    return;
  }

  elements.railList.innerHTML = rails
    .map((vehicle) => {
      const color = routeDisplayColor(vehicle);
      return `
        <button class="rail-card ${vehicle.id === state.selectedVehicleId ? "active" : ""}" data-train-id="${vehicle.id}">
          <div class="rail-card-top">
            <div class="route-pill" style="${routePillStyle(color, vehicle.textColor)}">
              ${vehicle.routeShortName}
            </div>
            <div class="delay-pill">${formatDelay(vehicle.delayMinutes)}</div>
          </div>
          <div class="rail-headline">${vehicle.destination || vehicle.routeLongName}</div>
          <div class="rail-meta">${vehicle.nextStop || vehicle.currentStop || "Next stop unknown"}</div>
        </button>
      `;
    })
    .join("");

  elements.railList.querySelectorAll("[data-train-id]").forEach((button) => {
    button.addEventListener("click", () => {
      onSelectTrain(button.dataset.trainId);
    });
  });
}

export function renderSelection(simTimestamp, selectedVehicle, callbacks) {
  if (!selectedVehicle) {
    elements.selection.classList.add("hidden");
    elements.selection.innerHTML = "";
    return;
  }

  const displayColor = routeDisplayColor(selectedVehicle);
  const locationLine =
    selectedVehicle.currentStop || selectedVehicle.nextStop || "Between stops";
  const timingLine =
    selectedVehicle.delayMinutes === null ||
    selectedVehicle.delayMinutes === undefined
      ? "No live timing"
      : formatDelay(selectedVehicle.delayMinutes);

  elements.selection.classList.remove("hidden");
  elements.selection.innerHTML = `
    <div class="selection-head compact">
      <div>
        <div class="route-pill" style="${routePillStyle(displayColor, selectedVehicle.textColor)}">
          ${selectedVehicle.routeShortName}
        </div>
        <h2 class="selection-title" style="margin-top: 10px;">${selectedVehicle.destination || selectedVehicle.routeLongName}</h2>
        <div class="selection-subtitle">${locationLine}</div>
      </div>
      <div class="selection-pill-stack">
        <div class="delay-pill">${timingLine}</div>
        <div class="status-pill ${selectedVehicle.positionState}">${positionStateLabel(selectedVehicle.positionState)}</div>
      </div>
    </div>

    <div class="selection-actions compact">
      <button class="subbutton" data-selection-action="focus">Focus</button>
      ${
        selectedVehicle.mode === "regional_rail"
          ? '<button class="subbutton rail-accent" data-selection-action="next-train">Next</button>'
          : '<button class="subbutton" data-selection-action="live">Live</button>'
      }
    </div>

    <div class="detail-grid compact">
      <div class="detail">
        <div class="detail-label">Line</div>
        <div class="detail-value">${selectedVehicle.routeLongName}</div>
      </div>
      <div class="detail">
        <div class="detail-label">Next stop</div>
        <div class="detail-value">${selectedVehicle.nextStop || "Unknown"}</div>
      </div>
      <div class="detail">
        <div class="detail-label">Updated</div>
        <div class="detail-value">${formatDateTime(selectedVehicle.timestamp)}</div>
      </div>
      <div class="detail">
        <div class="detail-label">Source</div>
        <div class="detail-value">${positionStateDescription(selectedVehicle.positionState)}</div>
      </div>
    </div>
  `;

  elements.selection
    .querySelectorAll("[data-selection-action]")
    .forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.selectionAction;
        if (action === "focus") callbacks.focus();
        if (action === "next-train") callbacks.nextTrain();
        if (action === "live") callbacks.goLive();
      });
    });
}

export function updateSummary(simTimestamp, visibleVehicles, allVehicles) {
  const railCount = allVehicles.filter(
    (vehicle) => vehicle.mode === "regional_rail",
  ).length;
  const metroCoverage = state.data.coverage?.metro || {};
  const hiddenMetroRoutes = Object.values(metroCoverage).filter(
    (route) => route.activeTrips > 0 && route.locatedTrips === 0,
  );
  const partialMetroRoutes = Object.values(metroCoverage).filter(
    (route) => route.locatedTrips > 0 && route.missingTrips > 0,
  );

  if (state.live) {
    let subwayNote = "";
    if (hiddenMetroRoutes.length) {
      subwayNote = ` Subway without public GPS: ${hiddenMetroRoutes.map((route) => route.routeShortName).join(", ")}.`;
    } else if (partialMetroRoutes.length) {
      subwayNote = ` Partial subway GPS: ${partialMetroRoutes.map((route) => route.routeShortName).join(", ")}.`;
    }
    elements.liveSummary.textContent = `${formatNumber(visibleVehicles.length)} live vehicles · ${formatNumber(railCount)} regional rail.${subwayNote}`;
  } else {
    const hoursBack = (
      (REPLAY_WINDOW_SECONDS - state.timelineSeconds) /
      3600
    ).toFixed(1);
    elements.liveSummary.textContent = `Replay · ${hoursBack}h back`;
  }

  elements.clock.textContent = formatClock(simTimestamp);
  elements.subclock.textContent = formatDate(simTimestamp);
}
