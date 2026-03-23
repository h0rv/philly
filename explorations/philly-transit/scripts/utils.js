import { MODE_STYLE } from "./config.js";
import { state } from "./state.js";

export function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

export function naturalCompare(a, b) {
  return String(a || "").localeCompare(String(b || ""), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

export function formatClock(timestampSeconds) {
  const date = new Date(timestampSeconds * 1000);
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function formatDate(timestampSeconds) {
  const date = new Date(timestampSeconds * 1000);
  return new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  }).format(date);
}

export function formatDateTime(timestampSeconds) {
  return `${formatClock(timestampSeconds)} · ${formatDate(timestampSeconds)}`;
}

export function formatDelay(delayMinutes) {
  if (delayMinutes === null || delayMinutes === undefined)
    return "No live delay";
  if (delayMinutes === 0) return "On time";
  if (delayMinutes > 0) return `${delayMinutes} min late`;
  return `${Math.abs(delayMinutes)} min early`;
}

export function formatSpeed(speedMps) {
  return `${(speedMps * 2.23694).toFixed(0)} mph`;
}

export function formatCoordinates(coord) {
  if (!coord) return "Unknown";
  return `${coord[1].toFixed(4)}, ${coord[0].toFixed(4)}`;
}

export function withAlpha(color, alpha) {
  return [color[0], color[1], color[2], alpha];
}

export function statusLabel(status) {
  return status ? status.replaceAll("_", " ").toLowerCase() : "unknown";
}

export function wrapProgress(progress, lengthM, isLoop) {
  if (!lengthM) return 0;
  if (isLoop) {
    let wrapped = progress % lengthM;
    if (wrapped < 0) wrapped += lengthM;
    return wrapped;
  }
  const cycle = lengthM * 2;
  let value = progress % cycle;
  if (value < 0) value += cycle;
  if (value > lengthM) value = cycle - value;
  return value;
}

export function segmentDistance(a, b) {
  const latRef = (((a[1] + b[1]) / 2) * Math.PI) / 180;
  const metersPerDegLat = 111320;
  const metersPerDegLng = 111320 * Math.cos(latRef);
  const dx = (b[0] - a[0]) * metersPerDegLng;
  const dy = (b[1] - a[1]) * metersPerDegLat;
  return Math.hypot(dx, dy);
}

export function bearingBetween(a, b) {
  const lng1 = (a[0] * Math.PI) / 180;
  const lng2 = (b[0] * Math.PI) / 180;
  const lat1 = (a[1] * Math.PI) / 180;
  const lat2 = (b[1] * Math.PI) / 180;
  const y = Math.sin(lng2 - lng1) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(lng2 - lng1);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

export function luminance(color) {
  return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
}

export function normalizeRouteColor(color, mode) {
  const accent = MODE_STYLE[mode]?.color || color;
  const lum = luminance(color);
  if (lum > 220 || lum < 30) return accent;
  return color;
}

export function adjustColorForTheme(color) {
  if (state.theme !== "light") return color;
  const lum = luminance(color);
  if (lum > 180) {
    return [
      Math.max(32, color[0] - 90),
      Math.max(32, color[1] - 90),
      Math.max(32, color[2] - 90),
    ];
  }
  if (lum < 70) {
    return [
      Math.min(235, color[0] + 35),
      Math.min(235, color[1] + 35),
      Math.min(235, color[2] + 35),
    ];
  }
  return color;
}

export function selectionHighlightColor() {
  return state.theme === "light" ? [8, 145, 178, 235] : [34, 211, 238, 245];
}

export function routeDisplayColor(vehicle) {
  return adjustColorForTheme(normalizeRouteColor(vehicle.color, vehicle.mode));
}

export function labelDisplayColor(vehicle) {
  const color = routeDisplayColor(vehicle);
  if (state.theme === "light" && luminance(color) > 170) {
    return [
      Math.max(0, color[0] - 80),
      Math.max(0, color[1] - 80),
      Math.max(0, color[2] - 80),
    ];
  }
  return color;
}

export function getPositionState(vehicle, simTimestamp) {
  const ageSeconds = Math.max(0, simTimestamp - vehicle.timestamp);
  const isRailLike =
    vehicle.mode === "regional_rail" || vehicle.mode === "metro";
  const liveWindow = isRailLike ? 75 : 45;
  const extrapolatedWindow = isRailLike ? 360 : 180;
  const stopWindow = isRailLike ? 180 : 90;

  if (vehicle.positionSource === "stop_inferred") {
    return ageSeconds <= stopWindow ? "stop_inferred" : "stale";
  }
  if (vehicle.positionSource === "progress_inferred") {
    return ageSeconds <= extrapolatedWindow ? "extrapolated" : "stale";
  }
  if (ageSeconds <= liveWindow) return "live";
  if (ageSeconds <= extrapolatedWindow) return "extrapolated";
  return "stale";
}

export function positionStateLabel(positionState) {
  if (positionState === "live") return "GPS live";
  if (positionState === "stop_inferred") return "Stop inferred";
  if (positionState === "extrapolated") return "Estimated";
  return "Stale estimate";
}

export function positionStateDescription(positionState) {
  if (positionState === "live")
    return "Direct vehicle coordinate from SEPTA feed";
  if (positionState === "stop_inferred")
    return "No coordinate in feed; anchored to the reported stop";
  if (positionState === "extrapolated")
    return "Projected forward from the last real report along the route shape";
  return "Older projected position; useful as a hint, not exact ground truth";
}
