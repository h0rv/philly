export const MAP_STYLES = {
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
};

export const MODE_STYLE = {
  bus: { label: "Bus", color: [56, 189, 248] },
  trolley: { label: "Trolley", color: [74, 222, 128] },
  metro: { label: "Metro", color: [251, 146, 60] },
  regional_rail: { label: "Regional Rail", color: [167, 139, 250] },
};

export const ALL_MODES = ["bus", "trolley", "metro", "regional_rail"];
export const REPLAY_WINDOW_SECONDS = 24 * 60 * 60;
export const TRAIL_SECONDS = 45;
export const MAP_BOOT_TIMEOUT_MS = 12000;
export const HUD_REFRESH_MS = 250;

export const VISUAL_SCALE = {
  bus: { length: 1.0, width: 1.0, height: 1.0, halo: 0, trail: 1 },
  trolley: { length: 1.15, width: 1.1, height: 1.08, halo: 0, trail: 1.15 },
  metro: { length: 1.3, width: 1.3, height: 1.18, halo: 40, trail: 1.25 },
  regional_rail: {
    length: 1.95,
    width: 2.0,
    height: 1.38,
    halo: 140,
    trail: 1.65,
  },
};
