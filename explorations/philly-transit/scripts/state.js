import { ALL_MODES } from "./config.js";

export const state = {
  data: null,
  map: null,
  overlay: null,
  theme: "dark",
  shapeMeta: {},
  activeModes: new Set(ALL_MODES),
  selectedVehicleId: null,
  live: true,
  playing: false,
  playbackRate: 15,
  timelineSeconds: 24 * 60 * 60,
  liveAnchorMs: performance.now(),
  lastFrameMs: 0,
  currentVisibleAnimated: [],
  currentAllAnimated: [],
  pendingSelectionClear: false,
  lastHudUpdateMs: 0,
  uiDirty: true,
  showLeftPanel: true,
  showRightPanel: true,
  focusMode: false,
};

export const elements = {};

const DOM_IDS = [
  "loading",
  "error",
  "stats",
  "modes",
  "liveBtn",
  "replayBtn",
  "playBtn",
  "themeBtn",
  "toggleLeftBtn",
  "toggleRightBtn",
  "focusModeBtn",
  "viewToolbar",
  "exitFocusBtn",
  "leftColumn",
  "rightColumn",
  "shell",
  "metroCoverage",
  "timeline",
  "clock",
  "subclock",
  "rangeStart",
  "rangeEnd",
  "speedRow",
  "helperText",
  "notes",
  "modeBadge",
  "selection",
  "footerMeta",
  "replayShell",
  "liveSummary",
  "railCount",
  "railList",
  "railStatus",
  "prevTrainBtn",
  "nextTrainBtn",
  "focusTrainBtn",
  "railOnlyBtn",
];

export function cacheDom() {
  DOM_IDS.forEach((id) => {
    elements[id] = document.getElementById(id);
  });
}

export function markUiDirty() {
  state.uiDirty = true;
}
