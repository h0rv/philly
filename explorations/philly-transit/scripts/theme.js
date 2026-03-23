import { MAP_STYLES } from "./config.js";
import { state, elements, markUiDirty } from "./state.js";

export function applyTheme(theme, { updateMap = true } = {}) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;

  if (elements.themeBtn) {
    elements.themeBtn.textContent =
      theme === "dark" ? "Light mode" : "Dark mode";
  }

  if (state.map && updateMap) {
    state.map.setStyle(MAP_STYLES[theme]);
  }

  markUiDirty();
}

export function toggleTheme() {
  applyTheme(state.theme === "dark" ? "light" : "dark");
}
