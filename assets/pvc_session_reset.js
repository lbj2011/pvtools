/* PVcopilot session behavior: state persists while moving between PVTOOLS
   pages within the tab, but a browser RELOAD (F5 / Cmd+R) resets the tool
   completely. This runs on every full document load, before Dash mounts its
   components, and clears PVcopilot's session-scoped stores when — and only
   when — the load was a reload. */
(function () {
  try {
    var type = "navigate";
    if (performance.getEntriesByType) {
      var nav = performance.getEntriesByType("navigation");
      if (nav && nav.length) { type = nav[0].type; }
    } else if (performance.navigation) {
      type = performance.navigation.type === 1 ? "reload" : "navigate";
    }
    if (type === "reload") {
      [
        "mapped-vars-store",
        "data-columns-store",
        "data-source-store",
        "stored-data-file-name",
        "step-progress",
        "advanced-active-step",
        "ui-mode",
        "downsample-note",
        "session-cache-meta",
        "mapping-notes-store"
      ].forEach(function (k) { sessionStorage.removeItem(k); });
    }
  } catch (e) { /* never block page load over this */ }
})();
