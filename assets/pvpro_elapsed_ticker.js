// =============================================================================
// pvpro_elapsed_ticker.js
//
// Animates the "Elapsed: Ns" line of the PVPRO progress display at 1Hz, even
// though the Dash poll callback only re-renders the progress UI every 2s.
//
// How it works
// ------------
// The poll callback emits an element with a stable id and a data-attribute:
//
//   <div id="pvpro-elapsed-display" data-started-at="1779267247727">
//     Elapsed: 12s
//   </div>
//
// where `data-started-at` is the wall-clock millisecond timestamp at which
// the PVPRO worker thread was launched.  This script keeps one setInterval
// running for the lifetime of the page.  Every second the interval looks up
// the element (if present), reads its data-attribute, computes
// (Date.now() - startedAt) / 1000, and overwrites the element's text.
//
// Why this is safe
// ----------------
// * The script never touches the server, the worker thread, or the GIL --
//   it's pure DOM manipulation in the browser.  Zero effect on PVPRO speed.
// * When the poll callback re-renders the progress div every 2s, Dash
//   replaces the element in the DOM.  The new element has the SAME
//   data-started-at value, so when our timer runs again it computes the
//   same elapsed value (give or take a fraction) and the counter appears
//   to tick smoothly.  No flicker.
// * When PVPRO finishes the poll callback replaces the progress UI with
//   the result UI; the element with id="pvpro-elapsed-display" is gone
//   so the timer becomes a no-op (it just finds nothing to update).
//
// We start one timer on page load and never cancel it -- it's a single
// document.getElementById per second, which is essentially free.
// =============================================================================
(function () {
    "use strict";

    function tick() {
        var el = document.getElementById("pvpro-elapsed-display");
        if (!el) return;                   // progress UI not on screen, no-op
        var startedAtStr = el.dataset.startedAt;
        if (!startedAtStr) return;         // attribute missing / not yet set
        var startedAt = parseInt(startedAtStr, 10);
        if (!startedAt || isNaN(startedAt)) return;
        var elapsedSec = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
        el.textContent = "Elapsed: " + elapsedSec + "s";
    }

    // Run on a 1-second cadence.  Don't worry about timer drift over long
    // runs -- worst case the counter lags by 1s, which is invisible.
    setInterval(tick, 1000);
})();
