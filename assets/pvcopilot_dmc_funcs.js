// pvcopilot_dmc_funcs.js
// ---------------------------------------------------------------------------
// Custom dmc.Select option renderer for the "IDENTIFIED VARIABLES" mapping
// table. Renders each dropdown option as:  column-name (left)  ...  quality pill
// (right).  The quality label is read from each option's `quality` field, which
// build_variable_mapping_table() sets per column (e.g. "all-zero",
// "94% missing", "per-device", "wrong units", "constant").
//
// DEPLOYMENT: drop this file into your Dash app's  assets/  folder (the same
// folder as your CSS / logo). Dash auto-serves everything in assets/, so no
// import or <script> tag is needed. Referenced from Python via
//     dmc.Select(..., renderOption={"function": "renderVarMapOption"})
// ---------------------------------------------------------------------------

var dmcfuncs = window.dashMantineFunctions = window.dashMantineFunctions || {};

dmcfuncs.renderVarMapOption = function (input) {
    var option = input.option || {};
    var name = option.label != null ? option.label : option.value;
    var tag = option.quality || "";

    // Left: a small bullet marker, then the column name. The bullet appears on
    // EVERY option (both groups) so the list reads as a bulleted set of choices.
    var children = [
        React.createElement(
            "span",
            {
                key: "bullet",
                "aria-hidden": "true",
                style: {
                    flex: "0 0 auto",
                    color: "#94a3b8",
                    fontSize: "13px",
                    lineHeight: "1",
                },
            },
            "\u2022"   // •
        ),
        React.createElement(
            "span",
            {
                key: "name",
                style: {
                    fontFamily: "Arial, sans-serif",
                    fontSize: "13px",
                    color: "#1d1d1f",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                },
            },
            name
        ),
    ];

    // Right: a small quality pill, pushed to the far right with an auto margin.
    if (tag) {
        // Neutral (gray) = context, not a defect: a single-device channel,
        // which the renderer labels "one inverter" / "one MPPT" / ... (older
        // builds used "per-device"). Everything else (all-zero, "94% missing",
        // "no numeric data", "wrong units", "constant", ...) flags a real
        // data-quality issue -> muted amber. Test the RAW semantic tag, before
        // the lead word is added below.
        var isNeutral = /per-device|^one\s/i.test(tag);
        var fg = isNeutral ? "#57606a" : "#8a6d00";
        var bg = isNeutral ? "#f1f3f5" : "#fff6e0";
        var bd = isNeutral ? "#d7dce0" : "#f0dfa8";

        // Lead word: neutral context reads "Note:", a real data-quality issue
        // reads "Warning:" — so the pill's severity is clear from the text too,
        // not just the color.
        var pillText = (isNeutral ? "Note: " : "Warning: ") + tag;

        children.push(
            React.createElement(
                "span",
                {
                    key: "pill",
                    style: {
                        marginInlineStart: "auto", // <- pushes pill to the right
                        flex: "0 0 auto",
                        fontFamily: "Arial, sans-serif",
                        fontSize: "10.5px",
                        fontWeight: "600",
                        lineHeight: "1",
                        letterSpacing: "0.01em",
                        color: fg,
                        background: bg,
                        border: "1px solid " + bd,
                        borderRadius: "980px",
                        padding: "3px 8px",
                        whiteSpace: "nowrap",
                    },
                },
                pillText
            )
        );
    }

    return React.createElement(
        "div",
        {
            style: {
                display: "flex",
                alignItems: "center",
                width: "100%",
                gap: "8px",
            },
        },
        children
    );
};

// Keep the newest server event visible and let the user drag the complete
// progress monitor by its header. The translation is saved across Dash
// re-renders and page refreshes.
(function initPVCProgressMonitor() {
    if (window.__pvcProgressMonitorInit) return;
    window.__pvcProgressMonitorInit = true;

    var drag = null;
    var resizeDrag = null;
    var monitor = null;
    var saved = {x: 0, y: 0};
    var savedSize = null;
    var suppressToggleClick = false;
    var scrollState = {logFollow: true, bodyFollow: true, logTop: 0, bodyTop: 0};
    var lastLogNode = null;
    try { saved = JSON.parse(localStorage.getItem("pvc-monitor-position") || "{\"x\":0,\"y\":0}"); } catch (e) {}
    // Versioned key intentionally resets the former oversized default once.
    try { savedSize = JSON.parse(localStorage.getItem("pvc-monitor-size-v2") || "null"); } catch (e) {}

    function topBoundary() {
        var nav = document.querySelector("nav.navbar.sticky-top, .navbar.sticky-top, nav.navbar");
        if (!nav) return 8;
        var rect = nav.getBoundingClientRect();
        return (rect.bottom > 0 && rect.top < window.innerHeight) ? Math.max(8, rect.bottom + 8) : 8;
    }

    function applyPosition() {
        monitor = document.querySelector(".pvcopilot-root .pvc-monitor");
        if (monitor) monitor.style.transform = "translate(" + (saved.x || 0) + "px," + (saved.y || 0) + "px)";
    }

    function restoreScroll() {
        var log = document.getElementById("pvc-monitor-log");
        // Dash replaces the log element on every polling response. Restore the
        // saved position once for that new node, not on unrelated page/class
        // mutations (which previously kept pulling the user away mid-scroll).
        if (log && log !== lastLogNode) {
            lastLogNode = log;
            log.scrollTop = scrollState.logFollow ? log.scrollHeight : Math.min(scrollState.logTop, log.scrollHeight);
        }
        var body = document.querySelector(".pvc-monitor-panel.is-open .pvc-monitor-body");
        if (body) {
            body.scrollTop = scrollState.bodyFollow ? body.scrollHeight : Math.min(scrollState.bodyTop, body.scrollHeight);
        }
    }

    function setupResizablePanel() {
        var panel = document.querySelector(".pvc-monitor-panel.is-open");
        if (!panel || panel.dataset.pvcResizeReady === "1") return;
        panel.dataset.pvcResizeReady = "1";
        // Re-anchor from `bottom` to `top` so the lower-right resize handle
        // follows the pointer naturally in both directions.
        var parent = panel.closest(".pvc-monitor");
        var panelRect = panel.getBoundingClientRect();
        var parentRect = parent.getBoundingClientRect();
        var pillRect = document.getElementById("progress-monitor-toggle").getBoundingClientRect();
        panel.dataset.pvcPillGap = String(Math.max(4, pillRect.top - panelRect.bottom));
        panel.style.top = (panelRect.top - parentRect.top) + "px";
        panel.style.bottom = "auto";
        if (savedSize && savedSize.width && savedSize.height) {
            panel.style.width = Math.min(savedSize.width, window.innerWidth - 16) + "px";
            panel.style.height = Math.min(savedSize.height, window.innerHeight - 100) + "px";
        }
        anchorPanelAbovePill(panel);
        constrainMonitor();
    }

    function anchorPanelAbovePill(panel) {
        var parent = panel && panel.closest(".pvc-monitor");
        var pill = document.getElementById("progress-monitor-toggle");
        if (!panel || !parent || !pill) return;
        var parentRect = parent.getBoundingClientRect();
        var pillRect = pill.getBoundingClientRect();
        var height = panel.getBoundingClientRect().height;
        var gap = parseFloat(panel.dataset.pvcPillGap || "6");
        panel.style.top = (pillRect.top - parentRect.top - gap - height) + "px";
        panel.style.bottom = "auto";
    }

    function constrainMonitor() {
        monitor = document.querySelector(".pvcopilot-root .pvc-monitor");
        var panel = document.querySelector(".pvc-monitor-panel.is-open");
        var pill = document.getElementById("progress-monitor-toggle");
        if (!monitor || !pill) return;
        var dx = 0, dy = 0;
        var pillRect = pill.getBoundingClientRect();
        var panelRect = panel ? panel.getBoundingClientRect() : null;
        var minTop = topBoundary();
        var bottomLimit = window.innerHeight - 12;
        var gap = panel ? parseFloat(panel.dataset.pvcPillGap || "6") : 6;
        var minPanelHeight = Math.min(280, Math.max(220, bottomLimit - minTop - pillRect.height - gap));
        if (pillRect.bottom > bottomLimit) dy -= pillRect.bottom - bottomLimit;
        if (panel) {
            var minimumPillTop = minTop + minPanelHeight + gap;
            if (pillRect.top + dy < minimumPillTop) dy += minimumPillTop - (pillRect.top + dy);
        } else if (pillRect.top + dy < minTop) {
            dy += minTop - (pillRect.top + dy);
        }
        var leftEdge = panelRect ? Math.min(panelRect.left, pillRect.left) : pillRect.left;
        var rightEdge = panelRect ? Math.max(panelRect.right, pillRect.right) : pillRect.right;
        if (leftEdge < 8) dx += 8 - leftEdge;
        if (rightEdge > window.innerWidth - 8) dx -= rightEdge - (window.innerWidth - 8);
        if (dx || dy) {
            saved.x = (saved.x || 0) + dx; saved.y = (saved.y || 0) + dy;
            applyPosition();
            try { localStorage.setItem("pvc-monitor-position", JSON.stringify(saved)); } catch (e) {}
        }
        if (panel) {
            pillRect = pill.getBoundingClientRect();
            panelRect = panel.getBoundingClientRect();
            var maxHeight = Math.max(minPanelHeight, pillRect.top - gap - minTop);
            var panelMinWidth = Math.min(340, window.innerWidth - 16);
            var maxWidth = Math.max(panelMinWidth, window.innerWidth - 8 - panelRect.left);
            if (panelRect.height > maxHeight) panel.style.height = maxHeight + "px";
            if (panelRect.width > maxWidth) panel.style.width = maxWidth + "px";
            anchorPanelAbovePill(panel);
        }
    }

    document.addEventListener("scroll", function (event) {
        var target = event.target;
        if (!target || target === document) return;
        if (target.id === "pvc-monitor-log") {
            scrollState.logTop = target.scrollTop;
            scrollState.logFollow = target.scrollHeight - target.scrollTop - target.clientHeight < 32;
        } else if (target.classList && target.classList.contains("pvc-monitor-body")) {
            scrollState.bodyTop = target.scrollTop;
            scrollState.bodyFollow = target.scrollHeight - target.scrollTop - target.clientHeight < 32;
        }
    }, true);

    // Record manual intent before the browser emits `scroll`. This makes an
    // upward wheel/touch gesture win even if a polling render lands in the
    // same frame.
    document.addEventListener("wheel", function (event) {
        var log = event.target.closest && event.target.closest("#pvc-monitor-log");
        if (log && event.deltaY < 0) {
            scrollState.logFollow = false;
            scrollState.logTop = log.scrollTop;
        }
    }, {passive: true, capture: true});

    document.addEventListener("pointerdown", function (event) {
        var log = event.target.closest && event.target.closest("#pvc-monitor-log");
        if (log && !event.target.closest("#pvc-monitor-latest")) {
            scrollState.logFollow = false;
            scrollState.logTop = log.scrollTop;
        }
    }, true);

    document.addEventListener("pointerdown", function (event) {
        var grip = event.target.closest && event.target.closest(".pvc-monitor-resize-grip");
        if (grip) {
            var panel = grip.closest(".pvc-monitor-panel");
            var rect = panel.getBoundingClientRect();
            resizeDrag = {pointerId: event.pointerId, panel: panel, startX: event.clientX,
                          startY: event.clientY, width: rect.width, height: rect.height};
            try { grip.setPointerCapture(event.pointerId); } catch (e) {}
            event.preventDefault(); event.stopPropagation();
            return;
        }
        var head = event.target.closest && event.target.closest(".pvc-monitor-head");
        var pill = event.target.closest && event.target.closest("#progress-monitor-toggle");
        if ((!head && !pill) || (head && event.target.closest(".pvc-monitor-actions"))) return;
        monitor = (head || pill).closest(".pvc-monitor");
        if (!monitor) return;
        drag = {pointerId: event.pointerId, startX: event.clientX, startY: event.clientY,
                baseX: saved.x || 0, baseY: saved.y || 0,
                originX: event.clientX, originY: event.clientY, moved: false,
                source: pill ? "pill" : "head"};
        if (pill) monitor.classList.add("is-pill-dragging");
        try { (head || pill).setPointerCapture(event.pointerId); } catch (e) {}
        if (head) event.preventDefault();
    });

    document.addEventListener("pointermove", function (event) {
        if (resizeDrag && event.pointerId === resizeDrag.pointerId) {
            var panel = resizeDrag.panel;
            var pill = document.getElementById("progress-monitor-toggle");
            var rect = panel.getBoundingClientRect();
            var pillRect = pill.getBoundingClientRect();
            var minWidth = Math.min(340, window.innerWidth - 16);
            var maxWidth = Math.max(minWidth, window.innerWidth - 8 - rect.left);
            var gap = parseFloat(panel.dataset.pvcPillGap || "6");
            var minHeight = Math.min(280, Math.max(220, pillRect.top - gap - topBoundary()));
            var maxHeight = Math.max(minHeight, pillRect.top - gap - topBoundary());
            var width = Math.max(minWidth, Math.min(maxWidth, resizeDrag.width + event.clientX - resizeDrag.startX));
            var height = Math.max(minHeight, Math.min(maxHeight, resizeDrag.height + event.clientY - resizeDrag.startY));
            panel.style.width = width + "px"; panel.style.height = height + "px";
            anchorPanelAbovePill(panel);
            savedSize = {width: Math.round(width), height: Math.round(height)};
            event.preventDefault();
            return;
        }
        if (!drag || event.pointerId !== drag.pointerId || !monitor) return;
        var dx = event.clientX - drag.startX;
        var dy = event.clientY - drag.startY;
        if (Math.abs(event.clientX - drag.originX) + Math.abs(event.clientY - drag.originY) > 5) drag.moved = true;
        var openPanel = monitor.querySelector(".pvc-monitor-panel.is-open");
        var rect = openPanel ? openPanel.getBoundingClientRect() : document.getElementById("progress-monitor-toggle").getBoundingClientRect();
        var nextX = drag.baseX + dx;
        var nextY = drag.baseY + dy;
        var pillRect = document.getElementById("progress-monitor-toggle").getBoundingClientRect();
        var minTop = topBoundary();
        var leftEdge = Math.min(rect.left, pillRect.left) + dx;
        var rightEdge = Math.max(rect.right, pillRect.right) + dx;
        var topEdge = Math.min(rect.top, pillRect.top) + dy;
        var pillBottom = pillRect.bottom + dy;
        if (leftEdge < 8) nextX += 8 - leftEdge;
        if (rightEdge > window.innerWidth - 8) nextX -= rightEdge - (window.innerWidth - 8);
        if (topEdge < minTop) nextY += minTop - topEdge;
        if (pillBottom > window.innerHeight - 12) nextY -= pillBottom - (window.innerHeight - 12);
        saved = {x: nextX, y: nextY};
        monitor.style.transform = "translate(" + nextX + "px," + nextY + "px)";
        drag.startX = event.clientX; drag.startY = event.clientY;
        drag.baseX = nextX; drag.baseY = nextY;
    });

    function endDrag(event) {
        if (resizeDrag && (event.pointerId == null || event.pointerId === resizeDrag.pointerId)) {
            resizeDrag = null;
            try { localStorage.setItem("pvc-monitor-size-v2", JSON.stringify(savedSize)); } catch (e) {}
            constrainMonitor();
            return;
        }
        if (!drag || (event.pointerId != null && event.pointerId !== drag.pointerId)) return;
        if (drag.source === "pill" && drag.moved) {
            suppressToggleClick = true;
            window.setTimeout(function () { suppressToggleClick = false; }, 350);
        }
        if (monitor) monitor.classList.remove("is-pill-dragging");
        drag = null;
        try { localStorage.setItem("pvc-monitor-position", JSON.stringify(saved)); } catch (e) {}
    }
    document.addEventListener("pointerup", endDrag);
    document.addEventListener("pointercancel", endDrag);
    window.addEventListener("resize", function () { window.requestAnimationFrame(constrainMonitor); });
    window.addEventListener("scroll", function () { window.requestAnimationFrame(constrainMonitor); }, {passive: true});

    document.addEventListener("click", function (event) {
        var draggedToggle = event.target.closest && event.target.closest("#progress-monitor-toggle");
        if (draggedToggle && suppressToggleClick) {
            suppressToggleClick = false;
            event.preventDefault();
            event.stopImmediatePropagation();
        }
    }, true);

    document.addEventListener("click", function (event) {
        var toggle = event.target.closest && event.target.closest("#progress-monitor-toggle");
        if (toggle) {
            // Dash toggles the class just after this native click handler.
            window.setTimeout(function () { setupResizablePanel(); constrainMonitor(); restoreScroll(); }, 40);
        }
        var latest = event.target.closest && event.target.closest("#pvc-monitor-latest");
        if (latest) {
            var latestLog = document.getElementById("pvc-monitor-log");
            scrollState.logFollow = true;
            if (latestLog) {
                latestLog.scrollTop = latestLog.scrollHeight;
                scrollState.logTop = latestLog.scrollTop;
            }
            return;
        }
        var button = event.target.closest && event.target.closest("#progress-monitor-copy");
        if (!button) return;
        var lines = Array.prototype.map.call(document.querySelectorAll("#pvc-monitor-log .pvc-log-line"), function (line) {
            return (line.innerText || line.textContent || "").trim();
        });
        var text = lines.join("\n");
        var done = function () {
            var old = button.textContent;
            button.textContent = "Copied";
            window.setTimeout(function () { button.textContent = old; }, 1400);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(done).catch(function () { fallbackCopy(text); done(); });
        } else {
            fallbackCopy(text); done();
        }
    });

    function fallbackCopy(text) {
        var area = document.createElement("textarea");
        area.value = text; area.setAttribute("readonly", "");
        area.style.position = "fixed"; area.style.opacity = "0";
        document.body.appendChild(area); area.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(area);
    }

    var observer = new MutationObserver(function () {
        applyPosition();
        window.requestAnimationFrame(function () { setupResizablePanel(); constrainMonitor(); restoreScroll(); });
    });
    function start() {
        observer.observe(document.body, {childList: true, subtree: true,
                         attributes: false});
        applyPosition(); setupResizablePanel(); constrainMonitor(); restoreScroll();
        window.setTimeout(constrainMonitor, 60);
    }
    if (document.body) start(); else document.addEventListener("DOMContentLoaded", start, {once: true});
})();
