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