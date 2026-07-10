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

    // Left: the column name.
    var children = [
        React.createElement(
            "span",
            {
                key: "name",
                style: {
                    fontFamily:
                        "SFMono-Regular, ui-monospace, Menlo, monospace",
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
        var fg = "#8a6d00";
        var bg = "#fff6e0";
        var bd = "#f0dfa8";

        children.push(
            React.createElement(
                "span",
                {
                    key: "pill",
                    style: {
                        marginInlineStart: "auto", // <- pushes pill to the right
                        flex: "0 0 auto",
                        fontFamily:
                            "-apple-system, BlinkMacSystemFont, 'Segoe UI', " +
                            "Roboto, Helvetica, Arial, sans-serif",
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
                tag
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
                gap: "10px",
            },
        },
        children
    );
};