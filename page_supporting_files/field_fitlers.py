from dash import html, dcc


# ─────────────────────────────────────────────────────────────────────────────
# Reusable multi-select dropdown block (matches the PV Material Pathway page)
# ─────────────────────────────────────────────────────────────────────────────
def _dropdown_block(label_text, dropdown_id, options, default_values):
    return html.Div(
        [
            html.Label(
                label_text,
                className="filter-label",
                style={
                    "fontWeight": "700",
                    "marginBottom": "8px",
                    "fontSize": "15px",
                    "display": "block",
                    "color": "#1f2937",
                },
            ),
            dcc.Dropdown(
                id=dropdown_id,
                options=options,
                value=default_values,
                multi=True,
                clearable=True,
                placeholder="Select…",
                className="filter-dropdown",
                style={"fontSize": "14px"},
            ),
        ],
        style={"flex": "1 1 300px", "minWidth": "260px"},
    )


def build_filters(types, advanced_extra=None):

    return html.Div(

        [

            # ------------- BASIC FILTERS (one row of dropdown pills) -------------
            html.Div(
                [
                    _dropdown_block(
                        "PV Technology",
                        "pv-tech-filter",
                        [{"label": t, "value": t} for t in types],
                        types,
                    ),
                    _dropdown_block(
                        "Climate Zone",
                        "pv-climate-filter",
                        [
                            {"label": "Moderate", "value": "Moderate"},
                            {"label": "Desert", "value": "Desert"},
                            {"label": "Hot & Humid", "value": "Hot & Humid"},
                            {"label": "Snow", "value": "Snow"},
                        ],
                        ["Moderate", "Desert", "Hot & Humid", "Snow"],
                    ),
                    _dropdown_block(
                        "Scope of Study",
                        "scope-filter",
                        [
                            {"label": "Module level", "value": "module level"},
                            {"label": "System level", "value": "system level"},
                        ],
                        ["module level", "system level"],
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "20px",
                    "flexWrap": "wrap",
                    "alignItems": "flex-start",
                },
            ),

            # ---------- ADVANCED FILTERS row (AI button left, toggle right) ----------

            html.Div(
                [

                    html.Details(

                        [

                            html.Summary(
                                "Advanced Filters",
                                className="advanced-summary"
                            ),

                    html.Div(

                        [

                            html.Div(
                                [
                                    html.Label(
                                        "Degradation rate (% / year)",
                                        className="filter-label"
                                    ),

                                    html.Div(
                                        [
                                            html.Label("Min"),
                                            dcc.Input(
                                                id="rate-min",
                                                type="number",
                                                value=-20,
                                                className="filter-input"
                                            ),

                                            html.Label("Max"),
                                            dcc.Input(
                                                id="rate-max",
                                                type="number",
                                                value=5,
                                                className="filter-input"
                                            ),
                                        ],
                                        className="filter-controls"
                                    ),
                                ],
                                className="filter-grid-row"
                            ),

                            html.Div(
                                [
                                    html.Label(
                                        "Study duration (years)",
                                        className="filter-label"
                                    ),

                                    html.Div(
                                        [
                                            html.Label("Min"),
                                            dcc.Input(
                                                id="duration-min",
                                                type="number",
                                                value=0,
                                                className="filter-input"
                                            ),

                                            html.Label("Max"),
                                            dcc.Input(
                                                id="duration-max",
                                                type="number",
                                                value=50,
                                                className="filter-input"
                                            ),
                                        ],
                                        className="filter-controls"
                                    ),
                                ],
                                className="filter-grid-row"
                            ),

                            html.Div(
                                [
                                    html.Label(
                                        "System capacity (kW)",
                                        className="filter-label"
                                    ),

                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="capacity-report-filter",
                                                options=[
                                                    {"label": "Reported", "value": "reported"},
                                                    {"label": "Not reported", "value": "not_reported"},
                                                ],
                                                value=["reported", "not_reported"],
                                                inline=True,
                                                className="filter-options"
                                            ),

                                            html.Label("Min"),
                                            dcc.Input(
                                                id="capacity-min",
                                                type="number",
                                                value=0,
                                                className="filter-input"
                                            ),

                                            html.Label("Max"),
                                            dcc.Input(
                                                id="capacity-max",
                                                type="number",
                                                value=500,
                                                className="filter-input"
                                            ),
                                        ],
                                        className="filter-controls wrap"
                                    ),
                                ],
                                className="filter-grid-row"
                            ),

                            html.Div(
                                [
                                    html.Label(
                                        "Faults reported",
                                        className="filter-label"
                                    ),

                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="faults-filter",
                                                options=[
                                                    {"label": "Reported", "value": "reported"},
                                                    {"label": "Not reported", "value": "not_reported"},
                                                ],
                                                value=["reported", "not_reported"],
                                                inline=True,
                                                className="filter-options"
                                            )
                                        ],
                                        className="filter-controls"
                                    ),
                                ],
                                className="filter-grid-row"
                            ),

                        ],
                        className="advanced-panel"
                    ),

                ],
                style={"flex": "0 0 auto"},

            ),

                ],
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "gap": "16px",
                    "marginTop": "14px",
                    "flexWrap": "wrap",
                },
            ),

        ],

        className="filter-card"
    )