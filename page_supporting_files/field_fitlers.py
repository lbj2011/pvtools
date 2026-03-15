from dash import html, dcc


def build_filters(types):

    return html.Div(

        [

            html.H4(
                "Refine Results with Filters",
                className="section-title"
            ),

            # ---------------- BASIC FILTERS ---------------- #

            html.Div(
                [
                    html.Label("PV Technology", className="filter-label"),

                    html.Div(
                        [
                            dcc.Checklist(
                                id="pv-tech-filter",
                                options=[{"label": t, "value": t} for t in types],
                                value=types,
                                inline=True,
                                className="filter-options"
                            )
                        ],
                        className="filter-controls wrap"
                    ),
                ],
                className="filter-grid-row"
            ),

            html.Div(
                [
                    html.Label("Climate Zone", className="filter-label"),

                    html.Div(
                        [
                            dcc.Checklist(
                                id="pv-climate-filter",
                                options=[
                                    {"label": "Moderate", "value": "Moderate"},
                                    {"label": "Desert", "value": "Desert"},
                                    {"label": "Hot & Humid", "value": "Hot & Humid"},
                                    {"label": "Snow", "value": "Snow"},
                                ],
                                value=["Moderate", "Desert", "Hot & Humid", "Snow"],
                                inline=True,
                                className="filter-options"
                            )
                        ],
                        className="filter-controls wrap"
                    ),
                ],
                className="filter-grid-row"
            ),

            html.Div(
                [
                    html.Label("Scope of Study", className="filter-label"),

                    html.Div(
                        [
                            dcc.Checklist(
                                id="scope-filter",
                                options=[
                                    {"label": "Module level", "value": "module level"},
                                    {"label": "System level", "value": "system level"},
                                ],
                                value=["module level", "system level"],
                                inline=True,
                                className="filter-options"
                            )
                        ],
                        className="filter-controls wrap"
                    ),
                ],
                className="filter-grid-row"
            ),

            # ---------------- ADVANCED FILTERS ---------------- #

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

                ]

            ),

        ],

        className="filter-card"
    )