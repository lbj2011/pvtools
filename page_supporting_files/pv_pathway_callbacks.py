from dash.dependencies import Input, Output, ALL, State
import dash
from page_supporting_files.pv_pathway_graph_builder import build_elements
from dash import html
import plotly.express as px
import ast
import re
import numpy as np

allc = ['#8A257F','#D476EC','#3BB1FF', '#1E51BB']

CATEGORY_COLORS = {
    "stressor": allc[0],
    "mechanism": allc[1],
    "failure": allc[2],
    "performance_impact": allc[3],
}

KNOWN_COMPONENTS = {"cell", "encapsulant", "glass", "front sheet", "backsheet"}
KNOWN_MECHANISMS = {"pid", "crack", "corrosion", "hot spot", "delamination", "moisture ingress", "thermal cycling"}

def parse_list(x):
    if isinstance(x, list):
        return [str(i).lower() for i in x]

    if isinstance(x, np.ndarray):   # 👈 THIS is your key case
        return [str(i).lower() for i in x]

    if isinstance(x, str):
        return [x.lower()]

    return []

def is_valid_graph(lst):
    if not isinstance(lst, (list, tuple, np.ndarray)):
        return False

    ids = {d.get("id") for d in lst}

    for d in lst:
        for cid in d.get("child_ids", []):
            if cid not in ids:
                return False

    return True


def register_callbacks(app, DATA, INDEX_MAP, DF, file_list):

    # =========================
    # MAIN CALLBACK (UNIFIED)
    # =========================
    @app.callback(
        Output('graph', 'elements'),
        Output('pathway-buttons', 'children'),
        Output('selected-file', 'data'),
        Output('map-detail', 'children'),
        Output('map-detail', 'style'),

        Input({'type': 'file-btn', 'index': ALL}, 'n_clicks'),
        Input({'type': 'pathway-btn', 'index': ALL}, 'n_clicks'),
        Input("map", "clickData"),
        Input("component-filter", "value"),        # 👈 ADD
        Input("mechanism-filter", "value"),        # 👈 ADD

        State('selected-file', 'data'),
        State('map-detail', 'children'),
        State('map-detail', 'style'),
    )
    def update_graph(file_clicks, pathway_clicks, map_click, component_values, mechanism_values, stored_file,
                 prev_map_children, prev_map_style):

        ctx = dash.callback_context

        if not ctx.triggered:
            return [],  [], stored_file, "", {"display": "none"}

        triggered = ctx.triggered_id

        # 🔥 If no filters → clear everything
        if component_values == [] or mechanism_values == []:
            return [], [], stored_file, "", {"display": "none"}

        file_idx = stored_file if stored_file is not None else 0

        pathway_idx = 0

        print(f'file id:{stored_file}')

        map_detail_children = prev_map_children
        map_detail_style = prev_map_style

        # =========================
        # MAP CLICK
        # =========================
        if triggered == "map":
            if map_click:
                point = map_click["points"][0]

                # ✅ robust index extraction
                if "customdata" in point:
                    file_idx = point["customdata"][0]
                else:
                    lat = point["lat"]
                    lon = point["lon"]

                    match = DF[
                        (DF["latitude"] == lat) &
                        (DF["longitude"] == lon)
                    ]

                    if match.empty:
                        return [], [], stored_file, ""

                    file_idx = match.index[0]

                pathway_idx = 0

                row = DF.iloc[file_idx]

                # ✅ CLEAN DETAIL BOX
                row = DF.iloc[file_idx]

                map_detail_children = [

                    # Title
                    html.H5(
                        row.get("title", row["eid"]),
                        style={
                            "marginBottom": "8px",
                            "display": "-webkit-box",
                            "WebkitLineClamp": 5,  # number of lines
                            "WebkitBoxOrient": "vertical",
                            "overflow": "hidden"
                        }
                    ),

                    # Year + DOI + Location
                    html.P(
                        [
                            html.B("Year: "), str(row.get("year", "N/A")), html.Br(),

                            html.B("DOI: "),
                            html.A(
                                str(row.get("doi", "N/A")),
                                href=f"https://doi.org/{row.get('doi', '')}",
                                target="_blank"  # opens in new tab
                            ),
                            html.Br(),

                            html.B("City: "), str(row.get("major_affiliation_city", "N/A")), html.Br(),
                            html.B("Country: "), str(row.get("major_affiliation_country", "N/A"))
                        ],
                        style={
                            "marginBottom": "10px",
                            "fontSize": "13px",
                            "color": "#D9D9D9"
                        }
                    ),

                    html.Hr(),

                    # Components
                    html.Div(
                        [
                            html.B("Module components:"),
                            html.P(
                                str(row.get("components", "N/A")),
                                style={"marginTop": "4px", "fontSize": "13px"}
                            ),
                            html.B("Major degradation:"),
                            html.P(
                                str(row.get("major_mechanisms_faults", "N/A")),
                                style={"marginTop": "4px", "fontSize": "13px"}
                            )
                        ],
                        style={"marginBottom": "10px"}
                    ),

                    # Summary
                    html.Div(
                        [
                            html.B("Summary:"),
                            html.P(
                                str(row.get("summary_x", "N/A")),
                                style={"marginTop": "4px", "fontSize": "13px", "lineHeight": "1.5"}
                            )
                        ]
                    )
                ]

                map_detail_style = {
                    "width": "300px",

    # 🔥 GLASS EFFECT
    "background": "rgba(20, 96,130, 0.55)",
    "backdropFilter": "blur(12px)",

    # ✨ soft border glow
    "border": "1px solid rgba(255,255,255,0.1)",

    # depth
    "boxShadow": "0 8px 32px rgba(0,0,0,0.6)",

    "padding": "16px",
    "borderRadius": "14px",

    # text color for dark theme
    "color": "#ffffff",

    "overflowY": "auto",
    "flex": "1",
    "zIndex": 999,
                    
                }

        # =========================
        # FILE BUTTON CLICK
        # =========================
        elif isinstance(triggered, dict) and triggered["type"] == "file-btn":
            file_idx = triggered["index"]
            pathway_idx = 0

        # =========================
        # PATHWAY BUTTON CLICK
        # =========================
        elif isinstance(triggered, dict) and triggered["type"] == "pathway-btn":
            pathway_idx = triggered["index"]

        # =========================
        # BUILD GRAPH
        # =========================
        file_name = INDEX_MAP[file_idx]
        json_data = DATA[file_name]

        elements, num_pathways = build_elements(json_data, pathway_idx)

        pathway_buttons = [
            html.Button(
                f"Pathway {i+1}",
                id={'type': 'pathway-btn', 'index': i},
                style={
                    "margin": "4px",
                    "backgroundColor": "#a8aeb4",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 16px",   # 👈 bigger button
                    "fontSize": "16px",      # 👈 larger text
                    "fontWeight": "600",
                    "borderRadius": "6px",
                    "cursor": "pointer"
                }
            )
            for i in range(num_pathways)
        ]

        return (
            elements,
            pathway_buttons,
            file_idx,
            map_detail_children,
            map_detail_style
        )

    # =========================
    # MAP FIGURE
    # =========================
    @app.callback(
    Output("map", "figure", allow_duplicate=True),
        Input("selected-file", "data"),
        Input("component-filter", "value"),
        Input("mechanism-filter", "value"),
        prevent_initial_call=True
    )
    def update_map(selected_idx, component_values, mechanism_values):

        print(selected_idx)

        df = DF.copy()

        # df = df[df["pathways_graph"].apply(is_valid_graph)]

        # print(len(df))

        # Ensure columns exist
        if "components" not in df.columns or "major_mechanisms_faults" not in df.columns:
            raise ValueError("Expected columns missing in DF")
    
        # -------------------------
        # FILTER: components
        # -------------------------
        # Components
        if component_values:
            selected = [c.lower() for c in component_values]

            def match_components(comps):
                parsed = parse_list(comps)

                # normal match
                normal_match = any(sel in parsed for sel in selected if sel != "other")

                # "other" match = anything not in known list
                other_match = False
                if "other" in selected:
                    other_match = any(c not in KNOWN_COMPONENTS for c in parsed)

                return normal_match or other_match

            df = df[df["components"].apply(match_components)]

        # -------------------------
        # FILTER: mechanisms
        # -------------------------
        if mechanism_values:
            selected = [m.lower() for m in mechanism_values]

            def match_mechanisms(comps):
                parsed = parse_list(comps)

                normal_match = any(sel in parsed for sel in selected if sel != "other")

                other_match = False
                if "other" in selected:
                    other_match = any(m not in KNOWN_MECHANISMS for m in parsed)

                return normal_match or other_match

            df = df[df["major_mechanisms_faults"].apply(match_mechanisms)]

        # -------------------------
        # 🔥 CRITICAL: preserve columns
        # -------------------------
        if component_values == [] or mechanism_values == []:
            return px.scatter_mapbox(lat=[], lon=[]).update_layout(
                mapbox_style="carto-darkmatter",
                mapbox=dict(
                    zoom=2,
                    center={"lat": 20, "lon": 0}
                ),
                margin={"l":0, "r":0, "t":0, "b":0}
            )

        # restore index column AFTER filtering
        df = df.copy()
        df["index"] = df.index

        # optional: remove duplicates
        df = df.drop_duplicates(subset=["latitude", "longitude"])

        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            hover_name="eid",
            hover_data={
                "year": True,
                "major_affiliation_city": True,
                "major_affiliation_country": True,
                "latitude": False,
                "longitude": False
            },
            custom_data=["index"],
            zoom=2
        )

        # =========================
        # 🔵 Marker style
        # =========================
        fig.update_traces(
            marker=dict(
                size=16,      # 👈 larger points
                opacity=0.5,   # 👈 transparent
                color="#00B0F0"   # 👈 change here
            )
        )

        # =========================
        # 🔴 Highlight selected point
        # =========================
        if selected_idx is not None and selected_idx in df.index:
            row = df.loc[selected_idx]

            fig.add_scattermapbox(
                lat=[row["latitude"]],
                lon=[row["longitude"]],
                mode="markers",
                marker=dict(
                    size=22,
                    opacity=1,
                    color="#F234B3"
                ),
                name="selected"
            )

        # =========================
        # 🌍 Map behavior (NO REPEAT)
        # =========================
        fig.update_layout(
            # mapbox_style="carto-positron",  # clean + light
            mapbox_style="carto-darkmatter",
            margin={"l":0, "r":0, "t":0, "b":0},

            mapbox=dict(
                zoom=2,
                center={"lat": 20, "lon": 0},
            ),
            showlegend=False, 
            uirevision="constant"
        )

        return fig

    # =========================
    # GRAPH NODE DETAIL PANEL
    # =========================
    @app.callback(
        Output("detail-panel", "children"),

        Input("graph", "tapNodeData"),
        Input("graph", "mouseoverNodeData"),
        Input({'type': 'file-btn', 'index': ALL}, 'n_clicks'),
        Input({'type': 'pathway-btn', 'index': ALL}, 'n_clicks'),
        Input("map", "clickData"),
    )
    def show_details(click_data, hover_data, file_clicks, pathway_clicks, map_click):

        ctx = dash.callback_context

        if not ctx.triggered:
            return html.Div(
                "Click a paper on the map to show the pathway",
                style={
                    "color": "#aaa",
                    "fontSize": "13px",
                    "fontStyle": "italic",
                    "textAlign": "center",
                    "marginTop": "20px"
                }
            )

        triggered = ctx.triggered_id

        # ✅ 1. If pathway / file / map triggered → show default message
        if (
            triggered == "map" or
            (isinstance(triggered, dict) and triggered["type"] in ["file-btn", "pathway-btn"])
        ):
            return html.Div(
                "Hover or click a node to see details",
                style={
                    "color": "#999",
                    "fontSize": "13px",
                    "fontStyle": "italic"
                }
            )

        # ✅ 2. Only show node details when graph triggered
        if triggered == "graph":
            data = hover_data if hover_data else click_data

            if not data:
                return html.Div(
                    "Hover or click a node to see details",
                    style={
                        "color": "#999",
                        "fontSize": "13px",
                        "fontStyle": "italic",
                        "textAlign": "center",
                        "marginTop": "20px",
                        "minHeight": "200px"   # ✅ prevents shrink
                    }
                )

            return html.Div([
        
                # 🔷 Title
                html.H4(
                    data["label"],
                    style={
                        "marginBottom": "8px",
                        "fontWeight": "600",
                        "letterSpacing": "0.3px"
                    }
                ),

                # 🧊 Category cube + label
                html.Div(
                [
                    html.Div(
                        style={
                            "width": "12px",
                            "height": "12px",
                            "backgroundColor": CATEGORY_COLORS.get(data["category"], "#999"),
                            "borderRadius": "3px",
                            "marginRight": "8px",
                            "boxShadow": "0 0 6px rgba(0,0,0,0.3)"
                        }
                    ),
                    html.Span(
                        data["category"].replace("_", " ").title(),
                        style={
                            "fontSize": "13px",
                            "color": CATEGORY_COLORS.get(data["category"], "#999"),  # 👈 match color
                            "fontWeight": "600",
                            "letterSpacing": "0.3px"
                        }
                    )
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "12px"
                }
            ),

                html.Hr(style={"borderColor": "#eee"}),

                # 📝 Description (gray / softer)
                html.P(
                    data.get("title", ""),
                    style={
                        "color": "#777",
                        "fontSize": "13px",
                        "lineHeight": "1.6",
                        "marginTop": "10px"
                    }
                )

            ]
            )
        
    @app.callback(
        Output("map-summary", "children"),
        Output("map-summary", "style"),

        Input("component-filter", "value"),
        Input("mechanism-filter", "value"),
    )
    def update_map_summary(component_values, mechanism_values):

        df = DF.copy()

        # -------------------------
        # apply SAME filters as map
        # -------------------------
        if component_values:
            selected = [c.lower() for c in component_values]

            def match_components(comps):
                parsed = parse_list(comps)
                normal_match = any(sel in parsed for sel in selected if sel != "other")
                other_match = any(c not in KNOWN_COMPONENTS for c in parsed) if "other" in selected else False
                return normal_match or other_match

            df = df[df["components"].apply(match_components)]

        if mechanism_values:
            selected = [m.lower() for m in mechanism_values]

            def match_mechanisms(comps):
                parsed = parse_list(comps)
                normal_match = any(sel in parsed for sel in selected if sel != "other")
                other_match = any(m not in KNOWN_MECHANISMS for m in parsed) if "other" in selected else False
                return normal_match or other_match

            df = df[df["major_mechanisms_faults"].apply(match_mechanisms)]

        count = len(df)

        # -------------------------
        # EMPTY STATE
        # -------------------------
        if count == 0:
            return (
                html.Div([
                    html.B("No data selected", style={"fontSize": "15px"}),
                    html.P(
                        "Please adjust filters to show data on map.",
                        style={"fontSize": "13px", "color": "#777", "marginTop": "6px"}
                    )
                ]),
                {"display": "block"}
            )

        # -------------------------
        # NORMAL STATE
        # -------------------------
        return (
            html.Div([
                html.B(f"{count} data points selected", style={"fontSize": "15px"}),

                html.P(
                    "Click a data point to view details.",
                    style={
                        "fontSize": "13px",
                        "color": "#BFBFBF",
                        "marginTop": "6px",
                        "marginBottom": "0px"   # 👈 add this
                    }
                )
            ]),
            {
                # 🔥 GLASS EFFECT
    "background": "rgba(0,112,192, 0.55)",
    "backdropFilter": "blur(12px)",

    # ✨ soft border glow
    "border": "1px solid rgba(255,255,255,0.1)",

    # depth
    "boxShadow": "0 8px 32px rgba(0,0,0,0.6)",

    "padding": "16px",
    "borderRadius": "14px",

    # text color for dark theme
    "color": "#ffffff",

                "width": "300px",
                "padding": "10px",
                "zIndex": 20,
            }
        )