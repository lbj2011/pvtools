from dash.dependencies import Input, Output, ALL, State
import dash
from page_supporting_files.pv_pathway_graph_builder import (
    build_elements,
    load_nodes, build_graph, to_cytoscape_positioned
)
from dash import html
import plotly.express as px
import ast
import re
import numpy as np
import plotly.graph_objects as go
from dash_model_viewer import DashModelViewer

from page_supporting_files.pv_pathway_layout import allc, FAULT_BUTTONS

allc = ['#8A257F','#D476EC','#3BB1FF', '#1E51BB']

CATEGORY_COLORS = {
    "stressor": allc[0],
    "mechanism": allc[1],
    "failure": allc[2],
    "performance_impact": allc[3],
    "performance_loss": allc[3],   # alias used by common pathway nodes
}

KNOWN_COMPONENTS = {"cell", "encapsulant", "glass", "front sheet", "backsheet"}
KNOWN_MECHANISMS = {"pid", "crack", "corrosion", "hot spot", "delamination", "moisture ingress", "thermal cycling"}

# ─────────────────────────────────────────────────────────────────────────────
# COMMON PATHWAY DATA
# Loaded once at module level; keyed by fault name.
# Add entries here as data for other faults becomes available.
# ─────────────────────────────────────────────────────────────────────────────
def _load_common_elements(fault):
    """Return Cytoscape elements for a given fault, or None if not available."""
    DATA_PATHS = {
        "PID":   "data/common_pathway_data/pid_data",
        "Crack": "data/common_pathway_data/crack_data",
        # "Hot spot":     "data/common_pathway_data/hotspot_data",    # not ready
        # "Delamination": "data/common_pathway_data/delamination_data", # not ready
    }
    path = DATA_PATHS.get(fault)
    if path is None:
        return None
    nodes = load_nodes(path)
    G = build_graph(nodes)
    return to_cytoscape_positioned(nodes, G)

# Pre-load available faults so the callback is instant
_COMMON_ELEMENTS_CACHE = {f: _load_common_elements(f) for f in FAULT_BUTTONS}

# Raw node dicts keyed by fault → {node_id: node_dict} for wiki detail lookup
_COMMON_NODES_CACHE: dict = {}
for _f in FAULT_BUTTONS:
    _path = {
        "PID":   "data/common_pathway_data/pid_data",
        "Crack": "data/common_pathway_data/crack_data",
    }.get(_f)
    if _path:
        _COMMON_NODES_CACHE[_f] = load_nodes(_path)
    else:
        _COMMON_NODES_CACHE[_f] = {}


def parse_list(x):
    if isinstance(x, list):
        return [str(i).lower() for i in x]
    if isinstance(x, np.ndarray):
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

    # EID → DOI lookup for source chip links
    _EID_DOI = dict(zip(DF["eid"].astype(str), DF["doi"].astype(str)))

    # EID → "LastName et al." short author label
    def _first_author_label(author_str):
        if not author_str or str(author_str) == "nan":
            return ""
        first = str(author_str).split(";")[0].strip()   # "LastName, FirstName"
        last = first.split(",")[0].strip()               # "LastName"
        return last

    _EID_AUTHOR = {
        str(eid): _first_author_label(auth)
        for eid, auth in zip(DF["eid"], DF["author_names"])
    }
    _EID_YEAR = {
        str(eid): str(int(yr)) if str(yr) != "nan" else ""
        for eid, yr in zip(DF["eid"], DF["year"])
    }

    # =========================================================================
    # MAIN CALLBACK — per-paper pathway graph
    # =========================================================================
    @app.callback(
        Output('graph', 'elements'),
        Output('pathway-buttons', 'children'),
        Output('selected-file', 'data'),
        Output('map-detail', 'children'),
        Output('map-detail', 'style'),

        Input({'type': 'file-btn', 'index': ALL}, 'n_clicks'),
        Input({'type': 'pathway-btn', 'index': ALL}, 'n_clicks'),
        Input("map", "clickData"),
        Input("component-filter", "value"),
        Input("mechanism-filter", "value"),
        Input("map-close-btn", "n_clicks"),

        State('selected-file', 'data'),
        State('map-detail', 'children'),
        State('map-detail', 'style'),
    )
    def update_graph(file_clicks, pathway_clicks, map_click, component_values, mechanism_values,
                 close_clicks, stored_file, prev_map_children, prev_map_style):

        ctx = dash.callback_context

        if not ctx.triggered:
            return [], [], stored_file, "", {"display": "none"}

        triggered = ctx.triggered_id

        # ── close button: hide the detail panel ───────────────────────
        if triggered == "map-close-btn":
            return dash.no_update, dash.no_update, stored_file, "", {"display": "none"}

        if component_values == [] or mechanism_values == []:
            return [], [], stored_file, "", {"display": "none"}

        file_idx = stored_file if stored_file is not None else 0
        pathway_idx = 0

        print(f'file id:{stored_file}')

        map_detail_children = prev_map_children
        map_detail_style = prev_map_style

        # ── map click ─────────────────────────────────────────────────
        if triggered == "map":
            if map_click:
                point = map_click["points"][0]

                if "customdata" in point:
                    file_idx = point["customdata"][0]
                else:
                    lat = point["lat"]
                    lon = point["lon"]
                    match = DF[(DF["latitude"] == lat) & (DF["longitude"] == lon)]
                    if match.empty:
                        return [], [], stored_file, ""
                    file_idx = match.index[0]

                pathway_idx = 0
                row = DF.iloc[file_idx]

                map_detail_children = [
                    # ── close button ──────────────────────────────────
                    html.Button(
                        "✕",
                        id="map-close-btn",
                        n_clicks=0,
                        style={
                            "position": "absolute",
                            "top": "10px",
                            "right": "10px",
                            "background": "rgba(255,255,255,0.7)",
                            "border": "1px solid rgba(0,0,0,0.12)",
                            "borderRadius": "50%",
                            "width": "26px",
                            "height": "26px",
                            "fontSize": "12px",
                            "color": "#6b7280",
                            "cursor": "pointer",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "padding": "0",
                            "lineHeight": "1",
                            "zIndex": 1000,
                        }
                    ),
                    html.H5(
                        row.get("title", row["eid"]),
                        style={
                            "marginBottom": "4px",
                            "display": "-webkit-box",
                            'font-weight': '600',
                            "WebkitLineClamp": 3,
                            "WebkitBoxOrient": "vertical",
                            "overflow": "hidden",
                            "paddingRight": "28px",
                        }
                    ),
                    html.P([
                        html.B("Year: "), str(row.get("year", "N/A")), html.Br(),
                        html.B("Location: "),
                        str(row.get("major_affiliation_city", "N/A")), ', ',
                        str(row.get("major_affiliation_country", "N/A")),
                        html.Br(),
                        html.B("DOI: "),
                        html.A(
                            str(row.get("doi", "N/A")),
                            href=f"https://doi.org/{row.get('doi', '')}",
                            target="_blank"
                        ),
                    ], style={"marginBottom": "4px", "fontSize": "13px", "color": "#222324"}),
                    html.Hr(),
                    html.Div([
                        html.B("Module components:"),
                        html.P(str(row.get("components", "N/A")),
                               style={"marginTop": "0px", "fontSize": "13px"}),
                        html.B("Major degradation:"),
                        html.P(str(row.get("major_mechanisms_faults", "N/A")),
                               style={"marginTop": "0px", "fontSize": "13px"}),
                    ], style={"marginBottom": "0px"}),
                    html.Div([
                        html.B("Summary:"),
                        html.P(str(row.get("summary_x", "N/A")),
                               style={"marginTop": "0px", "fontSize": "13px", "lineHeight": "1.5"})
                    ]),

                    # ── View Pathway button ───────────────────────────
                    html.Button(
                        [
                            html.Span("View Degradation Pathway"),
                            html.Span(" →", style={"marginLeft": "6px", "fontSize": "15px"}),
                        ],
                        id="open-pathway-modal-btn",
                        n_clicks=0,
                        className="view-pathway-btn",
                        style={
                            "marginTop": "0px",
                            "display": "inline-flex",
                            "alignItems": "center",
                            "padding": "9px 16px",
                            "background": "#1f2937",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "cursor": "pointer",
                            "letterSpacing": "0.3px",
                            "transition": "background 0.15s ease, transform 0.1s ease",
                        }
                    ),

                    DashModelViewer(
                        id="pv-model",
                        src="/assets/Untitled4.glb",
                        alt="A 3D model of an exploded PV module",
                        cameraControls=True,
                        cameraOrbit="70deg 70deg 50%",
                        fieldOfView="90deg",
                        ar=False,
                        style={
                            "width": "100%", "height": "150px",
                            "marginTop": "2px",
                            "marginLeft": "-15px",
                        }
                    ),
                ]

                map_detail_style = {
                    "position": "relative",
                    "width": "350px",
                    "background": "rgba(255,255,255, 0.55)",
                    "backdropFilter": "blur(12px)",
                    "border": "1px solid rgba(255,255,255,0.1)",
                    "boxShadow": "0 8px 32px rgba(240,240,240,0.3)",
                    "padding": "16px",
                    "borderRadius": "14px",
                    "color": "#131314",
                    "overflowY": "auto",
                    "flex": "1",
                    "zIndex": 999,
                    "pointerEvents": "auto",
                    "userSelect": "text",
                }

        # ── file button click ──────────────────────────────────────────
        elif isinstance(triggered, dict) and triggered["type"] == "file-btn":
            file_idx = triggered["index"]
            pathway_idx = 0

        # ── pathway button click ───────────────────────────────────────
        elif isinstance(triggered, dict) and triggered["type"] == "pathway-btn":
            pathway_idx = triggered["index"]

        # ── build per-paper graph ──────────────────────────────────────
        file_name = INDEX_MAP[file_idx]
        json_data = DATA[file_name]

        elements, num_pathways = build_elements(json_data, pathway_idx)

        pathway_buttons = [
            html.Button(
                f"Pathway {i+1}",
                id={'type': 'pathway-btn', 'index': i},
                style={
                    "backgroundColor": "#f3f4f6",
                    "color": "#1f2937",
                    "border": "1px solid #d1d5db",
                    "padding": "5px 12px",
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "letterSpacing": "0.5px",
                }
            )
            for i in range(num_pathways)
        ]

        return elements, pathway_buttons, file_idx, map_detail_children, map_detail_style

    # =========================================================================
    # PATHWAY MODAL — open/close + update title and stats
    # =========================================================================
    @app.callback(
        Output("pathway-modal", "is_open"),
        Output("modal-paper-title", "children"),

        Input("open-pathway-modal-btn", "n_clicks"),
        Input("pathway-modal", "is_open"),

        State("selected-file", "data"),
        State("graph", "elements"),
        prevent_initial_call=True,
    )
    def toggle_pathway_modal(open_clicks, is_open, file_idx, elements):
        ctx = dash.callback_context
        triggered = ctx.triggered_id

        if triggered == "open-pathway-modal-btn" and open_clicks:
            title = ""
            if file_idx is not None and file_idx in INDEX_MAP:
                file_name = INDEX_MAP[file_idx]
                row = DF[DF["eid"] == file_name]
                if not row.empty:
                    title = row.iloc[0].get("title", file_name)

            return True, title

        return False, dash.no_update


    # =========================================================================
    @app.callback(
        Output('common-pathway-graph', 'elements'),
        Output('common-pathway-placeholder', 'children'),
        Output('common-pathway-placeholder', 'style'),
        Output({'type': 'fault-btn', 'index': ALL}, 'style'),

        Input({'type': 'fault-btn', 'index': ALL}, 'n_clicks'),
    )
    def update_common_pathway(fault_clicks):
        ctx = dash.callback_context

        # Determine which fault is selected (default PID on initial load)
        selected = 'PID'
        if ctx.triggered and ctx.triggered_id:
            triggered = ctx.triggered_id
            if isinstance(triggered, dict) and triggered.get('type') == 'fault-btn':
                selected = triggered['index']

        # Build button styles (active vs inactive)
        btn_styles = []
        for label in FAULT_BUTTONS:
            is_active = (label == selected)
            btn_styles.append({
                "padding": "10px 28px",
                "borderRadius": "24px",
                "border": f"1.5px solid {'#1f2937' if is_active else '#6b7280'}",
                "background": "#1f2937" if is_active else "white",
                "color": "white" if is_active else "#6b7280",
                "fontWeight": "700",
                "fontSize": "16px",
                "cursor": "pointer",
                "transition": "all 0.15s ease",
            })

        # Fetch elements from cache
        elements = _COMMON_ELEMENTS_CACHE.get(selected)

        if elements is not None:
            # Data available — show graph, hide placeholder
            placeholder_text = ""
            placeholder_style = {"display": "none"}
        else:
            # Data not ready — empty graph + show placeholder message
            elements = []
            placeholder_text = f"Data for '{selected}' pathways is coming soon."
            placeholder_style = {
                "position": "absolute",
                "top": "50%", "left": "50%",
                "transform": "translate(-50%, -50%)",
                "color": "#aaa",
                "fontSize": "14px",
                "fontStyle": "italic",
                "textAlign": "center",
                "pointerEvents": "none",
                "display": "block",
            }

        return elements, placeholder_text, placeholder_style, btn_styles

    # =========================================================================
    # MAP FIGURE
    # =========================================================================
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

        if "components" not in df.columns or "major_mechanisms_faults" not in df.columns:
            raise ValueError("Expected columns missing in DF")

        def get_component_group(comps):
            parsed = parse_list(comps)
            for c in parsed:
                c_lower = c.lower()
                if c_lower in KNOWN_COMPONENTS:
                    return c_lower
            return "other"

        df["parsed_components"] = df["components"].apply(parse_list)
        df_exploded = df.explode("parsed_components")
        df_exploded = df_exploded[df_exploded["parsed_components"].notna()]
        df_exploded["parsed_components"] = df_exploded["parsed_components"].astype(str).str.lower()
        df_exploded["component_group"] = df_exploded["parsed_components"].apply(
            lambda x: x if x in KNOWN_COMPONENTS else "other"
        )
        df_exploded = df_exploded.reset_index(drop=False)

        if component_values:
            selected = [c.lower() for c in component_values]

            def match_components(comps):
                parsed = parse_list(comps)
                normal_match = any(sel in parsed for sel in selected if sel != "other")
                other_match = any(c not in KNOWN_COMPONENTS for c in parsed) if "other" in selected else False
                return normal_match or other_match

            df_exploded = df_exploded[df_exploded["component_group"].apply(match_components)]

        if mechanism_values:
            selected = [m.lower() for m in mechanism_values]

            def match_mechanisms(comps):
                parsed = parse_list(comps)
                normal_match = any(sel in parsed for sel in selected if sel != "other")
                other_match = any(m not in KNOWN_MECHANISMS for m in parsed) if "other" in selected else False
                return normal_match or other_match

            df = df[df["major_mechanisms_faults"].apply(match_mechanisms)]

        if component_values == [] or mechanism_values == []:
            return px.scatter_mapbox(lat=[], lon=[]).update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(zoom=2, center={"lat": 20, "lon": 0}),
                margin={"l": 0, "r": 0, "t": 0, "b": 0}
            )

        df = df.copy()
        df["index"] = df.index
        df = df.drop_duplicates(subset=["latitude", "longitude"])

        color_map = {
            "cell": "#0b62a1",
            "front sheet": "#37adf1",
            "encapsulant": "#93dff4",
            "glass": "#ec8bbc",
            "backsheet": "#9b48b5",
            "other": "#bec0c0"
        }

        fig = px.scatter_mapbox(
            df_exploded,
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
            color="component_group",
            color_discrete_map=color_map,
            zoom=2,
        )

        fig.update_traces(marker=dict(size=16, opacity=0.3))

        fig.update_layout(
            legend_title_text="Component Type",
            legend=dict(
                orientation="v",
                x=0.95, y=0.02,
                xanchor="right", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )

        if selected_idx is not None and selected_idx in df.index:
            row = df.loc[selected_idx]
            fig.add_scattermapbox(
                lat=[row["latitude"]],
                lon=[row["longitude"]],
                mode="markers",
                marker=dict(size=22, opacity=1, color="#F4BC05"),
                name="selected"
            )

        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            mapbox=dict(
                zoom=2,
                center={"lat": 20, "lon": 0},
                bounds=dict(west=-360, east=360, south=-60, north=70),
                # minZoom=1.5,
                # maxZoom=10,
            ),
            uirevision="constant"
        )

        return fig

    # =========================================================================
    # NODE HOVER CLASS TOGGLE — adds 'mouseover' class for CSS hover effect
    # Works for both the modal graph and the common pathway graph
    # =========================================================================
    @app.callback(
        Output("graph", "elements", allow_duplicate=True),
        Input("graph", "mouseoverNodeData"),
        Input("graph", "mouseoutNodeData"),
        State("graph", "elements"),
        prevent_initial_call=True,
    )
    def toggle_node_hover_class(over_data, out_data, elements):
        if not elements:
            return dash.no_update
        ctx = dash.callback_context
        triggered = ctx.triggered_id

        result = []
        for el in elements:
            el = dict(el)
            if "source" in el.get("data", {}):
                result.append(el)
                continue
            classes = set((el.get("classes") or "").split())
            classes.discard("mouseover")
            if triggered == "graph" and over_data and el["data"]["id"] == over_data.get("id"):
                classes.add("mouseover")
            el["classes"] = " ".join(classes)
            result.append(el)
        return result

    @app.callback(
        Output("common-pathway-graph", "elements", allow_duplicate=True),
        Input("common-pathway-graph", "mouseoverNodeData"),
        Input("common-pathway-graph", "mouseoutNodeData"),
        State("common-pathway-graph", "elements"),
        prevent_initial_call=True,
    )
    def toggle_common_node_hover_class(over_data, out_data, elements):
        if not elements:
            return dash.no_update
        ctx = dash.callback_context
        triggered = ctx.triggered_id

        result = []
        for el in elements:
            el = dict(el)
            if "source" in el.get("data", {}):
                result.append(el)
                continue
            classes = set((el.get("classes") or "").split())
            classes.discard("mouseover")
            if triggered == "common-pathway-graph" and over_data and el["data"]["id"] == over_data.get("id"):
                classes.add("mouseover")
            el["classes"] = " ".join(classes)
            result.append(el)
        return result

    # =========================================================================
    # GRAPH NODE DETAIL PANEL
    # =========================================================================
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
                style={"color": "#aaa", "fontSize": "13px", "fontStyle": "italic",
                       "textAlign": "center", "marginTop": "20px"}
            )

        triggered = ctx.triggered_id

        if (
            triggered == "map" or
            (isinstance(triggered, dict) and triggered["type"] in ["file-btn", "pathway-btn"])
        ):
            return html.Div(
                "Hover or click a node to see details",
                style={"color": "#999", "fontSize": "13px", "fontStyle": "italic"}
            )

        if triggered == "graph":
            data = hover_data if hover_data else click_data

            if not data:
                return html.Div(
                    "Hover or click a node to see details",
                    style={"color": "#999", "fontSize": "13px", "fontStyle": "italic",
                           "textAlign": "center", "marginTop": "20px", "minHeight": "200px"}
                )

            return html.Div([
                html.P(data["label"], style={
                    "marginBottom": "6px", "fontWeight": "700",
                    "fontSize": "17px", "lineHeight": "1.4",
                    "color": "#1f2937", "letterSpacing": "0.2px",
                }),
                html.Div([
                    html.Div(style={
                        "width": "11px", "height": "11px",
                        "backgroundColor": CATEGORY_COLORS.get(data["category"], "#999"),
                        "borderRadius": "3px", "marginRight": "7px",
                        "flexShrink": "0",
                    }),
                    html.Span(
                        data["category"].replace("_", " ").title(),
                        style={
                            "fontSize": "13px",
                            "color": CATEGORY_COLORS.get(data["category"], "#999"),
                            "fontWeight": "600", "letterSpacing": "0.3px"
                        }
                    )
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
                html.Hr(style={"borderColor": "#eee", "margin": "8px 0"}),
                html.P(data.get("title", ""), style={
                    "color": "#666", "fontSize": "14px",
                    "lineHeight": "1.7", "marginTop": "8px"
                })
            ])

    # =========================================================================
    # MAP SUMMARY BADGE
    # =========================================================================
    @app.callback(
        Output("map-summary", "children"),
        Output("map-summary", "style"),

        Input("component-filter", "value"),
        Input("mechanism-filter", "value"),
    )
    def update_map_summary(component_values, mechanism_values):

        df = DF.copy()

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

        if count == 0:
            return (
                html.Div([
                    html.B("No data selected", style={"fontSize": "15px"}),
                    html.P("Please adjust filters to show data on map.",
                           style={"fontSize": "13px", "color": "#777", "marginTop": "6px"})
                ]),
                {"display": "block"}
            )

        return (
            html.Div([
                html.B(f"{count} data points selected", style={"fontSize": "15px"}),
            ]),
            {
                "background": "rgba(220,234,247, 0.55)",
                "backdropFilter": "blur(12px)",
                "border": "1px solid rgba(255,255,255,0.1)",
                "boxShadow": "0 8px 30px rgba(200,200,200,0.2)",
                "padding": "16px",
                "borderRadius": "14px",
                "color": "#42454c",
                "width": "350px",
                "zIndex": 20,
            }
        )

    # =========================================================================
    # COMMON PATHWAY WIKI PANEL
    # Triggered by: graph hover/click, connection chip click (via store), fault switch.
    # =========================================================================

    # ── Store: written by connection chip buttons, read by wiki callback ──
    @app.callback(
        Output("wiki-node-id", "data"),
        Input({'type': 'wiki-nav-btn', 'index': ALL}, 'n_clicks'),
        State({'type': 'wiki-nav-btn', 'index': ALL}, 'id'),
        prevent_initial_call=True,
    )
    def navigate_wiki(n_clicks_list, id_list):
        ctx = dash.callback_context
        if not ctx.triggered or not any(n for n in n_clicks_list if n):
            return dash.no_update
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "wiki-nav-btn":
            return triggered["index"]
        return dash.no_update

    @app.callback(
        Output("common-pathway-wiki", "children"),

        Input("common-pathway-graph", "mouseoverNodeData"),
        Input("common-pathway-graph", "tapNodeData"),
        Input("wiki-node-id", "data"),
        Input({'type': 'fault-btn', 'index': ALL}, 'n_clicks'),
    )
    def update_common_wiki(hover_data, click_data, wiki_nav_id, fault_clicks):

        _PLACEHOLDER = html.Div(
            "Hover or click a node to see details",
            style={
                "color": "#aaa", "fontSize": "13px",
                "fontStyle": "italic", "textAlign": "center",
                "marginTop": "20px",
            }
        )

        ctx = dash.callback_context
        if not ctx.triggered:
            return _PLACEHOLDER

        triggered = ctx.triggered_id

        # Fault button switched — reset panel
        if isinstance(triggered, dict) and triggered.get("type") == "fault-btn":
            return _PLACEHOLDER

        # Determine node_id to display:
        # priority: connection chip nav > graph hover > graph click
        node_id = None
        if triggered == "wiki-node-id" and wiki_nav_id:
            node_id = wiki_nav_id
        else:
            node_data = hover_data or click_data
            if node_data:
                node_id = node_data.get("id", "")

        if not node_id:
            return _PLACEHOLDER

        # Look up the node dict
        node = None
        for fault_nodes in _COMMON_NODES_CACHE.values():
            if node_id in fault_nodes:
                node = fault_nodes[node_id]
                break

        if node is None:
            return html.Div(f"No details found for: {node_id}",
                            style={"color": "#aaa", "fontSize": "13px"})

        import re

        # ── colour helpers ─────────────────────────────────────────────
        node_type  = node.get("node_type", "")
        type_color = CATEGORY_COLORS.get(node_type, "#999")

        # Map node_type → colour for connection chips
        # (uses same CATEGORY_COLORS; "performance_loss" maps to performance_impact colour)
        def node_type_color(nt):
            return CATEGORY_COLORS.get(nt, CATEGORY_COLORS.get("performance_impact", "#999"))

        def section_header(text):
            return html.P(text, style={
                "fontWeight": "700",
                "fontSize": "11px",
                "color": "#6b7280",
                "marginBottom": "6px",
                "marginTop": "14px",
                "textTransform": "uppercase",
                "letterSpacing": "0.8px",
            })

        def divider():
            return html.Hr(style={"borderColor": "#f0f0f0", "margin": "10px 0"})

        # ── type badge ─────────────────────────────────────────────────
        type_badge = html.Div([
            html.Div(style={
                "width": "10px", "height": "10px",
                "backgroundColor": type_color,
                "borderRadius": "3px",
                "marginRight": "6px",
                "flexShrink": "0",
            }),
            html.Span(
                node_type.replace("_", " ").title(),
                style={"fontSize": "12px", "color": type_color, "fontWeight": "600"}
            )
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})

        # ── title ──────────────────────────────────────────────────────
        title_block = html.H5(
            node.get("title", node_id),
            style={"fontWeight": "700", "marginBottom": "6px", "lineHeight": "1.4"}
        )

        # ── intro ──────────────────────────────────────────────────────
        intro = node.get("introduction", "").strip()
        intro_block = html.P(intro, style={
            "fontSize": "13px", "color": "#555", "lineHeight": "1.6", "marginBottom": "6px"
        }) if intro else None

        # ── connections (ABOVE key findings) ───────────────────────────
        parents  = node.get("parent_nodes", [])
        children = node.get("child_nodes", [])

        def conn_chip(n):
            """Clickable chip coloured by node category."""
            nt    = n.get("node_type", "")
            color = node_type_color(nt)
            label = n.get("display_name", n.get("id", ""))
            nid   = n.get("id", "")
            return html.Button(
                [
                    html.Span(
                        nt.replace("_", " ").title(),
                        style={
                            "fontSize": "9px",
                            "fontWeight": "700",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.4px",
                            "color": color,
                            "display": "block",
                            "marginBottom": "2px",
                        }
                    ),
                    html.Span(label, style={"fontSize": "12px", "color": "#1f2937"}),
                ],
                id={'type': 'wiki-nav-btn', 'index': nid},
                n_clicks=0,
                className="wiki-conn-chip",
                style={
                    "display": "inline-block",
                    "background": "white",
                    "borderRadius": "8px",
                    "padding": "5px 10px",
                    "fontSize": "12px",
                    "border": f"1.5px solid {color}",
                    "borderLeft": f"4px solid {color}",
                    "marginRight": "6px",
                    "marginBottom": "6px",
                    "textAlign": "left",
                    "cursor": "pointer",
                    "lineHeight": "1.3",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                    "width": "100%",
                    "--chip-hover-bg": f"{color}18",  # hex color + 18 = ~10% opacity
                }
            )

        conn_section = []
        if parents or children:
            conn_section = [divider(), section_header("Connections")]
            if parents:
                conn_section += [
                    html.P("← From", style={"fontSize": "11px", "color": "#9ca3af",
                                            "marginBottom": "6px", "fontWeight": "600"}),
                    html.Div([conn_chip(p) for p in parents],
                             style={"marginBottom": "8px", "flexWrap": "wrap", "display": "flex"}),
                ]
            if children:
                conn_section += [
                    html.P("→ To", style={"fontSize": "11px", "color": "#9ca3af",
                                          "marginBottom": "6px", "fontWeight": "600"}),
                    html.Div([conn_chip(c) for c in children],
                             style={"flexWrap": "wrap", "display": "flex"}),
                ]

        # ── key findings ───────────────────────────────────────────────
        findings = node.get("key_findings", [])
        finding_blocks = []

        def clean_detail_text(text):
            """Strip all citation markers and markdown from display text."""
            # Strip [[path|label]] inline refs (PID format)
            text = re.sub(r'\[\[.*?\]\]', '', text)
            # Strip [^key] footnote markers (crack format)
            text = re.sub(r'\[\^[^\]]+\]', '', text)
            # Strip **bold**
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            # Strip $math$
            text = re.sub(r'\$.*?\$', '', text)
            # Strip trailing --- separators
            text = re.sub(r'\n---\s*$', '', text)
            return text.strip()

        def parse_inline_sources(text):
            """Extract sources from [[path#pageN|label]] inline refs (PID format)."""
            sources = []
            pattern = r'\[\[([^\]|]+?)(?:#page=(\d+))?\|([^\]]+?)\]\]'
            for match in re.finditer(pattern, text):
                raw_path = match.group(1)
                page     = match.group(2) or ""
                label    = re.sub(r',\s*p\.\d+$', '', match.group(3)).strip()

                eid_match = re.search(r'(2-s2\.0-\d+)', raw_path)
                eid = eid_match.group(1) if eid_match else ""

                year_match = re.search(r'-(\d{4})-', raw_path)
                year = year_match.group(1) if year_match else ""

                fname = raw_path.split("/")[-1].replace(".pdf", "")
                after_year = re.sub(r'^.*?\d{4}-[A-Z]-', '', fname).strip()
                words = after_year.split()
                short_title = " ".join(words[:5]) + ("…" if len(words) > 5 else "")

                sources.append({"eid": eid, "label": label, "short": short_title, "year": year})
            return sources

        def parse_structured_sources(source_locations):
            """Extract sources from source_locations array (crack format)."""
            seen = set()
            sources = []
            for loc in (source_locations or []):
                src_path = loc.get("source_path", "")
                if not src_path or src_path == "_placeholder":
                    continue

                eid_match = re.search(r'(2-s2\.0-\d+)', src_path)
                eid = eid_match.group(1) if eid_match else ""

                if eid in seen:
                    continue
                seen.add(eid)

                # Prefer year from DF, fall back to filename
                year = _EID_YEAR.get(eid, "")
                if not year:
                    year_match = re.search(r'-(\d{4})-', src_path)
                    year = year_match.group(1) if year_match else ""

                # Build "LastName et al. (year)" — prefer DF author, fall back to loc.authors
                last_name = _EID_AUTHOR.get(eid, "")
                if not last_name:
                    raw_authors = loc.get("authors") or ""
                    last_name = raw_authors.split(",")[0].strip() if raw_authors else ""
                if last_name:
                    # Add "et al." if multiple authors in DF
                    author_str = str(DF[DF["eid"] == eid]["author_names"].values[0]) if eid in _EID_AUTHOR else ""
                    et_al = " et al." if ";" in author_str else ""
                    label = f"{last_name}{et_al} ({year})" if year else f"{last_name}{et_al}"
                else:
                    # Last resort: first word of filename title
                    fname = src_path.split("/")[-1].replace(".json", "").replace(".pdf", "")
                    after_year = re.sub(r'^.*?\d{4}-[A-Z]-', '', fname).replace("_", " ").strip()
                    first_word = after_year.split()[0] if after_year else "Unknown"
                    label = f"{first_word} et al. ({year})" if year else first_word

                # Tooltip: full title from filename
                fname = src_path.split("/")[-1].replace(".json", "").replace(".pdf", "")
                after_year = re.sub(r'^.*?\d{4}-[A-Z]-', '', fname).replace("_", " ").strip()
                words = after_year.split()
                short_title = " ".join(words[:5]) + ("…" if len(words) > 5 else "")

                sources.append({"eid": eid, "label": label, "short": short_title, "year": year})
            return sources

        def source_chip(src):
            doi = _EID_DOI.get(src["eid"], "")
            href = f"https://doi.org/{doi}" if doi else None
            return html.A(
                html.Span(src["label"], style={"fontWeight": "500"}),
                href=href,
                target="_blank",
                title=src["short"],
                className="wiki-source-chip",
                style={
                    "display": "inline-block",
                    "background": "#f3f4f6",
                    "border": "1px solid #d1d5db",
                    "borderRadius": "12px",
                    "padding": "2px 10px",
                    "fontSize": "10px",
                    "color": "#374151",
                    "marginRight": "4px",
                    "marginTop": "6px",
                    "fontWeight": "500",
                    "whiteSpace": "nowrap",
                    "textDecoration": "none",
                    "cursor": "pointer" if href else "default",
                    "transition": "background 0.15s ease, color 0.15s ease",
                }
            )

        for f in findings:
            claim  = f.get("claim", "").strip()
            detail = f.get("detail", "").strip()
            if not claim and not detail:
                continue

            detail_clean = clean_detail_text(detail)

            # Try inline [[...]] refs first (PID format), fall back to source_locations (crack format)
            inline_sources = parse_inline_sources(detail)
            if inline_sources:
                sources = inline_sources
            else:
                sources = parse_structured_sources(f.get("source_locations", []))

            source_row = html.Div(
                [source_chip(s) for s in sources],
                style={"display": "flex", "flexWrap": "wrap", "marginTop": "6px"}
            ) if sources else None

            finding_blocks.append(html.Div([
                html.P(claim, style={
                    "fontWeight": "600",
                    "fontSize": "13px",
                    "color": type_color,
                    "marginBottom": "3px",
                }),
                html.P(detail_clean, style={
                    "fontSize": "12px",
                    "color": "#555",
                    "lineHeight": "1.6",
                    "marginBottom": "0",
                    "whiteSpace": "pre-line",
                }),
                *([source_row] if source_row else []),
            ], style={
                "padding": "10px 12px",
                "background": "#f9fafb",
                "borderRadius": "8px",
                "borderLeft": f"3px solid {type_color}",
                "marginBottom": "8px",
            }))

        findings_section = [
            divider(),
            section_header("Key Findings"),
            *finding_blocks,
        ] if finding_blocks else []

        # ── assemble: findings first, then connections ─────────────────
        return html.Div([
            type_badge,
            title_block,
            *([intro_block] if intro_block else []),
            *findings_section,   # ← findings first
            *conn_section,       # ← connections below
        ])