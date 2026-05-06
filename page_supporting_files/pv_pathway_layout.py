from dash import html, dcc
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from dash_model_viewer import DashModelViewer

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS  (also imported by pv_pathway_callbacks)
# ─────────────────────────────────────────────────────────────────────────────
allc = ['#8A257F', '#D476EC', '#3BB1FF', '#1E51BB']

FAULT_BUTTONS = ['PID', 'Crack', 'Hot spot', 'Delamination']


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STYLESHEET
#
# attr_key   – node data field used for colour selectors
#              "category"  →  id='graph'                (per-paper pathway)
#              "type"      →  id='common-pathway-graph' (common pathway)
# perf_value – value of the performance node in that data set
#              "performance_impact"  /  "performance_loss"
# ─────────────────────────────────────────────────────────────────────────────
def create_stylesheet(attr_key="category", perf_value="performance_impact"):
    def sel(v):
        return f'[{attr_key} = "{v}"]'

    return [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'font-size': '14px',
                'font-weight': 'bold',
                'text-wrap': 'wrap',
                'text-max-width': '140px',
                'min-zoomed-font-size': 10,
                'color': 'white',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': '150px',
                'height': '50px',
                'shape': 'round-rectangle',
                'border-width': 1,
                'border-color': 'rgba(255,255,255,0.4)',
                'shadow-blur': 10,
                'shadow-color': 'rgba(0,0,0,0.3)',
                'shadow-opacity': 1,
                'shadow-offset-x': 0,
                'shadow-offset-y': 3,
                'transition-property': 'width, height, border-width, border-color, shadow-blur, shadow-opacity, shadow-offset-y, background-color',
                'transition-duration': '0.2s',
                'transition-timing-function': 'ease-out',
            }
        },
        # suppress default cyan tap flash
        {
            'selector': 'node:active',
            'style': {'overlay-opacity': 0}
        },
        # ── HOVER: glow + grow ───────────────────────────────────────
        {
            'selector': 'node.mouseover',
            'style': {
                'width': '165px',
                'height': '58px',
                'border-width': 3,
                'border-color': 'white',
                'shadow-blur': 30,
                'shadow-color': 'rgba(255,255,255,0.6)',
                'shadow-opacity': 1,
                'shadow-offset-x': 0,
                'shadow-offset-y': 0,
                'font-size': '15px',
            }
        },
        # ── SELECTED: strong accent ring ────────────────────────────
        {
            'selector': 'node:selected',
            'style': {
                'width': '165px',
                'height': '58px',
                'border-width': 4,
                'border-color': 'white',
                'shadow-blur': 40,
                'shadow-color': 'rgba(255,255,255,0.8)',
                'shadow-opacity': 1,
                'shadow-offset-x': 0,
                'shadow-offset-y': 0,
                'overlay-opacity': 0,
            }
        },
        {'selector': sel("stressor"),  'style': {'background-color': allc[0]}},
        {'selector': sel("mechanism"), 'style': {'background-color': allc[1]}},
        {'selector': sel("failure"),   'style': {'background-color': allc[2]}},
        {'selector': sel(perf_value),  'style': {'background-color': allc[3]}},
        # ── category-tinted glow on hover ───────────────────────────
        {
            'selector': f'[{attr_key} = "stressor"].mouseover',
            'style': {'shadow-color': allc[0]}
        },
        {
            'selector': f'[{attr_key} = "mechanism"].mouseover',
            'style': {'shadow-color': allc[1]}
        },
        {
            'selector': f'[{attr_key} = "failure"].mouseover',
            'style': {'shadow-color': allc[2]}
        },
        {
            'selector': f'[{attr_key} = "{perf_value}"].mouseover',
            'style': {'shadow-color': allc[3]}
        },
        {
            'selector': 'edge',
            'style': {
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'width': 2,
                'line-color': '#cbd5e1',
                'target-arrow-color': '#cbd5e1',
                'transition-property': 'width, line-color, target-arrow-color',
                'transition-duration': '0.2s',
            }
        },
        {
            'selector': 'edge:selected',
            'style': {
                'width': 3.5,
                'line-color': '#94a3b8',
                'target-arrow-color': '#94a3b8',
            }
        },
    ]


def create_stylesheet_common():
    """Larger nodes for the common pathway graph where more nodes are shown."""
    def sel(v):
        return f'[type = "{v}"]'

    return [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'font-size': '16px',
                'font-weight': 'bold',
                'text-wrap': 'wrap',
                'text-max-width': '160px',
                'min-zoomed-font-size': 10,
                'color': 'white',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': '180px',
                'height': '60px',
                'shape': 'round-rectangle',
                'border-width': 1,
                'border-color': 'rgba(255,255,255,0.4)',
                'shadow-blur': 10,
                'shadow-color': 'rgba(0,0,0,0.3)',
                'shadow-opacity': 1,
                'shadow-offset-x': 0,
                'shadow-offset-y': 3,
                'transition-property': 'width, height, border-width, border-color, shadow-blur, shadow-color, shadow-offset-y, opacity, background-color, background-opacity, border-opacity',
                'transition-duration': '0.2s',
                'transition-timing-function': 'ease-out',
            }
        },
        {
            'selector': 'node:active',
            'style': {'overlay-opacity': 0}
        },
        {
            'selector': 'node.mouseover',
            'style': {
                'width': '196px',
                'height': '68px',
                'border-width': 3,
                'border-color': 'white',
                'shadow-blur': 30,
                'shadow-color': 'rgba(255,255,255,0.6)',
                'shadow-opacity': 1,
                'shadow-offset-x': 0,
                'shadow-offset-y': 0,
                'font-size': '17px',
            }
        },
        {
            'selector': 'node:selected',
            'style': {
                'width': '196px',
                'height': '68px',
                'border-width': 4,
                'border-color': 'white',
                'shadow-blur': 40,
                'shadow-color': 'rgba(255,255,255,0.8)',
                'shadow-opacity': 1,
                'shadow-offset-x': 0,
                'shadow-offset-y': 0,
                'overlay-opacity': 0,
            }
        },
        # ── CLICKED: persistent ring; per-category border colour
        # so the highlight is unmistakable on dark and light fills
        # alike. We include both `.clicked` and `.clicked:selected`
        # variants — when the user taps a node, Cytoscape auto-marks
        # it `:selected`, and the bare `node:selected` rule above
        # would otherwise force its own border-color to white. ──────
        {
            'selector': 'node.clicked, node.clicked:selected',
            'style': {
                'width': '196px',
                'height': '68px',
                'border-width': 5,
                'shadow-blur': 45,
                'shadow-color': '#f59e0b',
                'shadow-opacity': 1,
                'shadow-offset-x': 0,
                'shadow-offset-y': 0,
                'font-size': '17px',
                'overlay-opacity': 0,
            }
        },
        # per-category border colours — listed for both `.clicked` and
        # `.clicked:selected` to override `node:selected`'s white border
        {'selector': '[type = "stressor"].clicked, [type = "stressor"].clicked:selected',
         'style': {'border-color': "#f386ec"}},
        {'selector': '[type = "performance_loss"].clicked, [type = "performance_loss"].clicked:selected',
         'style': {'border-color': "#aad8fe"}},
        {'selector': '[type = "mechanism"].clicked, [type = "mechanism"].clicked:selected',
         'style': {'border-color': "#7c1280"}},
        {'selector': '[type = "failure"].clicked, [type = "failure"].clicked:selected',
         'style': {'border-color': "#2275f2"}},
        {'selector': sel("stressor"),        'style': {'background-color': allc[0]}},
        {'selector': sel("mechanism"),       'style': {'background-color': allc[1]}},
        {'selector': sel("failure"),         'style': {'background-color': allc[2]}},
        {'selector': sel("performance_loss"),'style': {'background-color': allc[3]}},
        # category-tinted glow on hover
        {'selector': '[type = "stressor"].mouseover',        'style': {'shadow-color': allc[0]}},
        {'selector': '[type = "mechanism"].mouseover',       'style': {'shadow-color': allc[1]}},
        {'selector': '[type = "failure"].mouseover',         'style': {'shadow-color': allc[2]}},
        {'selector': '[type = "performance_loss"].mouseover','style': {'shadow-color': allc[3]}},
        {
            'selector': 'edge',
            'style': {
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'width': 2,
                'line-color': '#cbd5e1',
                'target-arrow-color': '#cbd5e1',
                'transition-property': 'width, line-color, target-arrow-color, opacity',
                'transition-duration': '0.2s',
            }
        },
        {
            'selector': 'edge:selected',
            'style': {
                'width': 3.5,
                'line-color': '#94a3b8',
                'target-arrow-color': '#94a3b8',
            }
        },
        # ── FADED: nodes/edges outside the clicked node's connected
        # component. Per-category selectors so specificity beats the
        # base `node` color rule. ────────────────────────────────────
        {
            'selector': 'node.faded',
            'style': {
                'background-opacity': 0.2,
                'border-opacity': 0,
                'text-opacity': 1,
                'shadow-opacity': 0,
            }
        },
        {'selector': '[type = "stressor"].faded',         'style': {'color': "#FFFFFF"}},
        {'selector': '[type = "mechanism"].faded',        'style': {'color': "#FFFFFF"}},
        {'selector': '[type = "failure"].faded',          'style': {'color': "#FFFFFF"}},
        {'selector': '[type = "performance_loss"].faded', 'style': {'color': "#FFFFFF"}},
        {
            'selector': 'edge.faded',
            'style': {
                'opacity': 0.25,
            }
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
def create_legend():
    def item(color, label):
        return html.Div([
            html.Div(style={
                "width": "14px", "height": "14px",
                "backgroundColor": color,
                "marginRight": "8px",
                "display": "inline-block",
                "borderRadius": "3px",
            }),
            html.Span(label, style={"color": "#6b7280", "fontSize": "13px"})
        ], style={"marginBottom": "5px", "display": "flex", "alignItems": "center"})

    return html.Div([
        html.H6("Legend", style={"margin": "0 0 10px 0", "color": "#4b5563"}),
        item(allc[0], "Stressor"),
        item(allc[1], "Mechanism"),
        item(allc[2], "Failure"),
        item(allc[3], "Performance Impact"),
    ], style={
        "position": "absolute",
        "left": "10px",
        "bottom": "10px",
        "background": "rgba(255,255,255,0.5)",
        "padding": "10px",
        "border": "1px solid #ddd",
        "backdropFilter": "blur(6px)",
        "borderRadius": "8px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.2)",
        "zIndex": 10,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SHARED WRAPPER STYLES
# ─────────────────────────────────────────────────────────────────────────────
_GRAPH_WRAPPER_STYLE = {
    "width": "100%",
    "height": "450px",
    "border": "1px solid #e5e7eb",
    "borderRadius": "10px",
    "padding": "6px",
    "background": "#fafafa",
}

_INNER_RELATIVE_STYLE = {
    "position": "relative",
    "width": "100%",
    "height": "100%",
}

_CONTAINER_STYLE = {
    "marginTop": "20px",
    "border": "1px solid #e5e7eb",
    "borderRadius": "12px",
    "padding": "10px",
    "background": "white",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAP SECTION
# ─────────────────────────────────────────────────────────────────────────────
def create_map_section():
    return html.Div([
        dcc.Graph(
            id="map",
            style={"height": "700px", "width": "100%"},
            config={
                "displayModeBar": False,
                "scrollZoom": True,
            }
        ),
        dbc.Container(
            html.Div([
                html.Div(id="map-summary"),
                html.Div(
                    id="map-detail",
                    style={"position": "relative"},
                    children=[
                        # placeholder so Dash registers the button id on initial load
                        html.Button(id="map-close-btn", n_clicks=0,
                                    style={"display": "none"})
                    ]
                ),
            ], style={
                "position": "absolute",
                "top": "10px", "bottom": "10px", "left": "16px",
                "display": "flex", "flexDirection": "column", "gap": "10px",
            }),
            fluid=False,
            style={
                "position": "absolute",
                "top": 0, "left": 0, "right": 0, "height": "100%",
                "pointerEvents": "none", "zIndex": 10,
            }
        )
    ], style={
        "position": "relative",
        "width": "100vw",
        "marginLeft": "calc(-50vw + 50%)",
        "marginTop": "20px",
    })


# ─────────────────────────────────────────────────────────────────────────────
# PER-PAPER PATHWAY GRAPH  (id='graph')
# ─────────────────────────────────────────────────────────────────────────────
def create_graph_section():
    return html.Div([
        html.Div([
            cyto.Cytoscape(
                id='graph',
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '100%'},
                elements=[],
                stylesheet=create_stylesheet(
                    attr_key="category",
                    perf_value="performance_impact"
                ),
                minZoom=0.5,
                maxZoom=1.5,
                autoungrabify=True,   # nodes can't be dragged by the user
            ),
            html.Div(
                id="pathway-buttons",
                style={
                    "position": "absolute", "left": "0px", "top": "0px",
                    "background": "rgba(255,255,255,0.6)",
                    "borderRadius": "16px", "zIndex": 10,
                }
            ),
            create_legend(),
        ], style=_INNER_RELATIVE_STYLE)
    ], style=_GRAPH_WRAPPER_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# COMMON PATHWAY GRAPH  (id='common-pathway-graph')
# Fault buttons and graph elements are both managed by callbacks.
# ─────────────────────────────────────────────────────────────────────────────
_BTN_ACTIVE   = "#1f2937"   # dark charcoal — active/selected
_BTN_INACTIVE = "#6b7280"   # medium gray — border + text when inactive

def _fault_btn(label):
    is_active = (label == 'PID')
    return html.Button(
        label,
        id={'type': 'fault-btn', 'index': label},
        n_clicks=0,
        style={
            "padding": "10px 28px",
            "borderRadius": "24px",
            "border": f"1.5px solid {_BTN_ACTIVE if is_active else _BTN_INACTIVE}",
            "background": _BTN_ACTIVE if is_active else "white",
            "color": "white" if is_active else _BTN_INACTIVE,
            "fontWeight": "700",
            "fontSize": "16px",
            "cursor": "pointer",
            "transition": "all 0.15s ease",
        }
    )


def create_common_pathway_section():
    return html.Div([

        # ── cytoscape + placeholder + legend ─────────────────────────
        html.Div([
            cyto.Cytoscape(
                id='common-pathway-graph',
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '100%'},
                elements=[],                # populated by callback on load / btn click
                stylesheet=create_stylesheet_common(),
                zoom=0.7,
                minZoom=0.3,
                maxZoom=1.5,
                autoungrabify=True,   # nodes can't be dragged by the user
            ),

            # shown by callback when fault data is not yet available
            html.Div(
                id="common-pathway-placeholder",
                style={
                    "position": "absolute",
                    "top": "50%", "left": "50%",
                    "transform": "translate(-50%, -50%)",
                    "color": "#aaa",
                    "fontSize": "14px",
                    "fontStyle": "italic",
                    "textAlign": "center",
                    "pointerEvents": "none",
                    "display": "none",
                }
            ),

            # ── Unselect button: top-right overlay, shown only while
            # a node is highlighted. Clicking it clears the selection
            # and restores every node to its normal colour. ─────────
            html.Button(
                [
                    html.Span("Unselect node", style={
                        "fontSize": "13px",
                        "color": "#374151",
                        "fontWeight": "500",
                    }),
                    html.Span("✕", style={
                        "fontSize": "14px",
                        "fontWeight": "700",
                        "color": "#6b7280",
                        "lineHeight": "1",
                        "marginLeft": "8px",
                    }),
                ],
                id="common-pathway-unselect-btn",
                n_clicks=0,
                style={
                    "position": "absolute",
                    "top": "12px",
                    "right": "12px",
                    "display": "none",        # toggled by callback
                    "alignItems": "center",
                    "gap": "0px",
                    "padding": "6px 12px",
                    "background": "rgba(255,255,255,0.92)",
                    "border": "1px solid #d1d5db",
                    "borderRadius": "20px",
                    "cursor": "pointer",
                    "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                    "backdropFilter": "blur(4px)",
                    "transition": "all 0.15s ease",
                    "zIndex": 20,
                },
            ),

            create_legend(),
        ], style={
            # graph box only — fixed height, border, background
            "position": "relative",
            "width": "100%",
            "height": "700px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "10px",
            "padding": "6px",
            "background": "#fafafa",
        }),

    ])  # outer div has no fixed height; buttons + graph stack naturally


# ─────────────────────────────────────────────────────────────────────────────
# DETAIL PANEL
# ─────────────────────────────────────────────────────────────────────────────
def create_common_pathway_wiki():
    """Right-hand wiki/detail panel for common-pathway-graph node hover/click."""
    return html.Div(
        id="common-pathway-wiki",
        children=html.Div(
            "Click a node to see details",
            style={
                "color": "#aaa",
                "fontSize": "13px",
                "fontStyle": "italic",
                "textAlign": "center",
                "marginTop": "20px",
            }
        ),
        style={
            "width": "100%",
            "height": "700px",
            "overflowY": "auto",
            "background": "rgba(255,255,255,0.85)",
            "padding": "20px 16px",
            "borderRadius": "10px",
            "border": "1px solid #eee",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
        }
    )


def create_pathway_modal():
    """Modal containing the per-paper pathway graph with detail panel on the right."""

    return dbc.Modal(
        [
            dbc.ModalHeader(
                html.Div([
                    html.P("DEGRADATION PATHWAY", style={
                        "fontSize": "12px", "fontWeight": "700",
                        "letterSpacing": "1.5px", "color": "#9ca3af",
                        "marginBottom": "3px",
                    }),
                    html.Div(id="modal-paper-title", style={
                        "fontSize": "16px", "fontWeight": "600", "color": "#1f2937",
                    }),
                ], style={"width": "100%"}),
                close_button=True,
            ),

            dbc.ModalBody([
                # ── legend row ────────────────────────────────────────
                html.Div(
                    [html.Div([
                        html.Div(style={
                            "width": "13px", "height": "13px",
                            "backgroundColor": allc[i],
                            "borderRadius": "3px", "marginRight": "6px",
                            "display": "inline-block",
                        }),
                        html.Span(label, style={"fontSize": "14px", "color": "#6b7280"})
                    ], style={"display": "flex", "alignItems": "center", "marginRight": "18px"})
                    for i, label in enumerate(["Stressor", "Mechanism", "Failure", "Performance Impact"])],
                    style={"display": "flex", "flexWrap": "wrap", "marginBottom": "10px"}
                ),

                html.Hr(style={"margin": "0 0 12px 0", "borderColor": "#f0f0f0"}),

                # ── graph + detail: side by side on wide, stacked on narrow ──
                html.Div([

                    # graph box — height controlled by CSS class
                    html.Div([
                        cyto.Cytoscape(
                            id='graph',
                            layout={'name': 'preset'},
                            style={'width': '100%', 'height': '100%'},
                            elements=[],
                            stylesheet=create_stylesheet(
                                attr_key="category",
                                perf_value="performance_impact"
                            ),
                            minZoom=0.3,
                            maxZoom=2.0,
                            autoungrabify=True,   # nodes can't be dragged by the user
                        ),
                        html.Div(
                            id="pathway-buttons",
                            style={
                                "position": "absolute", "left": "8px", "top": "8px",
                                "display": "flex", "gap": "6px", "zIndex": 10,
                            }
                        ),
                    ], className="modal-graph-box"),

                    # detail panel
                    html.Div(
                        id="detail-panel",
                        className="pv-pathway-detail-panel modal-detail-panel",
                        style={
                            "overflowY": "auto",
                            "background": "#fff",
                            "padding": "18px",
                            "borderRadius": "10px",
                            "border": "1px solid #eee",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
                        }
                    ),

                ], className="modal-graph-detail-row"),
            ]),
        ],
        id="pathway-modal",
        is_open=False,
        centered=True,
        size="xl",
        className="pathway-modal-blur",
        style={"fontFamily": "inherit"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# DETAIL PANEL  (per-paper pathway, right column) — kept for import compatibility
# ─────────────────────────────────────────────────────────────────────────────
def create_detail_panel():
    return html.Div(
        id="detail-panel",
        className="pv-pathway-detail-panel",
        style={
            "width": "100%",
            "overflowY": "auto",
            "background": "rgba(255,255,255,0.85)",
            "padding": "20px 16px",
            "borderRadius": "10px",
            "border": "1px solid #eee",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION CONTAINERS
# ─────────────────────────────────────────────────────────────────────────────
def create_graph_container():
    return html.Div([
        dbc.Row([
            dbc.Col(create_graph_section(), xs=12, md=9),
            dbc.Col(create_detail_panel(),  xs=12, md=3),
        ], className="g-3 align-items-start")
    ], style=_CONTAINER_STYLE)


def create_common_pathway_container():
    return html.Div([

        # ── fault buttons: full container width, wraps on narrow screens ──
        html.Div(
            id="fault-button-row",
            children=[_fault_btn(f) for f in FAULT_BUTTONS],
            style={
                "display": "flex",
                "flexDirection": "row",
                "flexWrap": "wrap",
                "gap": "12px",
                "marginBottom": "14px",
                "padding": "0 6px",
            }
        ),

        dbc.Row([
            dbc.Col(create_common_pathway_section(), xs=12, md=8),
            dbc.Col(create_common_pathway_wiki(),    xs=12, md=4),
        ], className="g-3 align-items-start")
    ], style=_CONTAINER_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# COMING SOON SECTION — two preview cards side-by-side describing upcoming
# features. Same outer container style as the fault section so the page
# rhythm stays consistent.
# ─────────────────────────────────────────────────────────────────────────────
_SUBCARD_STYLE = {
    "background": "#fff",
    "border": "1px solid #e5e7eb",
    "borderRadius": "12px",
    "padding": "24px",
    "height": "100%",
    "display": "flex",
    "flexDirection": "column",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
    "transition": "transform 0.2s ease, box-shadow 0.2s ease",
}

# Blue pill — uses Tailwind-style blue palette so it reads as
# "informational / on the roadmap".
_COMING_SOON_BADGE_STYLE = {
    "display": "inline-block",
    "background": "#dbeafe",        # blue-100
    "color": "#1e40af",             # blue-800
    "border": "1px solid #93c5fd",  # blue-300
    "fontSize": "11px",
    "fontWeight": "700",
    "letterSpacing": "1px",
    "padding": "3px 10px",
    "borderRadius": "12px",
    "marginBottom": "12px",
    "alignSelf": "flex-start",
}

_GITHUB_BTN_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "padding": "7px 13px",
    "background": "#1f2937",
    "color": "white",
    "fontSize": "12px",
    "fontWeight": "600",
    "borderRadius": "8px",
    "textDecoration": "none",
    "alignSelf": "flex-start",
    "marginBottom": "10px",
    "transition": "background 0.15s ease",
}


def _bullet_list(items):
    """Render a tight bullet list with consistent typography."""
    return html.Ul(
        [html.Li(t, style={"marginBottom": "6px"}) for t in items],
        style={
            "fontSize": "15px",
            "color": "#4b5563",
            "lineHeight": "1.55",
            "paddingLeft": "20px",
            "marginBottom": "14px",
        },
    )


def _coming_soon_subcard(title, bullets, image_src, image_alt,
                         pre_image=None, image_href=None,
                         image_max_height="80px"):
    """One feature-preview card.
    `bullets`           : list of short strings rendered as a bullet list.
    `pre_image`         : optional element rendered between the bullets and
                          the image frame (e.g. a GitHub button above the
                          pvdeg pic).
    `image_href`        : if provided, wraps the image in a link to that URL.
    `image_max_height`  : per-card cap so individual logos can be sized
                          independently.
    """
    img = html.Img(
        src=image_src,
        alt=image_alt,
        style={
            "maxWidth": "100%",
            "maxHeight": image_max_height,
            "objectFit": "contain",
        },
    )

    if image_href:
        img = html.A(
            img,
            href=image_href,
            target="_blank",
            rel="noopener noreferrer",
            title=image_href,
            style={"display": "inline-block", "lineHeight": 0},
        )

    image_frame = html.Div(
        img,
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "flex-start",   # left-aligned
            "padding": "0",
        },
    )

    return html.Div(
        [
            html.Span("COMING SOON", style=_COMING_SOON_BADGE_STYLE),
            html.H4(title, style={
                "fontSize": "18px",
                "fontWeight": "700",
                "color": "#1f2937",
                "marginBottom": "10px",
            }),
            _bullet_list(bullets),
            *([pre_image] if pre_image else []),
            image_frame,
        ],
        style=_SUBCARD_STYLE,
    )


def create_coming_soon_container():
    chat_card = _coming_soon_subcard(
        title="Chat-based pathway analysis",
        bullets=[
            "Explore the dataset through natural conversation",
            "Ask about specific stressors, mechanisms, or failure modes",
            "Answers grounded in the underlying literature",
        ],
        image_src="/assets/pathway_chat_logo.png",
        image_alt="Pathway chat preview",
    )

    github_url = "https://github.com/NatLabRockies/PVDegradationTools"
    github_btn = html.A(
        [
            html.Span("View on GitHub", style={"marginRight": "6px"}),
            html.Span("↗", style={"fontSize": "12px"}),
        ],
        href=github_url,
        target="_blank",
        rel="noopener noreferrer",
        style=_GITHUB_BTN_STYLE,
    )

    pvdeg_card = _coming_soon_subcard(
        title="Integration with pvdeg",
        bullets=[
            "Connect the pathway knowledge base with pvdeg, an open-source degradation modeling tool",
            "Pathways help refine physics-based degradation models",
        ],
        image_src="/assets/pathway_pvdeg_logo.png",
        image_alt="pvdeg integration preview",
        pre_image=github_btn,         # button sits above the image
        image_href=github_url,        # image itself is also clickable
        image_max_height="50px",      # smaller pvdeg logo
    )

    return html.Div([
        dbc.Row([
            dbc.Col(chat_card,  xs=12, md=6),
            dbc.Col(pvdeg_card, xs=12, md=6),
        ], className="g-3 align-items-stretch"),
    ], style=_CONTAINER_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER SECTION
# ─────────────────────────────────────────────────────────────────────────────
def create_filter_section():
    def dropdown_block(label_text, dropdown_id, options, default_values):
        return html.Div([
            html.Label(label_text, style={
                "fontWeight": "700", "marginBottom": "8px", "fontSize": "17px",
            }),
            dcc.Dropdown(
                id=dropdown_id,
                options=options,
                value=default_values,
                multi=True,
                style={"color": "#999", "font-size": "14px"},
            )
        ], style={"flex": "1 1 400px", "minWidth": "300px"})

    return html.Div([
        html.Div([
            dropdown_block(
                "Filter by Degradation Components",
                "component-filter",
                [
                    {"label": "Frame",       "value": "frame"},
                    {"label": "Glass",       "value": "glass"},
                    {"label": "Encapsulant", "value": "encapsulant"},
                    {"label": "Cells",       "value": "cells"},
                    {"label": "Backsheet",   "value": "backsheet"},
                    {"label": "JBox-Diode",  "value": "jbox_diode"},
                    {"label": "Other",       "value": "other"},
                ],
                ["frame", "glass", "encapsulant", "cells",
                 "backsheet", "jbox_diode", "other"],
            ),
            dropdown_block(
                "Filter by Major Degradation Types",
                "mechanism-filter",
                [{"label": x, "value": x.lower()}
                 for x in ["Moisture ingress", "Crack", "Corrosion", "Delamination",
                            "UV degradation", "Hot spot", "PID", "Solder fatigue",
                            "Discoloration", "Soiling", "Shading", "LID",
                            "Bypass diode failure", "Glass breakage", "LeTID"]]
                + [{"label": "Other", "value": "other"}],
                ["moisture ingress", "crack", "corrosion", "delamination",
                 "uv degradation", "hot spot", "pid", "solder fatigue",
                 "discoloration", "soiling", "shading", "lid",
                 "bypass diode failure", "glass breakage", "letid", "other"],
            ),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"})
    ], style={
        "background": "#fff",
        "padding": "16px",
        "borderRadius": "12px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
        "marginTop": "20px",
    })


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
def create_layout(file_list):
    return html.Div([

        # pathway modal (lives outside map so it can overlay everything)
        create_pathway_modal(),

        dbc.Container([
            html.Div([
                dcc.Store(id="selected-file", data=0),
                dcc.Store(id="wiki-node-id", data=None),
                # placeholder buttons registered before any map click
                html.Button(id="open-pathway-modal-btn", n_clicks=0,
                            style={"display": "none"}),
                html.Hr(),
                html.Div(
                    html.H1(
                        "PV Module Degradation Pathway Explorer",
                        style={"fontWeight": 800, "letterSpacing": "-0.5px"},
                    ),
                    style={'width': '100%', 'padding': '0 10px', 'textAlign': 'center'}
                ),
                html.Hr(),
                create_filter_section(),
            ])
        ]),

        create_map_section(),

        dbc.Container([
            html.Div([
                html.H3(
                    'Material Degradation Pathways by Fault',
                    style={"marginTop": "20px", "fontWeight": 800},
                ),
                create_common_pathway_container(),
            ])
        ]),

        dbc.Container([
            html.Div([
                html.H3(
                    'New Functions & Features to Come',
                    style={"marginTop": "20px", "fontWeight": 800},
                ),
                create_coming_soon_container(),
            ])
        ]),

    ])