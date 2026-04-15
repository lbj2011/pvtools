from dash import html
import dash_cytoscape as cyto
from dash import dcc
import dash_bootstrap_components as dbc
from dash_model_viewer import DashModelViewer

# allc = ['#8A257F','#C781CD','#7AB8D7', '#368BB6']
allc = ['#8A257F','#D476EC','#3BB1FF', '#1E51BB']


def create_legend():
    def item(color, label):
        return html.Div([
            html.Div(
                style={
                    "width": "14px",
                    "height": "14px",
                    "backgroundColor": color,
                    "marginRight": "8px",
                    "display": "inline-block"
                }
            ),
            html.Span(
                label,
                style={
                    "color": "#6b7280",   # 👈 gray text
                    "fontSize": "13px"
                }
            )
        ], style={
            "marginBottom": "5px",
            "display": "flex",
            "alignItems": "center"
        })

    return html.Div([
        html.H6(
            "Legend",
            style={
                "margin": "0 0 10px 0",
                "color": "#4b5563"   # 👈 slightly darker gray for title
            }
        ),
        item(allc[0], "Stressor"),
        item(allc[1], "Mechanism"),
        item(allc[2], "Failure"),
        item(allc[3], "Performance Impact"),
    ],
    style={
        "position": "absolute",
        "left": "10px",
        "bottom": "10px",
        "background": "rgba(255, 255, 255, 0.5)",
        "padding": "10px",
        "border": "1px solid #ddd",
        "backdropFilter": "blur(6px)",
        "borderRadius": "8px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.2)",
        "zIndex": 10
    })

def create_stylesheet():
    return [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',

                # 🔥 FIXED readable size
                'font-size': '14px',
                'font-weight': 'bold',

                # wrap
                'text-wrap': 'wrap',
                'text-max-width': '140px',

                # prevent overflow
                'min-zoomed-font-size': 10,

                'color': 'white',
                'text-valign': 'center',
                'text-halign': 'center',

                'width': '150px',
                'height': '50px',

                'shape': 'round-rectangle',
                'border-width': 1,
                'border-color': 'white',

                # ✅ SHADOW
                'shadow-blur': 15,
                'shadow-color': 'black',
                'shadow-opacity': 0.3,
                'shadow-offset-x': 2,
                'shadow-offset-y': 2
            }
        },

        # Stressor
        {'selector': '[category = "stressor"]', 'style': {'background-color': allc[0]}},

        # Mechanism
        {'selector': '[category = "mechanism"]', 'style': {'background-color': allc[1]}},

        # Failure
        {'selector': '[category = "failure"]', 'style': {'background-color': allc[2]}},

        # Performance
        {'selector': '[category = "performance_impact"]', 'style': {'background-color': allc[3]}},

        {
            'selector': 'edge',
            'style': {
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'width': 2
            }
        }
    ]

def create_map_section():
    return html.Div(
        [

            # ✅ FULL-WIDTH MAP
            dcc.Graph(
                id="map",
                style={"height": "650px", "width": "100%"},
                config={
                    "displayModeBar": False,
                    "scrollZoom": True
                }
            ),

            # ✅ OVERLAY (aligned with container automatically)
            dbc.Container(
                html.Div(
                    [
                        html.Div(
                            id="map-summary",
                            # style={"pointerEvents": "auto"}   # 👈 allow interaction ONLY here
                        ),
                        html.Div(
                            id="map-detail",
                            # style={"pointerEvents": "auto"}   # 👈 allow interaction ONLY here
                        ),
                    ],
                    style={
                        "position": "absolute",
                        "top": "10px",
                        "bottom": "10px",
                        "left": "16px",
                        "width": "300px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "10px",
                    }
                ),
                fluid=False,   # 👈 THIS IS KEY (keeps alignment)
                style={
                    "position": "absolute",
                    "top": 0,
                    "left": 0,
                    "right": 0,
                    "height": "100%",
                    "pointerEvents": "none",
                    "zIndex": 10,   # ✅ KEY FIX
                }
            )
        ],
        style={
            "position": "relative",

            # 👇 full-width breakout
            "width": "100vw",
            "marginLeft": "calc(-50vw + 50%)",

            "marginTop": "20px"
        }
    )
def create_graph_section():
    return html.Div(
        [
            html.Div(
                [
                    cyto.Cytoscape(
                        id='graph',
                        layout={'name': 'preset'},
                        style={'width': '100%', 'height': '100%'},
                        elements=[],
                        stylesheet=create_stylesheet(),
                        minZoom=0.5,
                        maxZoom=1.5,
                    ),

                    html.Div(
                        id="pathway-buttons",
                        style={
                            "position": "absolute",
                            "left": "0px",
                            "top": "0px",
                            "background": "rgba(255,255,255,0.6)",
                            "borderRadius": "16px",
                            "zIndex": 10
                        }
                    ),

                    create_legend()
                ],
                style={
                    "position": "relative",
                    "width": "100%",
                    "height": "100%",
                }
            )
        ],
        style={
            "width": "100%",
            "height": "450px",

            # ✅ NEW: light gray contour
            "border": "1px solid #e5e7eb",
            "borderRadius": "10px",

            # optional polish
            "padding": "6px",
            "background": "#fafafa",
        }
    )

def create_detail_panel():
    return html.Div(
        id="detail-panel",
        className="pv-pathway-detail-panel",   # 👈 scoped name
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

def create_graph_container():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        create_graph_section(),
                        xs=12, md=9
                    ),
                    dbc.Col(
                        create_detail_panel(),
                        xs=12, md=3
                    ),
                ],
                className="g-3 align-items-start"   # ✅ critical
            )
        ],
        style={
            "marginTop": "20px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "padding": "10px",
            "background": "white"
        }
    )

def create_filter_section():
    return html.Div(
        [
            html.Div(
                [
                    # =====================
                    # COMPONENT FILTER
                    # =====================
                    html.Div(
                        [
                            html.Label(
                                "Filter by Degradation Components",
                                style={
                                    "fontWeight": "600",
                                    "marginBottom": "6px",
                                    "fontSize": "14px",
                                }
                            ),
                            dcc.Dropdown(
                                id="component-filter",
                                options=[
                                    {"label": x.title(), "value": x}
                                    for x in ["cell", "encapsulant", "glass", "front sheet", "backsheet"]
                                ] + [{"label": "Other", "value": "other"}],
                                value=["cell", "encapsulant", "glass", "front sheet", "backsheet", "other"],
                                multi=True,
                                placeholder="Select components...",
                                style={
                                    "color": "#999",
                                    "font-size": "14px",
                                }
                            )
                            
                        ],
                        style={
                            "flex": "1 1 400px",   # 👈 key
                            "minWidth": "300px"
                        }
                    ),

                    # =====================
                    # MECHANISM FILTER
                    # =====================
                    html.Div(
                        [
                            html.Label(
                                "Filter by Major Degradation Types",
                                style={
                                    "fontWeight": "600",
                                    "marginBottom": "6px",
                                    "fontSize": "14px",
                                }
                            ),
                            dcc.Dropdown(
                                id="mechanism-filter",
                                options=[
                                    {"label": x, "value": x.lower()}
                                    for x in ["PID", "Crack", "Corrosion", "Hot spot", "Delamination", "Moisture ingress", "Thermal cycling"]
                                ] + [{"label": "Other", "value": "other"}],
                                value=["pid", "crack", "corrosion", "hot spot", "delamination", "moisture ingress", "thermal cycling", "other"],
                                multi=True,
                                placeholder="Select mechanisms...",
                                style={
                                    "color": "#999",
                                    "font-size": "14px",
                                }
                            )
                        ],
                        style={
                            "flex": "1 1 400px",
                            "minWidth": "300px"
                        }
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "20px",
                    "flexWrap": "wrap",   # 👈 enables stacking when needed
                }
            )
        ],
        style={
            "background": "#fff",
            "padding": "16px",
            "borderRadius": "12px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
            "marginTop": "20px"
        }
    )

def create_layout(file_list):
    return html.Div([

        # 👇 NORMAL CONTENT (keeps margins)
        dbc.Container([
            html.Div([

                dcc.Store(id="selected-file", data=0),

                html.Hr(),

                html.Div([
                    html.H1("PV Module Degradation Pathway Explorer (Demo)"),
                ], style={
                    'width': '100%',
                    'padding-left': '10px',
                    'padding-right': '10px',
                    'textAlign': 'center'
                }),


                create_filter_section(),

            ])
        ]),

        # 👇 FULL-WIDTH MAP (outside container)
        create_map_section(),

        # 👇 NORMAL CONTENT AGAIN
        dbc.Container([
            html.Div([

                html.H3('Material degradation pathways', style={"marginTop": "20px"}),

                create_graph_container()

            ])
        ]),

        dbc.Container([
            DashModelViewer(
                    id="pv-model",
                    # Use a URL or a path to your file in the 'assets' folder
                    src="/assets/Untitled4.glb",
                    alt="A 3D model of an exploded PV module",
                    cameraControls=True,   # Allows orbiting and zooming
                    cameraOrbit="70deg 70deg 65%",
                    fieldOfView="25deg",
                    ar=False,              # Enable if you want mobile users to view in AR
                    style={"width": "100%", "height": "400px",
                           "borderRadius": "10px",           # 圆角
                            "border": "1px solid #ddd",      # 边框线}
                    }
                ),
        ])

    ])