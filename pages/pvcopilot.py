import dash
from dash import dcc, html, Input, Output, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import dash_bootstrap_components as dbc
from app import app
from page_supporting_files.analysis_utils import parse_contents
from dash import callback_context as ctx
from io import StringIO
import traceback
from page_supporting_files.analysis_utils import make_overview_figures, normalize, low_irra_power_filter, aggregate_daily, compute_yoy, get_full_code
from page_supporting_files.analysis_utils import compute_lr, compute_hw, compute_arima, compute_csd
from page_supporting_files.pvcopilot_filter_functions import identify_outliers_iqr, clear_sky_filter, basic_value_filter
import base64
import time

# --- Define Color Variables ---
MAJOR_CARD_BACKGROUND = "#F8F8F8"
MAJOR_CARD_FONT_COLOR = "black"
BODY_CARD_BACKGROUND = "white" 
CODE_BLOCK_BACKGROUND = "#f8f9fa"


def _df_from_store(value):
    """
    Robustly reconstruct a DataFrame from a dcc.Store payload.

    Stores can hand the value back as a JSON string OR as an already-deserialized
    dict, depending on Dash/pandas versions. Newer pandas read_json refuses dicts,
    so we branch.

    Accepts:
      - dict in pandas split format: {'columns': [...], 'index': [...], 'data': [...]}
      - dict in to_dict() / records / records_dict shapes (best-effort)
      - JSON string in 'split' orientation
      - None / empty -> raises ValueError so callers can short-circuit
    """
    if value is None or value == {} or value == "":
        raise ValueError("No dataframe in store")

    if isinstance(value, dict):
        # Native split-orient dict
        if {"columns", "index", "data"} <= value.keys():
            return pd.DataFrame(**value)
        # Generic dict — let pandas guess
        return pd.DataFrame(value)

    if isinstance(value, str):
        return pd.read_json(StringIO(value), orient="split")

    # Last-resort: hand whatever it is to DataFrame and hope
    return pd.DataFrame(value)


def _no_data_alert(message):
    """Reusable warning alert shown inside a card's right-side output area
    when an action is taken before the prerequisite data is loaded."""
    return dbc.Alert(
        [html.Strong("⚠️ No data loaded.  "), message],
        color="warning",
        className="mb-0",
        style={"fontSize": "14px"},
    )


def get_layout():
    return layout

# --- Redesigned Application Layout (Headers Only Deep Blue) ---
layout = dbc.Container([
    
    html.Div([
    dcc.Store(id='mapped-vars-store', data={}),
    dcc.Store(id='dataframe-store', data={}),
    dcc.Store(id='dataframe-filtered', data={}),
    dcc.Store(id='code-read-store', data={}),
    dcc.Store(id="data-source-store", data=None),
    dcc.Store(id="stored-data-file-name", data=None),

    html.Hr(),
    html.Div([
        html.H1("PV-Copilot", className="page-title"), 
    ], className="page-title-container"),
    html.Hr(),

    html.Div([

        # Floating button
        html.Button(
            "Get the code",
            id="floating-btn",
            n_clicks=0,
            style={
                "position": "fixed",
                "bottom": "30px",
                "right": "30px",
                "padding": "10px 16px",
                "borderRadius": "30px",
                "backgroundColor": "#AE28C5",
                "color": "white",
                "fontSize": "18px",
                "fontWeight": "800",
                "border": "none",
                "boxShadow": "0px 4px 10px rgba(0,0,0,0.3)",
                "cursor": "pointer",
                "zIndex": 1000
            }
        ),

        # Floating content panel
        html.Div(
            id="floating-panel",
            children=[
                
                html.Div(id="panel-content", 
                         children=[
                             html.Div(
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "5px"
                                },
                                children=[

                                    dbc.Button(
                                        "Generate full code to run",
                                        id="generate-code-btn",
                                        color="primary",
                                        size="sm",
                                        style={"fontSize": "16px", "fontWeight": "500"}
                                    ),

                                    dbc.Button(
                                        "X",
                                        id="close-panel-btn",
                                        size="sm",
                                        style={
                                            "width": "20px",
                                            "height": "20px",
                                            "borderRadius": "50%",   # 👈 makes it a circle
                                            "padding": "0",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "backgroundColor": "#e0e0e0",  # light gray
                                            "color": "#fffcfc",
                                            "border": "none",
                                            "fontSize": "10px",
                                            "fontWeight": "600",
                                            "cursor": "pointer"
                                        }
                                    ),
                                ]
                            ),
                            html.Small(
                                        "(It typically takes 2-10 seconds)",
                                        className="text-muted small"
                                    ),
                            dcc.Loading(
                                id="code-loading",
                                type="circle",
                                children=html.Div(
                                    id="code-preview",
                                    style={"marginTop": "10px"}
                                )
                            ),
                            html.A(
                                "Download code (python)",
                                id="download-link",
                                href="",
                                download="generated_code.py",
                                style={"display": "none", "marginTop": "10px"}  # ✅ FIX duplicate display
                            )
            ])
              
            ],
            style={
                "position": "fixed",
                "bottom": "100px",
                "right": "30px",
                "width": "250px",
                "padding": "15px",
                "backgroundColor": "white",
                "borderRadius": "10px",
                "boxShadow": "0px 4px 15px rgba(0,0,0,0.2)",
                "display": "none",   # hidden initially
                "zIndex": 1000
            }
        )

    ]),

    dbc.Row([
        dbc.Col([
            dcc.Markdown(
            """
            #### **PV-Copilot**: Data in, results out.

            *LLM-powered PV data analysis tool*

            * **No Coding Experience Required:** Analyze your PV data without writing a single line of code.
            * **Cross-Tool Integration:** Use functions from different PV packages and tools.
            * **Instant Results:** Choose the functions you need and view the analysis results instantly.
            * **Downloadable Code:** Download ready-to-run Python code for local deployment and use.

             """.replace('    ', '')
            ),
        ], xs=12, sm=12, md=12, lg=8, xl=8),

        dbc.Col(
            [
                html.Iframe(
                    src="https://www.youtube.com/embed/QuTOc8Fb4g4",
                    style={
                        "width": "70%",
                        "height": "150px",
                        "border": "none",
                        "marginBottom": "20px",  # 👈 add this
                        "borderRadius": "12px",  # 👈 rounded corners
                        "overflow": "hidden"     # 👈 ensures corners clip properly
                    },
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",
                ),
            ],
            xs=12, sm=12, md=10, lg=4, xl=4,
            className="text-center text-lg-end"
        )
    ]),

    dbc.Alert(
        [
            html.Strong("Note: "),
            "This tool is currently under ",
            html.Strong("active development"),
            ". If you encounter issues, have suggestions, or would like to collaborate, please",
            html.A(
                " contact us",
                href="mailto:baojieli@lbl.gov",
                style={"fontWeight": "bold"}
            ),
            "."
        ],
        color="primary",
        className="mt-2"
    ),
    
    html.P(''),
    
    # --- ROW 1: Consolidated Data Upload and Summary Table (Header Deep Blue) ---
    dbc.Row(style={"marginBottom": "30px"}, children=[
        dbc.Col(dbc.Card(
            # Card body uses default style (black text, white background)
            children=[
            dbc.CardHeader(
                # Apply deep blue background and white text to the HEADER only
                html.H4("Data Upload & Summary", style={"color": MAJOR_CARD_FONT_COLOR}),
                style={"backgroundColor": MAJOR_CARD_BACKGROUND}
            ),
            dbc.CardBody(style={"backgroundColor": BODY_CARD_BACKGROUND}, children=[
                dbc.Row([
                    
                    # Left side: Upload and Analyze Button
                    dbc.Col(lg=4, md=12, sm=12, xs=12, children=[
                        html.Div(
                            "Upload your data (.csv, .xls, .parquet)",
                            style={
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "color": "#2c3e50",

                                "paddingBottom": "10px",   # 👈 bottom padding
                                "marginTop": "5px",
                            }
                        ),

                        html.Details([

                            # --- Summary (no box, just padded text) ---
                            html.Summary(
                                "Data requirements (click to expand)",
                                style={
                                    "cursor": "pointer",
                                    "color": "#B5B5B8",
                                    "fontSize": "14px",
                                    "fontWeight": "500",
                                }
                            ),

                            # --- Expanded content (THIS gets the box) ---
                            html.Div(
                                [
                                    html.Ul([
                                        html.Li([
                                            "Include columns for ",
                                            html.B("time, power, irradiance, and temperature")
                                        ]),
                                        html.Li([
                                            "Use ",
                                            html.B(">=2 years of data"),
                                            " for reliable degradation analysis"
                                        ]),
                                        html.Li([
                                            "Recommended time resolution: ",
                                            html.B("1–6 hours")
                                        ]),
                                    ], 
                                    style={
                                        "marginBottom": "0",
                                        "paddingLeft": "16px"
                                        })
                                ],
                                style={
                                    "marginTop": "8px",

                                    # 👇 CARD STYLE ONLY FOR EXPANDED AREA
                                    "padding": "12px 14px",
                                    "border": "1px solid #e0e0e0",
                                    "borderRadius": "10px",
                                    "backgroundColor": "#e8f3ff",
                                    "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",

                                    "color": "#1A64BE",
                                    "fontSize": "13px",
                                    "lineHeight": "1.6",
                                }
                            )
                        ]),

                        dcc.Upload(
                            id="upload-data",
                            accept=".csv, text/csv, .xls, .xlsx, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, .parquet",
                            children=html.Div(
                                [
                                    "Drag and Drop or ",
                                    html.A("Select Files", style={"color": "#0d6efd", "fontWeight": "500"})
                                ]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "textAlign": "center",

                                # 👇 THIS is what you want
                                "marginTop": "12px",   # ✅ space ABOVE

                                # styling
                                "paddingTop": "18px",
                                "borderWidth": "1.5px",
                                "borderStyle": "dashed",
                                "borderColor": "#ced4da",
                                "borderRadius": "8px",
                                "backgroundColor": "#ffffff",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                                "cursor": "pointer"
                            }
                        ),
                        html.Div(id="upload-status-output", style={"marginTop": "5px"}),
                        html.Div(
                        [
                            dbc.Button(
                                "Example Data 1",
                                id="load-example-btn-1",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="mt-2 me-2"
                            ),
                            dbc.Button(
                                "Example Data 2",
                                id="load-example-btn-2",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="mt-2 me-2"
                            ),
                            dbc.Button(
                                "Example Data 3",
                                id="load-example-btn-3",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="mt-2 me-2"
                            )
                        ],
                        style={"marginTop": "10px"}
                        ),
                        dbc.Button("Analyze Data", id="analyze-btn", color="primary", className="w-100 mt-3"),
                        html.Small(
                            "(Analysis typically takes 2-10 seconds)",
                            style={
                                "color": "#adb5bd",   # 👈 lighter gray
                                "marginTop": "6px",   # 👈 space above
                                "display": "block"    # 👈 ensures margin works properly
                            }
                        )
                    ]),
                    
                    # Right side: Data Summary Table
                    dbc.Col(
                        lg=8, md=12, sm=12, xs=12,
                        children=[
                            dcc.Loading(
                                id="loading-summary-and-figs",
                                type="circle",
                                color="#0d6efd",

                                children=html.Div([

                                    # -------------------------
                                    # Summary Table + figs
                                    # -------------------------
                                    html.Div(
                                        id="data-summary-output",
                                        className="p-2 border",
                                        style={
                                            "minHeight": "170px",
                                            "marginTop": "5px"
                                        }
                                    )
                                ])
                            )
                        ]
                    )
                ])
            ])
        ]), xs=12, md=12),
    ]),

    # --- ROW 2: Analysis Options (Outer Header Deep Blue) ---
    dbc.Row(style={"marginBottom": "30px"}, children=[
        dbc.Col(dbc.Card(
            # Card body uses default style
            children=[
            dbc.CardHeader(
                # Apply deep blue background and white text to the HEADER only
                html.H4("Data filter", style={"color": MAJOR_CARD_FONT_COLOR}),
                style={"backgroundColor": MAJOR_CARD_BACKGROUND}
            ),
            dbc.CardBody(style={"backgroundColor": BODY_CARD_BACKGROUND}, children=[
                dbc.Row([
                    
                    dbc.Col(
                        
                        [
                            html.H6("Choose the filters:"),

                            # Use a hidden Checklist to keep the same id/value interface for the callback
                            dbc.Checklist(
                                id="filter-options",
                                options=[
                                    {"label": "", "value": "timezone"},
                                    {"label": "", "value": "low-irra-power"},
                                    {"label": "", "value": "outlier"},
                                    {"label": "", "value": "clearsky"},
                                ],
                                value=['timezone', "low-irra-power", "outlier", "clearsky"],
                                inline=False,
                                style={"display": "none"}
                            ),

                            # --- Manual per-filter rows ---
                            # 1. Timezone
                            html.Div([
                                dbc.Checkbox(id="cb-timezone", value=True, className="me-2 d-inline-block"),
                                html.Span("Time zone & DST correction"),
                            ], style={"marginBottom": "6px"}),

                            # 2. Low irradiance/power filter + inline customize
                            html.Div([
                                dbc.Checkbox(id="cb-low-irra-power", value=True, className="me-2 d-inline-block"),
                                html.Span("Low irradiance/power filter"),
                                html.Details([
                                    html.Summary("Customize parameters", style={
                                        "cursor": "pointer", "color": "#adb5bd",
                                        "fontSize": "13px", "fontWeight": "500", "marginTop": "4px",
                                        "marginLeft": "22px",
                                    }),
                                    html.Div([

                                        html.Div([
                                            html.Label("γ — temperature coefficient of power (/°C)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                            dcc.Input(id="param-gamma", type="number", value=-0.004, step=0.001,
                                                      style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                        ], style={"marginBottom": "10px"}),

                                        html.Div([
                                            html.Label("Min. irradiance threshold (W/m²)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                            html.Div("Excludes data below this irradiance level, where measurement uncertainty is high and the system may not be in normal operation.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                            dcc.Input(id="param-irr-thresh", type="number", value=300, step=10, min=0,
                                                      style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                        ], style={"marginBottom": "10px"}),

                                        html.Div([
                                            html.Label("Min. power / irradiance ratio", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                            html.Div("Rejects points where DC power is abnormally low relative to irradiance (e.g., shading, inverter faults). Condition: P > ratio × G.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                            dcc.Input(id="param-power-ratio", type="number", value=0.02, step=0.005, min=0,
                                                      style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                        ], style={"marginBottom": "0px"}),

                                        # Hidden inputs kept for callback compatibility
                                        dcc.Input(id="param-norm-lower", type="number", value=0.01, style={"display": "none"}),
                                        dcc.Input(id="param-norm-upper-pct", type="number", value=99, style={"display": "none"}),

                                    ], style={
                                        "marginTop": "8px", "marginLeft": "22px",
                                        "padding": "10px 12px",
                                        "backgroundColor": "#f0f6ff",
                                        "borderRadius": "8px",
                                        "border": "1px solid #c8dff8",
                                        "color": "#1a1a2e",
                                        "fontSize": "12px",
                                    })
                                ]),
                            ], style={"marginBottom": "6px"}),

                            # 3. Outlier removal + inline customize
                            html.Div([
                                dbc.Checkbox(id="cb-outlier", value=True, className="me-2 d-inline-block"),
                                html.Span("Outlier removal"),
                                html.Details([
                                    html.Summary("Customize parameters", style={
                                        "cursor": "pointer", "color": "#adb5bd",
                                        "fontSize": "13px", "fontWeight": "500", "marginTop": "4px",
                                        "marginLeft": "22px",
                                    }),
                                    html.Div([

                                        html.Div([
                                            html.Label("IQR multiplier (k)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                            html.Div([
                                                "Defines the fence width: bounds = [Q1 − k·IQR, Q3 + k·IQR]. ",
                                                "The standard Tukey fence uses k = 1.5. ",
                                                "Larger values (e.g., k = 3) yield a more permissive filter; smaller values are stricter."
                                            ], style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                            dcc.Input(id="param-iqr-multiplier", type="number", value=1.5, step=0.1, min=0.1,
                                                      style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                        ], style={"marginBottom": "0px"}),

                                    ], style={
                                        "marginTop": "8px", "marginLeft": "22px",
                                        "padding": "10px 12px",
                                        "backgroundColor": "#f0f6ff",
                                        "borderRadius": "8px",
                                        "border": "1px solid #c8dff8",
                                        "color": "#1a1a2e",
                                        "fontSize": "12px",
                                    })
                                ]),
                            ], style={"marginBottom": "6px"}),

                            # 4. Clear-sky filter + inline customize
                            html.Div([
                                dbc.Checkbox(id="cb-clearsky", value=True, className="me-2 d-inline-block"),
                                html.Span("Clear-sky filter"),
                                html.Details([
                                    html.Summary("Customize parameters", style={
                                        "cursor": "pointer", "color": "#adb5bd",
                                        "fontSize": "13px", "fontWeight": "500", "marginTop": "4px",
                                        "marginLeft": "22px",
                                    }),
                                    html.Div([

                                        html.Div([
                                            html.Label("Smoothness threshold", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                            html.Div([
                                                "Minimum per-day smoothness score (0–1). Based on the L1-norm of the 2nd-order temporal difference of the intraday irradiance profile. ",
                                                "Higher values are stricter (only very smooth bell-shaped days pass). ",
                                                "Recommended: 0.3–0.6 for hourly data, 0.7–0.9 for sub-hourly data."
                                            ], style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                            dcc.Input(id="param-cs-smooth", type="number", value=0.3, step=0.05, min=0.0, max=1.0,
                                                      style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                        ], style={"marginBottom": "10px"}),

                                        html.Div([
                                            html.Label("Energy threshold", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                            html.Div([
                                                "Minimum seasonally-normalized daily irradiance score (0–1). ",
                                                "Computed as the ratio of daily irradiance sum to a rolling 90th-percentile baseline (±30-day window). ",
                                                "A value of 0.5 retains days with at least 50% of the local seasonal maximum irradiance."
                                            ], style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                            dcc.Input(id="param-cs-energy", type="number", value=0.5, step=0.05, min=0.0, max=1.0,
                                                      style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                        ], style={"marginBottom": "0px"}),

                                    ], style={
                                        "marginTop": "8px", "marginLeft": "22px",
                                        "padding": "10px 12px",
                                        "backgroundColor": "#f0f6ff",
                                        "borderRadius": "8px",
                                        "border": "1px solid #c8dff8",
                                        "color": "#1a1a2e",
                                        "fontSize": "12px",
                                    })
                                ]),
                            ], style={"marginBottom": "6px"}),

                            # Sync individual checkboxes → hidden checklist (clientside)
                            dcc.Store(id="_cb-sync-dummy"),

                            dbc.Button("Filter data", id="filter-btn", color="primary", className="w-100 mt-3"),

                            html.Details([
                                html.Summary(
                                    "Filter detail:",
                                    style={
                                        "cursor": "pointer",
                                        "color": "#B5B5B8",
                                        "fontSize": "14px",
                                        "fontWeight": "500",
                                    }
                                ),

                                html.Div([

                                    # Each filter item is individually collapsible
                                    html.Details([
                                        html.Summary(html.B("Basic value filter (always applied)"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                        html.Div(
                                            "Applied automatically before all other filters. Removes physically implausible sensor readings: "
                                            "irradiance outside [0, 1500] W/m², "
                                            "module temperature outside [−40, 100] °C, "
                                            "and DC power below −1 W. "
                                            "Catches sensor faults (e.g. irradiance = 34,000 W/m²) that would corrupt normalization and clear-sky scoring.",
                                            style={"marginTop": "4px", "paddingLeft": "12px"}
                                        ),
                                    ], style={"marginBottom": "6px"}),

                                    html.Details([
                                        html.Summary(html.B("Time zone & DST correction"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                        html.Div(
                                            "Corrects timestamps for local time zone offsets and Daylight Saving Time (DST) transitions. "
                                            "Ensures the datetime index is monotonic and properly localized before any temporal analysis.",
                                            style={"marginTop": "4px", "paddingLeft": "12px"}
                                        ),
                                    ], style={"marginBottom": "6px"}),

                                    html.Details([
                                        html.Summary(html.B("Low irradiance/power filter"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                        html.Div([
                                            "Removes non-representative operating points using three simultaneous conditions: "
                                            "① irradiance above a minimum threshold; "
                                            "② power exceeding a minimum fraction of irradiance; "
                                            "③ temperature-corrected normalized power"
                                            " (norm = P / [G · (1 + γ(T − 25))] × 1000)",
                                            html.Sup("[1]"),
                                            " within a valid range. Points failing any condition are excluded."
                                        ], style={"marginTop": "4px", "paddingLeft": "12px"}),
                                    ], style={"marginBottom": "6px"}),

                                    html.Details([
                                        html.Summary(html.B("Outlier removal"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                        html.Div([
                                            "Detects statistical outliers on the temperature-corrected normalized power signal using the IQR method",
                                            html.Sup("[2]"),
                                            ". Points outside [Q1 − k·IQR, Q3 + k·IQR] (default k = 1.5, Tukey's fence) are flagged and excluded from downstream degradation analysis."
                                        ], style={"marginTop": "4px", "paddingLeft": "12px"}),
                                    ], style={"marginBottom": "6px"}),

                                    html.Details([
                                        html.Summary(html.B("Clear-sky filter"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                        html.Div([
                                            "Applied to the raw irradiance signal before power normalization, preserving the full intraday bell-shaped profile needed for smoothness scoring. "
                                            "Follows the approach of Meyers et al.",
                                            html.Sup("[3]"),
                                            " The algorithm is resolution-aware: ",
                                            html.Br(),
                                            html.B("Sub-daily data (≥4 readings/day): "),
                                            "① a smoothness score derived from the L1-norm of the 2nd-order temporal difference of the intraday irradiance signal "
                                            "(smooth bell-shaped profiles score high); "
                                            "② a seasonally-normalized daily energy score (ratio of daily irradiance sum to a rolling 90th-percentile baseline, ±30-day window). "
                                            "A day is classified as clear only if both scores exceed their respective thresholds (AND rule). ",
                                            html.Br(),
                                            html.B("Coarse/downsampled data (<4 readings/day): "),
                                            "smoothness cannot be reliably estimated from sparse samples, so the filter falls back to energy-only mode — "
                                            "retaining days whose seasonally-normalized irradiance exceeds the energy threshold."
                                        ], style={"marginTop": "4px", "paddingLeft": "12px"}),
                                    ], style={"marginBottom": "6px"}),

                                    # References (collapsible)
                                    html.Hr(style={"borderColor": "#c8dff8", "margin": "6px 0"}),
                                    html.Details([
                                        html.Summary("References", style={
                                            "cursor": "pointer",
                                            "fontSize": "11px", "fontWeight": "700",
                                            "color": "#1A64BE",
                                        }),
                                        html.Ol([
                                            html.Li([
                                                "IEC 60891:2021 — Photovoltaic devices: Procedures for temperature and irradiance corrections to measured I-V characteristics. ",
                                                html.A("webstore.iec.ch",
                                                       href="https://webstore.iec.ch/en/publication/61766",
                                                       target="_blank",
                                                       style={"color": "#0d6efd"}),
                                                "."
                                            ], style={"marginBottom": "4px"}),
                                            html.Li([
                                                "Kim, G. G., Hyun, J. H., Choi, J. H., Bhang, B. G., & Ahn, H. K. (2023). Quality analysis of photovoltaic system using descriptive statistics of power performance index. ",
                                                html.Em("IEEE Access"),
                                                ", 11, 28427–28438. ",
                                                html.A("10.1109/ACCESS.2023.3257373",
                                                       href="https://doi.org/10.1109/ACCESS.2023.3257373",
                                                       target="_blank",
                                                       style={"color": "#0d6efd"}),
                                                "."
                                            ], style={"marginBottom": "4px"}),
                                            html.Li([
                                                "B. E. Meyers, E. Apostolaki-Iosifidou, and L. Schelhas, \"Solar Data Tools: Automatic Solar Data Processing Pipeline,\" ",
                                                html.Em("2020 47th IEEE Photovoltaic Specialists Conference (PVSC)"),
                                                ", Calgary, AB, Canada, 2020, pp. 0655–0656, doi: ",
                                                html.A("10.1109/PVSC45281.2020.9300847",
                                                       href="https://doi.org/10.1109/PVSC45281.2020.9300847",
                                                       target="_blank",
                                                       style={"color": "#0d6efd"}),
                                                "."
                                            ]),
                                        ], style={"paddingLeft": "16px", "marginTop": "6px", "marginBottom": "0", "fontSize": "11px", "color": "#4a6fa5", "lineHeight": "1.5"})
                                    ])

                                ], style={
                                    "marginTop": "10px",
                                    "padding": "12px 14px",
                                    "border": "1px solid #e0e0e0",
                                    "borderRadius": "10px",
                                    "backgroundColor": "#e8f3ff",
                                    "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                                    "color": "#1A64BE",
                                    "fontSize": "13px",
                                    "lineHeight": "1.6",
                                }),
                            ], style={"marginTop": "10px"})
                        ],
                    xs=12, lg=4),

                    # Right side: Data Summary Table
                    dbc.Col(
                        lg=8, md=12, sm=12, xs=12,
                        children=[
                            dcc.Loading(
                                id="data-filter-result",
                                type="circle",
                                color="#0d6efd",

                                children=html.Div([

                                    html.Div(
                                        id="data-filter-output",
                                        className="p-2 border",
                                        style={
                                            "minHeight": "170px",
                                            "marginTop": "5px"
                                        }
                                    )
                                ])
                            )
                        ]
                    )
                    
                    
                    # 3. Figures block (Inner Card)
                    # dbc.Col(dbc.Card(children=[
                    #     dbc.CardHeader(html.H5("Figures to Show")),
                    #     dbc.CardBody([
                    #         dbc.Checklist(
                    #             id="figure-options",
                    #             options=[
                    #                 {"label": "Power vs time", "value": "power_time"},
                    #                 {"label": "Outliers vs time", "value": "outliers_time"},
                    #                 {"label": "Distribution of rate", "value": "rate_dist"},
                    #                 {"label": "SDM parameter vs time", "value": "sdm_param"},
                    #             ],
                    #             value=["power_time"],
                    #             inline=False
                    #         )
                    #     ])
                    # ]), xs=12, lg=4)
                    
                ])
            ])
        ]), xs=12, md=12),
    ]),

    # --- ROW 3: Analysis Output (Header Deep Blue) ---
    dbc.Card(
        # Card body uses default style
        children=[
        dbc.CardHeader(
            # Apply deep blue background and white text to the HEADER only
            html.H4("Degradation Analysis", style={"color": MAJOR_CARD_FONT_COLOR}),
            style={"backgroundColor": MAJOR_CARD_BACKGROUND}
        ),
        dbc.CardBody(style={"backgroundColor": BODY_CARD_BACKGROUND}, children=[
            # Run Button - Centered and reduced width (width=6)
            dbc.Row(
                justify="start",
                className="mt-4", # Add some top margin for visibility
                children=[

                    dbc.Col([
                        html.H6("Choose the metric:"),

                        # dcc.RadioItems with full HTML labels — gives true single-selection
                        # and lets "Customize parameters" sit directly under each metric label.
                        dcc.RadioItems(
                            id="metric-selected-visible",
                            value="YOY",
                            options=[
                                {
                                    "label": html.Div([
                                        html.B("YoY (Year-over-Year)", className="mathjax-ignore"),
                                        html.Details([
                                            html.Summary("Customize parameters", style={"cursor": "pointer", "color": "#adb5bd", "fontSize": "13px", "fontWeight": "500", "marginTop": "4px", "marginLeft": "4px"}),
                                            html.Div([
                                                html.Div("Rolling trend window (days)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                                html.Div("Window for the rolling-mean trend line on the plot.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                                dcc.Input(id="param-yoy-window", type="number", value=30, step=5, min=7,
                                                          style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "marginBottom": "8px", "color": "#000"}),
                                                html.Div("IQR multiplier k", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                                html.Div("Ratios outside [Q1 − k·IQR, Q3 + k·IQR] are excluded before computing the median rate.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                                dcc.Input(id="param-yoy-iqr", type="number", value=1.5, step=0.1, min=0.5,
                                                          style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                            ], style={"padding": "8px 10px", "backgroundColor": "#f0f6ff", "borderRadius": "6px", "border": "1px solid #c8dff8", "marginTop": "4px", "marginLeft": "4px"}),
                                        ]),
                                    ]),
                                    "value": "YOY",
                                },
                                {
                                    "label": html.Div([
                                        html.B("LR (Linear regression)", className="mathjax-ignore"),
                                        html.Div("No tunable parameters.", style={"fontSize": "11px", "color": "#adb5bd", "marginLeft": "4px", "fontStyle": "italic", "marginTop": "2px"}),
                                        dcc.Input(id="param-yoy-iqr-dummy", style={"display": "none"}),
                                    ]),
                                    "value": "LR",
                                },
                                {
                                    "label": html.Div([
                                        html.B("HW (Holt-Winters)", className="mathjax-ignore"),
                                        html.Details([
                                            html.Summary("Customize parameters", style={"cursor": "pointer", "color": "#adb5bd", "fontSize": "13px", "fontWeight": "500", "marginTop": "4px", "marginLeft": "4px"}),
                                            html.Div([
                                                html.Div("Seasonal period (months)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                                html.Div("Number of periods per seasonal cycle. Use 12 for monthly-aggregated data.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                                dcc.Input(id="param-hw-period", type="number", value=12, step=1, min=2,
                                                          style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                            ], style={"padding": "8px 10px", "backgroundColor": "#f0f6ff", "borderRadius": "6px", "border": "1px solid #c8dff8", "marginTop": "4px", "marginLeft": "4px"}),
                                        ]),
                                    ]),
                                    "value": "HW",
                                },
                                {
                                    "label": html.Div([
                                        html.B("ARIMA (Auto Regressive Integrated Moving Average)", className="mathjax-ignore"),
                                        html.Details([
                                            html.Summary("Customize parameters", style={"cursor": "pointer", "color": "#adb5bd", "fontSize": "13px", "fontWeight": "500", "marginTop": "4px", "marginLeft": "4px"}),
                                            html.Div([
                                                html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "8px"}, children=[
                                                    html.Div([
                                                        html.Div("p (AR order)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                                        html.Div("Autoregressive lag.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                                        dcc.Input(id="param-arima-p", type="number", value=1, step=1, min=0,
                                                                  style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                                    ], style={"flex": "1"}),
                                                    html.Div([
                                                        html.Div("d (diff)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                                        html.Div("Differencing order.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                                        dcc.Input(id="param-arima-d", type="number", value=1, step=1, min=0,
                                                                  style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                                    ], style={"flex": "1"}),
                                                    html.Div([
                                                        html.Div("q (MA)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                                        html.Div("Moving-average order.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                                        dcc.Input(id="param-arima-q", type="number", value=0, step=1, min=0,
                                                                  style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                                    ], style={"flex": "1"}),
                                                ]),
                                                html.Div("Seasonal period s (months)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                                html.Div("Cycle length in SARIMA(p,d,q)(0,1,1,s).", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                                dcc.Input(id="param-arima-s", type="number", value=12, step=1, min=2,
                                                          style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                            ], style={"padding": "8px 10px", "backgroundColor": "#f0f6ff", "borderRadius": "6px", "border": "1px solid #c8dff8", "marginTop": "4px", "marginLeft": "4px"}),
                                        ]),
                                    ]),
                                    "value": "ARIMA",
                                },
                                {
                                    "label": html.Div([
                                        html.B("CSD (Classical Seasonal Decomposition)", className="mathjax-ignore"),
                                        html.Details([
                                            html.Summary("Customize parameters", style={"cursor": "pointer", "color": "#adb5bd", "fontSize": "13px", "fontWeight": "500", "marginTop": "4px", "marginLeft": "4px"}),
                                            html.Div([
                                                html.Div("Seasonal period (months)", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "2px"}),
                                                html.Div("Length of seasonal cycle. Use 12 for monthly-aggregated daily data.", style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                                                dcc.Input(id="param-csd-period", type="number", value=12, step=1, min=2,
                                                          style={"width": "100%", "fontSize": "12px", "padding": "4px 6px", "borderRadius": "4px", "border": "1px solid #ced4da", "color": "#000"}),
                                            ], style={"padding": "8px 10px", "backgroundColor": "#f0f6ff", "borderRadius": "6px", "border": "1px solid #c8dff8", "marginTop": "4px", "marginLeft": "4px"}),
                                        ]),
                                    ]),
                                    "value": "CSD",
                                },
                            ],
                            labelStyle={"display": "block", "marginBottom": "6px", "cursor": "pointer", "color": "inherit"},
                            labelClassName="metric-radio-label",
                            inputStyle={"marginRight": "8px", "marginTop": "3px", "accentColor": "#0d6efd"},
                            style={"marginBottom": "8px"},
                        ),

                                                # Clientside: sync RadioButtons → hidden RadioItems
                        dcc.Store(id="_rb-sync-dummy"),

                        dbc.Button(
                            "Calculate degradation",
                            id="run-btn", color="primary", className="w-100 mt-3"
                        ),

                        # ── Metric detail: description + equation + ref ────────
                        html.Details([
                            html.Summary(
                                "Metric detail:",
                                style={"cursor": "pointer", "color": "#B5B5B8", "fontSize": "14px", "fontWeight": "500"}
                            ),
                            html.Div([

                                html.Details([
                                    html.Summary(html.B("YoY (Year-over-Year)"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                    html.Div([
                                        "Compares daily irradiance-weighted power to the same calendar day one year prior. "
                                        "The degradation rate is the median of all year-over-year ratios after IQR-based outlier removal.",
                                        html.Div(r"$$R_i = \frac{P(t)}{P(t-1\,\text{yr})} - 1, \quad R_d = \text{median}(R_i) \times \frac{100\%}{\text{yr}}$$",
                                                 style={"color": "#555", "margin": "6px 0", "overflowX": "auto"}),
                                        html.Div([html.Sup("[1] "), "Jordan, D. et al., IEEE J. Photovoltaics 8(2), 525–531, 2018. ",
                                            html.A("10.1109/JPHOTOV.2017.2779779", href="https://doi.org/10.1109/JPHOTOV.2017.2779779",
                                                   target="_blank", style={"color": "#0d6efd", "fontSize": "11px"})],
                                            style={"fontSize": "11px", "color": "#4a6fa5", "marginTop": "6px"}),
                                    ], style={"marginTop": "4px", "paddingLeft": "12px"}),
                                ], style={"marginBottom": "6px"}),

                                html.Details([
                                    html.Summary(html.B("LR (Linear Regression)"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                    html.Div([
                                        "Fits an ordinary least-squares line to the daily power time series. "
                                        "The degradation rate is the slope normalised by mean power.",
                                        html.Div(r"$$P(t) = \beta_0 + \beta_1 t, \quad R_d = \frac{\beta_1}{\bar{P}} \times \frac{100\%}{\text{yr}}$$",
                                                 style={"color": "#555", "margin": "6px 0", "overflowX": "auto"}),
                                        html.Div("No tunable parameters.", style={"fontSize": "11px", "color": "#888", "fontStyle": "italic"}),
                                    ], style={"marginTop": "4px", "paddingLeft": "12px"}),
                                ], style={"marginBottom": "6px"}),

                                html.Details([
                                    html.Summary(html.B("HW (Holt-Winters)"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                    html.Div([
                                        "Additive Holt-Winters exponential smoothing decomposes the signal into level, trend, and seasonal components. "
                                        "A linear regression on the fitted values yields the degradation rate.",
                                        html.Div(r"$$\hat{y}(t) = L(t) + T(t) + S(t), \quad R_d = \frac{\text{slope}(\hat{y})}{\bar{\hat{y}}} \times \frac{100\%}{\text{yr}}$$",
                                                 style={"color": "#555", "margin": "6px 0", "overflowX": "auto"}),
                                        html.Div([html.Sup("[2] "), "Phinikarides, A. et al., Renew. Sustain. Energy Rev. 40, 143–152, 2014. ",
                                            html.A("10.1016/j.rser.2014.07.155", href="https://doi.org/10.1016/j.rser.2014.07.155",
                                                   target="_blank", style={"color": "#0d6efd", "fontSize": "11px"})],
                                            style={"fontSize": "11px", "color": "#4a6fa5", "marginTop": "6px"}),
                                    ], style={"marginTop": "4px", "paddingLeft": "12px"}),
                                ], style={"marginBottom": "6px"}),

                                html.Details([
                                    html.Summary(html.B("ARIMA"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                    html.Div([
                                        "Fits a SARIMA(p,d,q)(0,1,1,s) model. A linear regression on the fitted values extracts the degradation rate.",
                                        html.Div(r"$$\text{SARIMA}(p,d,q)(0,1,1,s), \quad R_d = \frac{\text{slope}(\hat{y})}{\bar{\hat{y}}} \times \frac{100\%}{\text{yr}}$$",
                                                 style={"color": "#555", "margin": "6px 0", "overflowX": "auto"}),
                                        html.Div([html.Sup("[2] "), "Phinikarides, A. et al., Renew. Sustain. Energy Rev. 40, 143–152, 2014. ",
                                            html.A("10.1016/j.rser.2014.07.155", href="https://doi.org/10.1016/j.rser.2014.07.155",
                                                   target="_blank", style={"color": "#0d6efd", "fontSize": "11px"})],
                                            style={"fontSize": "11px", "color": "#4a6fa5", "marginTop": "6px"}),
                                    ], style={"marginTop": "4px", "paddingLeft": "12px"}),
                                ], style={"marginBottom": "6px"}),

                                html.Details([
                                    html.Summary(html.B("CSD (Classical Seasonal Decomposition)"), style={"cursor": "pointer", "marginBottom": "2px"}),
                                    html.Div([
                                        "Decomposes the daily power series additively into trend, seasonal, and residual. "
                                        "A linear regression on the extracted trend gives the degradation rate.",
                                        html.Div(r"$$P(t) = T(t) + S(t) + R(t), \quad R_d = \frac{\text{slope}(T)}{\bar{T}} \times \frac{100\%}{\text{yr}}$$",
                                                 style={"color": "#555", "margin": "6px 0", "overflowX": "auto"}),
                                        html.Div([html.Sup("[2] "), "Phinikarides, A. et al., Renew. Sustain. Energy Rev. 40, 143–152, 2014. ",
                                            html.A("10.1016/j.rser.2014.07.155", href="https://doi.org/10.1016/j.rser.2014.07.155",
                                                   target="_blank", style={"color": "#0d6efd", "fontSize": "11px"})],
                                            style={"fontSize": "11px", "color": "#4a6fa5", "marginTop": "6px"}),
                                    ], style={"marginTop": "4px", "paddingLeft": "12px"}),
                                ], style={"marginBottom": "0px"}),

                            ], style={
                                "marginTop": "10px",
                                "padding": "12px 14px",
                                "border": "1px solid #e0e0e0",
                                "borderRadius": "10px",
                                "backgroundColor": "#e8f3ff",
                                "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                                "color": "#1A64BE",
                                "fontSize": "13px",
                                "lineHeight": "1.6",
                            }),
                        ], style={"marginTop": "10px"}),

                        ],
                        lg=4, md=12, sm=12, xs=12
                    ),

                    # 2. Slider Column (Right Side)
                    dbc.Col(
                        [
                            dcc.Loading(
                                type="circle",
                                color="#0d6efd",

                                children=html.Div([

                                    html.Div(
                                        id="degradation-output",
                                        className="p-2 border",
                                        style={
                                            "minHeight": "170px",
                                            "marginTop": "5px"
                                        }
                                    )
                                ])
                            )
                            
                        ],
                        lg=8, md=12, sm=12, xs=12,
                    )
                ]
            )
           
        ])
    ])
], style={
        # 'paddingLeft': '18%',   
        # 'paddingRight': '12%'   
})
])



app.clientside_callback(
    """
    function(tz, lip, out, cs) {
        var vals = [];
        if (tz)  vals.push("timezone");
        if (lip) vals.push("low-irra-power");
        if (out) vals.push("outlier");
        if (cs)  vals.push("clearsky");
        return vals;
    }
    """,
    Output("filter-options", "value"),
    Input("cb-timezone", "value"),
    Input("cb-low-irra-power", "value"),
    Input("cb-outlier", "value"),
    Input("cb-clearsky", "value"),
)

# (RadioButton sync removed — using dbc.RadioItems directly)


# ==================================================
# upload data
# ==================================================

@app.callback(
    Output("upload-status-output", "children"),
    Output("data-source-store", "data"),
    Output("data-summary-output", "children"),
    Output("stored-data-file-name", "data"),
    Input("upload-data", "filename"),
    prevent_initial_call=False
)
def update_upload_status(filename):
    """Displays a status message when a file is uploaded."""
    if filename:

        msg = dbc.Alert(
            [
                html.I(className="bi bi-check-circle-fill me-2"),  # Bootstrap icon
                html.Span(f"File selected: '{filename}'")
            ],
            color="success",
            className="d-flex align-items-center shadow-sm rounded px-3 py-2 slide-in-top",
            style={"fontSize": "0.9rem"}
        )
        return [msg, 'upload', '', filename]
    
    # Return empty div on initial load or if upload fails/resets
    return [html.Div("Awaiting file...", className="text-muted small"), None, '', None]


# ==================================================
# run data filter
# ==================================================

@app.callback(
    Output("data-filter-output", "children"),
    Output("dataframe-filtered", "data"),

    Input("filter-btn", "n_clicks"),
    Input("upload-data", "filename"),
    Input("load-example-btn-1", "n_clicks"),
    Input("load-example-btn-2", "n_clicks"),
    Input("load-example-btn-3", "n_clicks"),

    State("filter-options", "value"),
    State("mapped-vars-store", "data"),
    State("dataframe-store", "data"),
    State("param-gamma", "value"),
    State("param-irr-thresh", "value"),
    State("param-power-ratio", "value"),
    State("param-norm-lower", "value"),
    State("param-norm-upper-pct", "value"),
    State("param-iqr-multiplier", "value"),
    State("param-cs-smooth", "value"),
    State("param-cs-energy", "value"),

    prevent_initial_call=True
)
def run_filter(filter_clicks, upload_clicks,
        example1_clicks, example2_clicks, example3_clicks, selected_filters, mapped_variables_dict, df_json,
        gamma, irr_thresh, power_ratio, norm_lower, norm_upper_pct, iqr_multiplier,
        cs_smooth, cs_energy):

    trigger = ctx.triggered_id

    # No data loaded yet
    if not df_json:   # None or empty {}
        # Only show the warning if the user actually clicked the Filter button.
        # For other triggers (upload reset, example reset), just clear silently.
        if trigger == "filter-btn":
            return [
                _no_data_alert(
                    "Please click 'Analyze Data' first to load your dataset before filtering."
                ),
                None,
            ]
        return ['', None]

    if trigger == "upload-data" or (trigger and trigger.startswith("load-example-btn")):
        return ['', None]

    # =========================
    # Load dataframe
    # =========================
    df = _df_from_store(df_json)

    # =========================
    # Get irradiance column
    # =========================
    irra_key = mapped_variables_dict["Irradiance"] if mapped_variables_dict else None

    if irra_key is None or irra_key not in df.columns:
        return "❌ Irradiance column not found. Please map it first."

    # =========================
    # Core processing
    # =========================
    gamma = gamma if gamma is not None else -0.004
    irr_thresh = irr_thresh if irr_thresh is not None else 300
    power_ratio = power_ratio if power_ratio is not None else 0.02
    norm_lower = norm_lower if norm_lower is not None else 0.01
    norm_upper_pct = norm_upper_pct if norm_upper_pct is not None else 99

    # =========================
    # Step 0: Basic value filter (always applied, not optional)
    # Removes physically implausible sensor readings before anything else.
    # =========================
    print("[basic_value_filter] applying range sanity checks...")
    bv_normal, bv_outlier = basic_value_filter(df, mapped_variables_dict)
    df = df.loc[bv_normal].copy()
    print(f"[basic_value_filter] kept {len(df):,} / {len(bv_normal) + len(bv_outlier):,} pts")

    # =========================
    # Clear-sky filter: run on RAW df BEFORE normalization
    # so the full intraday irradiance shape is preserved for
    # smoothness scoring (bell-curve detection).
    # =========================
    clearsky_mask = pd.Series(True, index=df.index)
    if "clearsky" in selected_filters:
        cs_smooth = cs_smooth if cs_smooth is not None else 0.3
        cs_energy = cs_energy if cs_energy is not None else 0.5
        normal_idx, outlier_idx = clear_sky_filter(df, irra_key,
                                                    smoothness_threshold=cs_smooth,
                                                    energy_threshold=cs_energy)
        clearsky_mask = df.index.isin(normal_idx)

    df_filtered = normalize(df, mapped_variables_dict, gamma=gamma)

    # start with all data as valid, apply clear-sky mask immediately
    current_mask = pd.Series(clearsky_mask, index=df_filtered.index)

    filter_stats = []

    # =========================
    # Optional filters
    # =========================
    if "timezone" in selected_filters:
        try:
            df_filtered.index = pd.to_datetime(df_filtered.index)
            df_filtered.index = df_filtered.index.tz_localize("UTC").tz_convert("US/Pacific")
            filter_stats.append("Timezone corrected (UTC → US/Pacific)")
        except Exception:
            filter_stats.append("⚠️ Timezone correction failed")

    if "clearsky" in selected_filters:
        removed = (~clearsky_mask).sum()
        filter_stats.append(f"Clear-sky filter removed {removed} points")

    if "low-irra-power" in selected_filters:
        normal_idx, outlier_idx = low_irra_power_filter(
            df_filtered, mapped_variables_dict,
            irr_thresh=irr_thresh, power_ratio=power_ratio,
            norm_lower=norm_lower, norm_upper_pct=norm_upper_pct
        )

        mask = df_filtered.index.isin(normal_idx)
        removed = (~mask & current_mask).sum()

        current_mask &= mask
        filter_stats.append(f"Low irra-power filter removed {removed} points")

    if "outlier" in selected_filters:
        iqr_multiplier = iqr_multiplier if iqr_multiplier is not None else 1.5
        normal_idx, outlier_idx = identify_outliers_iqr(df_filtered, "norm", iqr_multiplier=iqr_multiplier)

        mask = df_filtered.index.isin(normal_idx)
        removed = (~mask & current_mask).sum()

        current_mask &= mask
        filter_stats.append(f"IQR outlier filter removed {removed} points")

    # =========================
    # Final indices
    # =========================
    normal_indices = df_filtered.index[current_mask]
    outlier_indices = df_filtered.index[~current_mask]

    # =========================
    # Counts
    # =========================
    n_total = len(df_filtered)
    n_good = len(normal_indices)
    n_bad = len(outlier_indices)

    # =========================
    # Pie chart (clean + compact)
    # =========================
    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=["High-quality data", "Filtered data"],
                values=[n_good, n_bad],
                hole=0.5,
                marker=dict(colors=["#0070C0", "#A6CAEC"]),
                textinfo="percent",
                hoverinfo="label+percent"
            )
        ]
    )

    pie_fig.update_layout(
        height=120,
        margin=dict(t=10, b=10, l=10, r=10),

        showlegend=True,

        legend=dict(
            orientation="v",   # vertical legend
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,            # push legend slightly outside chart
            font=dict(size=11)
        )
    )

    # =========================
    # Scatter plot (unchanged)
    # =========================
    scatter_fig = go.Figure()

    scatter_fig.add_trace(
        go.Scattergl(
            x=df_filtered.loc[outlier_indices].index,
            y=df_filtered.loc[outlier_indices]["norm"],
            mode="markers",
            marker=dict(size=5, opacity=0.3, color="#A6CAEC"),
            name="Filtered data"
        )
    )

    scatter_fig.add_trace(
        go.Scattergl(
            x=df_filtered.loc[normal_indices].index,
            y=df_filtered.loc[normal_indices]["norm"],
            mode="markers",
            marker=dict(size=5, opacity=0.4, color="#0070C0"),
            name="High-quality data"
        )
    )

    scatter_fig.update_layout(
        title="Normalized Power Over Time",
        xaxis_title="Time",
        yaxis_title="Normalized Power (W)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=60),  # 🔼 increase bottom margin
        height=350,

        legend=dict(
            orientation="h",        # horizontal legend
            yanchor="top",
            y=-0.25,                # push below chart
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        )
    )

    # =========================
    # summary block
    # =========================
    summary_block = html.Div([

        html.H5("Filtering Summary", style={"marginBottom": "10px"}),

        html.Ul([
            html.Li([
                html.Span("Total data points: "),
                html.B(f"{n_total}")
            ]),

            html.Li([
                html.Span("High-quality data: "),
                html.B(f"{n_good} ({n_good/n_total:.1%})")
            ]),

            html.Li([
                html.Span("Filtered data: "),
                html.B(f"{n_bad} ({n_bad/n_total:.1%})")
            ]),
        ], style={
            "paddingLeft": "20px",
            "marginBottom": "10px"
        }),

        html.Details([
            html.Summary(
                "Show filtering details",
                style={
                    "color": "gray",
                    "cursor": "pointer",
                    "fontSize": "0.9rem"
                }
            ),
            html.Ul(
                [html.Li(s) for s in filter_stats],
                style={"marginTop": "8px", "color": "gray",}
            )
        ])

    ], style={
        "paddingLeft": "15px"
    })

    # =========================
    # Layout
    # =========================
    filter_layout = html.Div([

        dbc.Row([
            dbc.Col(summary_block, md=6),
            dbc.Col(dcc.Graph(figure=pie_fig), md=6)
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=scatter_fig), md=12)
        ])
    ], className="slide-in-up")

    df_filtered_store = df_filtered.loc[normal_indices]

    return [filter_layout,
            df_filtered_store.to_json(date_format="iso", orient="split")
            ]

# ==================================================
# run degradation calculation
# ==================================================
@app.callback(
    Output("degradation-output", "children"),
    Output("run-btn", "disabled", allow_duplicate=True),
    Output("run-btn", "children", allow_duplicate=True),

    Input("run-btn", "n_clicks"),
    Input("upload-data", "filename"),
    Input("load-example-btn-1", "n_clicks"),
    Input("load-example-btn-2", "n_clicks"),
    Input("load-example-btn-3", "n_clicks"),

    State("dataframe-filtered", "data"),
    State("mapped-vars-store", "data"),
    State("metric-selected-visible", "value"),
    State("param-yoy-window", "value"),
    State("param-yoy-iqr", "value"),
    State("param-hw-period", "value"),
    State("param-arima-p", "value"),
    State("param-arima-d", "value"),
    State("param-arima-q", "value"),
    State("param-arima-s", "value"),
    State("param-csd-period", "value"),

    prevent_initial_call=True
)
def analyze_uploaded_data_callback(
        degradation_clicks, upload_clicks,
        example1_clicks, example2_clicks, example3_clicks,
        df_filtered_json,
        mapped_variables_dict,
        selected_metric,
        yoy_window, yoy_iqr,
        hw_period,
        arima_p, arima_d, arima_q, arima_s,
        csd_period,
):
    
    trigger = ctx.triggered_id
    
    if trigger in [
        "load-example-btn-1",
        "load-example-btn-2",
        "load-example-btn-3",
        "upload-data"
    ]:
        return ['', False, "Analyze Data"]

    # No filtered data yet
    if not df_filtered_json:   # None or empty {}
        if trigger == "run-btn":
            return [
                _no_data_alert(
                    "Please click 'Filter data' first before running degradation analysis."
                ),
                False,
                "Analyze Data",
            ]
        return ['', False, "Analyze Data"]
    
    df_filtered = _df_from_store(df_filtered_json)

    irra_key = mapped_variables_dict["Irradiance"] if mapped_variables_dict else None

    if irra_key is None or irra_key not in df_filtered.columns:
        return ["❌ Irradiance column not found. Please map it first.",
                False,
            "Analyze Data"]

    daily_data = aggregate_daily(df_filtered, irra_key)

    if selected_metric == "YOY":
        rd, fig = compute_yoy(daily_data,
                              rolling_window=yoy_window if yoy_window else 30,
                              iqr_multiplier=yoy_iqr if yoy_iqr else 1.5)

    elif selected_metric == "LR":
        rd, fig = compute_lr(daily_data)

    elif selected_metric == "HW":
        rd, fig = compute_hw(daily_data,
                             period=hw_period if hw_period else 12)

    elif selected_metric == "ARIMA":
        rd, fig = compute_arima(daily_data,
                                p=arima_p if arima_p is not None else 1,
                                d=arima_d if arima_d is not None else 1,
                                q=arima_q if arima_q is not None else 0,
                                seasonal_period=arima_s if arima_s else 12)

    elif selected_metric == "CSD":
        rd, fig = compute_csd(daily_data,
                              period=csd_period if csd_period else 12)

    else:
        raise ValueError(f"Unknown metric: {selected_metric}")

    # =========================
    # Duration calculation
    # =========================
    start_date = df_filtered.index.min()
    end_date = df_filtered.index.max()

    duration_days = (end_date - start_date).days
    duration_years = duration_days / 365.25

    # =========================
    # Summary block
    # =========================
    summary_block = html.Div([

        html.H5("Degradation Summary", style={"marginBottom": "10px"}),

        html.Ul([

            html.Li([
                html.Span("Metric: "),
                html.B(selected_metric)
            ]),

            html.Li([
                html.Span("Annual power degradation rate: "),
                html.B(f"{rd/100:.2%}/year") 
            ]),

            html.Li([
                html.Span("Measurement duration: "),
                html.B(f"{duration_years:.1f} years ")
            ]),

        ], style={
            "paddingLeft": "20px",
            "marginBottom": "10px"
        })

    ], style={
        "paddingLeft": "15px"
    })

    degradation_layout = html.Div([

        dbc.Row([
            dbc.Col(summary_block, md=6)
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig), md=12)
        ])
    ], className="slide-in-up")

    return [degradation_layout,
        False,
            "Analyze Data"]


app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks || n_clicks === 0) {
            return [false, "Analyze Data"];
        }
        return [true, "Analyzing…"];
    }
    """,
    [
        Output("analyze-btn", "disabled"),
        Output("analyze-btn", "children")
    ],
    Input("analyze-btn", "n_clicks"),
    prevent_initial_call=True
)

# ==================================================
# data upload and plot raw data
# ==================================================
@app.callback(
    Output("data-summary-output", "children", allow_duplicate=True),
    Output("mapped-vars-store", "data"),
    Output("dataframe-store", "data"),
    Output("code-read-store", "data"),
    Output("analyze-btn", "disabled", allow_duplicate=True),
    Output("analyze-btn", "children", allow_duplicate=True),
    Output("data-source-store", "data", allow_duplicate=True),
    Output("upload-status-output", "children", allow_duplicate=True),
    Output("stored-data-file-name", "data", allow_duplicate=True),
    Input("analyze-btn", "n_clicks"),
    Input("load-example-btn-1", "n_clicks"),
    Input("load-example-btn-2", "n_clicks"),
    Input("load-example-btn-3", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("dataframe-store", "data"),
    State("data-source-store", "data"),
    State("stored-data-file-name", "data"),
    prevent_initial_call=True
)
def analyze_uploaded_data_callback(
        analyze_clicks,
        example_clicks_1,
        example_clicks_2,
        example_clicks_3,
        contents,
        filename,
        stored_df_json,
        data_source,
        stored_file_name
):

    trigger = ctx.triggered_id

    # ------------------------------------------------
    # 1. Load example dataset (no analysis yet)
    # ------------------------------------------------
    if trigger in ["load-example-btn-1", "load-example-btn-2", "load-example-btn-3"]:

        file_map = {
            "load-example-btn-1": "sys_1278_downsampled.parquet",
            "load-example-btn-2": "sys_1403_part1_downsampled.parquet",
            "load-example-btn-3": "sys_1422_downsampled.parquet",
        }

        example_filename = file_map.get(trigger)

        try:
            df = pd.read_parquet(f"data/{example_filename}")
            df_json = df.to_json(date_format='iso', orient='split')

            output_msg = dbc.Alert(
                [
                    html.I(className="bi bi-check-circle-fill me-2"),
                    html.Span(f"{example_filename} loaded successfully")
                ],
                color="success",
                className="d-flex align-items-center shadow-sm rounded px-3 py-2 slide-in-top",
                style={"fontSize": "0.9rem"}
            )

        except Exception as e:
            return (
                html.Div(f"Error loading example data: {e}", className="alert alert-danger"),
                {}, None, "", False, "Analyze Data", None, '', example_filename
            )

        return (
            html.Div("", className="text-muted"),
            {}, df_json, "", False, "Analyze Data", 'example', output_msg, example_filename
        )

    # ------------------------------------------------
    # 2. Analyze Data button clicked
    # ------------------------------------------------
    if trigger == "analyze-btn":
        print(data_source)

        if data_source == "upload" and contents is not None:
            df, summary_table, mapped_variables_dict, code_read = parse_contents(contents, filename)

            if df is None:
                return summary_table, {}, None, "", False, "Analyze Data", None, '', stored_file_name

        elif data_source == "example" and stored_df_json is not None:
            try:
                df = _df_from_store(stored_df_json)

                df, summary_table, mapped_variables_dict, code_read = parse_contents(df=df)

            except Exception as e:
                return (
                    html.Div(f"Error processing stored dataset: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Analyze Data", None, '', stored_file_name
                )

        else:
            return (
                _no_data_alert(
                    "Please upload a file or click one of the example buttons, "
                    "then click 'Analyze Data'."
                ),
                {}, None, "", False, "Analyze Data", None, '', filename
            )

        # Convert dataframe to JSON
        try:
            df_json = df.to_json(date_format='iso', orient='split')

        except Exception as e:
            return (
                html.Div(
                    f"Error converting DataFrame to JSON: {e}",
                    className="alert alert-danger"
                ),{ }, None, "", False, "Analyze Data", None, '', stored_file_name
            )

        # ----------------------------------
        # Generate figures of raw data
        # ----------------------------------
        figures_output = html.Div()

        try:
            if df is not None and mapped_variables_dict:
                figures_output, err = make_overview_figures(df, mapped_variables_dict)
                figures_output = html.Div(figures_output) 
        except Exception:
            figures_output = html.Div("Figure generation failed.", className="text-danger")

        # -------------------------
        # Merge output: table + figs
        # -------------------------
        combined_output = html.Div([

            # Summary
            html.Div(
                summary_table,
                style={'fontSize': '10pt'},
                # className="p-2 border"
            ),

            # Figures title
            html.H5("Figures of raw data", className="mt-2"),

            # Figures
            figures_output

        ], className="slide-in-up")

        return (
            combined_output,
            mapped_variables_dict,
            df_json,
            code_read,
            False,
            "Analyze Data", None, '', stored_file_name
        )

    # ------------------------------------------------
    # Fallback
    # ------------------------------------------------
    return "", {}, None, "", html.Div(), False, "Analyze Data", None , '', stored_file_name


app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks || n_clicks === 0) {
            return [false, "Analyze Data"];
        }
        return [true, "Analyzing…"];
    }
    """,
    [
        Output("run-btn", "disabled"),
        Output("run-btn", "children")
    ],
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True
)

# ==================================================
# get the code panel
# ==================================================
@app.callback(
    Output("floating-panel", "style"),
    Input("floating-btn", "n_clicks"),
    Input("close-panel-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_panel(open_clicks, close_clicks):
    trigger = ctx.triggered_id

    base_style = {
        "position": "fixed",
        "bottom": "100px",
        "right": "30px",
        "width": "300px",
        "padding": "15px",
        "backgroundColor": "white",
        "borderRadius": "10px",
        "boxShadow": "0px 4px 15px rgba(0,0,0,0.2)",
        "zIndex": 1000
    }

    if trigger == "floating-btn":
        return {**base_style, "display": "block"}

    if trigger == "close-panel-btn":
        return {**base_style, "display": "none"}

    return {**base_style, "display": "none"}


@app.callback(
    Output("code-preview", "children", allow_duplicate=True),
    Output("download-link", "href", allow_duplicate=True),
    Output("download-link", "style", allow_duplicate=True),
    Input("upload-data", "filename"),
    Input("analyze-btn", "n_clicks"),
    Input("load-example-btn-1", "n_clicks"),
    Input("load-example-btn-2", "n_clicks"),
    Input("load-example-btn-3", "n_clicks"),
    prevent_initial_call=True,
)
def clear_code_panel_on_new_data(*_):
    """Reset the generated-code preview and hide the download link
    whenever the user uploads a new file, clicks Analyze, or loads an example."""
    hidden_style = {"display": "none", "marginTop": "10px"}
    return None, "", hidden_style


@app.callback(
    Output("code-preview", "children", allow_duplicate=True),
    Output("download-link", "href", allow_duplicate=True),
    Output("download-link", "style", allow_duplicate=True),
    Input("generate-code-btn", "n_clicks"),
    State("stored-data-file-name", "data"),
    State("mapped-vars-store", "data"),
    State("filter-options", "value"),
    State("metric-selected-visible", "value"),
    prevent_initial_call=True
)
def generate_code(n,filename, mapped_variables_dict, selected_filters, selected_metric):
    print(filename)

    # Generate code (this triggers loading spinner automatically)
    clean_code = get_full_code(filename, mapped_variables_dict, selected_filters, selected_metric)

    # Small delay so the loading spinner is visible and the reveal feels smooth
    time.sleep(2)

    # Preview (first ~20 lines)
    preview_lines = "\n".join(clean_code.splitlines()[:20]) + "\n..."

    preview = html.Div(
        children=[
            # Inject the keyframes once; harmless if rendered multiple times
            dcc.Markdown(
                """<style>
                @keyframes pvcopilot-fade-in {
                    from { opacity: 0; transform: translateY(4px); }
                    to   { opacity: 1; transform: translateY(0); }
                }
                </style>""",
                dangerously_allow_html=True,
            ),
            html.Pre(
                preview_lines,
                style={
                    "whiteSpace": "pre-wrap",
                    "fontSize": "12px",
                    "backgroundColor": "#f8f9fa",
                    "padding": "8px",
                    "borderRadius": "6px",
                    "maxHeight": "200px",
                    "overflowY": "auto",
                    "animation": "pvcopilot-fade-in 0.4s ease-out both",
                },
            ),
        ]
    )

    # Create downloadable file
    b64 = base64.b64encode(clean_code.encode()).decode()
    href = f"data:text/plain;base64,{b64}"

    # Show download button ONLY after code is ready
    download_style = {
        "display": "inline-block",
        "marginTop": "10px",
        "color": "#0070C0",
        "cursor": "pointer",
        "animation": "pvcopilot-fade-in 0.4s ease-out 0.1s both",
    }

    return preview, href, download_style

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)