
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
from page_supporting_files.analysis_utils import parse_contents, generate_degradation_code_and_execute, plot_power_vs_time, generate_full_code, plot_outlier_vs_time, get_filtered_display_string, build_data_summary_block
from dash import callback_context as ctx
from io import StringIO
import traceback
from page_supporting_files.analysis_utils import make_overview_figures, normalize, low_irra_power_filter, aggregate_daily, compute_yoy, get_full_code
from page_supporting_files.pvcopilot_filter_functions import identify_outliers_iqr
import base64

# --- Define Color Variables ---
MAJOR_CARD_BACKGROUND = "#F8F8F8"
MAJOR_CARD_FONT_COLOR = "black"
BODY_CARD_BACKGROUND = "white" 
CODE_BLOCK_BACKGROUND = "#f8f9fa"

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

    html.Hr(),
    html.Div([
        html.H1("PV-Copilot"), 
    ], style={
        'width': '100%',
        'padding-left': '10px',
        'padding-right': '10px',
        'textAlign': 'center'}),
    html.Hr(),

    html.Div([

        # Floating button
        html.Button(
            "Get the code ⬇",
            id="floating-btn",
            n_clicks=0,
            style={
                "position": "fixed",
                "bottom": "30px",
                "right": "30px",
                "padding": "10px 16px",
                "borderRadius": "30px",
                "backgroundColor": "#000000",
                "color": "white",
                "fontSize": "16px",
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
                            dbc.Button("Generate full code to run", id="generate-code-btn", color="primary", size="sm", 
                                    style={"fontSize": "16px", "fontWeight": "500"}),
                            html.Br(),  # 👈 add break
                            html.Small(
                                        "(It typically takes 2-6 seconds)",
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
                    "Download Code",
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
        ], xs=12, sm=12, md=12, lg=9, xl=9),

        dbc.Col([
            html.Img(src=app.get_asset_url('pvcopilot_logo.png'),
            style={'width': '90%'}),
        ], xs=9, sm=8, md=6, lg=3, xl=3, className="text-end"),
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
                        html.Label("Upload your data (.csv, .xls, .parquet)"),
                        dcc.Upload(
                            id="upload-data",
                            accept=".csv, text/csv, .xls, .xlsx, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, .parquet",
                            children=html.Div(["Drag and Drop or ", html.A("Select Files", style={"color": "blue"})]),
                            style={"width": "100%", "height": "60px", "lineHeight": "60px",
                                   "borderWidth": "1px", "borderStyle": "dashed",
                                   "textAlign": "center"}
                        ),
                        html.Div(id="upload-status-output", style={"marginTop": "5px"}),
                        html.Div(
                        [
                            dbc.Button(
                                "Example Data 1",
                                id="load-example-btn",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="mt-2"
                            )
                        ],
                        style={"marginTop": "10px"}
                        ),
                        dbc.Button("Analyze Data", id="analyze-btn", color="primary", className="w-100 mt-3"),
                        html.Small(
                            "(Analysis typically takes 2-4 seconds)",
                            className="text-muted small"
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
                            dbc.Checklist(
                                id="filter-options",
                                options=[
                                    {"label": "Time zone & DST correction", "value": "timezone"},
                                    {"label": "Low irradiance/power filter", "value": "low-irra-power"},
                                    {"label": "Outlier removal", "value": "outlier"},
                                    {"label": "Clear-sky filter", "value": "clearsky"},
                                ],
                                value=['timezone', "low-irra-power", "outlier"],
                                inline=False
                            ),
                            dbc.Button("Filter data", id="filter-btn", color="primary", className="w-100 mt-3")
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
                        # degradation metric 
                        dbc.RadioItems(
                            id="metric-selected",
                            options=[
                                {"label": "YoY (Year-over-Year)", "value": "yoy"},
                                {"label": "Linear regression", "value": "linear", "disabled": True},
                                {"label": "PV-Pro", "value": "pvpro", "disabled": True},
                                {"label": "PVUSA", "value": "pvusa", "disabled": True},
                            ],
                            value="yoy",
                            inline=False
                        ),

                        # LLM Temperature
                        # html.Label("LLM Temperature: 1.0", id="temp-label", className="fw-bold"),
                        # dcc.Slider(
                        #     id='temp-slider',
                        #     min=0,
                        #     max=1,
                        #     step=0.1,
                        #     # SETTING THE START VALUE TO 1.0
                        #     value=1.0, 
                        #     # ONLY show marks for 0, 0.5, and 1
                        #     marks={
                        #         0: {'label': '0'},
                        #         0.5: {'label': '0.5'},
                        #         1: {'label': '1'},
                        #     },
                        #     className="mb-4"
                        # ),

                        dbc.Button(
                            "RUN ANALYSIS",
                            id="run-btn", color="primary", className="w-100 mt-3"
                        ),
                        html.Small(
                            "(Analysis typically takes 2-4 seconds)",
                            className="text-muted small"
                        )],
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



# ==================================================
# upload data
# ==================================================

@app.callback(
    Output("upload-status-output", "children"),
    Output("data-source-store", "data"),
    Output("data-summary-output", "children"),
    Input("upload-data", "filename"),
    prevent_initial_call=False
)
def update_upload_status(filename):
    """Displays a status message when a file is uploaded."""
    if filename:
        return [dbc.Alert(
            [
                html.I(className="bi bi-check-circle-fill me-2"),  # Bootstrap icon
                html.Span(f"File selected: '{filename}'")
            ],
            color="success",
            className="d-flex align-items-center shadow-sm rounded px-3 py-2",
            style={"fontSize": "0.9rem"}
        ), 'upload', '']
    
    # Return empty div on initial load or if upload fails/resets
    return [html.Div("Awaiting file...", className="text-muted small"), None, '']


# ==================================================
# run data filter
# ==================================================

@app.callback(
    Output("data-filter-output", "children"),
    Output("dataframe-filtered", "data"),
    Input("filter-btn", "n_clicks"),
    Input("upload-data", "filename"),
    Input("load-example-btn", "n_clicks"),
    State("filter-options", "value"),
    State("mapped-vars-store", "data"),
    State("dataframe-store", "data"), 
    prevent_initial_call=True
)
def run_filter(filter_clicks, upload_clicks,
        example_clicks, selected_filters, mapped_variables_dict, df_json):

    trigger = ctx.triggered_id

    if df_json is None:
        return ['', None]
    
    if trigger == 'load-example-btn' or trigger == 'upload-data':
        return ['', None]

    # =========================
    # Load dataframe
    # =========================
    df = pd.read_json(df_json, orient='split')

    # =========================
    # Get irradiance column
    # =========================
    irra_key = mapped_variables_dict["Irradiance"] if mapped_variables_dict else None

    if irra_key is None or irra_key not in df.columns:
        return "❌ Irradiance column not found. Please map it first."

    # =========================
    # Core processing
    # =========================
    df_filtered = normalize(df, mapped_variables_dict)

    # start with all data as valid
    current_mask = pd.Series(True, index=df_filtered.index)

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

    if "low-irra-power" in selected_filters:
        normal_idx, outlier_idx = low_irra_power_filter(df_filtered, mapped_variables_dict)

        mask = df_filtered.index.isin(normal_idx)
        removed = (~mask & current_mask).sum()

        current_mask &= mask
        filter_stats.append(f"Low irra-power filter removed {removed} points")

    if "outlier" in selected_filters:
        normal_idx, outlier_idx = identify_outliers_iqr(df_filtered, "norm")

        mask = df_filtered.index.isin(normal_idx)
        removed = (~mask & current_mask).sum()

        current_mask &= mask
        filter_stats.append(f"IQR outlier filter removed {removed} points")

    if "clearsky" in selected_filters:
        normal_idx, outlier_idx = clear_sky(df_filtered, "norm")

        mask = df_filtered.index.isin(normal_idx)
        removed = (~mask & current_mask).sum()

        current_mask &= mask
        filter_stats.append(f"Clear-sky filter removed {removed} points")

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
            html.Li(f"Total data points: {n_total}"),
            html.Li(f"High-quality data: {n_good} ({n_good/n_total:.1%})"),
            html.Li(f"Filtered data: {n_bad} ({n_bad/n_total:.1%})"),
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
                style={"marginTop": "8px"}
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
    ])

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
    Input("load-example-btn", "n_clicks"),
    State("dataframe-filtered", "data"),
    State("mapped-vars-store", "data"),
    State("metric-selected", "value"),
    
    prevent_initial_call=True
)
def analyze_uploaded_data_callback(
        degradation_clicks, upload_clicks,
        example_clicks,
        df_filtered_json,
        mapped_variables_dict,
        selected_metric,

):
    
    trigger = ctx.triggered_id
    
    if trigger == 'load-example-btn' or trigger == 'upload-data':
        return ['', False, "Analyze Data"]
    
    df_filtered = pd.read_json(df_filtered_json, orient='split')

    irra_key = mapped_variables_dict["Irradiance"] if mapped_variables_dict else None

    if irra_key is None or irra_key not in df_filtered.columns:
        return ["❌ Irradiance column not found. Please map it first.",
                False,
            "Analyze Data"]

    daily_data = aggregate_daily(df_filtered, irra_key)

    rd, yoy_dist = compute_yoy(daily_data)

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
                html.Span("Annual degradation rate: "),
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

    trend = daily_data.rolling(30, center=True).mean()

    trend_fig = go.Figure()

    # Daily points
    trend_fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data,
            mode="markers",
            marker=dict(size=5, opacity=0.7, color="#A6CAEC"),
            name="Daily aggregated power"
        )
    )

    # Trend line
    trend_fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend,
            mode="lines",
            line=dict(color="#0070C0", width=2),
            name="Trend (30-day)"
        )
    )

    trend_fig.update_layout(
        title="Trend",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    degradation_layout = html.Div([

        dbc.Row([
            dbc.Col(summary_block, md=6),
            # dbc.Col(dcc.Graph(figure=pie_fig), md=6)
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=trend_fig), md=12)
        ])
    ])

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
    Input("analyze-btn", "n_clicks"),
    Input("load-example-btn", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("dataframe-store", "data"),
    State("data-source-store", "data"),
    prevent_initial_call=True
)
def analyze_uploaded_data_callback(
        analyze_clicks,
        example_clicks,
        contents,
        filename,
        stored_df_json,
        data_source
):

    trigger = ctx.triggered_id

    # ------------------------------------------------
    # 1. Load example dataset (no analysis yet)
    # ------------------------------------------------
    if trigger == "load-example-btn":

        try:
            df = pd.read_parquet("data/sys_1278_downsampled.parquet")
            df_json = df.to_json(date_format='iso', orient='split')
            output_msg = dbc.Alert(
                [
                    html.I(className="bi bi-check-circle-fill me-2"),
                    html.Span("Example dataset 1 selected")
                ],
                color="success",
                className="d-flex align-items-center shadow-sm rounded px-3 py-2",
                style={"fontSize": "0.9rem"}
            )

        except Exception as e:
            return (
                html.Div(f"Error loading example data: {e}", className="alert alert-danger"),
                {}, None, "", False, "Analyze Data", None, ''
            )

        return (
            html.Div("",className="text-muted"),
            {}, df_json, "", False, "Analyze Data", 'example', output_msg
        )

    # ------------------------------------------------
    # 2. Analyze Data button clicked
    # ------------------------------------------------
    if trigger == "analyze-btn":
        print(data_source)

        if data_source == "upload" and contents is not None:
            df, summary_table, mapped_variables_dict, code_read = parse_contents(contents, filename)

            if df is None:
                return summary_table, {}, None, "", False, "Analyze Data", None, ''

        elif data_source == "example" and stored_df_json is not None:
            try:
                if isinstance(stored_df_json, dict):
                    df = pd.DataFrame(**stored_df_json)
                else:
                    df = pd.read_json(stored_df_json, orient='split')

                df, summary_table, mapped_variables_dict, code_read = parse_contents(df=df)

            except Exception as e:
                return (
                    html.Div(f"Error processing stored dataset: {e}", className="alert alert-danger"),
                    {}, None, "", False, "Analyze Data", None, ''
                )

        else:
            return (
                html.Div("Upload a file or load the example dataset, then click 'Analyze Data'."),
                {}, None, "", False, "Analyze Data", None, ''
            )

        # Convert dataframe to JSON
        try:
            df_json = df.to_json(date_format='iso', orient='split')

        except Exception as e:
            return (
                html.Div(
                    f"Error converting DataFrame to JSON: {e}",
                    className="alert alert-danger"
                ),{ }, None, "", False, "Analyze Data", None, ''
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

        ])

        return (
            combined_output,
            mapped_variables_dict,
            df_json,
            code_read,
            False,
            "Analyze Data", None, ''
        )

    # ------------------------------------------------
    # Fallback
    # ------------------------------------------------
    return "", {}, None, "", html.Div(), False, "Analyze Data", None , ''


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


@app.callback(
    Output("floating-panel", "style"),
    Input("floating-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_panel(n):
    if n % 2 == 1:
        return {
            "position": "fixed",
            "bottom": "100px",
            "right": "30px",
            "width": "300px",
            "padding": "15px",
            "backgroundColor": "white",
            "borderRadius": "10px",
            "boxShadow": "0px 4px 15px rgba(0,0,0,0.2)",
            "display": "block",
            "zIndex": 1000
        }
    return {"display": "none"}


@app.callback(
    Output("code-preview", "children"),
    Output("download-link", "href"),
    Output("download-link", "style"),
    Input("generate-code-btn", "n_clicks"),
    State("upload-data", "filename"),
    State("mapped-vars-store", "data"),
    State("filter-options", "value"),
    State("metric-selected", "value"),
    prevent_initial_call=True
)
def generate_code(n,filename, mapped_variables_dict, selected_filters, selected_metric):

    # Generate code (this triggers loading spinner automatically)
    clean_code = get_full_code(filename, mapped_variables_dict, selected_filters, selected_metric)

    # Preview (first ~20 lines)
    preview_lines = "\n".join(clean_code.splitlines()[:20]) + "\n..."

    preview = html.Pre(
        preview_lines,
        style={
            "whiteSpace": "pre-wrap",
            "fontSize": "12px",
            "backgroundColor": "#f8f9fa",
            "padding": "8px",
            "borderRadius": "6px",
            "maxHeight": "200px",
            "overflowY": "auto"
        }
    )

    # Create downloadable file
    b64 = base64.b64encode(clean_code.encode()).decode()
    href = f"data:text/plain;base64,{b64}"

    # Show download button ONLY after code is ready
    download_style = {
        "display": "inline-block",
        "marginTop": "10px",
        "color": "#0070C0",
        "cursor": "pointer"
    }

    return preview, href, download_style

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)