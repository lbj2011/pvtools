
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
    dcc.Store(id='code-read-store', data={}),

    html.Hr(),
    html.Div([
        html.H1("PV-Copilot"), 
    ], style={
        'width': '100%',
        'padding-left': '10px',
        'padding-right': '10px',
        'textAlign': 'center'}),
    html.Hr(),

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
                        html.Label("Upload your data (.csv, .xls, .pkl)"),
                        dcc.Upload(
                            id="upload-data",
                            accept=".csv, text/csv, .xls, .xlsx, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, .pkl",
                            children=html.Div(["Drag and Drop or ", html.A("Select Files", style={"color": "blue"})]),
                            style={"width": "100%", "height": "60px", "lineHeight": "60px",
                                   "borderWidth": "1px", "borderStyle": "dashed",
                                   "textAlign": "center"}
                        ),
                        html.Div(id="upload-status-output", style={"marginTop": "5px"}),
                        html.Div(
                        [
                            dbc.Button(
                                "Try Example Data",
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
                                id="loading-summary",
                                type="circle",           # spinner type: "default", "circle", "dot", "cube"
                                color="#0d6efd",         # optional (Bootstrap primary)
                                children=html.Div(
                                    id="data-summary-output",
                                    className="p-2 border",
                                    style={"minHeight": "170px", "marginTop": "5px"}
                                )
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
                html.H4("Analysis Options", style={"color": MAJOR_CARD_FONT_COLOR}),
                style={"backgroundColor": MAJOR_CARD_BACKGROUND}
            ),
            dbc.CardBody(style={"backgroundColor": BODY_CARD_BACKGROUND}, children=[
                dbc.Row([
                    # 1. Filter block (Inner Card)
                    dbc.Col(dbc.Card(children=[
                        dbc.CardHeader(html.H5("Preprocessing (optional)")),
                        dbc.CardBody([
                            dbc.Checklist(
                                id="filter-options",
                                options=[
                                    {"label": "Outlier removal", "value": "outlier"},
                                    {"label": "Time zone & DST correction", "value": "timezone"},
                                    {"label": "Clear-sky filter", "value": "clearsky"},
                                ],
                                value=[],
                                inline=False
                            )
                        ])
                    ]), xs=12, lg=4),
                    
                    # 2. Degradation metric block (Inner Card)
                    dbc.Col(dbc.Card(children=[
                        dbc.CardHeader(html.H5("Degradation Metric/Function")),
                        dbc.CardBody([
                            dbc.RadioItems(
                                id="metric-select",
                                options=[
                                    {"label": "YoY (Year-over-Year)", "value": "yoy"},
                                    {"label": "Linear regression", "value": "linear"},
                                    {"label": "PV-Pro", "value": "pvpro"},
                                    {"label": "PVUSA", "value": "pvusa"},
                                ],
                                value="yoy",
                                inline=False
                            )
                        ])
                    ]), xs=12, lg=4),
                    
                    # 3. Figures block (Inner Card)
                    dbc.Col(dbc.Card(children=[
                        dbc.CardHeader(html.H5("Figures to Show")),
                        dbc.CardBody([
                            dbc.Checklist(
                                id="figure-options",
                                options=[
                                    {"label": "Power vs time", "value": "power_time"},
                                    {"label": "Outliers vs time", "value": "outliers_time"},
                                    {"label": "Distribution of rate", "value": "rate_dist"},
                                    {"label": "SDM parameter vs time", "value": "sdm_param"},
                                ],
                                value=["power_time"],
                                inline=False
                            )
                        ])
                    ]), xs=12, lg=4)
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
            html.H4("Analysis Output", style={"color": MAJOR_CARD_FONT_COLOR}),
            style={"backgroundColor": MAJOR_CARD_BACKGROUND}
        ),
        dbc.CardBody(style={"backgroundColor": BODY_CARD_BACKGROUND}, children=[
            # Run Button - Centered and reduced width (width=6)
            dbc.Row(
                justify="start",
                className="mt-4", # Add some top margin for visibility
                children=[
                    # 1. Button Column (Left Side)
                    dbc.Col([
                        dbc.Button(
                            "Click to RUN ANALYSIS",
                            id="run-btn",
                            color="success",
                            className="w-100"
                        ),
                        html.Small(
                            "(Analysis typically takes 2-4 seconds)",
                            className="text-muted small"
                        )],
                        lg=4, md=6, sm=6, xs=6
                    ),

                    # 2. Slider Column (Right Side)
                    dbc.Col(
                        [
                            # Label that will be updated by the callback
                            # Initial value is set to 1.0
                            html.Label("LLM Temperature: 1.0", id="temp-label", className="fw-bold"),
                            dcc.Slider(
                                id='temp-slider',
                                min=0,
                                max=1,
                                step=0.1,
                                # SETTING THE START VALUE TO 1.0
                                value=1.0, 
                                # ONLY show marks for 0, 0.5, and 1
                                marks={
                                    0: {'label': '0'},
                                    0.5: {'label': '0.5'},
                                    1: {'label': '1'},
                                },
                                className="mb-4"
                            ),
                        ],
                        lg=4, md=6, sm=6, xs=6
                    )
                ]
            )
            ,

            html.Hr(),

            # Results, Figures, and Code Snippet in the same row
            dbc.Row(
                children=[

                    # 1️⃣ Results Column
                    dbc.Col(
                        lg=3, md=12, sm=12, xs=12,
                        children=[
                            html.H4("Results Summary", className="mt-2"),
                            dcc.Loading(
                                id="loading-results",
                                type="circle",
                                color="#0d6efd",
                                children=html.Div(
                                    id="results-block",
                                    className="mb-4 p-3 border",
                                    style={"minHeight": "250px"}
                                )
                            ),
                        ]
                    ),

                    # 2️⃣ Code Column
                    dbc.Col(
                        lg=4, md=12, sm=12, xs=12,
                        children=[

                            html.Div(
                                [
                                    html.H4(
                                        "Code Snippet (for reproduction)",
                                        className="mt-2",
                                        style={"flex": "1"}
                                    ),

                                    dcc.Clipboard(
                                        target_id="code-block",
                                        title="Copy code",
                                        style={
                                            "cursor": "pointer",
                                            "fontSize": 20,
                                            "color": "#6c757d",
                                            "marginTop": "10px"
                                        }
                                    )
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "space-between"
                                }
                            ),

                            dcc.Loading(
                                id="loading-code",
                                type="circle",
                                color="#0d6efd",
                                children=html.Pre(
                                    id="code-block",
                                    style={
                                        "backgroundColor": CODE_BLOCK_BACKGROUND,
                                        "padding": "15px",
                                        "overflowX": "auto",
                                        "border": "1px solid #dee2e6",
                                        "minHeight": "250px",
                                        "maxHeight": "400px",
                                        "overflowY": "auto"
                                    }
                                )
                            ),
                        ],
                    ),

                    # 3️⃣ Figures Column
                    dbc.Col(
                        lg=5, md=12, sm=12, xs=12,
                        children=[
                            html.H4("Generated Figures", className="mt-2"),
                            dcc.Loading(
                                id="loading-figures",
                                type="circle",
                                color="#0d6efd",
                                children=html.Div(
                                    id="figures-block",
                                    className="mb-4 p-3 border",
                                    style={"minHeight": "250px"}
                                )
                            ),
                        ]
                    ),
                ]
            ),

            html.Details(
                open=False,   # folded by default
                children=[

                    html.Summary(
                        "Execution Status & Logs (click to expand)",
                        style={
                            "fontWeight": "bold",
                            "cursor": "pointer",
                            "padding": "6px"
                        }
                    ),

                    dcc.Loading(
                        type="circle",
                        children=html.Pre(
                            id="status-log",
                            style={
                                "backgroundColor": "#f8f9fa",
                                "padding": "15px",
                                "border": "1px solid #dee2e6",
                                "marginTop": "10px",
                                "maxHeight": "300px",
                                "overflowY": "auto",
                                "fontSize": "13px",
                                "whiteSpace": "pre-wrap"
                            }
                        )
                    )
                ]
            ),
        ])
    ])
], style={
        # 'paddingLeft': '18%',   
        # 'paddingRight': '12%'   
})
])

# --- Callbacks ---

# NEW: Callback to update the upload status text
@app.callback(
    Output("upload-status-output", "children"),
    Input("upload-data", "filename"),
    prevent_initial_call=False
)
def update_upload_status(filename):
    """Displays a status message when a file is uploaded."""
    if filename:
        return html.Div(f"File uploaded successfully: {filename}", className="text-success small")
    # Return empty div on initial load or if upload fails/resets
    return html.Div("Awaiting file...", className="text-muted small")

# --- Callback to Update Label ---
@app.callback(
    Output('temp-label', 'children'),
    [Input('temp-slider', 'value')]
)
def update_output(value):
    """Updates the LLM Temperature label with the slider's current value."""
    # Format the value to one decimal place for cleaner display
    return f"LLM Temperature: {value:.1f}"

# --- load_example_data ---

@app.callback(
    Output("dataframe-store", "data", allow_duplicate=True),
    Output("upload-status-output", "children", allow_duplicate=True),
    Input("load-example-btn", "n_clicks"),
    prevent_initial_call=True
)
def load_example_data(n):

    df = pd.read_csv("data/pmp.csv")
    return (
        df.to_json(date_format="iso", orient="split"),
        html.Div("Loaded example dataset: 'pmp.csv'", className="text-success small")
    )
# --- analyze button ---
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


@app.callback(
    Output("data-summary-output", "children"),
    Output("mapped-vars-store", "data"),
    Output("dataframe-store", "data"),
    Output("code-read-store", "data"),
    Output("analyze-btn", "disabled", allow_duplicate=True),
    Output("analyze-btn", "children", allow_duplicate=True),
    Input("analyze-btn", "n_clicks"),
    Input("load-example-btn", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("dataframe-store", "data"),
    prevent_initial_call=True
)
def analyze_uploaded_data_callback(
        analyze_clicks,
        example_clicks,
        contents,
        filename,
        stored_df_json
):

    trigger = ctx.triggered_id

    # ------------------------------------------------
    # 1. Load example dataset (no analysis yet)
    # ------------------------------------------------
    if trigger == "load-example-btn":

        try:
            df = pd.read_csv("data/pmp.csv")
            df_json = df.to_json(date_format='iso', orient='split')

        except Exception as e:
            return (
                html.Div(f"Error loading example data: {e}", className="alert alert-danger"),
                {},
                None,
                "",
                False,
                "Analyze Data"
            )

        return (
            html.Div(
                "",
                className="text-muted"
            ),
            {},
            df_json,
            "",
            False,
            "Analyze Data"
        )

    # ------------------------------------------------
    # 2. Analyze Data button clicked
    # ------------------------------------------------
    if trigger == "analyze-btn":

        # Case A: Uploaded file
        if contents is not None:

            df, summary_table, mapped_variables_dict, code_read = parse_contents(contents, filename)

            if df is None:
                return summary_table, {}, None, "", False, "Analyze Data"

        # Case B: Example dataset already loaded
        elif stored_df_json is not None:

            try:
                df = pd.read_json(stored_df_json, orient='split')
                df, summary_table, mapped_variables_dict, code_read = parse_contents(df=df)

            except Exception as e:
                return (
                    html.Div(f"Error processing stored dataset: {e}", className="alert alert-danger"),
                    {},
                    None,
                    "",
                    False,
                    "Analyze Data"
                )

        # Case C: No data at all
        else:
            return (
                html.Div(
                    "Upload a file or load the example dataset, then click 'Analyze Data'."
                ),
                {},
                None,
                "",
                False,
                "Analyze Data"
            )

        # Convert dataframe to JSON
        try:
            df_json = df.to_json(date_format='iso', orient='split')
        except Exception as e:
            return (
                html.Div(
                    f"Error converting DataFrame to JSON: {e}",
                    className="alert alert-danger"
                ),
                {},
                None,
                "",
                False,
                "Analyze Data"
            )

        return (
            html.Div(summary_table, style={'fontSize': '10pt'}),
            mapped_variables_dict,
            df_json,
            code_read,
            False,
            "Analyze Data"
        )

    # ------------------------------------------------
    # Fallback
    # ------------------------------------------------
    return "", {}, None, "", False, "Analyze Data"


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
    Output("results-block", "children"),
    Output("figures-block", "children"),
    Output("code-block", "children"),
    Output("status-log", "children"),
    Output("run-btn", "disabled", allow_duplicate=True),
    Output("run-btn", "children", allow_duplicate=True),
    Input("run-btn", "n_clicks"),
    State('filter-options', 'value'),
    State('figure-options', 'value'),
    State("dataframe-store", "data"),
    State("mapped-vars-store", "data"),
    State("code-read-store", "data"),
    State("upload-data", "contents"),
    State("filter-options", "value"),
    State("metric-select", "value"),
    State("figure-options", "value"),
    State('temp-slider', 'value'),
    prevent_initial_call=True
)
def run_full_analysis(
    n_clicks,
    filter_options,
    figure_options,
    df_json,
    mapped_variables_dict,
    code_read,
    contents,
    filters,
    metric,
    figures,
    llm_temp
):

    log_text = "[INFO] Starting analysis\n"

    # -------------------------
    # Initialize safe defaults
    # -------------------------
    df = None
    df_processed = None
    rd = None
    code = None
    outlier_indices = []
    nan_indices = []

    results = html.Div()
    figures_output = html.Div()
    code_snippet = "Code not generated."

    # ------------------------------------------------
    # 0. Check dataset availability
    # ------------------------------------------------
    if df_json is None:

        log_text += "[ERROR] No dataset available\n"

        return (
            html.Div([
                html.P("Upload a file or load example data, then click 'Analyze Data'.")
            ]),
            html.Div(),
            "",
            log_text,
            False,
            "RUN ANALYSIS"
        )

    # ------------------------------------------------
    # 1. Reconstruct dataframe
    # ------------------------------------------------
    try:

        log_text += "[STEP 1] Reconstructing dataframe from JSON\n"

        df = pd.read_json(df_json, orient='split')

        log_text += f"[INFO] Dataframe loaded successfully. Rows: {len(df)}\n"

    except Exception:

        log_text += "[ERROR] Failed to reconstruct dataframe\n"
        log_text += traceback.format_exc()

    # ------------------------------------------------
    # 2. Run degradation analysis
    # ------------------------------------------------
    if df is not None:

        try:

            log_text += "[STEP 2] Running degradation analysis\n"

            rd, outlier_indices, nan_indices, df_processed, code, log_sub, execution_success = \
                generate_degradation_code_and_execute(
                    df,
                    mapped_variables_dict,
                    llm_temp,
                    filter_options
                )
            
            if not execution_success and code:
                log_text += "[WARNING] Code generated but execution failed\n"

            log_text += log_sub

            warning = None

            if not execution_success and code:
                warning = dbc.Alert(
                    "⚠ Code was generated but execution failed. See logs below.",
                    color="warning",
                    className="mt-2"
                )

        except Exception:

            log_text += "[ERROR] Degradation analysis failed\n"
            log_text += traceback.format_exc()

    # ------------------------------------------------
    # 3. Degradation rate component
    # ------------------------------------------------
    rd_component = html.Div()

    if isinstance(rd, (int, float)):

        rd_component = dbc.Alert(
            html.Span([
                "Degradation Rate: ",
                html.B(f"{rd:.2f}%"),
                " per year"
            ]),
            color="success",
            className="mt-3"
        )

        log_text += f"[INFO] Degradation rate computed: {rd:.2f}% per year\n"

    elif rd is not None:

        rd_component = dbc.Alert(
            "Error: Degradation rate invalid.",
            color="danger",
            className="mt-3"
        )

        log_text += "[WARNING] Degradation rate invalid\n"

    # ------------------------------------------------
    # 4. Filter summary
    # ------------------------------------------------
    filters_display = ""

    try:

        log_text += "[STEP 3] Generating filter summary\n"

        filters_display = get_filtered_display_string(
            filters=filters,
            outlier_indices=outlier_indices
        )

    except Exception:

        log_text += "[WARNING] Failed to generate filter summary\n"
        log_text += traceback.format_exc()

        filters_display = "Filter summary unavailable."

    # ------------------------------------------------
    # 5. Data summary block
    # ------------------------------------------------
    summary_component = html.Div()

    if df_processed is not None:

        try:

            log_text += "[STEP 4] Building data summary\n"

            summary_component = build_data_summary_block(
                df_processed,
                outlier_indices,
                filters
            )

        except Exception:

            log_text += "[ERROR] Data summary failed\n"
            log_text += traceback.format_exc()

            summary_component = html.Div(
                "Error generating summary.",
                className="text-danger"
            )

    else:

        summary_component = html.Div(
            "Processed dataframe unavailable."
        )

    # ------------------------------------------------
    # 6. Results block
    # ------------------------------------------------
    results = html.Div([
        warning,   # <-- add here

        html.P([
            html.Strong("Degradation Metric: "),
            metric.upper() if metric else "N/A"
        ]),

        html.P([html.Strong("Filters Applied:"), html.Br()]),
        dcc.Markdown(filters_display, dangerously_allow_html=True),

        html.P([html.Strong("Data Summary:")]),
        summary_component,

        rd_component
    ])

    # ------------------------------------------------
    # 7. Generate figures
    # ------------------------------------------------
    figures_output_list = []

    if df_processed is not None:

        try:

            log_text += "[STEP 5] Generating figures\n"

            if figure_options and 'power_time' in figure_options:

                log_text += "[INFO] Generating Power vs Time plot\n"

                figures_output_list.append(
                    dcc.Graph(
                        id='power-vs-time-plot',
                        figure=plot_power_vs_time(
                            df_processed,
                            mapped_variables_dict,
                            rd
                        )
                    )
                )

            if figure_options and 'outliers_time' in figure_options:

                log_text += "[INFO] Generating Outliers vs Time plot\n"

                figures_output_list.append(
                    dcc.Graph(
                        id='outlier-vs-time-plot',
                        figure=plot_outlier_vs_time(
                            df_processed,
                            mapped_variables_dict,
                            nan_indices,
                            outlier_indices
                        )
                    )
                )

        except Exception:

            log_text += "[ERROR] Figure generation failed\n"
            log_text += traceback.format_exc()

    figures_output = html.Div(figures_output_list)

    # ------------------------------------------------
    # 8. Generate code snippet
    # ------------------------------------------------
    try:

        log_text += "[STEP 6] Generating reproducible code snippet\n"

        if code:

            code_snippet = generate_full_code(code, code_read)

            log_text += "[INFO] Code snippet generated successfully\n"

        else:

            code_snippet = "No generated code available."

            log_text += "[WARNING] No code returned from analysis\n"

    except Exception:

        log_text += "[WARNING] Failed to generate code snippet\n"
        log_text += traceback.format_exc()

        code_snippet = "Code snippet unavailable."

    # ------------------------------------------------
    # Finished
    # ------------------------------------------------
    log_text += "[SUCCESS] Callback finished\n"

    return (
        results,
        figures_output,
        code_snippet,
        log_text,
        False,
        "RUN ANALYSIS"
    )

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)