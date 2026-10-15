import dash
from dash import dcc, html, Input, Output, dash_table, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import dash_bootstrap_components as dbc
from app import app
from geopy.distance import distance
from collections import Counter
import math
from utils.data_loader import safe_get_df
import ast
import json
import os
import openai
import re
from page_supporting_files.field_chat import apply_filters, get_filter_from_llm
from dash import callback_context
from page_supporting_files.field_fitlers import build_filters
from dash import ctx

df_raw = safe_get_df()

# df_raw = pd.read_pickle('data_250318.pkl')
# df_raw = pd.read_pickle('data_250924.pkl')

df = df_raw[(df_raw['duration'] < 100) & (df_raw['rate'] <= 3)]

# ─────────────────────────────────────────────────────────────────────
# City search support — load GeoNames cities15000 once at import time.
# Run build_cities_csv.py to (re)generate data/cities15000.csv (~1 MB).
# ─────────────────────────────────────────────────────────────────────
_CITIES_CSV = os.path.join("data", "cities15000.csv")

def _load_cities():
    """Load cities CSV into a DataFrame; return empty if file missing."""
    try:
        cdf = pd.read_csv(
            _CITIES_CSV,
            keep_default_na=False,   # 'NA' is Namibia's ISO code
            na_values=[""],
        )
        # Sort by population so popular cities surface first in the dropdown
        if "population" in cdf.columns:
            cdf = cdf.sort_values("population", ascending=False).reset_index(drop=True)
        return cdf
    except FileNotFoundError:
        print(f"[field_degradation] {_CITIES_CSV} not found — "
              "run build_cities_csv.py to enable city search.")
        return pd.DataFrame(columns=["name", "asciiname", "country_code",
                                     "latitude", "longitude", "population"])

_CITIES_DF = _load_cities()

# Minimal ISO-3166-1 alpha-2 → country name lookup (covers the codes that
# actually appear in cities15000). Falls back to the raw code if missing.
_ISO_COUNTRY = {
    "AD":"Andorra","AE":"UAE","AF":"Afghanistan","AG":"Antigua & Barbuda","AI":"Anguilla",
    "AL":"Albania","AM":"Armenia","AO":"Angola","AQ":"Antarctica","AR":"Argentina",
    "AS":"American Samoa","AT":"Austria","AU":"Australia","AW":"Aruba","AX":"Åland Is.",
    "AZ":"Azerbaijan","BA":"Bosnia & Herzegovina","BB":"Barbados","BD":"Bangladesh",
    "BE":"Belgium","BF":"Burkina Faso","BG":"Bulgaria","BH":"Bahrain","BI":"Burundi",
    "BJ":"Benin","BL":"St. Barthélemy","BM":"Bermuda","BN":"Brunei","BO":"Bolivia",
    "BQ":"Caribbean NL","BR":"Brazil","BS":"Bahamas","BT":"Bhutan","BV":"Bouvet Is.",
    "BW":"Botswana","BY":"Belarus","BZ":"Belize","CA":"Canada","CC":"Cocos Is.",
    "CD":"DR Congo","CF":"Central African Rep.","CG":"Congo","CH":"Switzerland",
    "CI":"Côte d'Ivoire","CK":"Cook Is.","CL":"Chile","CM":"Cameroon","CN":"China",
    "CO":"Colombia","CR":"Costa Rica","CU":"Cuba","CV":"Cape Verde","CW":"Curaçao",
    "CX":"Christmas Is.","CY":"Cyprus","CZ":"Czechia","DE":"Germany","DJ":"Djibouti",
    "DK":"Denmark","DM":"Dominica","DO":"Dominican Rep.","DZ":"Algeria","EC":"Ecuador",
    "EE":"Estonia","EG":"Egypt","EH":"W. Sahara","ER":"Eritrea","ES":"Spain",
    "ET":"Ethiopia","FI":"Finland","FJ":"Fiji","FK":"Falkland Is.","FM":"Micronesia",
    "FO":"Faroe Is.","FR":"France","GA":"Gabon","GB":"United Kingdom","GD":"Grenada",
    "GE":"Georgia","GF":"French Guiana","GG":"Guernsey","GH":"Ghana","GI":"Gibraltar",
    "GL":"Greenland","GM":"Gambia","GN":"Guinea","GP":"Guadeloupe","GQ":"Eq. Guinea",
    "GR":"Greece","GS":"S. Georgia","GT":"Guatemala","GU":"Guam","GW":"Guinea-Bissau",
    "GY":"Guyana","HK":"Hong Kong","HM":"Heard Is.","HN":"Honduras","HR":"Croatia",
    "HT":"Haiti","HU":"Hungary","ID":"Indonesia","IE":"Ireland","IL":"Israel",
    "IM":"Isle of Man","IN":"India","IO":"British Indian Ocean","IQ":"Iraq","IR":"Iran",
    "IS":"Iceland","IT":"Italy","JE":"Jersey","JM":"Jamaica","JO":"Jordan","JP":"Japan",
    "KE":"Kenya","KG":"Kyrgyzstan","KH":"Cambodia","KI":"Kiribati","KM":"Comoros",
    "KN":"St. Kitts & Nevis","KP":"North Korea","KR":"South Korea","KW":"Kuwait",
    "KY":"Cayman Is.","KZ":"Kazakhstan","LA":"Laos","LB":"Lebanon","LC":"St. Lucia",
    "LI":"Liechtenstein","LK":"Sri Lanka","LR":"Liberia","LS":"Lesotho","LT":"Lithuania",
    "LU":"Luxembourg","LV":"Latvia","LY":"Libya","MA":"Morocco","MC":"Monaco",
    "MD":"Moldova","ME":"Montenegro","MF":"St. Martin","MG":"Madagascar","MH":"Marshall Is.",
    "MK":"North Macedonia","ML":"Mali","MM":"Myanmar","MN":"Mongolia","MO":"Macao",
    "MP":"N. Mariana Is.","MQ":"Martinique","MR":"Mauritania","MS":"Montserrat",
    "MT":"Malta","MU":"Mauritius","MV":"Maldives","MW":"Malawi","MX":"Mexico",
    "MY":"Malaysia","MZ":"Mozambique","NA":"Namibia","NC":"New Caledonia","NE":"Niger",
    "NF":"Norfolk Is.","NG":"Nigeria","NI":"Nicaragua","NL":"Netherlands","NO":"Norway",
    "NP":"Nepal","NR":"Nauru","NU":"Niue","NZ":"New Zealand","OM":"Oman","PA":"Panama",
    "PE":"Peru","PF":"French Polynesia","PG":"Papua New Guinea","PH":"Philippines",
    "PK":"Pakistan","PL":"Poland","PM":"St. Pierre","PN":"Pitcairn","PR":"Puerto Rico",
    "PS":"Palestine","PT":"Portugal","PW":"Palau","PY":"Paraguay","QA":"Qatar",
    "RE":"Réunion","RO":"Romania","RS":"Serbia","RU":"Russia","RW":"Rwanda",
    "SA":"Saudi Arabia","SB":"Solomon Is.","SC":"Seychelles","SD":"Sudan","SE":"Sweden",
    "SG":"Singapore","SH":"St. Helena","SI":"Slovenia","SJ":"Svalbard","SK":"Slovakia",
    "SL":"Sierra Leone","SM":"San Marino","SN":"Senegal","SO":"Somalia","SR":"Suriname",
    "SS":"South Sudan","ST":"São Tomé","SV":"El Salvador","SX":"Sint Maarten","SY":"Syria",
    "SZ":"Eswatini","TC":"Turks & Caicos","TD":"Chad","TF":"French S. Terr.","TG":"Togo",
    "TH":"Thailand","TJ":"Tajikistan","TK":"Tokelau","TL":"Timor-Leste","TM":"Turkmenistan",
    "TN":"Tunisia","TO":"Tonga","TR":"Turkey","TT":"Trinidad & Tobago","TV":"Tuvalu",
    "TW":"Taiwan","TZ":"Tanzania","UA":"Ukraine","UG":"Uganda","UM":"US Minor Is.",
    "US":"United States","UY":"Uruguay","UZ":"Uzbekistan","VA":"Vatican","VC":"St. Vincent",
    "VE":"Venezuela","VG":"British Virgin Is.","VI":"US Virgin Is.","VN":"Vietnam",
    "VU":"Vanuatu","WF":"Wallis & Futuna","WS":"Samoa","XK":"Kosovo","YE":"Yemen",
    "YT":"Mayotte","ZA":"South Africa","ZM":"Zambia","ZW":"Zimbabwe",
}

def _city_label(name, country_code):
    return f"{name}, {_ISO_COUNTRY.get(country_code, country_code)}"

def _row_to_option(row):
    return {
        "label": _city_label(row["name"], row["country_code"]),
        "value": f"{row['latitude']:.4f},{row['longitude']:.4f}",
    }

# Pre-loaded options shown when the dropdown is opened with no search term —
# just the most-populated cities. Searching swaps these out via callback.
_INITIAL_CITY_OPTIONS = [
    _row_to_option(row) for _, row in _CITIES_DF.head(200).iterrows()
]


# ─────────────────────────────────────────────────────────────────────
# Shared chart styling — keeps every figure on this page visually consistent
# ─────────────────────────────────────────────────────────────────────
_CHART_FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"


def _chart_tile(title, graph, *, subtitle=None):
    """Wrap a dcc.Graph in a soft, rounded card with a clean title row."""
    header_children = [
        html.Div(title, style={
            "fontFamily": _CHART_FONT, "fontSize": "14px",
            "fontWeight": "600", "color": "#111827", "letterSpacing": "0.2px",
        }),
    ]
    if subtitle:
        header_children.append(
            html.Div(subtitle, style={
                "fontFamily": _CHART_FONT, "fontSize": "12px",
                "color": "#6b7280", "marginTop": "2px",
            })
        )
    return html.Div(
        [
            html.Div(header_children, style={"marginBottom": "6px"}),
            graph,
        ],
        style={
            "padding": "14px 16px 8px 16px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "backgroundColor": "#ffffff",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
            "height": "100%",
        },
    )


types = ['mono-c-Si', 'multi-c-Si', 'a-Si', 'CIGS', 'CdTe', 'HIT','other']
filter_text_style = {
    'fontFamily': 'Arial',
    'fontSize': '17px',  # Default size, can be overridden per element
    'color': 'black'
    }


def get_layout():
    return layout


def normalize_faults(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [f.strip() for f in x.split(',')]
    return None

# Layout
layout = dbc.Container([

    html.Hr(),
    html.Div([
        html.H1(
            "Global PV System Field Performance",
            className="page-title"
        ),
    ], className="page-title-container"),
    html.Hr(),

    html.H3("Overview", className="level-2-title"),
    html.P(''),
    dbc.Row([
        dbc.Col([
            dcc.Markdown("""This tool visualizes **global PV field degradation** extracted from scientific literature using **Large Language Models (LLMs)**.

                    - Source papers (~3,900 papers) were retrieved from **Scopus** using keyword-based searches.  
                    - **ChatGPT and Gemini** were applied to automatically extract degradation information.  
                    - Degradation rate is reported as a **negative value when power decreases**.   
                         
                """.replace('    ', '')
                 ),
                 html.Details([
                    html.Summary(
                        "Resources",
                        style={"color": "#8A8A8A"}
                    ),
                    html.Ul([
                        html.Li(html.A("GitHub Repository", href="https://github.com/DuraMAT/PV-LLM", target="_blank")),
                        html.Li(html.A("Download Raw Data (DuraMAT Datahub)", href="https://datahub.duramat.org/project/mapping-pv-degradation-by-llm", target="_blank")),
                    ])
                ])
        ], xs=12, sm=12, md=12, lg=9, xl=9),

        dbc.Col([
            html.Img(src=app.get_asset_url('llm_logo.jpg'),
            style={'width': '90%'}),
        ], xs=9, sm=8, md=6, lg=3, xl=3),
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
        # color="primary",
        className="mt-2 custom-alert"
    ),
    
    html.P(''),
    html.H2("Degradation rate map", className="level-1-title"),
    html.P(''),
    dbc.Card([
        dbc.CardBody([
            html.Div(   
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H4(
                                    [
                                        "Ask Questions to Filter the Data ",
                                        dbc.Badge("Beta", color="#00B0F0", className="ms-2"),
                                    ],
                                    className="section-title mb-0",
                                    style={"margin": "0"},
                                ),
                                xs=12,
                                md="auto",
                            ),

                            dbc.Col(
                                html.Span(
                                    "(⏱ May take 5–20 seconds)",
                                    style={
                                        "color": "#A6A6A6",
                                        "fontSize": "14px",
                                        "fontWeight": "400",
                                    },
                                ),
                                xs=12,
                                md="auto",
                                className="align-self-center",
                            ),
                        ],
                        align="center",
                        className="g-2",
                    ),
                    html.P(''),
                    dcc.Store(id="chat-filtered-data"),

                    # INPUT ROW
                    html.Div(
                        [
                            dcc.Input(
                                id="chat-input",
                                type="text",
                                placeholder="Ask a question about the PV degradation dataset...",
                                className="chat-input-style",
                            ),

                            html.Button(
                                "Send",
                                id="chat-submit",
                                n_clicks=0,
                                disabled=True,
                                className="send-button",
                            ),

                            html.Button(
                                "Reset",
                                id="chat-reset",
                                n_clicks=0,
                                className="reset-button"
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "16px",
                        },
                    ),

                    # Example questions
                    html.Div(
                        [
                            html.Span(
                                "Examples question:",
                                style={
                                    "fontSize": "13px",
                                    "color": "#666",
                                    "alignSelf": "center",
                                    "marginRight": "4px",
                                },
                            ),

                            html.Button(
                                "Show studies with <-5% degradation",
                                id="q1",
                                n_clicks=0,
                                className="example-btn",
                            ),
                            html.Button(
                                "Show cases at offshore area",
                                id="q2",
                                n_clicks=0,
                                className="example-btn",
                            ),
                            html.Button(
                                "Tell me studies in Asia",
                                id="q3",
                                n_clicks=0,
                                className="example-btn",
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                    ),
                    html.P(''),

                    # RESPONSE TEXT
                    dcc.Loading(
                        type="circle",
                        children=html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(id="response-text")
                                            ],
                                            className="response-container"
                                        ),
                                        html.Div(id="filtered-table"),
                                    ],
                                    id="response-panel",
                                    style={"display": "none"}   # hidden at start
                                )
                            ]
                        ),
                    ),
                    html.P(''),

                    html.Details(
                    [
                        html.Summary(
                            "Show LLM reasoning / filter result",
                            style={
                                "cursor": "pointer",
                                "fontWeight": "300",
                                "fontSize": "13px",
                                "color": "#666",
                            },
                        ),
                        html.Pre(
                            id="llm-result",
                            style={
                                "whiteSpace": "pre-wrap",
                                "backgroundColor": "rgba(255,255,255,0.6)",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "fontSize": "13px",
                                "marginTop": "8px",
                                "border": "1px solid #e5e7eb",
                            },
                        ),
                    ],
                    open=False,  # folded by default
                    style={"marginBottom": "16px"},
                ),

                ],
                className="llm-card",
                style={
                    # "margin": "0 auto",
                    # "fontFamily": "Inter, system-ui, sans-serif",
                }
            ),
            html.P(''),

            html.Div(
            [
                build_filters(types)
            ]),
    
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='map')
                    ]#, style={'width': '70%', 'display': 'inline-block', 'marginLeft': '0px'}
                    ),
                ], xs=12, sm=12, md=12, lg=8, xl=8),
            
                dbc.Col([
                    html.Div([
                        html.H4("Details", className="section-title-detail"),
                        html.P("(Select a data point to show)", className="section-subtitle"),
                        html.Div(
            dash_table.DataTable(
                id='table',
                columns=[
                    {'name': 'Attribute', 'id': 'attribute'},
                    {'name': 'Value', 'id': 'value', 'presentation': 'markdown'}
                ],
                data=[],
                style_as_list_view=True,
            ),
            className="details-table"
        )
                    ]
                    ),
            ], xs=12, sm=12, md=12, lg=4, xl=4),
    ], className="g-2")  # Adds spacing between columns
    ]),
    ]),

    html.P(''),
    html.H2('Analysis', className="level-1-title"),
    html.P('(based on filtered data points)',
           style={"color": "#6b7280", "fontFamily": _CHART_FONT,
                  "fontSize": "13px", "marginTop": "-8px", "marginBottom": "14px"}),
    dbc.Card([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Degradation Rate Distribution",
                             style={"fontFamily": _CHART_FONT,
                                    "fontSize": "15px",
                                    "fontWeight": "600",
                                    "color": "#111827",
                                    "marginBottom": "2px"}),
                    html.Div(id="data_info",
                             style={"fontFamily": _CHART_FONT,
                                    "fontSize": "12px",
                                    "color": "#6b7280",
                                    "marginBottom": "8px"}),
                    dcc.Graph(id='histogram',
                              config={"displayModeBar": False},
                              style={'width': '100%'})
                ]),
            ], xs=12, sm=12, md=6, lg=6, xl=6),
            dbc.Col([
                html.Div([
                    html.Div("Degradation Rate vs Exposure Length",
                             style={"fontFamily": _CHART_FONT,
                                    "fontSize": "15px",
                                    "fontWeight": "600",
                                    "color": "#111827",
                                    "marginBottom": "2px"}),
                    html.Div("Each point is one reported study",
                             style={"fontFamily": _CHART_FONT,
                                    "fontSize": "12px",
                                    "color": "#6b7280",
                                    "marginBottom": "8px"}),
                    dcc.Graph(id='rate-duration-scatter',
                              config={"displayModeBar": False},
                              style={'width': '100%'})
                ]),
            ], xs=12, sm=12, md=6, lg=6, xl=6),
        ], className="g-3", style={'marginLeft': '0px', 'marginRight': '0px'})
    ], style={"padding": "16px"}),

    html.P(''),
    html.H2('Regional performance', className="level-1-title"),
    html.P(''),
    dbc.Card([
        dbc.Row([
            dbc.Col([
                html.P(''),

                # ── Location selector card ──────────────────────────────
                html.Div([
                    html.Div([
                        html.Span("Select your location",
                                  style={"fontFamily": "Arial",
                                         "fontSize": "17px",
                                         "fontWeight": "600",
                                         "color": "#1f2937"}),
                    ], style={"display": "flex", "alignItems": "center",
                              "marginBottom": "14px"}),

                    # City search
                    html.Label("Search city",
                               style={"fontFamily": "Arial", "fontSize": "13px",
                                      "color": "#6b7280", "marginBottom": "4px",
                                      "display": "block"}),
                    dcc.Dropdown(
                        id="city-search",
                        options=_INITIAL_CITY_OPTIONS,
                        placeholder="🔍  Type to search worldwide cities...",
                        clearable=True,
                        searchable=True,
                        optionHeight=36,
                        style={"marginBottom": "14px", "fontSize": "14px"},
                    ),

                    # Lat / Lon inline
                    dbc.Row([
                        dbc.Col([
                            html.Label("Latitude",
                                       style={"fontFamily": "Arial", "fontSize": "13px",
                                              "color": "#6b7280", "marginBottom": "4px",
                                              "display": "block"}),
                            dbc.InputGroup([
                                dcc.Input(
                                    id="lat-input",
                                    type="number",
                                    value=39.7392,
                                    step=0.0001,
                                    className="form-control",
                                    style={"borderRadius": "6px 0 0 6px"},
                                ),
                                dbc.InputGroupText("°",
                                    style={"backgroundColor": "#f9fafb",
                                           "color": "#6b7280",
                                           "fontSize": "13px"}),
                            ], size="sm"),
                        ], xs=6),
                        dbc.Col([
                            html.Label("Longitude",
                                       style={"fontFamily": "Arial", "fontSize": "13px",
                                              "color": "#6b7280", "marginBottom": "4px",
                                              "display": "block"}),
                            dbc.InputGroup([
                                dcc.Input(
                                    id="lon-input",
                                    type="number",
                                    value=-104.9903,
                                    step=0.0001,
                                    className="form-control",
                                    style={"borderRadius": "6px 0 0 6px"},
                                ),
                                dbc.InputGroupText("°",
                                    style={"backgroundColor": "#f9fafb",
                                           "color": "#6b7280",
                                           "fontSize": "13px"}),
                            ], size="sm"),
                        ], xs=6),
                    ], className="g-2", style={"marginBottom": "12px"}),

                    # Radius
                    html.Label("Search radius",
                               style={"fontFamily": "Arial", "fontSize": "13px",
                                      "color": "#6b7280", "marginBottom": "4px",
                                      "display": "block"}),
                    dbc.InputGroup([
                        dcc.Input(
                            id="radius-input",
                            type="number",
                            value=100,
                            min=1,
                            step=1,
                            className="form-control",
                            style={"borderRadius": "6px 0 0 6px"},
                        ),
                        dbc.InputGroupText("miles",
                            style={"backgroundColor": "#f9fafb",
                                   "color": "#6b7280",
                                   "fontSize": "13px"}),
                    ], size="sm", style={"marginBottom": "8px"}),

                ], style={
                    "padding": "16px 18px",
                    "border": "1px solid #e5e7eb",
                    "borderRadius": "12px",
                    "backgroundColor": "#fafbfc",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.04)",
                    "marginBottom": "12px",
                }),

                # Map figure
                dcc.Graph(
                    id='location-map',
                    figure={},
                    config={'scrollZoom': True},
                )
            ], xs=12, sm=12, md=4, lg=4, xl=4, className="h-100"),
                
            dbc.Col([
                html.P(''),
                html.Div([
                    html.Span("Performance within this radius",
                              style={"fontFamily": _CHART_FONT,
                                     "fontSize": "17px",
                                     "fontWeight": "600",
                                     "color": "#1f2937"}),
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "12px", "marginTop": "0"}),
                html.Div(id='pie-charts-container')  # placeholder for callback output
            ], xs=12, sm=12, md=8, lg=8, xl=8, className="h-100",
               style={'marginLeft': '0px'}),
        ], className="g-3", style={'marginLeft': '0px', 'marginRight': '0px'})
    ], style={"padding": "16px"}),

    html.P(''),
    dcc.Markdown("""Dataset version: 2025/9/24
                 
                """.replace('    ', ''), style={'color': 'lightgray'}
    ),
    # Contributor: Baojie Li
    # Project members: Baojie Li, Martin Springer, Dirk Jordan, Anubhav Jain
    
])

@app.callback(
    Output("chat-input", "value", allow_duplicate=True),
    Input("q1", "n_clicks"),
    Input("q2", "n_clicks"),
    Input("q3", "n_clicks"),
    prevent_initial_call=True,
)
def fill_input(q1, q2, q3):
    button_id = ctx.triggered_id

    questions = {
        "q1": "Show studies with <-5% degradation",
        "q2": "Show cases at offshore area",
        "q3": "Tell me studies in Asia",
    }

    return questions.get(button_id, "")

@app.callback(
    Output("chat-submit", "disabled"),
    Input("chat-submit", "n_clicks"),
    Input("chat-input", "n_submit"),   # ← add this
    prevent_initial_call=True
)
def disable_button_on_action(n_clicks, n_submit):
    return True

@app.callback(
    Output("chat-submit", "disabled", allow_duplicate=True),
    Input("chat-input", "value"),
    Input("chat-input", "n_submit"),
    prevent_initial_call=True
)
def toggle_send_button(value, n_submit):
    ctx = callback_context
    prop_id = ctx.triggered[0]["prop_id"]

    # If Enter pressed → disable
    if "n_submit" in prop_id:
        return True

    # If typing → enable/disable based on content
    if "value" in prop_id:
        return not (value and value.strip())

    return True

@app.callback(
    Output("response-text", "children"),
    Output("filtered-table", "children"),
    Output("chat-input", "value"),
    Output("chat-submit", "children"),
    Output("chat-submit", "disabled", allow_duplicate=True),
    Output("response-panel", "style"),
    Output("chat-filtered-data", "data"),
    Output("llm-result", "children"),   # ← ADD THIS
    Input("chat-submit", "n_clicks"),
    Input("chat-reset", "n_clicks"),
    Input("chat-input", "n_submit"),   # ← ADD THIS
    State("chat-input", "value"),
    prevent_initial_call=True
)
def handle_chat(submit_clicks, reset_clicks, enter_submit, question):

    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    hidden = {"display": "none"}
    visible = {
        "display": "block",
        "backgroundColor": "rgba(255,255,255,0.9)",
        "padding": "14px 16px",
        "borderRadius": "10px",
        "marginBottom": "15px",
        "fontSize": "15px",
        "border": "1px solid #D1E7F6",
    }

    if trigger == "chat-reset":
        return "", "", "", "Send", True, hidden, None, ""

    if trigger in ["chat-submit", "chat-input"]:

        if not question:
            return "", "", "", "Send", True, hidden, None, ""

        result = get_filter_from_llm(question)
        msg_text = result.get("reason", "")
        message = html.Div([
                html.Span("Response", className="field-chat-box-metric-label"),
                html.Span(msg_text, className="field-chat-box-metric-text")
            ], className="field-chat-box-metric-row")

        # pretty debug view
        llm_debug = json.dumps(result, indent=2)

        if not result.get("can_be_answered_with_dataframe", False):
            return message, "", "", "Send", True, visible, None, llm_debug

        df_filtered = apply_filters(df, result.get("filter_tree"))

        n_rows = len(df_filtered)

        message = html.Div([
            html.Div([
                html.Span("Filtered Data Points", className="field-chat-box-metric-label"),
                html.Span(f"{n_rows}", className="field-chat-box-metric-value")
            ], className="field-chat-box-metric-row"),

            html.Div([
                html.Span("Response", className="field-chat-box-metric-label"),
                html.Span(msg_text, className="field-chat-box-metric-text")
            ], className="field-chat-box-metric-row")
        ])

        table = dash_table.DataTable(
            data=df_filtered.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_filtered.columns],
            page_size=10,
            style_table={
                "overflowX": "auto",
                "width": "100%",
            },
            style_cell={
                "fontSize": "12px",
                "textAlign": "left",
                "padding": "4px 8px",
                "lineHeight": "1.1",
                "whiteSpace": "nowrap",
                "height": "auto",
            },
            style_header={
                "backgroundColor": "#f0f2f6",
                "fontWeight": "800",
                "textAlign": "left",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#fafafa",
                }
            ],
        )

        return (
            message,
            table,
            "",
            "Send",
            True,
            visible,
            df_filtered.index.tolist(),
            llm_debug
        )
    
    return "", "", "", "Send", True, hidden, None, ""


@app.callback(
    [
        Output('map', 'figure'),
        Output('histogram', 'figure'),
        Output("data_info", "children"),
        Output('rate-duration-scatter', 'figure')
    ],
    [
        Input('pv-tech-filter', 'value'),
        Input('pv-climate-filter', 'value'),
        Input('scope-filter', 'value'),
        Input('faults-filter', 'value'),
        Input('capacity-report-filter', 'value'),  # NEW
        Input('capacity-min', 'value'),             # NEW
        Input('capacity-max', 'value'),             # NEW
        Input('rate-min', 'value'),
        Input('rate-max', 'value'),
        Input('duration-min', 'value'),
        Input('duration-max', 'value'),
        Input("chat-filtered-data", "data")
    ]
)
def update_map_and_histogram(
    selected_types,
    selected_zones,
    selected_scopes,
    faults_filter,
    capacity_report_filter,
    capacity_min,
    capacity_max,
    rate_min,
    rate_max,
    duration_min,
    duration_max,
    chat_filtered_index
):
    # If any categorical filter is empty → show nothing
    if (
        not selected_types or
        not selected_zones or
        not selected_scopes or
        not capacity_report_filter or
        not faults_filter
    ):
        empty_fig = {
            "data": [],
            "layout": {
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "annotations": [{
                    "text": "No data selected",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14}
                }]
            }
        }

        return empty_fig, empty_fig, "No data selected", empty_fig
    
    filtered_df = df[
        df['pv tech'].isin(selected_types) &
        df['PV zone'].isin(selected_zones) &
        df['scope of study'].isin(selected_scopes)
    ]

    # apply LLM filter if present
    if chat_filtered_index:
        filtered_df = filtered_df.loc[
            filtered_df.index.intersection(chat_filtered_index)
        ]

    FAULTS_COL = 'faults_list'

    # Faults masks
    reported_faults_mask = filtered_df[FAULTS_COL].apply(
        lambda x: len(x) > 0
    )

    not_reported_faults_mask = filtered_df[FAULTS_COL].apply(
        lambda x: len(x) == 0
    )

    faults_mask = False

    if 'reported' in faults_filter:
        faults_mask |= reported_faults_mask

    if 'not_reported' in faults_filter:
        faults_mask |= not_reported_faults_mask

    filtered_df = filtered_df[faults_mask]

    SYSTEM_CAP_COL = 'system capacity in watts'

    reported_mask = filtered_df[SYSTEM_CAP_COL].notna()
    not_reported_mask = filtered_df[SYSTEM_CAP_COL].isna()

    capacity_mask = False

    # Reported systems
    if 'reported' in capacity_report_filter:
        cap_min_w = (capacity_min or 0) * 1e3
        cap_max_w = (capacity_max or float('inf')) * 1e3

        capacity_mask |= (
            reported_mask &
            (filtered_df[SYSTEM_CAP_COL] >= cap_min_w) &
            (filtered_df[SYSTEM_CAP_COL] <= cap_max_w)
        )

    # Not reported systems
    if 'not_reported' in capacity_report_filter:
        capacity_mask |= not_reported_mask

    filtered_df = filtered_df[capacity_mask]
    
    if rate_min is not None:
        filtered_df = filtered_df[filtered_df['rate'] >= rate_min]
    if rate_max is not None:
        filtered_df = filtered_df[filtered_df['rate'] <= rate_max]
    if duration_min is not None:
        filtered_df = filtered_df[filtered_df['duration'] >= duration_min]
    if duration_max is not None:
        filtered_df = filtered_df[filtered_df['duration'] <= duration_max]
    
    fig = px.scatter_map(
        filtered_df,
        lat='latitude',
        lon='longitude',
        color='rate',
        # size='rate_abs',
        size_max=60,
        zoom=3,
        color_continuous_scale=px.colors.sequential.GnBu[::-3],
        hover_data={"rate": True, "publish year": True, 
                    "country": True, "paper id": True,
                    "duration": True, "confidence level": True,
                    "PV zone": True}
    )

    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.8
        )
    )

    fig.update_layout(
        # title=dict(text='Worldwide PV degradation rate'),
        autosize=True,
        hovermode='closest',
        showlegend=False,
        map=dict(
            bearing=0,
            center=dict(
                lat=20,
                lon=10
            ),
            pitch=0,
            zoom=0.9,
            style='light'
        ),
        
        height=500,
        margin=dict(l=2, t=10),
        coloraxis_colorbar=dict(
            title="Rate<br>(%/year)<br>"  # Correct way to set the title
        )
    )
    
    mean_rate = filtered_df['rate'].mean()
    median_rate = filtered_df['rate'].median()
    std_rate = filtered_df['rate'].std() # Calculate standard deviation
    
    hist_fig = px.histogram(
        filtered_df,
        x='rate',
        nbins=50,
    )

    def _stat_pill(label, value, color="#374151"):
        return html.Span(
            [
                html.Span(label, style={"color": "#6b7280", "marginRight": "4px"}),
                html.Span(value, style={"color": color, "fontWeight": "600"}),
            ],
            style={
                "display": "inline-flex",
                "padding": "2px 10px",
                "marginRight": "6px",
                "fontSize": "12px",
                "fontFamily": _CHART_FONT,
                "backgroundColor": "#f3f4f6",
                "borderRadius": "999px",
            },
        )

    data_info = html.Div([
        _stat_pill("n", f"{len(filtered_df):,}"),
        _stat_pill("Mean", f"{mean_rate:.2f}%/y", color="#10b981"),
        _stat_pill("Median", f"{median_rate:.2f}%/y", color="#1e3a8a"),
    ], style={"marginBottom": "6px"})

    # Calculate KDE
    kde = gaussian_kde(filtered_df['rate'])

    # Generate x values for KDE curve
    x_values = np.linspace(filtered_df['rate'].min(), filtered_df['rate'].max(), 200)  # More points for smoother curve

    # Evaluate KDE at x values
    y_values = kde(x_values)

    # Scale y_values to match the histogram counts (important for visual fit)
    counts, bin_edges = np.histogram(filtered_df['rate'], bins=50)
    y_values = y_values * (counts.max() / y_values.max())  # Scale to max count

    # Add the KDE trace
    hist_fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        name='Kernel Density Estimate',
        yaxis="y",
        line=dict(color='black')
    ))

    # Set color + opacity
    hist_fig.update_traces(marker_color="#0070C0", opacity=0.3)

    # Drop NaNs to avoid issues
    values = filtered_df['rate'].dropna()

    # Compute histogram counts using numpy
    counts, bins = np.histogram(values, bins=25)

    # Max y-value = tallest bar
    max_count = counts.max()

    hist_fig.add_trace(go.Scatter(x=[mean_rate, mean_rate], y=[0, max_count],
        mode='lines', line=dict(color='#009546', dash='dash'), name='Mean'))
    hist_fig.add_trace(go.Scatter(x=[median_rate, median_rate], y=[0, max_count],
        mode='lines', line=dict(color='#002877', dash='dash'), name='Median'))

    hist_fig.update_layout(
        xaxis_title="Degradation Rate (%/year)",
        yaxis_title="Number of cases",
        legend=dict(
            title="Legend",
            orientation="v",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0.01,
        ),
        margin=dict(l=2, t=20)
    )

    duration_fig = px.scatter(filtered_df, x='duration', y='rate',
                              marginal_x="histogram", marginal_y="histogram",
                              hover_data={"rate": True, "publish year": True,
                                            "country": True, "paper id": True,
                                            "duration": True, "confidence level": True})
    # Set scatter marker color and opacity
    duration_fig.update_traces(marker=dict(color="#0070C0"), opacity=0.5)

    # Marginal histograms (x and y)
    duration_fig.update_traces(
        marker=dict(color="#00B050"),
        opacity=0.6,
        selector=dict(type="histogram")
    )

    duration_fig.update_layout(
        xaxis_title='Exposure length (year)',
        yaxis_title='Degradation Rate (%/year)',
    )
    return fig, hist_fig, data_info, duration_fig

# Callback to update table
@app.callback(
    Output('table', 'data'),
    [
        Input('map', 'clickData'),
        Input('pv-tech-filter', 'value'),
        Input('pv-climate-filter', 'value'),
        Input('rate-min', 'value'),
        Input('rate-max', 'value'),
        Input('duration-min', 'value'),
        Input('duration-max', 'value')
    ]
)
def display_click_data(clickData, tech_filter, climate_filter,
                       rate_min, rate_max, duration_min, duration_max,
                       ):

    # if no map point clicked → clear the table
    if not clickData:
        return []

    # if filters changed, Dash will re-run callback → clear table
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] != 'map':
        return []

    # otherwise: show info for clicked point
    point = clickData['points'][0]
    selected_data = df[(df['longitude'] == point['lon']) & 
                       (df['latitude'] == point['lat'])].iloc[0]

    doi = selected_data['doi']
    doi_link = f"[{doi}](https://doi.org/{doi})"

    return [
    {'attribute': 'DOI', 'value': doi_link},
    {'attribute': 'Year', 'value': selected_data['publish year']},
    {'attribute': 'Title', 'value': selected_data['title']},
    {'attribute': 'Type', 'value': selected_data['document type']},
    {'attribute': 'Country', 'value': selected_data['country']},
    {'attribute': 'Climate zone', 'value': selected_data['PV zone']}, 
    # {'attribute': 'Paper ID', 'value': selected_data['paper id']},
    {'attribute': 'Rate', 'value': f'{selected_data["rate"]}%/year'},
    {'attribute': 'PV tech', 'value': selected_data['pv tech']},
    {'attribute': 'Duration', 'value': f'{selected_data["duration"]} years'},
    {'attribute': 'System capacity', 'value': f'{selected_data["system capacity"]/1000} kW'},
    {'attribute': 'Note', 'value': selected_data['note']},
]

@app.callback(
    Output('location-map', 'figure'),
    [Input('lat-input', 'value'),
     Input('lon-input', 'value'),
     Input('radius-input', 'value')]
)
def make_map(lat, lon, radius_miles):
    # --- Step 1: Boolean mask for points inside the radius ---
    inside_mask = df.apply(
        lambda row: distance(
            (lat, lon), (row['latitude'], row['longitude'])
        ).miles <= radius_miles,
        axis=1
    )

    df_inside = df[inside_mask]
    df_outside = df[~inside_mask]

    # --- Step 2: Create figure manually with 2 scattermapbox traces ---
    fig = go.Figure()

    # Outside points (grey)
    fig.add_trace(go.Scattermapbox(
        lat=df_outside['latitude'],
        lon=df_outside['longitude'],
        mode='markers',
        marker=dict(size=10, color='lightgrey'),
        text=df_outside.apply(lambda r: f"Rate: {r['rate']}<br>Year: {r['publish year']}<br>"
                                       f"Country: {r['country']}<br>Paper ID: {r['paper id']}<br>"
                                       f"Duration: {r['duration']}<br>"f"PV tech: {r['pv tech']}<br>"
                                       f"PV zone: {r['PV zone']}", axis=1),
        hoverinfo='text',
        showlegend=False
    ))

    # Inside points (red, with hover info)
    fig.add_trace(go.Scattermapbox(
        lat=df_inside['latitude'],
        lon=df_inside['longitude'],
        mode='markers',
        marker=dict(size=20, color='#00B0F0'),
        text=df_inside.apply(lambda r: f"Rate: {r['rate']}<br>Year: {r['publish year']}<br>"
                                       f"Country: {r['country']}<br>Paper ID: {r['paper id']}<br>"
                                       f"Duration: {r['duration']}<br>"f"PV tech: {r['pv tech']}<br>"
                                       f"PV zone: {r['PV zone']}", axis=1),
        hoverinfo='text',
        showlegend=False,
        opacity=0.6 
    ))

    # --- Step 3: Circle around center ---
    circle_lats, circle_lons = [], []
    for bearing in range(0, 361, 5):  # 5° steps
        dest = distance(miles=radius_miles).destination((lat, lon), bearing)
        circle_lats.append(dest.latitude)
        circle_lons.append(dest.longitude)

    fig.add_trace(go.Scattermapbox(
        lat=circle_lats,
        lon=circle_lons,
        mode='lines',
        line=dict(width=2, color='#0070C0'),
        fill='toself',
        fillcolor='rgba(0,112,192,0.1)',
        showlegend=False
    ))

    # --- Step 4: Update layout ---
    if radius_miles is None:
        zoom = 6
    else:
        # Add padding so circle fits comfortably
        padded_radius = radius_miles * 0.7
        zoom = 8 - math.log(padded_radius / 10, 2)   # heuristic formula
        zoom = max(1, min(zoom, 12))  # clamp to [1,12]

    fig.update_layout(
        mapbox=dict(
            center=dict(lat=lat, lon=lon),
            zoom=zoom,
            style="carto-positron"
        ),
        margin=dict(l=0, r=0, t=0, b=10),
        showlegend=False,
        height=350
    )

    return fig

@app.callback(
    Output('pie-charts-container', 'children'),
    [Input('lat-input', 'value'),
     Input('lon-input', 'value'),
     Input('radius-input', 'value')]
)
def update_pie_charts(lat, lon, radius_miles):
    if lat is None or lon is None or radius_miles is None:
        return html.P("No location selected.", style={'fontStyle': 'italic'})

    # --- filter df within radius ---
    inside_mask = df.apply(
        lambda row: distance((lat, lon), (row['latitude'], row['longitude'])).miles <= radius_miles,
        axis=1
    )
    df_inside = df[inside_mask]

    df_inside['faults_list'] = df_inside['faults_list'].dropna().apply(normalize_faults)

    if df_inside.empty:
        return html.Div(
            [
                html.Div("No data points found within the selected radius.",
                         style={"color": "#6b7280", "fontSize": "14px",
                                "fontFamily": _CHART_FONT}),
                html.Div("Try increasing the search radius or moving the center.",
                         style={"color": "#9ca3af", "fontSize": "12px",
                                "fontFamily": _CHART_FONT, "marginTop": "4px"}),
            ],
            style={"textAlign": "center", "padding": "60px 20px",
                   "border": "1px dashed #e5e7eb", "borderRadius": "12px",
                   "backgroundColor": "#fafbfc"},
        )

    n_points = len(df_inside)

    tiles_row1 = []
    tiles_row2 = []

    # --- Pie chart 1: PV tech ---
    if not df_inside['pv tech'].dropna().empty:
        fig1 = px.pie(df_inside, names='pv tech',
                      color_discrete_sequence=px.colors.sequential.GnBu_r)
        fig1.update_layout(margin=dict(l=20, r=20, t=20, b=10), height=240)
        tiles_row1.append(_chart_tile(
            "PV Technologies",
            dcc.Graph(figure=fig1, config={"displayModeBar": False}),
            subtitle=f"{n_points} sites in radius",
        ))

    # --- Pie chart 2: PV climate zone ---
    if not df_inside['PV zone'].dropna().empty:
        fig2 = px.pie(df_inside, names='PV zone',
                      color_discrete_sequence=px.colors.sequential.GnBu_r)
        fig2.update_layout(margin=dict(l=20, r=20, t=20, b=10), height=240)
        tiles_row1.append(_chart_tile(
            "PV Climate Zones",
            dcc.Graph(figure=fig2, config={"displayModeBar": False}),
            subtitle="Köppen-style zones",
        ))

    # --- Pie chart 3: Faults (top 8) ---
    if not df_inside['faults_list'].dropna().empty:
        fault_counts = Counter(
            fault
            for sublist in df_inside['faults_list'].dropna()
            for fault in sublist
            if fault and fault.strip().lower() not in ['not reported', 'na', 'n/a', 'none']
        )
        if fault_counts:
            top_faults = fault_counts.most_common(8)
            fault_df = pd.DataFrame(top_faults, columns=['Fault', 'Count'])
            fig3 = px.pie(fault_df, names='Fault', values='Count',
                          color_discrete_sequence=px.colors.sequential.GnBu_r)
            fig3.update_layout(margin=dict(l=20, r=20, t=20, b=10), height=240)
            tiles_row2.append(_chart_tile(
                "Faults",
                dcc.Graph(figure=fig3, config={"displayModeBar": False}),
                subtitle="Top 8 reported issues",
            ))

    # --- Box plot: degradation rate per PV tech ---
    if not df_inside['pv tech'].dropna().empty:
        fig4 = px.box(
            df_inside,
            x='pv tech', y='rate',
            color='pv tech',
            color_discrete_sequence=px.colors.sequential.GnBu_r,
            labels={"pv tech": "PV technology", "rate": "Rd (%/year)"},
        )
        fig4.update_layout(margin=dict(l=20, r=20, t=20, b=10),
                           height=240, showlegend=False)
        tiles_row2.append(_chart_tile(
            "Rate Distribution by PV Tech",
            dcc.Graph(figure=fig4, config={"displayModeBar": False}),
        ))

    if not tiles_row1 and not tiles_row2:
        return html.P("No valid values found for PV tech, PV zone, or faults.",
                      style={'fontStyle': 'italic'})

    children = []
    if tiles_row1:
        children.append(
            dbc.Row(
                [dbc.Col(t, xs=12, md=6) for t in tiles_row1],
                className="g-3", style={"marginBottom": "12px"},
            )
        )
    if tiles_row2:
        children.append(
            dbc.Row(
                [dbc.Col(t, xs=12, md=6) for t in tiles_row2],
                className="g-3",
            )
        )
    return html.Div(children)


# ─────────────────────────────────────────────────────────────────────
# City search → update lat/lon inputs
# ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("city-search", "options"),
    Input("city-search", "search_value"),
    prevent_initial_call=True,
)
def update_city_options(search):
    """Filter the cities list by what the user is typing.

    Matches the user's query against the ASCII-folded name only ('asciiname'
    is the GeoNames-supplied accent-stripped form), so "munch" hits "München"
    and "sao" hits "São Paulo". Returns up to 50 results, biggest cities first.
    """
    if _CITIES_DF.empty:
        return []
    if not search or len(search.strip()) < 2:
        return _INITIAL_CITY_OPTIONS
    s = search.strip().lower()
    mask = _CITIES_DF["asciiname"].str.lower().str.startswith(s)
    matches = _CITIES_DF[mask].head(50)
    return [_row_to_option(r) for _, r in matches.iterrows()]


@app.callback(
    Output("lat-input", "value"),
    Output("lon-input", "value"),
    Input("city-search", "value"),
    prevent_initial_call=True,
)
def update_latlon_from_city(city_value):
    """Selected city option's value is 'lat,lon' — split and push to inputs."""
    if not city_value:
        return dash.no_update, dash.no_update
    try:
        lat_str, lon_str = city_value.split(",")
        return float(lat_str), float(lon_str)
    except (ValueError, AttributeError):
        return dash.no_update, dash.no_update


# Run app
if __name__ == '__main__':
    app.run_server(debug=True)