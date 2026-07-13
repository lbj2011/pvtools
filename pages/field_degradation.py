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
# Chat run/cancel tracking. Each LLM query gets an incrementing run id;
# pressing Stop records the current run id as cancelled, so when the
# (uninterruptible) LLM call finally returns we can discard its result
# instead of overwriting the reset. NOTE: this only takes effect while
# the call is running if the server can handle a second request
# concurrently (threaded gunicorn / Flask dev server), since the LLM
# call blocks the worker.
import threading
_chat_lock = threading.Lock()
_chat_state = {"run_id": 0, "cancel_id": 0}

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


# ─────────────────────────────────────────────────────────────────────
# SHAP analysis — figure data is precomputed offline (see extract_shap.py)
# from the LightGBM+SHAP pipeline and shipped as data/shap_fig_data.json.
# Here we only load that file and render Plotly figures in the blue/green
# scheme (green = more degradation / negative SHAP, blue = less degradation).
# ─────────────────────────────────────────────────────────────────────
def _load_shap_data():
    """Locate data/shap_fig_data.json robustly, regardless of where the app is
    launched from. Walks up from the cwd, this file, and field_chat's location,
    checking <ancestor>/data/shap_fig_data.json at each level."""
    starts = []
    try:
        starts.append(os.getcwd())
    except OSError:
        pass
    starts.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from page_supporting_files import field_chat as _fc
        starts.append(os.path.dirname(os.path.abspath(_fc.__file__)))
    except Exception:
        pass

    found = None
    for start in starts:
        d = start
        for _ in range(6):  # walk up a few levels
            cand = os.path.join(d, "data", "shap_fig_data.json")
            if os.path.isfile(cand):
                found = cand
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        if found:
            break

    if not found:
        return None
    try:
        with open(found, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None

SHAP_DATA = _load_shap_data()

# map a ranking-bar label ("PV technology") to its factor key ("pv_tech")
_SHAP_LABEL_TO_VALUE = (
    {s["label"]: s["value"] for s in SHAP_DATA["selector"]} if SHAP_DATA else {}
)

# blue/green diverging scale (matches the GnBu map palette)
_SHAP_GREEN = (0x2c, 0xa2, 0x5f)   # more degradation (negative SHAP)
_SHAP_PALE  = (0xea, 0xf3, 0xf0)   # ~0
_SHAP_BLUE  = (0x08, 0x68, 0xac)   # less degradation (positive SHAP)
SHAP_COLORSCALE = [[0.0, "#2ca25f"], [0.5, "#eaf3f0"], [1.0, "#0868ac"]]

def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def _shap_hex(v, vmax):
    vmax = vmax if vmax and vmax > 0 else 1.0
    t = max(-1.0, min(1.0, v / vmax))
    c = _lerp(_SHAP_PALE, _SHAP_GREEN, -t) if t < 0 else _lerp(_SHAP_PALE, _SHAP_BLUE, t)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_shap_ranking_fig():
    """Horizontal bar of mean|SHAP| per factor, coloured by category (blue/green)."""
    if not SHAP_DATA:
        return go.Figure()
    rk = list(reversed(SHAP_DATA["ranking"]))  # smallest first -> largest on top
    kind_color = {"design": "#0868ac", "measurement": "#41b6c4", "fault": "#78c679"}
    labels = [r["label"] for r in rk]
    vals = [r["value"] for r in rk]
    colors = [kind_color.get(r["kind"], "#0868ac") for r in rk]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=1.2)),
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        textfont=dict(size=12, color="#444"),
        cliponaxis=False, showlegend=False,
        hovertemplate="%{y}<br>impact = %{x:.2f}<extra>click to view</extra>",
    ))
    # category legend swatches
    for c, lab in [("#0868ac", "System & site"), ("#41b6c4", "Measurement"),
                   ("#78c679", "Fault")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=11, color=c, symbol="square"),
            name=lab, showlegend=True))
    fig.update_layout(
        margin=dict(l=6, r=24, t=6, b=34),
        xaxis_title="Average degradation impact",
        font=dict(family=_CHART_FONT, size=13),
        plot_bgcolor="white", paper_bgcolor="white",
        height=300,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="#eef0f2", zeroline=False,
                   range=[0, (max(vals) if vals else 1) * 1.18]),
        yaxis=dict(showgrid=False),
    )
    return fig


def build_shap_raincloud_fig(factor):
    fac = SHAP_DATA["factors"][factor]
    levels = fac["levels"]
    vmax = max([1e-6] + [abs(l["median"]) for l in levels])
    fig = go.Figure()
    for lv in levels:
        col = _shap_hex(lv["median"], vmax)
        fig.add_trace(go.Violin(
            y=lv["values"], x=[lv["label"]] * len(lv["values"]),
            name=lv["label"], side="positive", width=0.9,
            points="all", pointpos=-0.55, jitter=0.25,
            marker=dict(color=col, size=4, opacity=0.5),
            line_color=col, fillcolor=_rgba(col, 0.35),
            box_visible=True, meanline_visible=True,
            showlegend=False, scalemode="width",
            hovertemplate=f'{lv["label"]} (n={lv["n"]})<br>SHAP=%{{y:.3f}}<extra></extra>',
        ))
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"))
    # colorbar (green = more degradation, blue = less) — matches the duration fig
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=[0], colorscale=SHAP_COLORSCALE, cmin=-vmax, cmax=vmax,
                    opacity=0, showscale=True,
                    colorbar=dict(title=dict(text="Median SHAP<br>(%/yr)",
                                             font=dict(size=11)),
                                  tickfont=dict(size=10), thickness=12, len=0.7)),
        showlegend=False, hoverinfo="skip",
    ))
    for c, lab in [("#0868ac", "Less degradation"), ("#2ca25f", "More degradation")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=c, symbol="square"),
            name=lab, showlegend=True))
    fig.update_layout(
        title=dict(text=fac.get("title", ""), x=0.02, xanchor="left",
                   font=dict(size=15, color="#111827")),
        margin=dict(l=10, r=10, t=44, b=42),
        font=dict(family=_CHART_FONT, size=13),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="SHAP value (%/yr)", height=390,
        violingap=0.25, violingroupgap=0,
        legend=dict(orientation="h", y=1.0, x=1, xanchor="right", font=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#eef0f2", zeroline=False),
        xaxis=dict(showgrid=False),
    )
    return fig


def build_shap_scatter_fig(factor):
    fac = SHAP_DATA["factors"][factor]
    py = fac["points_y"]
    vmax = max([1e-6] + [abs(v) for v in py])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fac["points_x"], y=py, mode="markers",
        marker=dict(color=py, colorscale=SHAP_COLORSCALE, cmin=-vmax, cmax=vmax,
                    size=6, opacity=0.55, line=dict(width=0),
                    colorbar=dict(title=dict(text="SHAP<br>(%/yr)", font=dict(size=11)),
                                  tickfont=dict(size=10), thickness=12, len=0.7)),
        hovertemplate="%{x}<br>SHAP=%{y:.3f}<extra></extra>", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fac["line_x"], y=fac["line_y"], mode="lines+markers",
        line=dict(color="black", width=2), marker=dict(color="black", size=7),
        name="Binned median", showlegend=True,
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"))
    fig.update_layout(
        title=dict(text=fac.get("title", ""), x=0.02, xanchor="left",
                   font=dict(size=15, color="#111827")),
        margin=dict(l=10, r=10, t=44, b=42),
        font=dict(family=_CHART_FONT, size=13),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title=fac.get("xlabel", "Study duration (year)"),
        yaxis_title="SHAP (%/yr)", height=390,
        legend=dict(orientation="h", y=1.0, x=1, xanchor="right", font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="#eef0f2", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#eef0f2", zeroline=False),
    )
    return fig


def build_shap_factor_fig(factor):
    if not SHAP_DATA or factor not in SHAP_DATA.get("factors", {}):
        return go.Figure()
    fac = SHAP_DATA["factors"][factor]
    if fac.get("kind") == "scatter":
        return build_shap_scatter_fig(factor)
    return build_shap_raincloud_fig(factor)


def build_shap_body():
    """The two-column SHAP layout: ranking (left) + selector & factor fig (right)."""
    sub_style = {"fontFamily": _CHART_FONT, "fontSize": "15px",
                 "fontWeight": "600", "color": "#111827", "marginBottom": "8px"}

    short = {"pv_tech": "PV tech", "duration": "Duration", "faults": "Faults",
             "pv_zone": "Climate", "mounting": "Mounting", "scope_of_study": "Scope"}

    if not SHAP_DATA:
        pills, default_factor = [], None
        note = html.Div(
            "SHAP figure data not found — add data/shap_fig_data.json.",
            style={"color": "#9ca3af", "fontSize": "13px",
                   "fontFamily": _CHART_FONT, "marginBottom": "10px"})
    else:
        default_factor = SHAP_DATA["selector"][0]["value"]
        pills = [
            html.Button(
                short.get(s["value"], s["label"]),
                id={"type": "shap-pill", "index": s["value"]},
                n_clicks=0,
                className="shap-pill" + (" shap-pill-active"
                                         if s["value"] == default_factor else ""),
            )
            for s in SHAP_DATA["selector"]
        ]
        note = None

    return html.Div([
        note,
        dcc.Store(id="shap-selected", data=default_factor),
        dbc.Row([
            dbc.Col([
                html.Div("Importance ranking of factors", style=sub_style),
                dcc.Graph(id="shap-ranking-graph",
                          figure=build_shap_ranking_fig(),
                          config={"displayModeBar": False},
                          style={"width": "100%"}),
            ], xs=12, sm=12, md=5, lg=5, xl=5),

            dbc.Col([
                html.Div(pills, className="shap-pill-group"),
                dcc.Graph(id="shap-factor-graph",
                          figure=build_shap_factor_fig(default_factor),
                          config={"displayModeBar": False},
                          style={"width": "100%"}),
                html.Div(
                    "A positive SHAP value is associated with less severe "
                    "degradation; a negative value with more severe degradation.",
                    style={"fontFamily": _CHART_FONT, "fontSize": "12.5px",
                           "color": "#6b7280", "marginTop": "6px",
                           "textAlign": "center"}),
            ], xs=12, sm=12, md=7, lg=7, xl=7),
        ], className="g-3", style={"marginLeft": "0px", "marginRight": "0px"}),
    ])


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

# ─────────────────────────────────────────────────────────────────────
# Detail-panel + summary-badge styles (mirror the PV Pathway page so the
# two tools feel like one product). Kept as module constants so the
# click callback can reuse the exact same look.
# ─────────────────────────────────────────────────────────────────────
MAP_DETAIL_STYLE = {
    "position": "relative",
    "width": "350px",
    "maxHeight": "calc(100% - 20px)",
    "background": "rgba(255,255,255,0.55)",
    "backdropFilter": "blur(12px)",
    "WebkitBackdropFilter": "blur(12px)",
    "border": "1px solid rgba(255,255,255,0.6)",
    "boxShadow": "0 8px 32px rgba(140,140,140,0.25)",
    "padding": "16px 18px",
    "borderRadius": "14px",
    "color": "#131314",
    "overflowY": "auto",
    "zIndex": 999,
    "pointerEvents": "auto",
    "userSelect": "text",
}
MAP_DETAIL_HIDDEN = {"display": "none"}


# Layout
layout = html.Div([

  dbc.Container([

    html.Hr(),
    html.Div([
        html.H1(
            "Global PV System Field Performance",
            className="page-title"
        ),
    ], className="page-title-container"),
    html.Hr(),

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

    html.H2("Degradation rate map", className="level-1-title",
            style={"marginTop": "8px", "marginBottom": "6px"}),

    # ── Controls card: structured filters + AI ask-box, above the
    # full-width map (so the map can use the whole page). ──
    dbc.Card([
        dbc.CardBody([

            dcc.Store(id="chat-filtered-data"),

            # Structured filters (the box). The AI question button now
            # lives OUTSIDE this box, just below it.
            html.Div(
            [
                build_filters(types)
            ]),

            # ── AI ask button — sits outside the filter box ──
            html.Div(
                html.Button(
                    [
                        html.Span("Ask Questions to Filter the Data"),
                        dbc.Badge("Beta", color="#00B0F0",
                                  className="ms-2",
                                  style={"verticalAlign": "middle"}),
                        html.Span(id="ai-toggle-caret", children=" ▸",
                                  style={"marginLeft": "8px",
                                         "fontSize": "13px"}),
                    ],
                    id="ai-toggle-btn",
                    n_clicks=0,
                    className="ai-toggle-btn",
                ),
                style={"marginTop": "14px"},
            ),

            # ── Collapsible AI panel (full-width, below the filter card) ──
            dbc.Collapse(
                html.Div(
                [
                    # INPUT + EXAMPLES + RESPONSE, with a manual frosted overlay
                    # (spinner + timing note + Stop button) that we control directly
                    # so the Stop button can clear it immediately.
                    html.Div(
                    [
                        # ── loading overlay (hidden until a query is sent) ──
                        html.Div(
                            [
                                html.Button(
                                    "✕ Stop",
                                    id="chat-stop-btn",
                                    n_clicks=0,
                                    className="ai-stop-btn",
                                ),
                                html.Div(className="ai-run-spinner"),
                                html.Div("⏱ May take 5–20 seconds",
                                         className="ai-run-msg"),
                            ],
                            id="ai-loading-overlay",
                            className="ai-loading-overlay",
                            style={"opacity": "0", "visibility": "hidden",
                                   "pointerEvents": "none"},
                        ),

                        # ── interactive content ──
                        html.Div(
                        [
                            # INPUT ROW (Send / Reset stay on the input's line)
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

                            # RESPONSE
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
                            ),
                        ]
                        ),
                    ],
                    style={"position": "relative"},
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
                    id="llm-reasoning-details",
                    open=False,  # folded by default
                    style={"marginBottom": "16px", "display": "none"},  # hidden until analysis runs
                ),

                ],
                className="llm-card",
                style={
                    # "margin": "0 auto",
                    # "fontFamily": "Inter, system-ui, sans-serif",
                }
            ),
                id="ai-collapse",
                is_open=False,
            ),

        ], style={"padding": "0"}),  # end CardBody
    ], style={"border": "none", "boxShadow": "none",
              "backgroundColor": "transparent"}),

  ]),  # ── end first dbc.Container (everything above the map) ──

    # =================================================================
    # FULL-WIDTH MAP  (breaks out of the container, edge to edge)
    # A glass summary badge floats top-left; a glass detail panel slides
    # in on the left when a data point is clicked — same language as the
    # PV Material Pathway explorer.
    # =================================================================
    html.Div(
        [
            dcc.Graph(
                id='field-map',
                style={"height": "640px", "width": "100%"},
                config={"displayModeBar": False, "scrollZoom": True},
            ),

            # overlay layer — constrained to the page's content column
            # (non-fluid Container) so the panels line up with the filter
            # card above instead of hugging the screen edge.
            dbc.Container(
                html.Div(
                    [
                        html.Div(
                            id="field-map-detail",
                            style=MAP_DETAIL_HIDDEN,
                            children=[
                                # placeholder so Dash registers the id on load
                                html.Button(id="field-map-close-btn", n_clicks=0,
                                            style={"display": "none"})
                            ],
                        ),
                    ],
                    style={
                        "position": "absolute",
                        "top": "16px", "bottom": "16px", "left": "16px",
                        "display": "flex", "flexDirection": "column", "gap": "12px",
                        "maxHeight": "calc(100% - 32px)",
                    },
                ),
                fluid=False,
                style={
                    "position": "absolute",
                    "top": 0, "left": 0, "right": 0, "height": "100%",
                    "pointerEvents": "none", "zIndex": 10,
                },
            ),
        ],
        style={
            "position": "relative",
            "width": "100vw",
            "marginLeft": "calc(-50vw + 50%)",
            "marginTop": "16px",
            "marginBottom": "48px",
        },
    ),

  dbc.Container([   # ── resume container for everything below the map ──
    html.H2('Analysis', className="level-1-title"),
    dbc.Card([

        # ── Part 1: Degradation Rate Overview (title + data-scope toggle) ──
        html.Div(
            [
                html.Div("Degradation Rate Overview",
                         style={"fontFamily": _CHART_FONT, "fontSize": "18px",
                                "fontWeight": "700", "color": "#1f2937"}),
                dcc.RadioItems(
                    id="analysis-data-scope",
                    options=[
                        {"label": "Filtered data", "value": "filtered"},
                        {"label": "All data", "value": "all"},
                    ],
                    value="filtered",
                    inline=True,
                    className="scope-radio",
                ),
            ],
            style={"display": "flex", "alignItems": "center",
                   "justifyContent": "space-between", "flexWrap": "wrap",
                   "gap": "10px", "marginBottom": "16px"},
        ),

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
        ], className="g-3", style={'marginLeft': '0px', 'marginRight': '0px'}),

        # ── divider between the two parts ──
        html.Hr(style={"border": "none", "borderTop": "1px solid #eef0f2",
                       "margin": "24px 0 20px 0"}),

        # ── Part 2: SHAP Analysis (content added later) ──
        html.Div(
            [
                html.Div("SHAP Analysis",
                         style={"fontFamily": _CHART_FONT, "fontSize": "18px",
                                "fontWeight": "700", "color": "#1f2937",
                                "marginBottom": "4px"}),
                html.Div(
                    "How individual factors (e.g. climate zone, PV technology, "
                    "mounting, and system age) contribute to the degradation rate, "
                    "quantified with SHAP values.",
                    style={"fontFamily": _CHART_FONT, "fontSize": "13px",
                           "color": "#6b7280", "marginBottom": "12px",
                           "lineHeight": "1.5"}),
                html.Div(
                    id="shap-analysis-content",
                    children=build_shap_body(),
                ),
            ],
        ),

    ], style={"padding": "20px", "borderRadius": "18px"}),

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
    ], style={"padding": "16px", "borderRadius": "18px"}),

    html.P(''),
    dcc.Markdown("""Dataset version: 2025/9/24
                 
                """.replace('    ', ''), style={'color': 'lightgray'}
    ),
    # Contributor: Baojie Li
    # Project members: Baojie Li, Martin Springer, Dirk Jordan, Anubhav Jain

  ]),  # ── end resumed dbc.Container ──

])      # ── end outer html.Div(layout) ──


# ── Toggle the AI question panel open/closed ─────────────────────────
@app.callback(
    Output("ai-collapse", "is_open"),
    Output("ai-toggle-caret", "children"),
    Input("ai-toggle-btn", "n_clicks"),
    State("ai-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_ai_panel(n_clicks, is_open):
    new_open = not is_open
    return new_open, (" ▾" if new_open else " ▸")


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
    Output("ai-loading-overlay", "style", allow_duplicate=True),
    Input("chat-submit", "n_clicks"),
    Input("chat-input", "n_submit"),   # ← add this
    State("chat-input", "value"),
    prevent_initial_call=True
)
def disable_button_on_action(n_clicks, n_submit, value):
    # only reveal the loading overlay when there is actually a question
    overlay = (
        {"opacity": "1", "visibility": "visible", "pointerEvents": "auto"}
        if (value and value.strip()) else dash.no_update
    )
    return True, overlay


@app.callback(
    Output("response-text", "children", allow_duplicate=True),
    Output("filtered-table", "children", allow_duplicate=True),
    Output("chat-input", "value", allow_duplicate=True),
    Output("chat-submit", "children", allow_duplicate=True),
    Output("chat-submit", "disabled", allow_duplicate=True),
    Output("response-panel", "style", allow_duplicate=True),
    Output("chat-filtered-data", "data", allow_duplicate=True),
    Output("llm-result", "children", allow_duplicate=True),
    Output("llm-reasoning-details", "style", allow_duplicate=True),
    Output("ai-loading-overlay", "style", allow_duplicate=True),
    Input("chat-stop-btn", "n_clicks"),
    prevent_initial_call=True,
)
def stop_chat(n_clicks):
    # Mark the currently-running query as cancelled so its result is discarded
    # when it eventually returns, and reset the chat UI immediately.
    with _chat_lock:
        _chat_state["cancel_id"] = _chat_state["run_id"]

    hidden = {"display": "none"}
    details_hidden = {"marginBottom": "16px", "display": "none"}
    overlay_hidden = {"opacity": "0", "visibility": "hidden", "pointerEvents": "none"}
    return "", "", "", "Send", True, hidden, None, "", details_hidden, overlay_hidden


# ── SHAP: a pill click or a ranking-bar click sets the selected factor ──
@app.callback(
    Output("shap-selected", "data"),
    Input({"type": "shap-pill", "index": dash.ALL}, "n_clicks"),
    Input("shap-ranking-graph", "clickData"),
    State("shap-selected", "data"),
    prevent_initial_call=True,
)
def set_shap_selected(pill_clicks, click_data, current):
    trig = ctx.triggered_id
    if trig == "shap-ranking-graph":
        if click_data:
            label = click_data["points"][0].get("y")
            return _SHAP_LABEL_TO_VALUE.get(label, current)
        return current
    if isinstance(trig, dict) and trig.get("type") == "shap-pill":
        return trig.get("index", current)
    return current


# ── SHAP: render the selected factor's figure and highlight its pill ──
@app.callback(
    Output("shap-factor-graph", "figure"),
    Output({"type": "shap-pill", "index": dash.ALL}, "className"),
    Input("shap-selected", "data"),
    State({"type": "shap-pill", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def render_shap(selected, pill_ids):
    selected = selected or (SHAP_DATA["selector"][0]["value"] if SHAP_DATA else None)
    classes = ["shap-pill shap-pill-active" if pid["index"] == selected
               else "shap-pill" for pid in pill_ids]
    return build_shap_factor_fig(selected), classes

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
    Output("llm-reasoning-details", "style"),
    Output("ai-loading-overlay", "style", allow_duplicate=True),
    Input("chat-submit", "n_clicks"),
    Input("chat-reset", "n_clicks"),
    Input("chat-input", "n_submit"),   # ← ADD THIS
    State("chat-input", "value"),
    prevent_initial_call=True
)
def handle_chat(submit_clicks, reset_clicks, enter_submit, question):

    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # register this run so a later Stop can invalidate it
    with _chat_lock:
        _chat_state["run_id"] += 1
        my_run_id = _chat_state["run_id"]

    def _was_stopped():
        with _chat_lock:
            return my_run_id <= _chat_state["cancel_id"]

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

    # "Show LLM reasoning / filter result" — only visible once an analysis runs
    details_hidden = {"marginBottom": "16px", "display": "none"}
    details_visible = {"marginBottom": "16px", "display": "block"}

    overlay_hidden = {"opacity": "0", "visibility": "hidden", "pointerEvents": "none"}

    if trigger == "chat-reset":
        return "", "", "", "Send", True, hidden, None, "", details_hidden, overlay_hidden

    if trigger in ["chat-submit", "chat-input"]:

        if not question:
            return "", "", "", "Send", True, hidden, None, "", details_hidden, overlay_hidden

        result = get_filter_from_llm(question)

        # If Stop was pressed while this call was running, throw the result
        # away and leave the reset (from stop_chat) in place.
        if _was_stopped():
            return (dash.no_update,) * 10

        msg_text = result.get("reason", "")
        message = html.Div([
                html.Span("Response", className="field-chat-box-metric-label"),
                html.Span(msg_text, className="field-chat-box-metric-text")
            ], className="field-chat-box-metric-row")

        # pretty debug view
        llm_debug = json.dumps(result, indent=2)

        if not result.get("can_be_answered_with_dataframe", False):
            return message, "", "", "Send", True, visible, None, llm_debug, details_visible, overlay_hidden

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
            llm_debug,
            details_visible,
            overlay_hidden,
        )
    
    return "", "", "", "Send", True, hidden, None, "", details_hidden, overlay_hidden


@app.callback(
    [
        Output('field-map', 'figure'),
        Output('histogram', 'figure'),
        Output("data_info", "children"),
        Output('rate-duration-scatter', 'figure'),
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
        Input("chat-filtered-data", "data"),
        Input("analysis-data-scope", "value"),
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
    chat_filtered_index,
    analysis_scope,
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
            size=14,
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
                lon=0
            ),
            pitch=0,
            zoom=2,
            style='light'
        ),

        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Rate<br>(%/year)<br>",
            x=0.99, xanchor="right",
            y=0.5, len=0.7,
            thickness=14,
            bgcolor="rgba(255,255,255,0.6)",
            outlinewidth=0,
        )
    )
    
    # The Part-1 analysis figures can reflect either the filtered subset or the
    # entire dataset, controlled by the "Filtered data / All data" toggle.
    analysis_df = df if (analysis_scope == "all") else filtered_df

    mean_rate = analysis_df['rate'].mean()
    median_rate = analysis_df['rate'].median()
    std_rate = analysis_df['rate'].std() # Calculate standard deviation
    
    hist_fig = px.histogram(
        analysis_df,
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
        _stat_pill("Mean", f"{mean_rate:.2f}%/y", color="#10b981"),
        _stat_pill("Median", f"{median_rate:.2f}%/y", color="#1e3a8a"),
    ], style={"marginBottom": "6px"})

    # Calculate KDE
    kde = gaussian_kde(analysis_df['rate'])

    # Generate x values for KDE curve
    x_values = np.linspace(analysis_df['rate'].min(), analysis_df['rate'].max(), 200)  # More points for smoother curve

    # Evaluate KDE at x values
    y_values = kde(x_values)

    # Scale y_values to match the histogram counts (important for visual fit)
    counts, bin_edges = np.histogram(analysis_df['rate'], bins=50)
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
    values = analysis_df['rate'].dropna()

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

    duration_fig = px.scatter(analysis_df, x='duration', y='rate',
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

# ─────────────────────────────────────────────────────────────────────
# Floating detail panel — slides in over the map on click, hides on close
# or when any filter changes. Styled to match the PV Pathway explorer.
# ─────────────────────────────────────────────────────────────────────
def _detail_row(label, value, *, is_link=False, href=None):
    """One label/value line inside the glass detail panel."""
    if is_link and href:
        value_el = html.A(value, href=href, target="_blank",
                          style={"color": "#0a539c", "wordBreak": "break-all"})
    else:
        value_el = html.Span(str(value), style={"color": "#1f2d3d"})
    return html.Div(
        [
            html.Span(label, style={
                "minWidth": "92px", "color": "#5b6b7a",
                "fontWeight": "600", "fontSize": "12.5px",
                "marginRight": "10px", "flexShrink": "0",
            }),
            html.Span(value_el, style={"fontSize": "12.5px", "lineHeight": "1.4"}),
        ],
        style={"display": "flex", "alignItems": "flex-start",
               "padding": "7px 0", "borderTop": "1px solid rgba(0,0,0,0.06)"},
    )


@app.callback(
    Output('field-map-detail', 'children'),
    Output('field-map-detail', 'style'),
    [
        Input('field-map', 'clickData'),
        Input('field-map-close-btn', 'n_clicks'),
        Input('pv-tech-filter', 'value'),
        Input('pv-climate-filter', 'value'),
        Input('rate-min', 'value'),
        Input('rate-max', 'value'),
        Input('duration-min', 'value'),
        Input('duration-max', 'value'),
    ],
    prevent_initial_call=True,
)
def display_click_data(clickData, close_clicks, tech_filter, climate_filter,
                       rate_min, rate_max, duration_min, duration_max):

    placeholder = [html.Button(id="field-map-close-btn", n_clicks=0,
                               style={"display": "none"})]

    triggered = (dash.callback_context.triggered[0]['prop_id'].split('.')[0]
                 if dash.callback_context.triggered else None)

    # close button, filter change, or no click → hide panel
    if triggered != 'field-map' or not clickData:
        return placeholder, MAP_DETAIL_HIDDEN

    # show info for the clicked point (match on lat/lon as before)
    point = clickData['points'][0]
    match = df[(df['longitude'] == point['lon']) &
               (df['latitude'] == point['lat'])]
    if match.empty:
        return placeholder, MAP_DETAIL_HIDDEN
    sel = match.iloc[0]

    doi = sel['doi']
    try:
        cap_kw = f'{sel["system capacity"] / 1000:g} kW'
    except (TypeError, ValueError):
        cap_kw = "Not reported"

    # publish year is stored as a float (e.g. 2020.0) — show it as an integer
    try:
        year_display = str(int(float(sel.get("publish year"))))
    except (TypeError, ValueError):
        year_display = "N/A"

    children = [
        # close button (matches pathway page)
        html.Button(
            "✕",
            id="field-map-close-btn",
            n_clicks=0,
            style={
                "position": "absolute", "top": "12px", "right": "12px",
                "background": "rgba(255,255,255,0.7)",
                "border": "1px solid rgba(0,0,0,0.12)",
                "borderRadius": "50%", "width": "26px", "height": "26px",
                "fontSize": "12px", "color": "#6b7280", "cursor": "pointer",
                "display": "flex", "alignItems": "center",
                "justifyContent": "center", "padding": "0",
                "lineHeight": "1", "zIndex": 1000,
            },
        ),
        html.P("DEGRADATION DATA POINT", style={
            "fontSize": "11px", "fontWeight": "700", "letterSpacing": "1.3px",
            "color": "#9ca3af", "margin": "0 0 4px 0",
        }),
        html.H5(
            str(sel.get("title", "Study")),
            style={
                "marginBottom": "10px", "paddingRight": "30px",
                "fontWeight": "600", "fontSize": "15px", "color": "#1f2937",
                "display": "-webkit-box", "WebkitLineClamp": 3,
                "WebkitBoxOrient": "vertical", "overflow": "hidden",
            },
        ),
        # headline metric
        html.Div(
            [
                html.Span("Degradation rate", style={
                    "fontSize": "12px", "color": "#5b6b7a", "fontWeight": "600"}),
                html.Span(f'{sel["rate"]:.2f} %/year', style={
                    "fontSize": "20px", "fontWeight": "700", "color": "#0a539c"}),
            ],
            style={"display": "flex", "flexDirection": "column", "gap": "1px",
                   "padding": "10px 12px", "marginBottom": "8px",
                   "background": "rgba(255,255,255,0.55)", "borderRadius": "10px"},
        ),
        # details
        _detail_row("Year", year_display),
        _detail_row("Type", sel.get("document type", "N/A")),
        _detail_row("Country", sel.get("country", "N/A")),
        _detail_row("Climate zone", sel.get("PV zone", "N/A")),
        _detail_row("PV tech", sel.get("pv tech", "N/A")),
        _detail_row("Duration", f'{sel.get("duration", "N/A")} years'),
        _detail_row("System capacity", cap_kw),
        _detail_row("DOI", doi, is_link=True, href=f"https://doi.org/{doi}"),
        _detail_row("Note", sel.get("note", "N/A")),
    ]

    style = dict(MAP_DETAIL_STYLE)
    style["display"] = "block"
    return children, style

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