"""
Main page for pvtools website.

Lazy layout loading while keeping callbacks registered.
Works with Dash + Gunicorn + Heroku.
"""

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import time
from flask import request
import importlib

from app import app

# Required for Heroku
server = app.server


# ------------------------------------------------
# Register callbacks by importing pages once
# ------------------------------------------------
import pages.home
import pages.field_degradation
import pages.pv_image
import pages.pvcopilot
import pages.string_length_calculator
import pages.iv_correction_tool
import pages.pv_climate_stressors


# ------------------------------------------------
# Lazy page loader
# ------------------------------------------------
def load_page(page_name):
    module = importlib.import_module(f"pages.{page_name}")
    return module.get_layout()


# ------------------------------------------------
# Routes
# ------------------------------------------------
routes = {
    "/": "home",
    "/home": "home",
    "/pv-image": "pv_image",
    "/pv-copilot": "pvcopilot",
    "/field-degradation": "field_degradation",
    "/iv-curve-correction-tool": "iv_correction_tool",
    "/string-length-calculator": "string_length_calculator",
    "/pv-climate-stressors": "pv_climate_stressors",
}


# ------------------------------------------------
# App layout (no page imports here!)
# ------------------------------------------------
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])


# ------------------------------------------------
# Header
# ------------------------------------------------
header = dbc.Container([
    html.P(""),
    dbc.Row([
        dbc.Col([
            html.A(
                html.Img(
                    src=app.get_asset_url('LBL_Masterbrand_logo_with_Tagline-01.jpg'),
                    style={'height': 50, 'max-width': '100%', 'height': 'auto'}
                ),
                href="https://www.lbl.gov/"
            )
        ], xs=5, sm=4, md=3, lg=2),

        dbc.Col([
            html.A(
                html.Img(
                    src=app.get_asset_url('duramat_logo.png'),
                    style={'height': 50, 'max-width': '90%', 'height': 'auto'}
                ),
                href="https://www.duramat.org/"
            )
        ], xs=5, sm=4, md=3, lg=2, className="text-end")
    ], justify="between"),
    html.P("")
])


# ------------------------------------------------
# Navbar
# ------------------------------------------------
navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Tools",
            toggle_style={"color": "white"},
            children=[
                dbc.DropdownMenuItem(
                    "LLM-Powered PV Image Analysis", href='/pv-image'
                ),
                dbc.DropdownMenuItem(
                    "PVcopilot", href='/pv-copilot'
                ),
                dbc.DropdownMenuItem(
                    "Worldwide PV Field Performance", href='/field-degradation'
                ),
                dbc.DropdownMenuItem(
                    "IV Curve Correction Tool", href='/iv-curve-correction-tool'
                ),
                dbc.DropdownMenuItem(
                    "String Length Calculator", href='/string-length-calculator'
                ),
                dbc.DropdownMenuItem(
                    "Photovoltaic Climate Stressors", href='/pv-climate-stressors'
                ),
            ],
        ),
    ],
    brand=html.Span([
        html.A(
            "PVTOOLS",
            href="/",
            style={
                "color": "white",
                "fontWeight": "bold",
                "textDecoration": "none",
                "marginRight": "10px"
            }
        ),
        html.A(
            html.Img(
                src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
                style={"height": "30px", "filter": "invert(1)"}
            ),
            href="https://github.com/lbj2011/pvtools",
            target="_blank"
        )
    ],
    style={"display": "flex", "alignItems": "center"}),
    sticky="top",
    color="#000000",
    dark=True,
)


# ------------------------------------------------
# Footer
# ------------------------------------------------
footer = html.Div(
    style={
        'backgroundColor': '#000000',
        'color': 'white',
        'padding': '30px 0',
        'marginTop': '50px',
        'textAlign': 'center'
    },
    children=html.Div(
        className="container",
        style={'margin': '0 auto', 'fontSize': '14px'},
        children=[

            html.P(
                "Funding provided by U.S. DOE Office of Critical Minerals "
                "and Energy Innovation (CMEI) Solar Energy Technologies "
                "Office (SETO) and as part of the Durable Module Materials "
                "Consortium 2 (DuraMAT 2).",
                style={
                    'fontSize': '12px',
                    'color': '#cccccc',
                    'marginBottom': '20px',
                    'lineHeight': '1.6'
                }
            ),

            html.P(
                "Authors: Baojie Li, Todd Karin, Xin Chen, Anubhav Jain"
            ),

            html.P([
                "Open-source code and feedback on ",
                html.A(
                    "GitHub",
                    href="https://github.com/lbj2011/pvtools",
                    target="_blank",
                    style={
                        'color': '#dddddd',
                        'textDecoration': 'none',
                        'fontWeight': 'bold'
                    }
                ),
            ]),

            html.P(
                "© 2026 PVTOOLS | Lawrence Berkeley National Laboratory",
                style={'fontWeight': 'bold'}
            )
        ]
    )
)


# ------------------------------------------------
# Router
# ------------------------------------------------
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):

    start = time.time()

    page = routes.get(pathname)

    if page:
        try:
            body = load_page(page)
            print(f"DEBUG: Loading {pathname} took {time.time() - start:.2f} seconds")
            
        except Exception as e:
            body = html.Div([
                html.H3("Error loading page"),
                html.Pre(str(e))
            ])
    else:
        body = html.H3("404")

    return [
        header,
        navbar,
        html.Br(),
        body,
        footer
    ]


# ------------------------------------------------
# Run locally
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
