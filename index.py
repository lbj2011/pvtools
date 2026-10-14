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
import traceback
import sys

from app import app

# Required for Heroku
server = app.server


# ------------------------------------------------
# Register callbacks by importing pages once
# ------------------------------------------------
# A page that fails to import (e.g. a missing package) must not take the whole site down:
# it is skipped, logged, and its route shows the error instead.
BROKEN_PAGES = {}
for _p in ("home", "field_degradation", "pv_image", "pvcopilot", "pvcopilot_260701",
           "string_length_calculator", "iv_correction_tool", "pv_climate_stressors", "pv_pathway",
           "privacy", "admin_usage"):
    try:
        importlib.import_module(f"pages.{_p}")
    except Exception as _e:
        BROKEN_PAGES[_p] = f"{type(_e).__name__}: {_e}"
        print(f"ERROR: page {_p} not loaded: {BROKEN_PAGES[_p]}")

# Internal dataset-review tool at /dataset-review/ (a separate Dash app on the same server).
# Deliberately not linked from the navbar or any page: internal testing only.
try:
    import dataset_review
    dataset_review.mount(server)
except Exception as _e:  # never let it break the public site
    print(f"dataset review not mounted: {_e}")

# ------------------------------------------------
# Lazy page loader
# ------------------------------------------------
def load_page(page_name):
    if page_name in BROKEN_PAGES:
        return html.Div([html.H3("This tool is temporarily unavailable."),
                         html.P("It will be back after the next update.")])
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
    "/pv-copilot-260701": "pvcopilot_260701",
    "/field-degradation": "field_degradation",
    "/iv-curve-correction-tool": "iv_correction_tool",
    "/string-length-calculator": "string_length_calculator",
    "/pv-climate-stressors": "pv_climate_stressors",
    "/pv-pathway": "pv_pathway",
    "/privacy": "privacy",
    "/admin": "admin_usage",
}

# ------------------------------------------------
# Usage logging (which tool, which action -- never the user's data).
# See utils/usage_log.py for exactly what is and is not recorded.
# ------------------------------------------------
try:
    from utils.usage_log import init_usage_logging
    # the admin dashboard itself is not counted as tool usage
    init_usage_logging(server, {k: v for k, v in routes.items() if k != "/admin"})
except Exception as _e:  # logging must never take the site down
    print(f"usage logging not enabled: {_e}")


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
# Tools menu: bold entries; hovering one shows its cover picture and a short
# description in a preview card beside the menu (styles: assets/nav_menu.css).
TOOL_MENU = [
    # (menu label, href, cover image in assets/, short description)
    ("Global PV System Field Performance", "/field-degradation", "field_front.jpg",
     "Interactive tool for global PV degradation performance."),
    ("LLM-Powered PV Image Analysis", "/pv-image", "pv_image_logo.png",
     "Fault detection using PV visible, EL, and IR images with LLMs."),
    ("PV degradation pathway", "/pv-pathway", "pv_pathway_cover.png",
     "Explore degradation pathways of PV modules from literature."),
    ("PVcopilot", "/pv-copilot", "pvcopilot_cover.png",
     "Data in, degradation results out — LLM-assisted automatic PV data analysis."),
    ("IV Curve Correction Tool", "/iv-curve-correction-tool", "ivcorrection.png",
     "IV curve correction based on IEC 60891:2021."),
    ("String Length Calculator", "/string-length-calculator", "string_length_screenshot.png",
     "Calculate maximum PV string length for a given location."),
    ("Photovoltaic Climate Stressors", "/pv-climate-stressors", "pvcz_screenshow2.jpg",
     "Explore environmental stress distribution for PV systems."),
]


def _tool_menu_item(label, href, image, desc):
    return dbc.DropdownMenuItem(
        [
            html.Span(label, className="pvt-tool-label"),
            html.Div([
                html.Img(src=app.get_asset_url(image), alt="",
                         className="pvt-preview-img"),
                html.Div(label, className="pvt-preview-title"),
                html.Div(desc, className="pvt-preview-desc"),
            ], className="pvt-preview"),
        ],
        href=href,
        className="pvt-tool-item",
    )


navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Tools",
            toggle_style={"color": "white", "fontWeight": 700},
            className="pvt-tools-menu",
            align_end=True,   # keep the menu (and its preview card) on screen
            children=[
                *[_tool_menu_item(*t) for t in TOOL_MENU],
                dbc.DropdownMenuItem(divider=True),
                # separate Dash app on the same server -> full page load (external_link)
                dbc.DropdownMenuItem(
                    "Dataset review (internal)", href='/dataset-review/', external_link=True,
                    className='pvt-internal-item'
                ),
                dbc.DropdownMenuItem(
                    "Usage dashboard (admin)", href='/admin',
                    className='pvt-internal-item'
                ),
            ],
        ),
    ],
    brand=html.Span([
        html.A(
            "PVTOOLS",
            href="/",
            className="pvt-brand-logo",   # same look as the footer logo (assets/nav_menu.css)
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
footer = html.Footer(
    className="pvt-footer",
    children=[
        html.Div(className="pvt-footer-glow"),
        html.Div(
            className="container pvt-footer-inner",
            children=[
                html.Div(
                    className="pvt-footer-grid",
                    children=[
                        # Brand
                        html.Div(className="pvt-footer-col pvt-footer-brand", children=[
                            html.Div("PVTOOLS", className="pvt-footer-logo"),
                            html.P(
                                "Open, physics- and AI-based tools for PV "
                                "system design, performance and reliability.",
                                className="pvt-footer-tagline",
                            ),
                            html.Div(className="pvt-footer-partners", children=[
                                html.A(html.Img(src="/assets/berkeley_lab_logo_trim.png",
                                                alt="Berkeley Lab"),
                                       href="https://www.lbl.gov", target="_blank"),
                                html.A(html.Img(src="/assets/duramat_logo.png",
                                                alt="DuraMAT",
                                                className="pvt-partner-badge"),
                                       href="https://www.duramat.org", target="_blank"),
                            ]),
                        ]),
                        # Team
                        html.Div(className="pvt-footer-col", children=[
                            html.H6("Team", className="pvt-footer-heading"),
                            html.Ul(className="pvt-footer-list", children=[
                                html.Li(n) for n in
                                ["Baojie Li", "Todd Karin", "Xin Chen", "Anubhav Jain"]
                            ]),
                        ]),
                        # Resources
                        html.Div(className="pvt-footer-col", children=[
                            html.H6("Resources", className="pvt-footer-heading"),
                            html.Ul(className="pvt-footer-list", children=[
                                html.Li(html.A(
                                    "GitHub",
                                    href="https://github.com/lbj2011/pvtools",
                                    target="_blank",
                                    className="pvt-footer-link pvt-ico-github")),
                                html.Li(html.A(
                                    "Send feedback",
                                    href="mailto:pvtools.lbl@gmail.com",
                                    className="pvt-footer-link pvt-ico-mail")),
                            ]),
                        ]),
                    ],
                ),
                html.Div(className="pvt-footer-bottom", children=[
                    html.P(
                        "Funding provided by U.S. DOE Office of Critical Minerals "
                        "and Energy Innovation (CMEI) Solar Energy Technologies "
                        "Office (SETO) and as part of the Durable Module Materials "
                        "Consortium 2 (DuraMAT 2).",
                        className="pvt-footer-funding",
                    ),
                    html.P("© 2026 PVTOOLS · Lawrence Berkeley National Laboratory",
                           className="pvt-footer-copy"),
                ]),
            ],
        ),
    ],
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
            
        except Exception as e:
            # Full traceback goes to the server log only; users see a plain
            # message (a traceback exposes server paths and code structure).
            print(f"ERROR loading {pathname}:\n{traceback.format_exc()}")
            body = html.Div([
                html.H3("Error loading page"),
                html.P("Something went wrong on our side. Please try again later "
                       "or report it on GitHub."),
            ])
    else:
        body = html.H3("404")

    return [
        header,
        navbar,
        html.Br(),
        html.Div(body, className="pvt-page-body"),   # bottom gap before the footer: assets/footer.css
        footer
    ]


# ------------------------------------------------
# Run locally
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
