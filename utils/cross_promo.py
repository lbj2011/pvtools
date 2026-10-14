"""
utils/cross_promo.py -- slim, closable bar pointing visitors of one tool to
other PVTOOLS tools (styles: assets/cross_promo.css).

    from utils.cross_promo import promo_bar
    promo_bar("string-length")

Closing it is remembered in the browser (localStorage), so it stays hidden on
later visits. Links carry ?from=<source page>.
"""
from dash import html, dcc, Input, Output, State

from app import app

_LINKS = [  # (name, href, short description, thumbnail in assets/)
    ("PV-Copilot", "/pv-copilot", "degradation rate from your data", "pvcopilot_logo.png"),
    ("Global PV degradation", "/field-degradation", "field data worldwide", "field_front.jpg"),
]


def promo_bar(source):
    links = []
    for name, href, desc, img in _LINKS:
        links.append(html.A([
            html.Img(src=f"/assets/{img}", alt="",
                     className="pvx-thumb" + (" pvx-thumb-logo" if "logo" in img else "")),
            html.Span([html.B(name), html.Span(desc, className="pvx-desc")], className="pvx-text"),
            html.Span("→", className="pvx-arrow"),
        ], href=f"{href}?from={source}", className="pvx-link"))
    return html.Div([
        dcc.Store(id="pvx-dismissed", storage_type="local"),
        html.Span("Also from PVTOOLS:", className="pvx-label"),
        html.Span(links, className="pvx-links"),
        html.Button("×", id="pvx-close", n_clicks=0, className="pvx-close",
                    title="Hide", **{"aria-label": "Hide"}),
    ], id="pvx-bar", className="pvx-bar")


# Hide on click, and stay hidden on later visits (no server round trip).
app.clientside_callback(
    """
    function(n, dismissed) {
        const nu = window.dash_clientside.no_update;
        if (n) { return [{display: "none"}, true]; }
        return [dismissed ? {display: "none"} : {}, nu];
    }
    """,
    Output("pvx-bar", "style"),
    Output("pvx-dismissed", "data"),
    Input("pvx-close", "n_clicks"),
    State("pvx-dismissed", "data"),
)
