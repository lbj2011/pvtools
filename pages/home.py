
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, callback

from app import app

# Initialize the app - incorporate css
# external_stylesheets = [dbc.themes.BOOTSTRAP]
# app = Dash(__name__, external_stylesheets=external_stylesheets)

green_button_style = {
    'backgroundColor': '#92D050',
    'borderColor': '#92D050',
    'color': 'white',
    'fontWeight': 'bold'  # ✅ makes the text bold
}

def make_tool_card(title, image, description, link, badge=None):

    header = html.Span([
        html.B(title),
        dbc.Badge(badge, color="success", className="ms-2") if badge else None
    ])

    return dbc.Card(

        [
            dbc.CardImg(
                src=app.get_asset_url(image),
                top=True,
                className="tool-card-img"
            ),

            dbc.CardBody(
                [
                    html.H5(header, className="tool-card-title"),

                    html.P(description, className="tool-card-desc"),

                    dbc.Button(
                        "Launch Tool",
                        href=link,
                        style=green_button_style,
                        className="tool-card-btn"
                    )
                ],
                className="tool-card-body"
            )
        ],

        className="tool-card shadow-sm h-100"
    )

layout = dbc.Container([

    # ---------- ABOUT ----------
    dbc.Row(
        dbc.Col([
            html.H3("About", className="mt-4"),
            html.P(
                """
                PVTOOLS is a set of web applications and Python libraries for
                photovoltaic-specific applications. This work is funded by the
                DuraMAT consortium.
                """,
                className="about-text"
            ),
        ], width=12),
        className="mb-4"
    ),

    html.H3("Tools", className="mt-4"),

    # ---------- TOOL CARDS ----------
    dbc.Row([

        dbc.Col(make_tool_card(
            "PV Image Analysis Using LLMs",
            "pv_image_logo.png",
            "Fault detection using PV visible, EL, and IR images with LLMs.",
            "pv-image",
            badge="New!"
        ), xs=12, sm=6, lg=4),

        dbc.Col(make_tool_card(
            "Global PV Field Performance",
            "field_front.jpg",
            "Interactive tool for global PV degradation performance.",
            "field-degradation",
            badge="New!"
        ), xs=12, sm=6, lg=4),

        dbc.Col(make_tool_card(
            "IV Curve Correction Tool",
            "ivcorrection.png",
            "IV curve correction based on IEC 60891:2021.",
            "iv-curve-correction-tool"
        ), xs=12, sm=6, lg=4),

        dbc.Col(make_tool_card(
            "String Length Calculator",
            "string_length_screenshot.png",
            "Calculate maximum PV string length for a given location.",
            "string-length-calculator"
        ), xs=12, sm=6, lg=4),

        dbc.Col(make_tool_card(
            "Photovoltaic Climate Zones",
            "pvcz_screenshow2.jpg",
            "Explore environmental stress distribution for PV systems.",
            "pv-climate-stressors"
        ), xs=12, sm=6, lg=4),

    ], className="g-4")

])


if __name__ == "__main__":
    app.layout = html.Div(layout)
    app.run_server()
    app.run(debug=True)