"""
Main page for pvtools website. Run this script to start.

10/09/2023
toddkarin
baojie li

"""

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import time


from app import app

# Line is important for Heroku.w
server = app.server

# Load layouts for different pages
from pages import home, field_degradation, pv_image, pvcopilot
from pages import string_length_calculator, iv_correction_tool, pv_climate_stressors

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content',children=[home.layout])
])


header = dbc.Container([
    html.P(''),
    dbc.Row([
        dbc.Col([
            html.A(
                html.Img(
                    src=app.get_asset_url('LBL_Masterbrand_logo_with_Tagline-01.jpg'),
                    style={'height': 50, 'max-width': '100%', 'height': 'auto'}),
                href="https://www.lbl.gov/"
            )
            ], xs=5, sm=4, md=3, lg=2, xl=2),
        dbc.Col([
            html.A(
                html.Img(
                    src=app.get_asset_url('duramat_logo.png'),
                    style={'height': 50, 'max-width': '90%', 'height': 'auto'}), #added
                href="https://www.duramat.org/"
            )
            ], xs=5, sm=4, md=3, lg=2, xl=2, className="text-end")

        ],justify='between'
        ),
    html.P(''),
])

navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Tools",
            toggle_style={"color": "white"},
            children=[
                dbc.DropdownMenuItem("LLM-Powered PV Image Analysis", href='/pv-image'),
                dbc.DropdownMenuItem("PVcopilot", href='/pv-copilot'),
                dbc.DropdownMenuItem("Worldwide PV Field Performance", href='/field-degradation'),
                dbc.DropdownMenuItem("IV Curve Correction Tool", href='/iv-curve-correction-tool'),
                dbc.DropdownMenuItem("String Length Calculator", href='/string-length-calculator'),
                dbc.DropdownMenuItem("Photovoltaic Climate Stressors", href='/pv-climate-stressors'),
            ],
        ),
    ],

    brand=html.Span(
        [
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
                    style={
                        "height": "30px",
                        "filter": "invert(1)"
                    }
                ),
                href="https://github.com/lbj2011/pvtools",
                target="_blank"
            )
        ],
        style={"display": "flex", "alignItems": "center"}
    ),

    sticky="top",
    color="#000000",
    dark=True,
)


 # Footer
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
        style={
            # 'maxWidth': '900px',
            'margin': '0 auto',
            'fontSize': '14px'
        },
        children=[

            # FUNDING FIRST
            html.P(
                "Funding provided by U.S. DOE Office of Critical Minerals and Energy Innovation (CMEI) Solar Energy Technologies Office (SETO) "
                "and as part of the Durable Module Materials Consortium 2 (DuraMAT 2) funded by the U.S. DOE Office CMEI SETO, Agreement 38259. " \
                "Lawrence Berkeley National Laboratory is funded by the DOE under award DE-AC02-05CH11231." ,               
                style={
                    'fontSize': '12px',
                    'color': '#cccccc',
                    'marginBottom': '20px',
                    'lineHeight': '1.6'
                }
            ),

            # AUTHORS
            html.P(
                "Authors: Baojie Li, Todd Karin, Xin Chen, Anubhav Jain",
                style={'margin': '4px 0'}
            ),

            # html.P([
            #     "Contact: ",
            #     html.A(
            #         "baojieli@lbl.gov",
            #         href="mailto:baojieli@lbl.gov",
            #         style={'color': '#dddddd', 'textDecoration': 'none'}
            #     )
            # ], style={'margin': '4px 0'}),

            html.P(
            [
                "Open-source code and feedback on ",
                html.A(
                    "GitHub",
                    href="https://github.com/lbj2011/pvtools",
                    target="_blank",
                    style={'color': '#dddddd', 'textDecoration': 'none','fontWeight': 'bold'}
                ),
            ],
            style={'margin': '4px 0'}
            ),

            html.P(
                "© 2026 PVTOOLS | Lawrence Berkeley National Laboratory",
                style={'margin': '4px 0','fontWeight': 'bold'}
            )
        ]
    )
)

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/pvdata':
        # body = pv_data.layout
        body = home.layout
        return [header, body]
    else:
        if pathname == '/string-length-calculator':
            
            body = string_length_calculator.layout
            # body = home.layout
        elif pathname == '/home':
            body = home.layout
        elif pathname == '/pv-climate-stressors':
            
            body = pv_climate_stressors.layout
        elif pathname == '/iv-curve-correction-tool':
            
            body = iv_correction_tool.layout
        elif pathname == '/field-degradation':
            body = field_degradation.layout
        elif pathname == '/pv-image':
            body = pv_image.layout
        elif pathname == '/pv-copilot':
            body = pvcopilot.layout
        elif pathname == '/':
            body = home.layout
        else:
            body = '404'

        return [header, navbar, html.P(''), body, footer]

if __name__ == '__main__':
    app.run(debug=True)