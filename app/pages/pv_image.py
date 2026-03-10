import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import dash_bootstrap_components as dbc
from . import app
import textwrap

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import pandas as pd
import openai
import json
import base64, os, json

# OpenAI API setup
api = 'sk-ZUWPaMt_MaPEPIe7ndcxfw'  # Avoid storing API keys in production code
client = openai.OpenAI(
    api_key=api,
    base_url="https://api.cborg.lbl.gov"
)

# Map valid IDs to real image file names
image_map = {
    'example1': 'example1.jpg',
    'example2': 'example2.jpg',
    'example3': 'example3.png',
    'example4': 'example4.jpg',
    'example5': 'example5.png',
    'example6': 'example6.jpg'
}

image_name_map = {
    'example1': 'Visible image (snow)',
    'example2': 'Visible image (bird dropping)',
    'example3': 'EL image (healthy)',
    'example4': 'EL image (crack)',
    'example5': 'IR image (healthy)',
    'example6': 'IR image (hotspot)'
}

def render_example_thumbnails(selected_id=None):
    """Render clickable thumbnails, highlighting the selected one"""
    return [
        html.Img(
            src=f'/assets/{filename}',
            id=example_id,
            n_clicks=0,
            style = {
                'width': '70px',
                'height': '70px',
                'objectFit': 'cover',
                'margin': '5px',
                'cursor': 'pointer',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                'border': '6px solid #B8FB6B' if example_id == selected_id else '3px solid transparent'
            }
        )
        for example_id, filename in image_map.items()
    ]

def encode_image_as_upload_format(image_path):
    """Read local file and return as upload-style base64 image string"""
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# Table data
table_header = [
    html.Thead(html.Tr([
        html.Th("Model"),
        html.Th("Visible images", colSpan=2, style={"textAlign": "center"}),
        html.Th("EL images", colSpan=2, style={"textAlign": "center"}),
        html.Th("IR images", colSpan=2, style={"textAlign": "center"}),
        html.Th("Average", colSpan=2, style={"textAlign": "center"})
    ])),
    html.Thead(html.Tr([
        html.Th(""),
        html.Th("Binary"), html.Th("Multi"),
        html.Th("Binary"), html.Th("Multi"),
        html.Th("Binary"), html.Th("Multi"),
        html.Th("Binary"), html.Th("Multi")
    ]))
]

# Table rows
table_body = html.Tbody([
    html.Tr([html.Td("Gemini-2.5-Pro"), html.Td("0.96"), html.Td("0.90"), html.Td("0.96"), html.Td("0.91"),
             html.Td("0.84"), html.Td("0.71"), html.Td("0.92"), html.Td("0.84")]),
    html.Tr([html.Td("GPT-5.1"), html.Td("0.96"), html.Td("0.92"), html.Td("0.98"), html.Td("0.82"),
             html.Td("0.80"), html.Td("0.83"), html.Td("0.91"), html.Td("0.86")]),
    html.Tr([html.Td("Claude-Sonnet"), html.Td("0.96"), html.Td("0.77"), html.Td("0.96"), html.Td("0.78"),
             html.Td("0.65"), html.Td("0.59"), html.Td("0.86"), html.Td("0.71")])
])

question = """
First, judge if this image is a PV cell/module/array visible or electroluminescence (EL) or infrared (IR)  images.

If it is a visible image of a PV module or array. Assess its condition and return the probabilities for each category as a dictionary with the following format:
{
  "pv_image": true or false,
  "pv_image_type": "visible",
  "probabilities": {
    "clean": <probability>,
    "snow": <probability>,
    "bird_droppings": <probability>,
    "dust_or_soiling": <probability>,
    "hail_crack": <probability>
  }
} 
"pv_image": a boolean indicating whether the input is a PV image
"pv_image_type": a string indicating the type of PV image ("visible", "EL", "IR", or "other")
Probabilities should sum to 1. 
Only return the JSON dictionary.

If it is an electroluminescence (EL) image of a photovoltaic (PV) cell, assess the condition of the cell and return the probabilities for each of the following categories in a JSON dictionary format:

{
  "pv_image": true or false,
  "pv_image_type": "EL",
  "probabilities": {
  "healthy": <probability>,
  "crack": <probability>
   }
}

"pv_image": a boolean indicating whether the input is a PV image
"pv_image_type": a string indicating the type of PV image ("visible", "EL", "IR", or "other")

Category definitions:
- "healthy": The cell appears structurally intact with no visible cracks or dark areas (excluding natural dark zones at the four corners or grid lines).
- "crack": The cell contains one or more visible cracks or black/dark regions, excluding the four corners and the grid lines. These may include hairline fractures, shattered zones, or abnormal dark areas indicating damage.

Probabilities should be numeric values between 0 and 1 and must sum to 1. Do not include any explanation or additional text. Only return the JSON dictionary.

If it is an infrared (IR) image of a photovoltaic (PV) module or array. Assess the thermal condition based on visible brightness patterns and return the probabilities for each of the following categories in a JSON dictionary format:

 {
  "pv_image": true or false,
  "pv_image_type": "IR",
 "probabilities": {
  "healthy": <probability>,
  "hotspot": <probability>
}
}
"pv_image": a boolean indicating whether the input is a PV image
"pv_image_type": a string indicating the type of PV image ("visible", "EL", "IR", or "other")

Category definitions:
- "healthy": The module shows no visible hotspots or brighter regions. Thermal distribution appears uniform.
- "hotspot": There is at least one visible brighter region (localized or multiple), indicating thermal anomaly.

Probabilities should be numeric values between 0 and 1 and must sum to 1. Do not include any explanation or additional text. Only return the JSON dictionary.

if it is not a PV image, still return a JSON file:
{
  "pv_image": false
  }
"pv_image": a boolean indicating whether the input is a PV image
Do not include any explanation or additional text. Only return the JSON dictionary.

"""

# app = dash.Dash(__name__)

layout = dbc.Container([
    html.Div([
        html.Hr(),
        html.Div([
            html.H1("Unified LLM-Based PV Image Diagnostic Framework (Demo)"),
        ], style={
            # 'background-color': 'lightblue',
            'width': '100%',
            'padding-left': '10px',
            'padding-right': '10px',
            'textAlign': 'center'}),
        html.Hr(),

        html.P(''),
        dbc.Row([
            
            dbc.Col([
                dcc.Markdown(
                    textwrap.dedent("""
                    This demo showcases a unified LLM-based framework for automated PV image diagnostics across heterogeneous images.

                    **1. Test your own PV image:**  
                    You can upload a **visible**, **electroluminescence (EL)**, or **infrared (IR)** image of a PV module or array. The LLM (ChatGPT-5.1) will instantly analyze the image and return diagnostic results based on the following categories:

                    - **Visible images** – Detects: *Clean*, *Soiling*, *Hail Damage*, *Snow Coverage*, *Bird Droppings*  
                    - **EL images** – Detects: *Healthy*, *Cell Crack*  
                    - **IR images** – Detects: *Healthy*, *Hotspot*

                    **2. Review current LLM performance:**  
                    A summary table shows the **F1 scores** of various LLMs on a curated PV image dataset containing visible, EL, and IR images. The results reflect performance across both binary and multi-class diagnostic tasks.
                    """)
                )
            ], xs=12, sm=12, md=12, lg=9, xl=9),

            dbc.Col([
                html.Img(src=app.get_asset_url('llm_logo.jpg'),
                style={'width': '80%'}),
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
        dbc.Card([
            dbc.CardHeader(html.H4("1. Test your own PV image")),

            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                'Drop or ',
                                html.A('Select an Image')
                            ]),
                            style={
                                'width': '70%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '2px', 'borderStyle': 'dashed',
                                'borderRadius': '10px', 'textAlign': 'center', 'marginBottom': '10px', 'cursor': 'pointer'
                            },
                            accept='image/*',
                            multiple=False
                        ),
                        html.Div(
                            'Upload PV visible, EL, or IR images (JPEG, PNG format)',
                            style={'fontSize': '0.9em', 'color': 'gray', 'marginBottom': '15px'}
                        ),

                        html.Div('Or select an example image below to analyze:', style={'marginBottom': '10px'}),

                        html.Div(
                            children=render_example_thumbnails(),  # ✅ use the helper to stay consistent
                            id='example-image-container',
                            style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '15px'}
                        ),

                        html.Div(id='upload-status'),
                        dcc.Store(id='image-display-flag', data=False),  # Track if image has been shown
                        dcc.Store(id='image-content-store'),  # stores uploaded or clicked image content
                        html.Div(id='output-image-upload')
                    ], xs=12, md=6),

                    dbc.Col([
                        html.P([]),
                        html.Button('Click to run the analysis', id='analyze-button', n_clicks=0,
                                    style={
                                        'marginBottom': '10px',
                                        'padding': '10px 20px',
                                        'backgroundColor': '#0070C0',
                                        'color': 'white',
                                        'fontWeight': 'bold',         # This makes the text bold
                                        'borderRadius': '10px',       # This gives soft, rounded edges
                                        'border': 'none'              # Optional: removes any default border
                                    }),
                        html.Div(
                            '(It takes about 3-8 seconds)',
                            style={'fontSize': '0.9em', 'color': 'gray', 'marginBottom': '15px'}
                        ),
                        dcc.Loading(id='loading-progress', type='default', children=html.Div(id='image-analysis-result'))
                    ], xs=12, md=6)
                ])

            ])
        ], className="my-4"),

        html.P(''),

        dbc.Card([
            dbc.CardHeader(html.H4("2. Dataset and Performance")),

            dbc.CardBody([
                dbc.Row([
                    # Left: PV image and link
                    dbc.Col([
                        html.H5("PV Image Test Dataset"),
                        html.A(
                            html.Img(
                                src="/assets/images.png",
                                style={"width": "90%", "height": "auto", "marginBottom": "10px"}
                            ),
                            href="https://github.com/DuraMAT/PV-LLM",
                            target="_blank"
                        ),
                        html.P([
                            "Example image from the test dataset. Learn more at ",
                            html.A("DuraMAT/PV-LLM", href="https://github.com/DuraMAT/PV-LLM", target="_blank")
                        ], style={'fontSize': '0.9em', 'color': 'gray'})
                    ], md=6),

                    # Right: Table
                    dbc.Col([
                        html.H5("LLM Performance"),

                        html.Div([
                            html.Table(
                                table_header + [table_body],
                                className="table table-bordered table-striped",
                                style={'fontSize': '0.9em', 'minWidth': '700px'}
                            )
                        ], style={'overflowX': 'auto'}),  # Enables horizontal scrolling on small screens

                        html.P("(Updated on 2026/2/6)", style={'fontSize': '0.9em', 'color': 'gray'})
                    ], md=6)
                ])
            ])
        ], className="my-4")  # Adds spacing above and below the outer card

        ], style={
    })
])




def analyze_image(base64_image):
    response = client.chat.completions.create(
        model="openai/gpt-5.1",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=1000,
    )

    res_text = response.choices[0].message.content.strip()
    result = json.loads(res_text)

    return result


@app.callback(
    [Output('upload-status', 'children'),
     Output('output-image-upload', 'children'),
     Output('image-analysis-result', 'children'),
     Output('example-image-container', 'children'),
     Output('image-display-flag', 'data'),
     Output('image-content-store', 'data')],   # ✅ new output
    [Input('upload-image', 'contents'),
     Input('analyze-button', 'n_clicks'),
     Input('example1', 'n_clicks'),
     Input('example2', 'n_clicks'),
     Input('example3', 'n_clicks'),
     Input('example4', 'n_clicks'),
     Input('example5', 'n_clicks'),
     Input('example6', 'n_clicks')],
    [State('upload-image', 'contents'),
    State('image-display-flag', 'data'),
    State('image-content-store', 'data')]     # ✅ new state
)
def unified_callback(upload_content, n_clicks, n1, n2, n3, n4, n5, n6, uploaded_image, image_displayed, stored_image):

    trigger_id = ctx.triggered_id

    status_msg = dash.no_update
    image_display = dash.no_update
    analysis_output = dash.no_update
    thumbnails = render_example_thumbnails()

    # Case 1: new image uploaded
    if trigger_id == 'upload-image' and upload_content:
        status_msg = html.Span(
            'Status: Your image is successfully uploaded',
            style={'color': 'gray', 'fontStyle': 'italic'}
        )
        image_display = html.Div([
            html.P(''),
            html.Div(html.Strong('Your image')),
            html.Img(
                src=upload_content,
                style={
                    'width': '250px',
                    'height': '250px',
                    'objectFit': 'cover',
                    'marginTop': '10px',
                    'borderRadius': '12px'
                }
            )
        ])
        image_displayed = True  # ✅ set image display flag
        stored_image = upload_content  # ✅ store uploaded image content
        analysis_output = ''  # ✅ clear results
        thumbnails = render_example_thumbnails()  # reset highlight

    # Case 2: example image clicked
    elif trigger_id in image_map:
        img_path = os.path.join('assets', image_map[trigger_id])
        encoded_image = encode_image_as_upload_format(img_path)
        app._example_base64 = encoded_image  # simulate upload

        status_msg = html.Span(
            f'Status: Example image "{image_name_map[trigger_id]}" selected',
            style={'color': 'gray', 'fontStyle': 'italic'}
        )
        image_display = html.Div([
            html.P(''),
            html.Div(html.Strong(f'Example image: {image_name_map[trigger_id]}')),
            html.Img(
                src=encoded_image,
                style={
                    'width': '300px',
                    'height': '300px',
                    'objectFit': 'cover',
                    'marginTop': '10px',
                    'borderRadius': '12px'
                }
            )
        ])
        image_displayed = True  # ✅ set image display flag
        stored_image = encoded_image  # ✅ store clicked image content
        analysis_output = ''  # ✅ clear results
        thumbnails = render_example_thumbnails(selected_id=trigger_id)

    # Case 3: Analyze clicked
    elif trigger_id == 'analyze-button' and n_clicks > 0:
        image_data = stored_image

        if not image_displayed:
            analysis_output = html.Div(
                'Please select or upload an image before analyzing.',
                style={'color': 'orange', 'fontWeight': 'bold', 'marginTop': '10px'}
            )
        else:
            try:
                _, base64_image = image_data.split(',')
                result = analyze_image(base64_image)

                if not result.get("pv_image", False):
                    analysis_output = html.Div([
                        html.Strong('This is not a PV image. Please upload a new one.',
                                    style={'color': 'orange', 'fontWeight': 'bold', 'marginTop': '10px'})
                    ])
                else:
                    pv_image_type = result.get("pv_image_type", "other")
                    prob_dict = result.get("probabilities", {})
                    predicted_category = max(prob_dict, key=prob_dict.get)

                    df = pd.DataFrame({
                        'Category': list(prob_dict.keys()),
                        'Probability': list(prob_dict.values())
                    })

                    fig = px.bar(df, x='Category', y='Probability', range_y=[0, 1])
                    fig.update_traces(marker_color='#259EEA')
                    fig.update_layout(height=300, margin=dict(l=30, r=30, t=30, b=30), autosize=True)

                    analysis_output = html.Div([
                        html.Strong(f'PV Image Type: {pv_image_type}', style={'display': 'block', 'marginBottom': '5px'}),
                        html.Strong(f'Predicted Category: {predicted_category}', style={'marginBottom': '10px'}),
                        dcc.Graph(figure=fig),
                        html.Pre(json.dumps(prob_dict, indent=2), style={'color': 'gray'})
                    ])
            except Exception as e:
                analysis_output = html.Div(f'Error during analysis: {str(e)}', style={'color': 'red'})

    return status_msg, image_display, analysis_output, thumbnails, image_displayed, stored_image


if __name__ == '__main__':
    app.run_server(debug=True)