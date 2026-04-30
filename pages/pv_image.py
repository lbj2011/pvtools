import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import dash_bootstrap_components as dbc
from app import app
import textwrap

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import pandas as pd
import openai
import json
import base64, os, json
import os

from dotenv import load_dotenv

load_dotenv()

cborg_API_KEY = os.getenv("cborg_api_key")
client = openai.OpenAI(
    api_key=cborg_API_KEY,
    base_url="https://api.cborg.lbl.gov"
)

def get_layout():
    return layout

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
            className='pv-thumb pv-thumb-selected' if example_id == selected_id else 'pv-thumb',
            style={
                'width': '48px',
                'height': '48px',
                'objectFit': 'cover',
                'margin': '3px',
                'cursor': 'pointer',
                'borderRadius': '6px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.15)',
                'border': '2px solid #0070C0' if example_id == selected_id else '2px solid transparent',
                'transition': 'all 0.18s ease',
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
First, judge if this image is a PV cell/module/array visible or electroluminescence (EL) or infrared (IR) image.

You MUST always include an "explanation" field in the JSON response. Keep the explanation SHORT: maximum 2 sentences and approximately 40 words total. Be concise and concrete:
1. State the key visual feature(s) you observed (1 sentence).
2. State the recommended action or caveat, if any (1 sentence, optional).
Do not repeat the category name multiple times. Do not write more than 2 sentences.

If it is a visible image of a PV module or array, assess its condition and return:
{
  "pv_image": true,
  "pv_image_type": "visible",
  "probabilities": {
    "clean": <probability>,
    "snow": <probability>,
    "bird_droppings": <probability>,
    "dust_or_soiling": <probability>,
    "hail_crack": <probability>
  },
  "explanation": "<your explanation here>"
}

If it is an electroluminescence (EL) image of a PV cell, return:
{
  "pv_image": true,
  "pv_image_type": "EL",
  "probabilities": {
    "healthy": <probability>,
    "crack": <probability>
  },
  "explanation": "<your explanation here>"
}

Category definitions for EL:
- "healthy": The cell appears structurally intact with no visible cracks or dark areas (excluding natural dark zones at the four corners or grid lines).
- "crack": The cell contains one or more visible cracks or black/dark regions, excluding the four corners and the grid lines. These may include hairline fractures, shattered zones, or abnormal dark areas indicating damage.

If it is an infrared (IR) image of a PV module or array, return:
{
  "pv_image": true,
  "pv_image_type": "IR",
  "probabilities": {
    "healthy": <probability>,
    "hotspot": <probability>
  },
  "explanation": "<your explanation here>"
}

Category definitions for IR:
- "healthy": The module shows no visible hotspots or brighter regions. Thermal distribution appears uniform.
- "hotspot": There is at least one visible brighter region (localized or multiple), indicating thermal anomaly.

If it is NOT a PV image, return:
{
  "pv_image": false,
  "explanation": "<short explanation describing what the image actually shows and why it is not a PV image>"
}

Probabilities must be numeric values between 0 and 1 and must sum to 1.
Return ONLY the JSON dictionary. Do not include markdown code fences, prose, or any text outside the JSON.
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
                    # ============================ STEP 1 ============================
                    dbc.Col([
                        html.Div([
                            html.Span('1', className='pv-step-badge'),
                            html.Span('Provide an image', className='pv-step-title'),
                        ], className='pv-step-header'),

                        dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                html.Div('⬆', style={
                                    'fontSize': '1.5em',
                                    'color': '#0070C0',
                                    'lineHeight': '1',
                                    'marginBottom': '4px',
                                }),
                                html.Div([
                                    html.Span('Drop an image here, or '),
                                    html.A('browse files', style={
                                        'color': '#0070C0',
                                        'fontWeight': '600',
                                        'textDecoration': 'underline',
                                    }),
                                ], style={'fontSize': '0.9em', 'color': '#333'}),
                            ], style={'textAlign': 'center'}),
                            className='pv-upload-box',
                            style={
                                'width': '100%',
                                'padding': '16px 10px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderColor': '#B8D4EA',
                                'borderRadius': '12px',
                                'backgroundColor': '#F8FBFD',
                                'textAlign': 'center',
                                'marginBottom': '6px',
                                'cursor': 'pointer',
                                'transition': 'all 0.2s ease',
                            },
                            accept='image/*',
                            multiple=False
                        ),
                        html.Div(
                            'Accepts visible, EL, or IR images (JPEG / PNG)',
                            style={'fontSize': '0.78em', 'color': '#7A7A7A', 'marginBottom': '14px'}
                        ),

                        html.Div('Or pick an example:', style={
                            'fontSize': '0.85em',
                            'fontWeight': '600',
                            'color': '#444',
                            'marginBottom': '6px',
                        }),

                        html.Div(
                            children=render_example_thumbnails(),
                            id='example-image-container',
                            style={
                                'display': 'flex',
                                'flexWrap': 'wrap',
                                'marginBottom': '10px',
                                'padding': '6px',
                                'backgroundColor': '#FAFBFC',
                                'borderRadius': '10px',
                                'border': '1px solid #ECEFF3',
                            }
                        ),

                        html.Div(id='upload-status', style={'marginTop': '6px'}),
                        dcc.Store(id='image-display-flag', data=False),
                        dcc.Store(id='image-content-store'),
                        html.Div(id='output-image-upload')
                    ], xs=12, md=6, lg=4, className='mb-3 pv-col-step1'),

                    # ============================ STEP 2 ============================
                    dbc.Col([
                        html.Div([
                            html.Span('2', className='pv-step-badge'),
                            html.Span('Choose a model', className='pv-step-title'),
                        ], className='pv-step-header'),

                        html.Div(
                            'Pick which LLM should analyze the image:',
                            style={'fontSize': '0.85em', 'color': '#555', 'marginBottom': '10px'}
                        ),

                        dcc.RadioItems(
                            id='model-selector',
                            options=[
                                {'label': 'ChatGPT-5.1', 'value': 'openai/gpt-5.1'},
                                {'label': 'Gemini Flash', 'value': 'gemini-flash'},
                                {'label': 'Claude Opus', 'value': 'claude-opus'},
                            ],
                            value='openai/gpt-5.1',
                            className='pv-model-radio',
                            inputStyle={'marginRight': '10px'},
                            labelStyle={
                                'display': 'flex',
                                'alignItems': 'center',
                                'marginBottom': '10px',
                                'cursor': 'pointer',
                                'fontSize': '0.95em',
                                'color': '#333',
                            },
                        ),
                    ], xs=12, md=6, lg=4, className='mb-3 pv-col-step2'),

                    # ============================ STEP 3 ============================
                    dbc.Col([
                        html.Div([
                            html.Span('3', className='pv-step-badge'),
                            html.Span('Run & view results', className='pv-step-title'),
                        ], className='pv-step-header'),

                        html.Button(
                            'Click to run the analysis',
                            id='analyze-button',
                            n_clicks=0,
                            className='pv-run-button',
                            style={
                                'marginBottom': '6px',
                                'padding': '10px 22px',
                                'backgroundColor': '#0070C0',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'borderRadius': '10px',
                                'border': 'none',
                                'cursor': 'pointer',
                                'boxShadow': '0 2px 6px rgba(0,112,192,0.25)',
                                'transition': 'all 0.18s ease',
                            }
                        ),
                        html.Div(
                            '(It takes about 3–8 seconds)',
                            style={'fontSize': '0.82em', 'color': 'gray', 'marginBottom': '14px'}
                        ),
                        dcc.Loading(id='loading-progress', type='default',
                                    children=html.Div(id='image-analysis-result'))
                    ], xs=12, md=12, lg=4, className='mb-3 pv-col-step3'),
                ], className='g-4')

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




# -----------------------------------------------------------------------------
# Visual styling helpers for the analysis result
# -----------------------------------------------------------------------------

# Single accent color used for ALL diagnostic categories, charts, and panels
ACCENT_COLOR = '#0070C0'        # primary brand blue
MUTED_BAR_COLOR = '#BFD7EA'     # light blue for non-winning bars
TEXT_COLOR = '#222222'
SUBTLE_TEXT_COLOR = '#555555'

# Per-image-type label (no icons; same accent color used everywhere)
IMAGE_TYPE_LABEL = {
    'visible': 'Visible',
    'EL':      'Electroluminescence',
    'IR':      'Infrared',
    'other':   'Other',
}

# Friendly, human-readable label for each diagnostic category
CATEGORY_LABELS = {
    'clean':           'Clean',
    'snow':            'Snow coverage',
    'bird_droppings':  'Bird droppings',
    'dust_or_soiling': 'Dust / soiling',
    'hail_crack':      'Hail damage',
    'healthy':         'Healthy',
    'crack':           'Cell crack',
    'hotspot':         'Hotspot',
}

# Short labels used only on the bar-chart x-axis to avoid overlap.
# Use <br> to wrap to two lines so labels stay readable without angling.
CATEGORY_SHORT_LABELS = {
    'clean':           'Clean',
    'snow':            'Snow',
    'bird_droppings':  'Bird<br>droppings',
    'dust_or_soiling': 'Dust /<br>soiling',
    'hail_crack':      'Hail<br>damage',
    'healthy':         'Healthy',
    'crack':           'Crack',
    'hotspot':         'Hotspot',
}


def _confidence_descriptor(p):
    """Return a confidence label for a probability value."""
    if p >= 0.80:
        return 'High confidence'
    if p >= 0.55:
        return 'Moderate confidence'
    return 'Low confidence'


def build_result_panel(result):
    """Render a polished, card-style analysis panel from the LLM result."""
    pv_image_type = result.get('pv_image_type', 'other')
    prob_dict = result.get('probabilities', {}) or {}
    explanation = (result.get('explanation') or '').strip()

    if not prob_dict:
        return html.Div([
            dbc.Alert(
                explanation or 'The model did not return any diagnostic probabilities.',
                color='warning'
            )
        ])

    predicted_category = max(prob_dict, key=prob_dict.get)
    top_prob = float(prob_dict[predicted_category])
    conf_label = _confidence_descriptor(top_prob)

    type_label = IMAGE_TYPE_LABEL.get(pv_image_type, IMAGE_TYPE_LABEL['other'])
    cat_label = CATEGORY_LABELS.get(
        predicted_category, predicted_category.replace('_', ' ').title()
    )

    # Header strip: image type pill + predicted category pill (no icons)
    header = html.Div([
        html.Div([
            html.Span('Image type: ', style={'color': SUBTLE_TEXT_COLOR}),
            html.Span(type_label, style={'fontWeight': '700', 'color': ACCENT_COLOR}),
        ], style={
            'display': 'inline-block',
            'backgroundColor': '#F4F8FB',
            'padding': '4px 10px',
            'borderRadius': '999px',
            'border': f'1px solid {ACCENT_COLOR}33',
            'marginRight': '8px',
            'marginBottom': '6px',
            'fontSize': '0.85em',
        }),
        html.Div([
            html.Span('Predicted: ', style={'color': SUBTLE_TEXT_COLOR}),
            html.Span(cat_label, style={'fontWeight': '700', 'color': ACCENT_COLOR}),
        ], style={
            'display': 'inline-block',
            'backgroundColor': f'{ACCENT_COLOR}15',
            'padding': '4px 10px',
            'borderRadius': '999px',
            'border': f'1px solid {ACCENT_COLOR}66',
            'marginBottom': '6px',
            'fontSize': '0.85em',
        }),
    ], style={'marginBottom': '10px'})

    # Compact result card: predicted category + confidence on one row
    big_card = html.Div([
        html.Div([
            html.Span(cat_label, style={
                'fontSize': '1.05em',
                'fontWeight': '700',
                'color': ACCENT_COLOR,
                'marginRight': '10px',
            }),
            html.Span(f'{top_prob*100:.0f}%', style={
                'fontSize': '1.35em',
                'fontWeight': '800',
                'color': ACCENT_COLOR,
                'marginRight': '10px',
            }),
            html.Span(conf_label, style={
                'display': 'inline-block',
                'padding': '2px 8px',
                'borderRadius': '999px',
                'backgroundColor': f'{ACCENT_COLOR}1F',
                'color': ACCENT_COLOR,
                'fontWeight': '600',
                'fontSize': '0.78em',
                'verticalAlign': 'middle',
            }),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'}),
    ], style={
        'background': f'{ACCENT_COLOR}0A',
        'border': f'1px solid {ACCENT_COLOR}40',
        'borderLeft': f'4px solid {ACCENT_COLOR}',
        'borderRadius': '10px',
        'padding': '10px 14px',
        'marginBottom': '10px',
    })

    # Bar chart — single accent color for the winner, muted blue for others.
    # Slimmer height + constrained bar width + bold axis labels.
    categories = list(prob_dict.keys())
    probabilities = [float(prob_dict[c]) for c in categories]
    pretty_categories = [CATEGORY_SHORT_LABELS.get(c, c.replace('_', ' ').title()) for c in categories]
    bar_colors = [
        ACCENT_COLOR if c == predicted_category else MUTED_BAR_COLOR
        for c in categories
    ]

    # Cap bar width so 2-category charts (EL/IR) don't render comically wide.
    n_bars = len(categories)
    if n_bars <= 2:
        bargap = 0.7
    elif n_bars == 3:
        bargap = 0.55
    else:
        bargap = 0.45

    fig = go.Figure(go.Bar(
        x=pretty_categories,
        y=probabilities,
        marker=dict(
            color=bar_colors,
            line=dict(color='rgba(0,0,0,0.08)', width=1),
        ),
        text=[f'{p*100:.0f}%' for p in probabilities],
        textposition='outside',
        textfont=dict(size=11, color='#333'),
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.2f}<extra></extra>',
        cliponaxis=False,
    ))
    # Many short labels use <br> to wrap to two lines, so we always keep them
    # horizontal. Slightly more bottom room when there are 4+ categories.
    bottom_margin = 55 if len(categories) >= 4 else 40

    fig.update_layout(
        height=240,
        bargap=bargap,
        margin=dict(l=50, r=20, t=20, b=bottom_margin),
        autosize=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            range=[0, 1.08],
            title=dict(text='Probability', font=dict(size=12, color=TEXT_COLOR)),
            tickfont=dict(size=11, color=TEXT_COLOR),
            gridcolor='#EEE',
            zerolinecolor='#DDD',
        ),
        xaxis=dict(
            title=None,
            tickangle=0,
            tickfont=dict(size=11, color=TEXT_COLOR),
            automargin=True,
        ),
        showlegend=False,
    )

    chart_card = html.Div([
        html.Div('Probability across all categories', style={
            'fontWeight': '600', 'color': TEXT_COLOR,
            'marginBottom': '2px', 'fontSize': '0.85em',
        }),
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
    ], style={
        'border': '1px solid #E5E9EF',
        'borderRadius': '10px',
        'padding': '8px 12px',
        'marginBottom': '10px',
        'backgroundColor': 'white',
    })

    # Explanation panel — same accent color, no icon.
    if explanation:
        explanation_card = html.Div([
            html.Div('LLM explanation', style={
                'fontWeight': '700', 'color': ACCENT_COLOR,
                'marginBottom': '6px', 'fontSize': '0.9em',
            }),
            html.Div(explanation, style={
                'color': TEXT_COLOR,
                'lineHeight': '1.5',
                'fontSize': '0.9em',
                'whiteSpace': 'pre-wrap',
            }),
        ], style={
            'borderRadius': '10px',
            'padding': '10px 14px',
            'marginBottom': '10px',
            'backgroundColor': f'{ACCENT_COLOR}0A',
            'border': f'1px solid {ACCENT_COLOR}33',
            'borderLeft': f'4px solid {ACCENT_COLOR}',
        })
    else:
        explanation_card = html.Div()

    # Collapsible raw JSON
    raw_payload = {
        'pv_image_type': pv_image_type,
        'predicted_category': predicted_category,
        'probabilities': {k: round(float(v), 3) for k, v in prob_dict.items()},
    }
    raw_section = html.Details([
        html.Summary('Show raw model output', style={
            'cursor': 'pointer', 'color': SUBTLE_TEXT_COLOR,
            'fontSize': '0.85em', 'marginBottom': '4px',
        }),
        html.Pre(
            json.dumps(raw_payload, indent=2),
            style={
                'backgroundColor': '#F7F8FA',
                'border': '1px solid #E5E9EF',
                'borderRadius': '8px',
                'padding': '8px 12px',
                'color': '#444',
                'fontSize': '0.8em',
                'marginTop': '4px',
                'whiteSpace': 'pre-wrap',
            }
        ),
    ])

    return html.Div([header, big_card, chart_card, explanation_card, raw_section])


# Yellow accent for warning/error cards
WARN_COLOR = '#E1A82D'           # amber/yellow border accent
WARN_BG = '#FFFBEA'              # very light yellow background
WARN_TITLE_COLOR = '#B7791F'     # darker amber for the title text


def build_warning_card(title, summary, detail=None):
    """Yellow warning card with an optional collapsible <details> block.

    title:   short bold heading shown at the top.
    summary: short user-friendly message shown by default.
    detail:  optional long technical text revealed only when expanded.
    """
    children = [
        html.Div(title, style={
            'fontWeight': '700',
            'color': WARN_TITLE_COLOR,
            'marginBottom': '6px',
            'fontSize': '0.95em',
        }),
        html.Div(summary, style={
            'color': TEXT_COLOR,
            'lineHeight': '1.5',
            'fontSize': '0.9em',
            'whiteSpace': 'pre-wrap',
        }),
    ]
    if detail:
        children.append(html.Details([
            html.Summary('Show technical details', style={
                'cursor': 'pointer',
                'color': WARN_TITLE_COLOR,
                'fontSize': '0.85em',
                'marginTop': '8px',
                'fontWeight': '600',
            }),
            html.Pre(
                str(detail),
                style={
                    'backgroundColor': '#FFF8DD',
                    'border': f'1px solid {WARN_COLOR}66',
                    'borderRadius': '8px',
                    'padding': '8px 12px',
                    'color': '#5A4500',
                    'fontSize': '0.78em',
                    'marginTop': '6px',
                    'whiteSpace': 'pre-wrap',
                    'wordBreak': 'break-word',
                    'maxHeight': '300px',
                    'overflow': 'auto',
                }
            ),
        ], style={'marginTop': '4px'}))

    return html.Div(children, style={
        'backgroundColor': WARN_BG,
        'border': f'1px solid {WARN_COLOR}66',
        'borderLeft': f'4px solid {WARN_COLOR}',
        'borderRadius': '10px',
        'padding': '10px 14px',
        'marginTop': '10px',
    })


def analyze_image(base64_image, model="openai/gpt-5.1"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=1000,
    )

    res_text = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wrapped the JSON
    if res_text.startswith("```"):
        res_text = res_text.strip("`")
        # remove a leading "json" language tag if present
        if res_text.lower().startswith("json"):
            res_text = res_text[4:].strip()
    # In case there is leading/trailing prose, slice to the outermost braces
    if "{" in res_text and "}" in res_text:
        res_text = res_text[res_text.index("{"): res_text.rindex("}") + 1]

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
    State('image-content-store', 'data'),
    State('model-selector', 'value')]     # ✅ selected model
)
def unified_callback(upload_content, n_clicks, n1, n2, n3, n4, n5, n6, uploaded_image, image_displayed, stored_image, selected_model):

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
                    'width': '180px',
                    'height': '180px',
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
                    'width': '180px',
                    'height': '180px',
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
                result = analyze_image(base64_image, model=selected_model or "openai/gpt-5.1")

                if not result.get("pv_image", False):
                    not_pv_explanation = (result.get('explanation') or '').strip()
                    analysis_output = build_warning_card(
                        title='This does not appear to be a PV image.',
                        summary=(not_pv_explanation
                                 or 'Please upload a visible, EL, or IR image of a PV module/cell.'),
                    )
                else:
                    analysis_output = build_result_panel(result)
            except Exception as e:
                err_text = str(e)
                # Try to extract a short, user-friendly summary from the error
                short_msg = 'The selected model could not analyze this image.'
                lower = err_text.lower()
                if 'image/png' in lower and 'image/jpeg' in lower:
                    short_msg = 'The selected model is strict about image format. Try ChatGPT-5.1, or re-save your file as JPEG.'
                elif 'no fallback model' in lower or 'model_group' in lower:
                    short_msg = 'The selected model is not currently available on this endpoint. Try a different model.'
                elif 'rate limit' in lower or '429' in err_text:
                    short_msg = 'Rate limit reached. Please wait a moment and try again.'
                elif 'timeout' in lower:
                    short_msg = 'The request timed out. Please try again.'

                analysis_output = build_warning_card(
                    title='Analysis failed.',
                    summary=short_msg,
                    detail=err_text,
                )

    return status_msg, image_display, analysis_output, thumbnails, image_displayed, stored_image


if __name__ == '__main__':
    app.run_server(debug=True)