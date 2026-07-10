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

# === OLD (CBORG) — commented out ===
# cborg_API_KEY = os.getenv("cborg_api_key")
# client = openai.OpenAI(
#     api_key=cborg_API_KEY,
#     base_url="https://api.cborg.lbl.gov"
# )

# === NEW (OpenRouter) ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


def get_layout():
    return layout

# Map valid IDs to real image file names (in assets/example images/)
# Adjust the display names in image_name_map if a label doesn't match a file.
image_map = {
    'example1': 'example images/visible_1.jpg',
    'example2': 'example images/visible_2.jpg',
    'example3': 'example images/visible_3.png',
    'example4': 'example images/el_1.png',
    'example5': 'example images/el_2.jpg',
    'example6': 'example images/el_3.jpg',
    'example7': 'example images/ir_1.png',
    'example8': 'example images/ir_2.jpg',
    'example9': 'example images/ir_3.jpg',
}

image_name_map = {
    'example1': 'Visible image (snow)',
    'example2': 'Visible image (bird dropping)',
    'example3': 'Visible image (hail damage)',
    'example4': 'EL image (healthy)',
    'example5': 'EL image (crack)',
    'example6': 'EL image (cracks)',
    'example7': 'IR image (healthy)',
    'example8': 'IR image (hotspot)',
    'example9': 'IR image (hotspots)',
}

# Example images arranged as three clusters scattered around the upload zone
EXAMPLE_CLUSTERS = [
    ('Example visible images', 'pv-cluster-visible',
     ['example1', 'example2', 'example3']),
    ('Example EL images', 'pv-cluster-el',
     ['example4', 'example5', 'example6']),
    ('Example IR images', 'pv-cluster-ir',
     ['example7', 'example8', 'example9']),
]


def render_example_thumbnails(selected_id=None):
    """Three floating image clusters positioned around the central upload zone.

    Each cluster = a light label + two slightly overlapping square images,
    each bobbing on its own phase (`pv-float-N`, see pv_image.css).
    Click IDs stay on the <img> elements, so the callback is unchanged.
    """
    clusters = []
    float_idx = 0
    for label, cluster_cls, ids in EXAMPLE_CLUSTERS:
        imgs = []
        for example_id in ids:
            float_idx += 1
            selected = (example_id == selected_id)
            imgs.append(html.Div(
                html.Img(
                    src=f"/assets/{image_map[example_id]}",
                    id=example_id,
                    n_clicks=0,
                    title=image_name_map[example_id],  # tooltip on hover
                    className=('pv-thumb pv-thumb-selected'
                               if selected else 'pv-thumb'),
                ),
                className=f'pv-float pv-float-{float_idx}',
            ))
        clusters.append(html.Div([
            html.Div(label, className='pv-cluster-label'),
            html.Div(imgs, className='pv-stack'),
        ], className=f'pv-cluster {cluster_cls}'))
    return clusters

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

    # ==================== FULL-BLEED HERO SECTION ====================
    html.Div(html.Div([

        # ---------------- Header ----------------
        html.Div([
            html.H1('Unified LLM-Based PV Image Diagnostic Framework',
                    className='pv-hero-title'),
            html.P('Fault detection from visible, EL & IR module imagery — '
                   'benchmarked across multimodal language models.',
                   className='pv-hero-sub'),
            html.Details([
                html.Summary('About this demo'),
                html.Div([
                    html.Div([
                        html.P([
                            'Upload a ', html.Strong('visible'), ', ',
                            html.Strong('EL'), ', or ', html.Strong('IR'),
                            ' image of a PV module — a multimodal LLM identifies the '
                            'image type, classifies its condition, and explains the call.',
                        ], className='pv-about-lead'),
                        html.Div([
                            html.Div([
                                html.Span('Visible', className='pv-mod-name'),
                                html.Span('clean · soiling · hail · snow · bird droppings',
                                          className='pv-mod-cats'),
                            ], className='pv-mod'),
                            html.Div([
                                html.Span('EL', className='pv-mod-name'),
                                html.Span('healthy · cell crack', className='pv-mod-cats'),
                            ], className='pv-mod'),
                            html.Div([
                                html.Span('IR', className='pv-mod-name'),
                                html.Span('healthy · hotspot', className='pv-mod-cats'),
                            ], className='pv-mod'),
                        ], className='pv-mods'),
                        html.P([
                            'The result table below reports F1 scores of each model on a '
                            'curated test set — methodology and full results in ',
                            html.A('our paper in Solar Energy',
                                   href='https://www.sciencedirect.com/science/article/pii/S0038092X26004895',
                                   target='_blank'),
                            '. This tool is under active development — ',
                            html.A('contact us', href='mailto:baojieli@lbl.gov'),
                            ' with feedback or to collaborate.',
                        ], className='pv-about-foot'),
                    ], className='pv-about-text'),
                    html.Img(src=app.get_asset_url('llm_logo.jpg'),
                             className='pv-about-logo'),
                ], className='pv-about-body'),
            ], className='pv-about'),
        ], className='pv-header'),

        # ---------------- Workspace: stage (left) + steps 2-3 (right) ----------------
        dbc.Row([

            # ======== LEFT: the constellation stage ========
            dbc.Col([
                html.Div('Step 1 · Choose an image', className='pv-step-label'),
                html.Div([
                    # Floating image clusters (rebuilt by the callback)
                    html.Div(
                        children=render_example_thumbnails(),
                        id='example-image-container',
                        className='pv-cluster-layer',
                    ),
                    # Central upload square
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            html.Div('↑', className='pv-upload-icon'),
                            html.Div('Upload an image', className='pv-upload-title'),
                            html.Div(['Drop a file here, or tap', html.Br(),
                                      'a sample around it'],
                                     className='pv-upload-text'),
                        ]),
                        className='pv-upload-zone',
                        accept='image/*',
                        multiple=False,
                    ),
                    # Selected / uploaded preview fills the central square
                    html.Div(id='output-image-upload',
                             className='pv-center-preview-slot'),
                    # Clear (✕) button — shown only when an image is loaded
                    html.Button('✕', id='clear-image-btn', n_clicks=0,
                                className='pv-clear-btn',
                                style={'display': 'none'}),
                ], className='pv-stage'),
                html.Div(id='upload-status', className='pv-status'),
                dcc.Store(id='image-display-flag', data=False),
                dcc.Store(id='image-content-store'),
            ], xs=12, lg=7, className='mb-4'),

            # ======== RIGHT: model + run + results ========
            dbc.Col([
                html.Div('Step 2 · Model', className='pv-step-label'),
                dcc.RadioItems(
                    id='model-selector',
                    options=[
                        {'label': 'ChatGPT-5.1', 'value': 'openai/gpt-5.1'},
                        {'label': 'Gemini 2.5 Flash', 'value': 'google/gemini-2.5-flash'},
                        {'label': 'Claude Haiku 4.5', 'value': 'anthropic/claude-haiku-4.5'},
                    ],
                    value='openai/gpt-5.1',
                    className='pv-model-radio',
                    inputStyle={'marginRight': '12px', 'marginTop': '4px'},
                    labelStyle={'cursor': 'pointer', 'fontSize': '0.95em'},
                ),
                html.Button('Run the analysis', id='analyze-button', n_clicks=0,
                            className='pv-run-button'),
                html.Div('Takes about 3–8 seconds', className='pv-hint',
                         style={'textAlign': 'center'}),

                html.Div('Step 3 · Results', className='pv-step-label pv-step-results'),
                dcc.Loading(id='loading-progress', type='default',
                            children=html.Div(id='image-analysis-result')),
            ], xs=12, lg={'size': 4, 'offset': 1}, className='mb-4'),

        ], className='g-5'),

    ], className='pv-inner'), className='pv-page'),

    # ==================== FULL-BLEED BOTTOM BAND ====================
    html.Div(html.Div(
        dbc.Row([

            dbc.Col(html.Div([
                html.Div('Image dataset', className='pv-card-title'),
                html.Div('Curated visible / EL / IR PV images',
                         className='pv-card-sub'),
                html.A(
                    html.Img(src='/assets/images.png', className='pv-dataset-img'),
                    href='https://github.com/DuraMAT/PV-LLM',
                    target='_blank',
                ),
                html.P([
                    'Example images from the test dataset. Learn more at ',
                    html.A('DuraMAT/PV-LLM',
                           href='https://github.com/DuraMAT/PV-LLM',
                           target='_blank'),
                ], className='pv-hint mt-2 mb-0'),
            ], className='pv-card'), xs=12, lg=5, className='mb-4 mb-lg-0'),

            dbc.Col(html.Div([
                html.Div([
                    html.Span('Result table · LLM performance',
                              className='pv-card-title'),
                    html.Span('Updated 2026/2/6', className='pv-card-date'),
                ], className='pv-card-titlerow'),
                html.Div('F1 scores · binary and multi-class tasks',
                         className='pv-card-sub'),
                html.Div([
                    html.Table(
                        table_header + [table_body],
                        className='table pv-table',
                        style={'fontSize': '0.88em', 'minWidth': '640px'},
                    ),
                ], style={'overflowX': 'auto'}),
                html.P([
                    'Reference: ',
                    html.A('our paper in Solar Energy',
                           href='https://www.sciencedirect.com/science/article/pii/S0038092X26004895',
                           target='_blank'),
                ], className='pv-hint mb-0'),
            ], className='pv-card'), xs=12, lg=7, className='mb-4 mb-lg-0'),

        ], className='g-4'),
        className='pv-inner'), className='pv-band'),

], fluid=True, className='px-0')


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
    """Compact, refined result panel: headline + probability bars + why."""
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

    # --- Headline: prediction + confidence, with the image type as meta ---
    headline = html.Div([
        html.Div(f'{type_label} image', className='pv-res-meta'),
        html.Div([
            html.Span(cat_label, className='pv-res-cat'),
            html.Span(f'{top_prob * 100:.0f}%', className='pv-res-pct'),
            html.Span(conf_label, className='pv-res-chip'),
        ], className='pv-res-headrow'),
    ], className='pv-res-head')

    # --- Slim horizontal probability bars, sorted high to low ---
    bar_rows = []
    for cat, p in sorted(prob_dict.items(), key=lambda kv: -float(kv[1])):
        p = float(p)
        label = CATEGORY_LABELS.get(cat, cat.replace('_', ' ').title())
        winner = (cat == predicted_category)
        bar_rows.append(html.Div([
            html.Div(label, className='pv-bar-label'),
            html.Div(
                html.Div(
                    className='pv-bar-fill' + (' pv-bar-win' if winner else ''),
                    style={'width': f'{max(p * 100, 2):.0f}%'},
                ),
                className='pv-bar-track',
            ),
            html.Div(f'{p * 100:.0f}%', className='pv-bar-val'),
        ], className='pv-bar-row'))
    bars_block = html.Div(bar_rows, className='pv-res-bars')

    # --- Short "why" block ---
    if explanation:
        why_block = html.Div([
            html.Div('Why', className='pv-res-why-label'),
            html.Div(explanation, className='pv-res-why-text'),
        ], className='pv-res-why')
    else:
        why_block = html.Div()

    # --- Collapsible raw model output ---
    raw_payload = {
        'pv_image_type': pv_image_type,
        'predicted_category': predicted_category,
        'probabilities': {k: round(float(v), 3) for k, v in prob_dict.items()},
    }
    raw_section = html.Details([
        html.Summary('Show raw model output', className='pv-raw-summary'),
        html.Pre(json.dumps(raw_payload, indent=2), className='pv-raw-pre'),
    ], className='pv-raw')

    return html.Div([headline, bars_block, why_block, raw_section],
                    className='pv-result')


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
     Output('image-content-store', 'data'),
     Output('clear-image-btn', 'style')],      # ✅ show/hide the ✕ button
    [Input('upload-image', 'contents'),
     Input('analyze-button', 'n_clicks'),
     Input('clear-image-btn', 'n_clicks'),     # ✅ clear the selection
     Input('example1', 'n_clicks'),
     Input('example2', 'n_clicks'),
     Input('example3', 'n_clicks'),
     Input('example4', 'n_clicks'),
     Input('example5', 'n_clicks'),
     Input('example6', 'n_clicks'),
     Input('example7', 'n_clicks'),
     Input('example8', 'n_clicks'),
     Input('example9', 'n_clicks')],
    [State('upload-image', 'contents'),
    State('image-display-flag', 'data'),
    State('image-content-store', 'data'),
    State('model-selector', 'value')]     # ✅ selected model
)
def unified_callback(upload_content, n_clicks, clear_clicks, n1, n2, n3, n4, n5, n6, n7, n8, n9, uploaded_image, image_displayed, stored_image, selected_model):

    trigger_id = ctx.triggered_id

    status_msg = dash.no_update
    image_display = dash.no_update
    clear_btn_style = dash.no_update
    analysis_output = dash.no_update
    thumbnails = render_example_thumbnails()

    # Case 1: new image uploaded
    if trigger_id == 'upload-image' and upload_content:
        status_msg = html.Span('Your image is loaded — press "Run the analysis"',
                               className='pv-status-msg')
        image_display = html.Div([
            html.Img(src=upload_content, className='pv-center-preview',
                     title='Your uploaded image'),
            html.Div([
                html.Div('Your image', className='pv-preview-name'),
                html.Div('ready to analyze', className='pv-preview-sub'),
            ], className='pv-preview-cap'),
        ], className='pv-preview-wrap')
        clear_btn_style = {'display': 'flex'}
        image_displayed = True  # ✅ set image display flag
        stored_image = upload_content  # ✅ store uploaded image content
        analysis_output = ''  # ✅ clear results
        thumbnails = render_example_thumbnails()  # reset highlight

    # Case 2: example image clicked
    elif trigger_id in image_map:
        img_path = os.path.join('assets', image_map[trigger_id])
        encoded_image = encode_image_as_upload_format(img_path)
        app._example_base64 = encoded_image  # simulate upload

        status_msg = html.Span(f'Selected: {image_name_map[trigger_id]} — press "Run the analysis"',
                               className='pv-status-msg')
        image_display = html.Div([
            html.Img(src=encoded_image, className='pv-center-preview',
                     title=image_name_map[trigger_id]),
            html.Div([
                html.Div(image_name_map[trigger_id], className='pv-preview-name'),
                html.Div('ready to analyze', className='pv-preview-sub'),
            ], className='pv-preview-cap'),
        ], className='pv-preview-wrap')
        clear_btn_style = {'display': 'flex'}
        image_displayed = True  # ✅ set image display flag
        stored_image = encoded_image  # ✅ store clicked image content
        analysis_output = ''  # ✅ clear results
        thumbnails = render_example_thumbnails(selected_id=trigger_id)

    # Case 3: ✕ clear button clicked — reset the stage
    elif trigger_id == 'clear-image-btn':
        status_msg = ''
        image_display = ''
        analysis_output = ''
        thumbnails = render_example_thumbnails()
        image_displayed = False
        stored_image = None
        clear_btn_style = {'display': 'none'}

    # Case 4: Analyze clicked
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

    return status_msg, image_display, analysis_output, thumbnails, image_displayed, stored_image, clear_btn_style


if __name__ == '__main__':
    app.run_server(debug=True)