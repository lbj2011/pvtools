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
from geopy.distance import distance
from collections import Counter
import math
from utils.data_loader import safe_get_df
import ast

df_raw = safe_get_df()

# df_raw = pd.read_pickle('data_250318.pkl')
# df_raw = pd.read_pickle('data_250924.pkl')

df = df_raw[(df_raw['duration'] < 100) & (df_raw['rate'] <= 3)]

types = ['mono-c-Si', 'multi-c-Si', 'a-Si', 'CIGS', 'CdTe', 'HIT','other']
filter_text_style = {
    'fontFamily': 'Arial',
    'fontSize': '17px',  # Default size, can be overridden per element
    'color': 'black'
    }

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
        html.H1("Worldwide PV Field Performance (Demo)"),
    ], style={
        # 'background-color': 'lightblue',
        'width': '100%',
        'padding-left': '10px',
        'padding-right': '10px',
        'textAlign': 'center'}),
    html.Hr(),

    html.H2('Overview'),
    html.P(''),
    dbc.Row([
        dbc.Col([
            dcc.Markdown("""This tool visualizes **worldwide PV field degradation** extracted from scientific literature using **Large Language Models (LLMs)**.

                    - Source papers (~3,900 papers) were retrieved from **Scopus** using keyword-based searches.  
                    - **Gpt and Gemini** were applied to automatically extract degradation information.  

                    **Note:** Degradation rate is reported as a **negative value when power decreases**.   

                    📂 **Resources:**  
                    - [GitHub Repository](https://github.com/DuraMAT/PV-LLM)  
                    - [Download Raw Data (DuraMAT Datahub)](https://datahub.duramat.org/project/mapping-pv-degradation-by-llm)  
                         
                """.replace('    ', '')
                 ),
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
        color="primary",
        className="mt-2"
    ),
    
    html.P(''),
    html.H2('Degradation rate map'),
    html.P(''),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
            dbc.Col([

                html.Div([
                    html.Label("• Filter by PV Technologies:", style={**filter_text_style, 'marginRight': '10px'}),
                    dcc.Checklist(
                        id='pv-tech-filter',
                        options=[{'label': t, 'value': t} for t in types],
                        value=types,
                        inline=True,
                        style={**filter_text_style},
                        inputStyle={"margin-right": "6px"},   # space between checkbox and text
                        labelStyle={"margin-right": "10px"}   # space between options
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px', 'marginLeft': '10px'}),

                html.Div([
                    html.Label("• Filter by PV Climate Zone:", style={**filter_text_style, 'marginRight': '10px'}),
                    dcc.Checklist(
                        id='pv-climate-filter',
                        options=[
                            {'label': 'Moderate', 'value': 'Moderate'},
                            {'label': 'Desert', 'value': 'Desert'},
                            {'label': 'Hot & Humid', 'value': 'Hot & Humid'},
                            {'label': 'Snow', 'value': 'Snow'}
                        ],
                        value=['Moderate', 'Desert', 'Hot & Humid', 'Snow'],  # all selected by default
                        inline=True,
                        style={**filter_text_style},
                        inputStyle={"margin-right": "6px"},
                        labelStyle={"margin-right": "10px"}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px', 'marginLeft': '10px'}),

                # Scope of study filter
                html.Div([
                    html.Label("• Filter by Scope of Study:",
                            style={**filter_text_style, 'marginRight': '10px'}),
                    dcc.Checklist(
                        id='scope-filter',
                        options=[
                            {'label': 'Module level', 'value': 'module level'},
                            {'label': 'System level', 'value': 'system level'}
                        ],
                        value=['module level', 'system level'],
                        inline=True,
                        style={**filter_text_style},
                        inputStyle={"margin-right": "6px"},
                        labelStyle={"margin-right": "10px"}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px', 'marginLeft': '10px'}),

                

                html.Details(
                    [
                        html.Summary(
                            "Advanced filters",
                            style={
                                **filter_text_style,
                                'cursor': 'pointer',
                                'fontWeight': '600',
                                'marginLeft': '10px'
                            }
                        ),

                        html.Div([
                            html.Label("• Filter by Rate Range (%/year):", style={**filter_text_style, 'marginRight': '10px'}),
                            html.Label("Min:", style={**filter_text_style, 'marginRight': '5px'}),
                            dcc.Input(id='rate-min', type='number', placeholder='Min Rate', value=-20, style={**filter_text_style, 'marginRight': '10px', 'width': '80px'}),
                            html.Label("Max:", style={**filter_text_style, 'marginRight': '5px'}),
                            dcc.Input(id='rate-max', type='number', placeholder='Max Rate', value=5, style={**filter_text_style, 'width': '80px'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px', 'marginLeft': '20px'}),
                        
                        html.Div([
                            html.Label("• Filter by Duration Range (year):", style={**filter_text_style, 'marginRight': '10px'}),
                            html.Label("Min:", style={**filter_text_style, 'marginRight': '5px'}),
                            dcc.Input(id='duration-min', type='number', placeholder='Min Duration', value=0, style={**filter_text_style, 'marginRight': '10px', 'width': '80px'}),
                            html.Label("Max:", style={**filter_text_style, 'marginRight': '5px'}),
                            dcc.Input(id='duration-max', type='number', placeholder='Max Duration', value=50, style={**filter_text_style, 'width': '80px'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px', 'marginLeft': '20px'}),
                        
                        html.Div([
                            html.Label(
                                "• Filter by System capacity (kW):",
                                style={**filter_text_style, 'marginRight': '8px'}
                            ),

                            # Reported / Not reported
                            dcc.Checklist(
                                id='capacity-report-filter',
                                options=[
                                    {'label': 'Not reported', 'value': 'not_reported'},
                                    {'label': 'Reported', 'value': 'reported'},
                                ],
                                value=['reported', 'not_reported'],   # default: both
                                inline=True,
                                style={**filter_text_style},
                                inputStyle={'marginRight': '3px'},
                                labelStyle={'marginRight': '8px'}
                            ),

                            # Min / Max inputs (kW)
                            html.Label("(", style={**filter_text_style, 'margin': '0 4px 0 10px'}),

                            html.Label("Min", style={**filter_text_style, 'marginRight': '4px'}),
                            dcc.Input(
                                id='capacity-min',
                                type='number',
                                value=0,
                                min=0,
                                style={**filter_text_style, 'width': '70px', 'marginRight': '6px'}
                            ),

                            html.Label("Max", style={**filter_text_style, 'marginRight': '4px'}),
                            dcc.Input(
                                id='capacity-max',
                                type='number',
                                value=500,
                                min=0,
                                style={**filter_text_style, 'width': '70px'}
                            ),
                            html.Label(")", style={**filter_text_style}),
                        ],
                        style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'flexWrap': 'wrap',
                            'marginLeft': '20px',
                            'marginBottom': '8px'
                        }),

                        html.Div(
                            [
                                html.Label(
                                    "• Filter by Faults:",
                                    style={**filter_text_style, 'marginRight': '6px'}
                                ),
                                dcc.Checklist(
                                    id='faults-filter',
                                    options=[
                                        {'label': 'Reported', 'value': 'reported'},
                                        {'label': 'Not reported', 'value': 'not_reported'}
                                    ],
                                    value=['reported', 'not_reported'],
                                    inline=True,
                                    style={**filter_text_style},
                                    inputStyle={'marginRight': '3px'},
                                    labelStyle={'marginRight': '8px'}
                                )
                            ],
                            style={
                                'display': 'flex',
                                'alignItems': 'center',
                                'marginLeft': '20px',
                                'marginBottom': '8px'
                            }
                        )
                    ]
                ),

                html.Div([
                    dcc.Graph(id='map')
                ]#, style={'width': '70%', 'display': 'inline-block', 'marginLeft': '0px'}
                ),
            ], xs=12, sm=12, md=12, lg=8, xl=8),
        
            dbc.Col([
                html.Div([
                    html.H4("Details", style={'fontFamily': 'Arial'}),
                    html.P('(Select a data point to show)'),
                    dash_table.DataTable(
                        id='table',
                        columns=[
                            {'name': 'Attribute', 'id': 'attribute'},
                            {'name': 'Value', 'id': 'value', 'presentation': 'markdown'}
                        ],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'fontFamily': 'Arial', 'verticalAlign': 'top' },
                        style_as_list_view=True,
                    )
                ]
                ),
            ], xs=12, sm=12, md=12, lg=4, xl=4),
    ], className="g-2")  # Adds spacing between columns
    ]),
    ]),

    html.P(''),
    html.H2('Analysis'),
    html.P('(based on filtered data points)'),
    html.P(''),
    dbc.Card([
        dbc.Row([
            dbc.Col([
                html.P(''),
                html.H4("Degradation Rate Distribution", style={'fontFamily': 'Arial'}),
                html.Div(id="data_info"),
                dcc.Graph(id='histogram', style={'width': '100%'})
            ], xs=12, sm=12, md=6, lg=6, xl=6, className="h-100"),  # Adjust width for responsiveness
            dbc.Col([
                html.P(''),
                html.H4("Degradation Rate vs Exposure Length", style={'fontFamily': 'Arial'}),
                dcc.Graph(id='rate-duration-scatter', style={'width': '100%'})
            ], xs=12, sm=12, md=6, lg=6, xl=6, className="h-100"),
        ], className="g-2", style={'marginLeft': '20px'})
    ]),

    html.P(''),
    html.H2('Regional performance'),
    html.P(''),
    dbc.Card([
        dbc.Row([
            dbc.Col([
                html.P(''),
                html.H4("Select your location:", style={'fontFamily': 'Arial'}),

                # Latitude & Longitude side by side
                html.Div([
                    html.Label("Latitude:", style={'fontFamily': 'Arial', 'marginRight': '6px'}),
                    dcc.Input(
                        id='lat-input',
                        type='number',
                        value=39.7392,  # Denver latitude
                        style={'width': '120px', 'marginRight': '20px'}
                    ),

                    html.Label("Longitude:", style={'fontFamily': 'Arial', 'marginRight': '6px'}),
                    dcc.Input(
                        id='lon-input',
                        type='number',
                        value=-104.9903,  # Denver longitude
                        style={'width': '120px'}
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),

                # Radius input on its own line
                html.Div([
                    html.Label("Radius (miles):", style={'fontFamily': 'Arial', 'marginRight': '6px'}),
                    dcc.Input(
                        id='radius-input',
                        type='number',
                        value=100,  # ✅ default radius
                        style={'width': '120px'}
                    ),
                ], style={'marginBottom': '15px'}),

                # Map figure
                dcc.Graph(
                    id='location-map',
                    figure={},
                    config={'scrollZoom': True},
                    # style={'height': '400px'}
                )
            ], xs=12, sm=12, md=4, lg=4, xl=4, className="h-100"),
                
            dbc.Col([
                html.P(''),
                html.H4("Performance within this radius:", style={'fontFamily': 'Arial'}),
                html.Div(id='pie-charts-container')  # placeholder for callback output
            ], xs=12, sm=12, md=8, lg=8, xl=8, className="h-100", style={'marginLeft': '0px'}),
        ], className="g-2", style={'marginLeft': '20px'})
    ]),

    html.P(''),
    dcc.Markdown("""Dataset version: 2025/9/24
                 
                """.replace('    ', ''), style={'color': 'lightgray'}
    ),
    # Contributor: Baojie Li
    # Project members: Baojie Li, Martin Springer, Dirk Jordan, Anubhav Jain
    
])

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
        Input('duration-max', 'value')
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
    duration_max
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

    data_info = f"(Selected data points: {len(filtered_df)}, Mean rate: {mean_rate:.2f}%/y, Median rate: {median_rate:.2f}%/y)"

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

    # max_count = filtered_df['rate'].value_counts(bins=50).max() 
  
    hist_fig.add_trace(go.Scatter(x=[mean_rate, mean_rate], y=[0, max_count], mode='lines', line=dict(color='#009546', dash='dash'), name='Mean'))
    hist_fig.add_trace(go.Scatter(x=[median_rate, median_rate], y=[0, max_count], mode='lines', line=dict(color='#002877', dash='dash'), name='Median'))
    
    hist_fig.update_layout(
        xaxis_title="Degradation Rate (%/year)",
        yaxis_title="Number of cases",
        legend=dict(
            title="Legend",
            orientation="v",  # Vertical orientation
            yanchor="top",     # Anchor to the *top* of the legend
            y=-0.2,          # Place the top of the legend slightly above the plot
            xanchor="left",    # Anchor to the *left* of the legend
            x=0.01         # Place the left of the legend slightly to the right of the plot
        ),
        margin=dict(l=2, t=20) 
    )

    duration_fig = px.scatter(filtered_df, x = 'duration', y = 'rate', 
                              marginal_x="histogram", marginal_y="histogram",
                              hover_data={"rate": True, "publish year": True, 
                                            "country": True, "paper id": True,
                                            "duration": True, "confidence level": True})
    # Set scatter marker color and opacity
    duration_fig.update_traces(marker=dict(color="#0070C0"), opacity=0.5)

    # Marginal histograms (x and y)
    duration_fig.update_traces(
        marker=dict(color="#00B050"),  # orange
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
        return html.P("No data points found within the selected radius.", style={'fontStyle': 'italic'})

    charts = []

    charts_row1 = []   # for PV tech + PV zone
    chart_row2 = []  # for faults

    # --- Pie chart 1: PV tech ---
    if not df_inside['pv tech'].dropna().empty:
        fig1 = px.pie(df_inside, names='pv tech', title="PV Technologies", color_discrete_sequence=px.colors.sequential.GnBu_r)
        fig1.update_layout(margin=dict(l=20, r=20, t=60, b=10))
        charts_row1.append(dcc.Graph(figure=fig1, style={'height': '250px'}))

    # --- Pie chart 2: PV zone ---
    if not df_inside['PV zone'].dropna().empty:
        fig2 = px.pie(df_inside, names='PV zone', title="PV Climate Zones", color_discrete_sequence=px.colors.sequential.GnBu_r)
        fig2.update_layout(margin=dict(l=20, r=20, t=60, b=10))
        charts_row1.append(dcc.Graph(figure=fig2, style={'height': '250px'}))

    # --- Pie chart 3: Faults list ---
    if not df_inside['faults_list'].dropna().empty:
        # Flatten list of faults and count
        fault_counts = Counter(
            fault
            for sublist in df_inside['faults_list'].dropna()
            for fault in sublist
            if fault and fault.strip().lower() not in ['not reported', 'na', 'n/a', 'none']
        )
        # Keep top 8 faults
        top_faults = fault_counts.most_common(8)
        fault_df = pd.DataFrame(top_faults, columns=['Fault', 'Count'])
        fig3 = px.pie(fault_df, names='Fault', values='Count', title="Faults", color_discrete_sequence=px.colors.sequential.GnBu_r)
        fig3.update_layout(margin=dict(l=20, r=20, t=60, b=10))
        chart_row2.append(dcc.Graph(figure=fig3, style={'height': '250px'}))

    # --- Box plot: rate per PV tech ---
    if not df_inside['pv tech'].dropna().empty:
        fig4 = px.box(
            df_inside,
            x='pv tech',
            y='rate',
            title="Rate Distribution by PV Tech",
            color='pv tech',  # optional: color by tech
            color_discrete_sequence=px.colors.sequential.GnBu_r,
            labels={
                "pv tech": "PV technology",
                "rate": "Rd (%/year)"
            }
        )
        fig4.update_layout(margin=dict(l=20, r=20, t=60, b=10), showlegend=False)
        chart_row2.append(dcc.Graph(figure=fig4, style={'height': '250px'}))

    # --- If nothing valid ---
    if not charts_row1 and chart_row2 is None:
        return html.P("No valid values found for PV tech, PV zone, or faults.", style={'fontStyle': 'italic'})

    # --- Return charts ---
    children = []
    if charts_row1:
        children.append(dbc.Row([dbc.Col(c, width=6) for c in charts_row1]))
    if charts_row1:
        children.append(dbc.Row([dbc.Col(c, width=6) for c in chart_row2]))
        # children.append(dbc.Row([dbc.Col(c, width=6) for c in charts_row1]))

    return html.Div(children)

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
