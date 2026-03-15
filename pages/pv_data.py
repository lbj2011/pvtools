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

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

df_raw = pd.read_parquet("data_250924.parquet")
# df_raw = pd.read_pickle('sum_df.pkl')
df = df_raw[df_raw['duration']<100]

types = ['mono-c-Si', 'poly-c-Si', 'a-Si', 'other']
filter_text_style = {
    'fontFamily': 'Arial',
    'fontSize': '17px',  # Default size, can be overridden per element
    'color': 'black'
    }

def get_layout():
    return layout

# Layout
layout = dbc.Container([

    html.Hr(),
    html.Div([
        html.H1("Worldwide PV Field Performance Database (PV-Data)"),
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
            dcc.Markdown("""This tool shows the worldwide PV field degradation extracted from literature using Large Language Models (LLMs).
                         
                * Papers are selected from **Scopus** using keyword search
                * **Gemini** is used to extract the degradation information
                
                Note that the degradation rate is presented as a **negative value if the power decreases**.
                
                """.replace('    ', '')
                 ),
        ], xs=12, sm=12, md=12, lg=9, xl=9),

        dbc.Col([
            html.Img(src=app.get_asset_url('pv_data_logo.png'),
            style={'width': '50%'}),
        ], xs=9, sm=8, md=6, lg=3, xl=3),
    ]),
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
                            style={**filter_text_style}
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px', 'marginLeft': '10px'}),
                
                html.Div([
                    html.Label("• Filter by Rate Range (%/year):", style={**filter_text_style, 'marginRight': '10px'}),
                    html.Label("Min:", style={**filter_text_style, 'marginRight': '5px'}),
                    dcc.Input(id='rate-min', type='number', placeholder='Min Rate', value=-30, style={**filter_text_style, 'marginRight': '10px', 'width': '80px'}),
                    html.Label("Max:", style={**filter_text_style, 'marginRight': '5px'}),
                    dcc.Input(id='rate-max', type='number', placeholder='Max Rate', value=30, style={**filter_text_style, 'width': '80px'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px', 'marginLeft': '10px'}),
                
                html.Div([
                    html.Label("• Filter by Duration Range (year):", style={**filter_text_style, 'marginRight': '10px'}),
                    html.Label("Min:", style={**filter_text_style, 'marginRight': '5px'}),
                    dcc.Input(id='duration-min', type='number', placeholder='Min Duration', value=0, style={**filter_text_style, 'marginRight': '10px', 'width': '80px'}),
                    html.Label("Max:", style={**filter_text_style, 'marginRight': '5px'}),
                    dcc.Input(id='duration-max', type='number', placeholder='Max Duration', value=50, style={**filter_text_style, 'width': '80px'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px', 'marginLeft': '10px'}),

                html.Div([
                    dcc.Graph(id='map2')
                ]#, style={'width': '70%', 'display': 'inline-block', 'marginLeft': '0px'}
                ),
            ], xs=12, sm=12, md=12, lg=8, xl=8),
        
            dbc.Col([
                html.Div([
                    html.H4("Details", style={'fontFamily': 'Arial'}),
                    html.P('(Select a data point to show)'),
                    dash_table.DataTable(
                        id='table2',
                        columns=[
                            {'name': 'Attribute', 'id': 'attribute'},
                            {'name': 'Value', 'id': 'value', 'presentation': 'markdown'}
                        ],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'fontFamily': 'Arial'},
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
                html.Div(id="data_info2"),
                dcc.Graph(id='histogram2', style={'width': '100%'})
            ], xs=12, sm=12, md=6, lg=6, xl=6),  # Adjust width for responsiveness
            dbc.Col([
                html.P(''),
                html.H4("Degradation Rate vs Exposure Length", style={'fontFamily': 'Arial'}),
                dcc.Graph(id='rate-duration-scatter2', style={'width': '100%'})
            ], xs=12, sm=12, md=6, lg=6, xl=6),
        ], className="g-2", style={'marginLeft': '20px'})
    ]),
    html.P(''),
    dcc.Markdown("""Dataset version: 2025/3/18
                 
                 Contributor: Baojie Li
                 
                 Project members: Baojie Li, Martin Springer, Dirk Jordan, Anubhav Jain
                 
                """.replace('    ', ''), style={'color': 'lightgray'}
    ),
    # 
    
])

# Callback to update map and histogram
@app.callback(
    [Output('map2', 'figure'), Output('histogram2', 'figure'), Output("data_info2", "children"), Output('rate-duration-scatter2', 'figure')],
    [Input('pv-tech-filter', 'value'),
     Input('rate-min', 'value'), Input('rate-max', 'value'),
     Input('duration-min', 'value'), Input('duration-max', 'value')]
)
def update_map_and_histogram(selected_types, rate_min, rate_max, duration_min, duration_max):
    filtered_df = df[df['pv tech'].isin(selected_types)]
    
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
        size='rate_abs',
        size_max=30,
        zoom=3,
        hover_data={"rate": True, "publish year": True, 
                    "country": True, "paper id": True,
                    "duration": True, "confidence level": True}
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

    data_info = f"(Mean rate: {mean_rate:.2f}%/y, Median rate: {median_rate:.2f}%/y)"

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

    max_count = filtered_df['rate'].value_counts(bins=50).max() 
  
    hist_fig.add_trace(go.Scatter(x=[mean_rate, mean_rate], y=[0, max_count], mode='lines', line=dict(color='red', dash='dash'), name='Mean'))
    hist_fig.add_trace(go.Scatter(x=[median_rate, median_rate], y=[0, max_count], mode='lines', line=dict(color='blue', dash='dash'), name='Median'))
    
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

    duration_fig.update_layout(
        xaxis_title='Exposure length (year)',
        yaxis_title='Degradation Rate (%/year)'
    )
    return fig, hist_fig, data_info, duration_fig

# Callback to update table
@app.callback(
    Output('table2', 'data'),
    Input('map2', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return []
    
    point = clickData['points'][0]
    selected_data = df[(df['longitude'] == point['lon']) & (df['latitude'] == point['lat'])].iloc[0]

    doi = selected_data['doi']
    doi_link = f"[{doi}](https://doi.org/{doi})"  # Markdown link
    
    return [
        {'attribute': 'DOI', 'value': doi_link},
        {'attribute': 'Year', 'value': selected_data['publish year']},
        {'attribute': 'Title', 'value': selected_data['title']},
        {'attribute': 'Type', 'value': selected_data['document type']},
        {'attribute': 'Paper ID', 'value': selected_data['paper id']},
        {'attribute': 'Rate', 'value': f'{selected_data["rate"]}%/year'},
        {'attribute': 'Confidence level', 'value': selected_data['confidence level']},
        {'attribute': 'PV tech', 'value': selected_data['pv tech']},
        {'attribute': 'Duration', 'value': f'{selected_data["duration"]} years'},
        {'attribute': 'Country', 'value': selected_data['country']},
        {'attribute': 'Note', 'value': selected_data['note']},
    ]


# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
