import pandas as pd
import base64
import io
import dash_bootstrap_components as dbc
from dash import html, dcc
import base64, os, json
import openai
import rdtools
import ast
import plotly.express as px
import plotly.graph_objects as go
from page_supporting_files.pvcopilot_filter_functions import auto_fix_timezone
import traceback
from dotenv import load_dotenv
import numpy as np
import re

load_dotenv(override=True)

cborg_API_KEY = os.getenv("cborg_api_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Configuration (from your prompt) ---
client = openai.OpenAI(
    api_key=cborg_API_KEY,
    base_url="https://api.cborg.lbl.gov"
)

client_gpt = openai.OpenAI(
    api_key= OPENAI_API_KEY
)

# ================================
# Read data
# ================================

def parse_contents(contents=None, filename=None, df=None):
    """
    Parses uploaded file OR an existing DataFrame (example dataset),
    identifies PV variables via LLM, and returns:

        (df, summary_table_div, mapped_variables_dict, code_read)
    """

    # -----------------------------
    # 1. Load dataframe
    # -----------------------------
    if df is None:

        if contents is None:
            return None, html.Div("Please upload a file to analyze."), {}, None

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                code_read = f"df = pd.read_csv('{filename}')"

            elif 'xls' in filename or 'xlsx' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
                code_read = f"df = pd.read_excel('{filename}')"

            elif 'parquet' in filename:
                df = pd.read_parquet(io.BytesIO(decoded))
                code_read = f"df = pd.read_parquet('{filename}')"

            else:
                return None, html.Div(
                    f"Unsupported file type: {filename}",
                    className="alert alert-danger"
                ), {}, None

        except Exception as e:
            return None, html.Div(
                f"There was an error processing the file: {e}",
                className="alert alert-danger"
            ), {}, None

    else:
        # Example dataset case
        code_read = "df = pd.read_csv('data/pmp.csv')"

    # ----------------------------------
    # 1.5 Detect if time is in index
    # ----------------------------------
    time_in_index = False

    if isinstance(df.index, pd.DatetimeIndex):
        time_in_index = True
    else:
        # Try converting index to datetime (non-destructive check)
        try:
            converted_index = pd.to_datetime(df.index, errors='coerce')
            if converted_index.notna().sum() > 0.9 * len(df):
                df.index = converted_index
                time_in_index = True
        except Exception:
            pass

    # ----------------------------------
    # 2A. Validate column names
    # ----------------------------------
    colnames = df.columns.tolist()

    # Case 1: pandas auto-generated integer column names
    case1_no_headers = colnames == list(range(len(colnames)))

    # Case 2: column names contain no alphabetic letters (numeric-only headers)
    case2_no_real_names = all(not any(c.isalpha() for c in str(name)) for name in colnames)

    if case1_no_headers or case2_no_real_names:
        return None, html.Div(
            "Uploaded file does not contain valid column names. "
            "Column names must include descriptive text (e.g. 'power', 'time').",
            className="alert alert-danger"
        ), {}, None

    # ----------------------------------
    # 3. Prepare LLM identification
    # ----------------------------------
    required_vars = ["DC Power", "AC Power", "Irradiance", "Module temperature"]

    if not time_in_index:
        required_vars.insert(1, "Time")

    prompt = f"""
    The following is a list of column names from a data file: {colnames}.
    Your task is to identify which column name corresponds to each of the following physical quantities:
    {', '.join(required_vars)}.

    Return the result as a JSON object:
    {{
      "variable_mapping": [
        {{"Metric": "DC Power", "Variable Name": "column_name_or_N/A"}},
        {{"Metric": "AC Power", "Variable Name": "column_name_or_N/A"}},
        {{"Metric": "Irradiance", "Variable Name": "column_name_or_N/A"}},
        {{"Metric": "Module temperature", "Variable Name": "column_name_or_N/A"}},
        {{"Metric": "Time", "Variable Name": "column_name_or_N/A"}},
        ...
      ]
    }}
    """

    # Default return values
    mapped_variables_dict = {}

    try:
        # Call LLM
        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )

        res_text = response.choices[0].message.content.strip()
        cleaned = res_text.lstrip("`").lstrip("json").rstrip("`")

        result = json.loads(cleaned)
        mapping_data = result.get("variable_mapping", [])

        mapping_df = pd.DataFrame(mapping_data)

        # ----------------------------------
        # If time is in index, override mapping
        # ----------------------------------
        # inject index time
        if time_in_index:
            mapping_df = mapping_df[mapping_df["Metric"] != "Time"]

            index_name = df.index.name
            display_name = index_name if index_name not in [None, ""] else "__index__"

            mapping_df.loc[len(mapping_df)] = {
                "Metric": "Time",
                "Variable Name": display_name
            }

        # ----------------------------------
        # Ensure all required variables appear (fill missing with N/A)
        # ----------------------------------
        existing_metrics = set(mapping_df["Metric"].tolist())

        for rv in required_vars:
            if rv not in existing_metrics:
                mapping_df.loc[len(mapping_df)] = {"Metric": rv, "Variable Name": "N/A"}

        # ----------------------------------
        # Build dict of recognized variables (skip N/A)
        # ----------------------------------
        mapped_variables_dict = {
            row["Metric"]: row["Variable Name"]
            for _, row in mapping_df.iterrows()
            if row["Variable Name"] != "N/A"
        }

        # ----------------------------------
        # Build summary table for display
        # ----------------------------------
        summary_table = html.Div([
            html.H5("Identified Variables"),
            html.Table(
                [
                    html.Thead(html.Tr([html.Th(c) for c in mapping_df.columns])),
                    html.Tbody([
                        html.Tr([html.Td(mapping_df.iloc[i][col]) for col in mapping_df.columns])
                        for i in range(len(mapping_df))
                    ])
                ],
                className="table table-striped"
            )
        ])

        # ----------------------------------
        # Check for missing Power/Time
        # ----------------------------------
        missing_msgs = []
        if mapping_df.loc[mapping_df["Metric"] == "DC Power", "Variable Name"].iloc[0] == "N/A":
            missing_msgs.append("⚠️ Power column not identified.")
        if mapping_df.loc[mapping_df["Metric"] == "Time", "Variable Name"].iloc[0] == "N/A":
            missing_msgs.append("⚠️ Time column not identified.")

        if missing_msgs:
            summary_table = html.Div([
                summary_table,
                html.Div(
                    "Degradation analysis requires both Time and Power columns.",
                    className="alert alert-warning"
                ),
                html.Div([html.Div(msg) for msg in missing_msgs])
            ])

    except Exception as e:
        summary_table = html.Div(
            f"Error during LLM analysis or parsing: {e}",
            className="alert alert-warning"
        )
        mapped_variables_dict = {}

    return df, summary_table, mapped_variables_dict, code_read


# ================================
# FIGURES OF RAW DATA
# ================================
def make_overview_figures(df, mapped_variables_dict, temp_col="temp_C"):

    figures = []
    errors = []

    # -------------------------
    # Color palette
    # -------------------------
    COLORS = {
        "power": "#0d6efd",        # blue
        "irradiance": "#f59f00",   # amber/orange
        "temp_raw": "#dc3545",     # red
    }

    # -------------------------
    # Shared layout config
    # -------------------------
    def apply_layout(fig, title, y_label):
        fig.update_layout(
            title=dict(text=title, x=0.01),
            template="plotly_white",
            height=180,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="Time",
            yaxis_title=y_label,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=-0.25,
                x=0.5,
                xanchor="center"
            )
        )
        return fig

    # -------------------------
    # 1. Power (blue)
    # -------------------------
    try:
        power_key = mapped_variables_dict.get("DC Power")

        if not power_key:
            raise ValueError("DC Power key not found")

        if power_key not in df.columns:
            raise ValueError(f"Column '{power_key}' not found")

        fig_power = go.Figure()

        fig_power.add_trace(go.Scattergl(
            x=df.index,
            y=df[power_key],
            mode="markers",
            name="Power Output",
            opacity=0.3,
            marker=dict(color=COLORS["power"], size=4)
        ))

        fig_power = apply_layout(fig_power, "Power vs Time", "Power (W)")
        figures.append(dcc.Graph(figure=fig_power))

    except Exception as e:
        errors.append(f"[Power Plot] {str(e)}")

    # -------------------------
    # 2. Irradiance (orange)
    # -------------------------
    try:
        irr_key = mapped_variables_dict.get("Irradiance")

        if not irr_key:
            raise ValueError("Irradiance key not found")

        if irr_key not in df.columns:
            raise ValueError(f"Column '{irr_key}' not found")

        fig_irr = go.Figure()

        fig_irr.add_trace(go.Scattergl(
            x=df.index,
            y=df[irr_key],
            mode="markers",
            name="Irradiance",
            opacity=0.3,
            marker=dict(color=COLORS["irradiance"], size=4)
        ))

        fig_irr = apply_layout(fig_irr, "Irradiance", "Irradiance (W/m²)")
        figures.append(dcc.Graph(figure=fig_irr))

    except Exception as e:
        errors.append(f"[Irradiance Plot] {str(e)}")

    # -------------------------
    # 3. Temperature (red + purple)
    # -------------------------
    try:
        temp_raw = mapped_variables_dict.get("Module temperature")

        if not temp_raw:
            raise ValueError("Module temperature key not found")

        if temp_raw not in df.columns:
            raise ValueError(f"Column '{temp_raw}' not found")

        fig_temp = go.Figure()

        fig_temp.add_trace(go.Scattergl(
            x=df.index,
            y=df[temp_raw],
            mode="markers",
            name="Module Temp raw",
            opacity=0.3,
            marker=dict(color=COLORS["temp_raw"], size=4)
        ))

        fig_temp = apply_layout(fig_temp, "Temperature", "Temperature (°C)")

        # --- Apply temperature limit ---
        ymax = df[temp_raw].max()
        ymin = df[temp_raw].min()

        if ymax > 150:
            fig_temp.update_yaxes(range=[ymin-20, 80])

        if ymin<-50:
            fig_temp.update_yaxes(range=[-40, ymax+20])

        figures.append(dcc.Graph(figure=fig_temp))

    except Exception as e:
        errors.append(f"[Temperature Plot] {str(e)}")

    return figures, errors


# ================================
# NORMALIZATION
# ================================
def normalize(df, mapped_variables_dict, gamma=-0.004):

    irr_key = mapped_variables_dict["Irradiance"]
    power_key = mapped_variables_dict["DC Power"]
    temp_C_key = mapped_variables_dict["Module temperature"]

    df['norm'] = df[power_key] / (
        df[irr_key] * (1 + gamma * (df[temp_C_key] - 25)))*1000

    df.loc[df[irr_key] < 50, 'norm'] = np.nan

    return df


# ================================
# Low irradiance & power filter
# ================================
def low_irra_power_filter(df, mapped_variables_dict):
    mask = pd.Series(True, index=df.index)

    irr_key = mapped_variables_dict["Irradiance"]
    power_key = mapped_variables_dict["DC Power"]

    # irradiance filter
    mask &= df[irr_key] > 300

    # power filter
    mask &= df[power_key] > 0.02 * df[irr_key]

    # norm range filter
    upper = df['norm'].quantile(0.99)
    mask &= df['norm'].between(0.01, upper)

    # ✅ indices
    normal_indices = df.index[mask]
    outlier_indices = df.index[~mask]

    return normal_indices, outlier_indices

# ================================
# DAILY AGGREGATION
# ================================
def aggregate_daily(df_f, irradiance_col):
    daily = (
        df_f[['norm', irradiance_col]]
        .dropna()
        .groupby(df_f.index.date)
        .apply(lambda x: np.sum(x['norm'] * x[irradiance_col]) / np.sum(x[irradiance_col]))
    )

    daily.index = pd.to_datetime(daily.index)

    return daily

# ================================
# YoY
# ================================
def compute_yoy(series, eps=1e-6):
    series = series.dropna()
    yoy = []

    for t in series.index:
        t_prev = t - pd.DateOffset(years=1)

        if t_prev in series.index:
            prev = series.loc[t_prev]
            curr = series.loc[t]

            if prev < eps:
                continue

            ratio = curr / prev - 1

            if np.isfinite(ratio):
                yoy.append(ratio)

    yoy = np.array(yoy)

    # --- Remove outliers using IQR ---
    if len(yoy) > 0:
        q1 = np.percentile(yoy, 25)
        q3 = np.percentile(yoy, 75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        yoy = yoy[(yoy >= lower) & (yoy <= upper)]

    rd = np.median(yoy) * 100 if len(yoy) > 0 else np.nan

    return rd, yoy

# ================================
# get full code
# ================================
def get_full_code(filename, mapped_variables_dict,selected_filters, selected_metric):

    with open("page_supporting_files/pvcopilot_functions_code.txt", "r", encoding="utf-8") as f:
            pvcopilot_functions_code = f.read().replace('\n', ' ').replace('"', "'")

    with open("page_supporting_files/pvcopilot_packages_code.txt", "r", encoding="utf-8") as f:
            pvcopilot_packages_code = f.read().replace('\n', ' ').replace('"', "'")

    prompt = f"""
        Your task is to generate a code:
        * load data as df where filename is {filename}, add comment user need to provide file path if necessary
        * define a dict 'mapped_variables_dict' from {mapped_variables_dict}
        * use functinon df_filtered = normalize(df, mapped_variables_dict)
        * if "low-irra-power" in selected_filters {selected_filters}:
            use function normal_idx, outlier_idx = low_irra_power_filter(df_filtered, mapped_variables_dict)
        * if "outlier" in selected_filters {selected_filters}:
            use function normal_idx, outlier_idx = identify_outliers_iqr(df_filtered, "norm")
        * merge all normal_idx, print the total number of points, normal ones, and outliers
        * define df_filtered_final with only normal_idx from df_filtered
        * use function daily_data = aggregate_daily(df_filtered_final, irra_key)
        * use function rd, yoy_dist = compute_yoy(daily_data)
        * print rd

        Note that: 
        * all these functions are already defined, just use them.
        * add comments to each part for user to understand.
        * add necessary packages on the beginning of code based on {pvcopilot_packages_code}

        No verbose.

        """

    # Call LLM
    response = client.chat.completions.create(
        model="openai/gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
    )

    res_text = response.choices[0].message.content.strip()
    clean_text = re.sub(r"```python\n(.*?)```", r"\1", res_text, flags=re.DOTALL).strip()

    with open("llm_response.txt", "w", encoding="utf-8") as f:
        f.write(clean_text)

    return clean_text
