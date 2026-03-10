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
import pvcopilot_files.pvcopilot_filter_functions as pvcopilot_filter_functions
from typing import List, Optional
import plotly.graph_objects as go
from pvcopilot_files.pvcopilot_filter_functions import auto_fix_timezone
import traceback
from dotenv import load_dotenv

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

# --- Configuration End ---

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

            elif 'pkl' in filename:
                df = pd.read_pickle(io.BytesIO(decoded))
                code_read = f"df = pd.read_pickle('{filename}')"

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
    required_vars = ["Power", "Time", "DC Voltage", "DC Current"]

    prompt = f"""
    The following is a list of column names from a data file: {colnames}.
    Your task is to identify which column name corresponds to each of the following physical quantities:
    {', '.join(required_vars)}.

    Return the result as a JSON object:
    {{
      "variable_mapping": [
        {{"Metric": "Power", "Variable Name": "column_name_or_N/A"}},
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
            html.H6("Identified Variables"),
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
        if mapping_df.loc[mapping_df["Metric"] == "Power", "Variable Name"].iloc[0] == "N/A":
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


def execute_generated_code(code, df, variable_dict):
    safe_locals = {
        "pd": pd,
        "rdtools": rdtools,
        "variable_dict": variable_dict,
        "df": df,
        # expose the function
        "auto_fix_timezone": auto_fix_timezone,
    }

    # Safety check — do not allow imports
    if "import" in code:
        raise ValueError("Generated code contains imports, not allowed.")

    exec(code, {}, safe_locals)
    return safe_locals

def generate_full_code(code, code_read):

    with open("pvcopilot_files/code_plot_power_t.txt", "r", encoding="utf-8") as f:
            code_plot = f.read().replace('\n', ' ').replace('"', "'")

    with open("pvcopilot_files/code_plot_lib.txt", "r", encoding="utf-8") as f:
            code_plot_lib = f.read().replace('\n', ' ').replace('"', "'")

    code_full = code["library_code"] + "\n" + code_plot_lib + "\n\n" + code_read + "\n\n"+ code["run_code"] + "\n\n"+ code_plot
    
    return code_full


def generate_degradation_code_and_execute(df, variable_dict, llm_temp, filter_options):

    log_text = "[INFO] Starting degradation code generation...\n"

    # -------------------------
    # Read function descriptions
    # -------------------------
    try:
        with open("pvcopilot_files/rdtools_function_description.txt", "r", encoding="utf-8") as f:
            rdtools_function_summary = f.read().replace('\n', ' ').replace('"', "'")
    except FileNotFoundError:
        rdtools_function_summary = "The rdtools degradation_year_on_year function takes time-indexed power data and returns the degradation rate."
        log_text += "[WARNING] rdtools description file missing.\n"

    try:
        with open("pvcopilot_files/filter_function_description.txt", "r", encoding="utf-8") as f:
            filter_function_summary = f.read().replace('\n', ' ').replace('"', "'")
    except FileNotFoundError:
        filter_function_summary = "xx"
        log_text += "[WARNING] filter description file missing.\n"


    # -------------------------
    # Build prompt
    # -------------------------
    outlier_filter = "outlier" in filter_options
    timezone_filter = "timezone" in filter_options

    outlier_filter_prompt = ""
    timezone_prompt = ""

    if outlier_filter:
        outlier_filter_prompt = """
        Identify outliers using IQR and negative value filter.
        Store indices as 'outlier_indices'
        """

    if timezone_filter:
        timezone_prompt = f"""
        Fix timezone using auto_fix_timezone.
        Function summary: {filter_function_summary}
        """

    prompt = f""" You MUST output ONLY a single Python dictionary. 
    There should be no explanation, no markdown, no comments, and no 'python' word outside of the dictionary structure. 
    The dictionary MUST contain exactly two keys: 
    
    1. **'run_code'**: The value must be a single string containing the Python code necessary to execute the task. 
    This code must NOT include any 'import' statements. 
    
    2. **'library_code'**: The value must be a single string containing all necessary 'import' statements (e.g., 'import pandas as pd', 'import rdtools'). 
    Context and Task for 'run_code': There is a pandas DataFrame named 'df'. 
    The libraries pandas (as pd), numpy (as np), and rdtools are already available for use in the 'run_code' string. 
    Note that do not change the length of df like by dropping rows. 
    
    **Follow these steps for the 'run_code' string strictly and add comments in the code:** 
    * Define a variable named **power_key** with value of {variable_dict['Power']}, and **time_key** with value of {variable_dict['Time']}. 
    * Set the column specified by time_key as df's index. - Ensure the index is converted to datetime format using pd.to_datetime(). 
    * {timezone_prompt} * Identify the nan/empty points, store the indices as 'nan_indices'. 
    * {outlier_filter_prompt} 
    * Use the rdtools.degradation_year_on_year function to calculate the degradation rate. 
        - Function summary: {rdtools_function_summary} 
        - Use the filterd data by excluding nan_indices and other indices 
        - Store the degradation rate (the **first element** of the tuple returned by the rdtools function) in a variable named **result**. 
    
    Output ONLY the final dictionary. 
    
    Example final output structure: {{ 'run_code': '...', 'library_code': '...' }} 
    
    """

    # -------------------------
    # LLM generation
    # -------------------------

    code = None
    execution_success = False

    try:
        log_text += "[STEP] Calling LLM...\n"

        response = client_gpt.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_temp
        )

        code = ast.literal_eval(response.choices[-1].message.content)

        log_text += "[INFO] LLM code generated.\n"

    except Exception:

        log_text += "[ERROR] LLM code generation failed\n"
        log_text += traceback.format_exc()

        # LLM failed → nothing to return
        return None, [], [], None, None, log_text, execution_success


    # -------------------------
    # Execute generated code
    # -------------------------

    rd = None
    df_processed = None
    nan_indices = []
    outlier_indices = []

    try:

        log_text += "[STEP] Executing generated code...\n"

        safe_locals = execute_generated_code(
            code["run_code"],
            df,
            variable_dict
        )

        rd = safe_locals["result"]
        df_processed = safe_locals["df"]
        nan_indices = safe_locals["nan_indices"]
        outlier_indices = safe_locals.get("outlier_indices", [])

        execution_success = True

        log_text += "[INFO] Code executed successfully.\n"

    except Exception:

        execution_success = False

        log_text += "[ERROR] Generated analysis code execution failed\n"
        log_text += traceback.format_exc()

        # Important: DO NOT raise
        # We still return the generated code


    # -------------------------
    # Final return (always return code)
    # -------------------------

    return rd, outlier_indices, nan_indices, df_processed, code, log_text, execution_success

POWER_COLOR = "#0D6EFD"
SLOPE_COLOR = "orange"
BACKGROUND_COLOR = "#F5F5F5"   # light grey

def plot_power_vs_time(df: pd.DataFrame, mapped_variables_dict: dict, Rd_pct: float = None):

    power_key = mapped_variables_dict["Power"]
    y = df[power_key].dropna()
    x = y.index

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    title = (
        f"Degradation trend (Rd = {Rd_pct:.2f}%/year)"
        if Rd_pct is not None else
        "Power vs Time"
    )

    # ---------------------------------------------------------
    # Base figure with PX
    # ---------------------------------------------------------
    fig = px.line(
        df,
        x=df.index,
        y=power_key,
        title=title,
        labels={"x": "Time", power_key: "Power Output"},
        color_discrete_sequence=[POWER_COLOR]
    )

    # --- FORCE legend to show for the power trace ---
    fig.update_traces(name="Power Output", showlegend=True)

    # ---------------------------------------------------------
    # Add degradation slope line
    # ---------------------------------------------------------
    if Rd_pct is not None and len(y) > 1:

        x_years = (x - x.min()).total_seconds() / (365.25 * 24 * 3600)

        first_part_n = max(1, int(len(y) * 0.10))
        baseline = y.iloc[:first_part_n].mean()

        slope = (Rd_pct / 100) * baseline
        degradation_line = baseline + slope * x_years

        fig.add_trace(go.Scatter(
            x=x,
            y=degradation_line,
            mode="lines",
            line=dict(color=SLOPE_COLOR, dash="dash", width=2),
            name="Degradation Trend",
            showlegend=True      # force legend here too
        ))

    # ---------------------------------------------------------
    # Layout and Legend Below
    # ---------------------------------------------------------
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=power_key,
        hovermode="x unified",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),

        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor="white",

        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.32,    # enough spacing to avoid overlap
            xanchor="center",
            x=0.3
        )
    )

    return fig


# --- Plotting colors ---
NORMAL_COLOR = "rgba(13, 110, 253, 0.8)"   # transparent blue
OUTLIER_COLOR = "orange"

def plot_outlier_vs_time(df: pd.DataFrame, mapped_variables_dict: dict, nan_indices, outlier_indices: list):
    """
    Scatter plot of power vs time, highlighting outliers.
    NaN points remain in df but are NOT plotted.
    """

    # --- Get mapped columns ---
    power_key = mapped_variables_dict["Power"]
    time_key = mapped_variables_dict["Time"]

    # SAFE COPY (df length unchanged)
    df = df.copy()

    # ---------------------------------------------------
    # Create Status column
    # ---------------------------------------------------
    df["Status"] = "Normal"
    df.loc[outlier_indices, "Status"] = "Outlier"
    df.loc[nan_indices, "Status"] = "NaN"

    # ---------------------------------------------------
    # Create df_plot (ONLY for plotting)
    # Remove NaN rows visually (but NOT from original df)
    # ---------------------------------------------------
    df_plot = df[df["Status"] != "NaN"].copy()

    # ---------------------------------------------------
    # Counts for legend (ignore NaNs)
    # ---------------------------------------------------
    normal_count = (df_plot["Status"] == "Normal").sum()
    outlier_count = (df_plot["Status"] == "Outlier").sum()

    # ---------------------------------------------------
    # Legend labels
    # ---------------------------------------------------
    df_plot["LegendLabel"] = df_plot["Status"].replace({
        "Normal":  f"Normal ({normal_count})",
        "Outlier": f"Outliers ({outlier_count})"
    })

    color_map = {
        f"Normal ({normal_count})": NORMAL_COLOR,
        f"Outliers ({outlier_count})": OUTLIER_COLOR,
    }

    # ---------------------------------------------------
    # Build the plot (ONLY df_plot is used)
    # ---------------------------------------------------
    fig = px.scatter(
        df_plot,
        y=power_key,
        color="LegendLabel",
        color_discrete_map=color_map,
        title="Outlier Analysis",
        labels={time_key: "Time", power_key: "Power Output"},
        height=300,
    )

    # Marker sizes
    fig.update_traces(
        selector=dict(name=f"Normal ({normal_count})"),
        marker=dict(size=4)
    )
    fig.update_traces(
        selector=dict(name=f"Outliers ({outlier_count})"),
        marker=dict(size=7)
    )

    # Remove legend title
    fig.update_layout(legend_title_text="")

    # Legend placement & layout
    fig.update_layout(
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=130),

        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.32,
            xanchor="center",
            x=0.3
        ),

        hovermode="closest"
    )

    return fig

def build_data_summary_block(df, outlier_indices=None, filters=None):

    filters = filters or []
    outlier_count = len(outlier_indices) if outlier_indices is not None else 0
    total_input = len(df)

    # Count NaN rows (any column NaN or specific column)
    nan_count = df.isna().any(axis=1).sum()

    # Time info
    start = df.index.min().date() if total_input else "N/A"
    end = df.index.max().date() if total_input else "N/A"
    dur = ((df.index.max() - df.index.min()).days / 365.25) if total_input else 0

    # Build text block
    text = (
        f"• Total input points: {total_input}\n"
        f"• NaN/empty points: {nan_count}\n"
        + (f"• Outliers removed: {outlier_count}\n" if "outlier" in filters else "")
        + f"• Final (filtered): {total_input - outlier_count - nan_count}\n"
        f"• Duration: {dur:.2f} years\n"
        f"• Date range: {start} to {end}"
    )

    return html.Pre(text, style={
        "backgroundColor": "#F5F5F5",
        "padding": "8px",
        "whiteSpace": "pre-wrap",
        "fontFamily": "monospace",
        "margin": 0,
    })

def get_filtered_display_string(filters, outlier_indices=None):
    """
    Generates a multi-line display string for applied filters, embedding the outlier
    count next to 'Outlier removal' when applicable. Each filter is shown on its own
    line with a ✓ symbol.
    """
    FILTER_NAME_MAP = {
        "outlier": "Outlier removal",
        "clearsky": "Clear-sky filter",
        "timezone": "Time zone correction",
    }

    # Safely calculate outlier count
    outlier_count = len(outlier_indices) if outlier_indices is not None else 0

    # Build lines with ✓ prefix, include outlier count only when relevant
    return "\n".join(
        f"• {FILTER_NAME_MAP.get(f, f)}\n"
        for f in filters
    ) if filters else "None"