import dash
from dash import dcc, html, Input, Output, dash_table, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import dash_bootstrap_components as dbc
from collections import Counter
import math
import ast
import json
import os
import openai
import re

cborg_API_KEY = os.getenv("cborg_api_key")

client = openai.OpenAI(
    api_key=cborg_API_KEY,
    base_url="https://api.cborg.lbl.gov"
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
COLUMN_INFO_PATH = os.path.join(BASE_DIR, "data", "data_column_info.json")

with open(COLUMN_INFO_PATH, "r") as f:
    column_info = json.load(f)

def build_prompt(question):

    return f"""
You are an assistant that converts natural language questions into dataframe filters
for a dataset about photovoltaic (PV) module degradation studies.

Your tasks:

1. Determine whether the user's question is related to photovoltaic (PV) systems,
solar modules, degradation studies, or research metadata.
2. Determine whether the dataframe likely contains information needed to answer the question.
3. If the dataframe can answer the question, generate dataframe filters.

The dataframe columns with descriptions and types are:

{json.dumps(column_info, indent=2)}

Allowed filter operators:
==, !=, >, <, >=, <=, contains

Filtering rules:

- Only use columns that exist in the dataframe.
- Use numeric comparisons for numeric columns.
- Use "contains" for partial text matching in text columns.
- Filters may require combinations of AND and OR.
- Use nested logical groups when needed.

Concept expansion rules:

- When the user mentions a concept that may appear under different terms
  in the dataset, expand it into related keywords using OR logic.
- Include common synonyms, variations, and related technical terms.
- This is especially important for location, PV technology types,
  environments, and faults.

Examples of concept expansion:

"offshore PV" may include:
- offshore
- ocean
- marine
- sea
- floating PV
- floating solar

"desert climates" may include:
- desert
- arid
- dry climate
- BWk

"Asia" may include countries such as:
- China
- India
- Japan
- Korea
- Mongolia

When expanding a concept, group the related filters using OR logic.

If multiple independent conditions must be satisfied simultaneously,
combine them with AND logic.

If no filtering is needed, return an empty condition list.

Return JSON ONLY in the following format:

{{
  "is_pv_related": true or false,
  "can_be_answered_with_dataframe": true or false,
  "reason": "A short message to display to the user explaining the result.",
  "filter_tree": {{
      "logic": "AND or OR",
      "conditions": [
          {{
              "column": "column_name",
              "operator": "==",
              "value": "example"
          }},
          {{
              "logic": "AND or OR",
              "conditions": [
                  {{
                      "column": "column_name",
                      "operator": "contains",
                      "value": "example"
                  }}
              ]
          }}
      ]
  }}
}}

Explanation of filter_tree:

- "logic": defines how the conditions inside are combined.
- "conditions": may contain either filter rules or nested logic groups.
- Leaf conditions contain:
  column, operator, value.

Rules for the "reason" field:

- This message will be shown directly to the user.
- If filters are generated, explain what data is being shown and indicate that
  the filtered results appear below.
- If the dataframe cannot answer the question, explain briefly why.
- If the question is not related to PV, politely inform the user.

Example:

User question:
"show offshore PV cases"

Output:

{{
  "is_pv_related": true,
  "can_be_answered_with_dataframe": true,
  "reason": "PV systems related to offshore or floating environments have been filtered. Matching records are shown below.",
  "filter_tree": {{
      "logic": "OR",
      "conditions": [
          {{"column": "mounting", "operator": "contains", "value": "offshore"}},
          {{"column": "mounting", "operator": "contains", "value": "floating"}},
          {{"column": "note", "operator": "contains", "value": "floating PV"}},
          {{"column": "location (raw text)", "operator": "contains", "value": "ocean"}},
          {{"column": "location (raw text)", "operator": "contains", "value": "marine"}}
      ]
  }}
}}

User question:
{question}
"""

def get_filter_from_llm(question):

    prompt = build_prompt(question)

    response = client.chat.completions.create(
        model="openai/gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # remove markdown code blocks if present
    content = re.sub(r"^```json", "", content)
    content = re.sub(r"```$", "", content).strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # fallback response if parsing fails
        return {
            "is_pv_related": False,
            "can_be_answered_with_dataframe": False,
            "reason": "Sorry, I couldn't understand the request. Please try rephrasing your question.",
            "filter_tree": []
        }

    # ensure required keys exist
    result.setdefault("is_pv_related", False)
    result.setdefault("can_be_answered_with_dataframe", False)
    result.setdefault("reason", "")
    result.setdefault("filter_tree", [])

    return result


def _evaluate_condition(df, condition):

    col = condition["column"]
    op = condition["operator"]
    val = condition["value"]

    if col not in df.columns:
        return pd.Series([True] * len(df), index=df.index)

    if op == "==":
        return df[col] == val

    elif op == "!=":
        return df[col] != val

    elif op == ">":
        return df[col] > val

    elif op == "<":
        return df[col] < val

    elif op == ">=":
        return df[col] >= val

    elif op == "<=":
        return df[col] <= val

    elif op == "contains":
        return df[col].astype(str).str.contains(str(val), case=False, na=False)

    else:
        return pd.Series([True] * len(df), index=df.index)


def _evaluate_filter_tree(df, node):

    logic = node.get("logic", "AND").upper()
    conditions = node.get("conditions", [])

    masks = []

    for cond in conditions:

        # nested logic group
        if "logic" in cond:
            masks.append(_evaluate_filter_tree(df, cond))

        # leaf filter
        else:
            masks.append(_evaluate_condition(df, cond))

    if not masks:
        return pd.Series([True] * len(df), index=df.index)

    mask = masks[0]

    for m in masks[1:]:

        if logic == "AND":
            mask = mask & m

        elif logic == "OR":
            mask = mask | m

    return mask


def apply_filters(df, filter_tree):

    if not filter_tree:
        return df

    mask = _evaluate_filter_tree(df, filter_tree)

    return df[mask]