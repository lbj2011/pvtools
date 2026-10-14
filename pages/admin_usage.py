"""
Admin usage dashboard (/admin): visualise, analyse and download the usage
logs written by utils/usage_log.py. Password protected.

Team / internal traffic
    Signing in marks the current browser (its pvtools_client_id cookie) as
    internal; more browsers or a whole network (IP hash) can be marked from
    the "Team / internal visitors" card. Internal traffic is excluded by
    default, including events from before the browser was marked. Requests
    from LBNL networks are also tagged (net="lbl") and can be excluded.

Password
    Not in the code or in git. The page reads a PBKDF2 hash of the password
    from the environment variable ADMIN_PASSWORD_HASH:
      - local: the .env file (gitignored, and excluded from the Docker image)
      - Heroku: Settings -> Config Vars -> ADMIN_PASSWORD_HASH
    Generate the hash for a (new) password with
        python -c "import hashlib,getpass;print(hashlib.pbkdf2_hmac('sha256',getpass.getpass().encode(),b'pvtools-admin-v1',200000).hex())"
    If the variable is missing, admin mode stays locked.
"""
import hashlib
import hmac
import os
import time

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv

from app import app
from utils import usage_log, usage_report

load_dotenv()

_PW_SALT = b"pvtools-admin-v1"
_PW_ITER = 200_000

PALETTE = ["#3B4CC0", "#7B61C9", "#4C8FD6", "#A07BD8", "#2E3A8C",
           "#6FB1E8", "#5B3E9E", "#C4A8EB", "#1F6FB2", "#8C8FD9"]
_CACHE_TTL_S = 300
_cache = {"t": 0.0, "rows": []}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def _pw_hash():
    return (os.environ.get("ADMIN_PASSWORD_HASH") or "").strip().lower()


def _token():
    """Session token = HMAC(server secret, password hash): nothing secret goes
    to the browser and changing the password signs everyone out."""
    key = (app.server.secret_key or "pvtools").encode()
    return hmac.new(key, _pw_hash().encode(), hashlib.sha256).hexdigest()


def _check_password(pw):
    if not _pw_hash():
        return False
    h = hashlib.pbkdf2_hmac("sha256", (pw or "").encode(), _PW_SALT, _PW_ITER).hex()
    return hmac.compare_digest(h, _pw_hash())


def _authed(tok):
    return bool(_pw_hash()) and isinstance(tok, str) and hmac.compare_digest(tok, _token())


def _flag_internal_cookie():
    """Readable-by-JS cookie so Google Analytics tags this browser's hits as
    traffic_type=internal (see ga_script in app.py)."""
    try:
        dash.callback_context.response.set_cookie(
            "pvtools_internal", "1", max_age=2 * 365 * 24 * 3600,
            secure=True, samesite="Lax")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load(force=False):
    if force or time.time() - _cache["t"] > _CACHE_TTL_S:
        try:
            _cache["rows"] = usage_report.load_events(days=400)
            _cache["err"] = None
        except Exception as e:
            _cache["err"] = f"{type(e).__name__}: {e}"
        _cache["t"] = time.time()
    return usage_report.to_frame(_cache["rows"] + usage_log.pending_events(),
                                 usage_report.load_internal())


def _filter(df, days, tools, flags=None):
    if df.empty:
        return df
    flags = flags or []
    if "no_internal" in flags:
        df = df[~df["internal"]]
    if "no_lbl" in flags:
        df = df[~df["lbl_net"]]
    if days and days != "all":
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))
        df = df[df["ts"] >= cutoff]
    if tools:
        df = df[df["tool"].isin(tools)]
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(fig, height=330):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(family="Arial", size=13),
                      legend=dict(orientation="h", y=-0.2, title=None),
                      colorway=PALETTE)
    return fig


def _empty(msg="No data in this range", height=200):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#888"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=height)


def _fig_daily(df):
    if df.empty:
        return _empty()
    d = (df.assign(day=df["ts_local"].dt.date)
           .groupby(["day", "tool"])["who"].nunique().rename("visitors").reset_index())
    fig = px.line(d, x="day", y="visitors", color="tool", markers=True,
                  color_discrete_sequence=PALETTE)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Unique visitors", rangemode="tozero")
    return _style(fig)


def _fig_tools(summary):
    if summary.empty:
        return _empty()
    s = summary.sort_values("visitors")
    fig = px.bar(s, x="visitors", y="tool", orientation="h", text="visitors",
                 color_discrete_sequence=PALETTE)
    fig.update_xaxes(title="Unique visitors")
    fig.update_yaxes(title=None)
    return _style(fig)


def _fig_funnel(df):
    if df.empty:
        return _empty()
    rows = []
    for tool, g in df.groupby("tool"):
        rows += [
            (tool, "Opened", g.loc[g["event"].isin(["visit", "page_view"]), "who"].nunique()),
            (tool, "Used a control", g.loc[g["event"] == "action", "who"].nunique()),
            (tool, "Uploaded data", g.loc[g["is_upload"], "who"].nunique()),
        ]
    f = pd.DataFrame(rows, columns=["tool", "stage", "visitors"])
    fig = px.bar(f, x="tool", y="visitors", color="stage", barmode="group",
                 color_discrete_sequence=[PALETTE[0], PALETTE[1], PALETTE[3]])
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Unique visitors")
    return _style(fig)


def _fig_heatmap(df):
    if df.empty:
        return _empty()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    t = df["ts_local"]
    pv = (pd.crosstab(t.dt.dayofweek, t.dt.hour)
            .reindex(index=range(7), columns=range(24), fill_value=0))
    fig = go.Figure(go.Heatmap(z=pv.values, x=list(range(24)), y=days,
                               colorscale=[[0, "#F4F3FB"], [0.5, "#7B61C9"], [1, "#2E3A8C"]],
                               hovertemplate="%{y} %{x}:00 — %{z} events<extra></extra>"))
    fig.update_xaxes(title="Hour (Pacific time)", dtick=3)
    fig.update_yaxes(autorange="reversed")
    return _style(fig)


def _fig_actions(df):
    a = df[df["event"] == "action"]
    if a.empty:
        return _empty("No actions in this range")
    t = (a.assign(label=a["tool"] + " · " + a["trigger"].fillna("?"))
          .groupby("label").size().nlargest(15).sort_values().rename("n").reset_index())
    fig = px.bar(t, x="n", y="label", orientation="h", text="n",
                 color_discrete_sequence=[PALETTE[1]])
    fig.update_traces(textposition="outside", textangle=0, cliponaxis=False)
    fig.update_xaxes(title="Times used")
    fig.update_yaxes(title=None)
    return _style(fig, height=420)


def _fig_counts(series, xlabel, color):
    s = series.dropna()
    if s.empty:
        return _empty("No data")
    c = s.value_counts().head(10).sort_values().rename("n").reset_index()
    c.columns = ["label", "n"]
    fig = px.bar(c, x="n", y="label", orientation="h", text="n",
                 color_discrete_sequence=[color])
    fig.update_traces(textposition="outside", textangle=0, cliponaxis=False)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=None)
    # height follows the number of bars, so one category is not a giant slab
    _style(fig, height=max(200, 110 + 34 * len(c)))
    fig.update_layout(bargap=0.35)
    return fig


def _fig_refs(df):
    """Where visits come from. A visit with no external referrer is a direct
    visit (typed URL, bookmark, email/Slack link, or an in-site link)."""
    v = df[df["event"] == "visit"]
    if v.empty:
        return _empty("No visits in this range")
    return _fig_counts(v["ref"].fillna("(direct / bookmark / link)"), "Visits", PALETTE[2])


def _fig_countries(df):
    c = df.dropna(subset=["country"]).drop_duplicates("who")["country"]
    if c.empty:
        return _empty("No country info (needs a CDN header)")
    return _fig_counts(c, "Visitors", PALETTE[4])


# Stage colours: one hue family per stage (blue / teal / violet / amber /
# rose) so neighbouring stages never look alike; UI events stay neutral grey.
STAGE_COLOR = {"data": "#3B4CC0", "filter": "#14A098", "degradation": "#7B4FC9",
               "pvpro": "#E8862F", "ai_code": "#D6457A", "ai_diag": "#3F9A4A",
               "ai_chat": "#B8860B", "ui": "#A3A9C2"}
_LANE_COLOR = {"UI": STAGE_COLOR["ui"], "Data": STAGE_COLOR["data"],
               "Filter": STAGE_COLOR["filter"], "Degradation": STAGE_COLOR["degradation"],
               "AI coding": STAGE_COLOR["ai_code"], "AI diagnostic": STAGE_COLOR["ai_diag"],
               "AI help chat": STAGE_COLOR["ai_chat"]}


def _fig_session(t):
    if t.empty:
        return _empty("No events")
    t = t.copy()
    t["srv_s"] = (t["ms"].fillna(0) / 1000).round(2)
    fig = go.Figure()
    # background PVPRO fits: a bar from start to finish on the Degradation lane
    for _, r in t[(t["x_stage"] == "pvpro_done") & t["x_run_s"].notna()].iterrows():
        t0 = r["ts_local"] - pd.Timedelta(seconds=float(r["x_run_s"]))
        fig.add_trace(go.Scatter(x=[t0, r["ts_local"]], y=["Degradation", "Degradation"],
                                 mode="lines", line=dict(color=STAGE_COLOR["pvpro"], width=10),
                                 opacity=0.35, hoverinfo="skip", showlegend=False))
    for lane in usage_report._LANES:
        d = t[t["lane"] == lane]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["ts_local"], y=d["lane"], mode="markers", name=lane,
            marker=dict(size=(8 + d["srv_s"].clip(upper=30) * 1.2), color=_LANE_COLOR[lane],
                        line=dict(width=1, color="white")),
            customdata=d[["label", "srv_s"]].values,
            hovertemplate="%{x|%H:%M:%S}<br><b>%{customdata[0]}</b>"
                          "<br>server time %{customdata[1]} s<extra></extra>"))
    e = t[t["is_error"]]
    if not e.empty:
        fig.add_trace(go.Scatter(x=e["ts_local"], y=e["lane"], mode="markers", name="error",
                                 marker=dict(symbol="x", size=14, color="#D64545"),
                                 customdata=e[["err"]].fillna("5xx").values,
                                 hovertemplate="error: %{customdata[0]}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=usage_report._LANES, title=None)
    fig.update_xaxes(title="Time (Pacific)")
    _style(fig, height=320)
    fig.update_layout(showlegend=False)   # the y-axis lanes already name each colour
    return fig


def _chip(label, value):
    return html.Div([html.Div(label, className="adm-chip-label"),
                     html.Div(value if value not in (None, "") else "—", className="adm-chip-value")],
                    className="adm-chip")


def _kpi(label, value, sub=None):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, style={"fontSize": "0.8rem", "color": "#666",
                               "textTransform": "uppercase", "letterSpacing": "0.04em"}),
        html.Div(value, style={"fontSize": "1.8rem", "fontWeight": 700, "color": "#2E3A8C"}),
        html.Div(sub or "", style={"fontSize": "0.8rem", "color": "#888"}),
    ]), className="h-100 shadow-sm"), xs=6, md=3, lg=True, className="mb-3")


def _kpis(df):
    if df.empty:
        return [_kpi("Visitors", "0")]
    days_per = df.assign(day=df["ts_local"].dt.date).groupby("who")["day"].nunique()
    acts = df[df["event"] == "action"]
    n_err = int(df["is_error"].sum())
    med = acts["ms"].median()
    return [
        _kpi("Visitors", f"{df['who'].nunique():,}",
             f"{int((days_per >= 2).sum()):,} returning"),
        _kpi("Sessions", f"{df['session'].nunique():,}"),
        _kpi("Page views", f"{int((df['event'] == 'page_view').sum()):,}"),
        _kpi("Actions", f"{len(acts):,}"),
        _kpi("Uploads", f"{int(df['is_upload'].sum()):,}"),
        _kpi("Errors", f"{n_err:,}",
             f"{100 * n_err / max(len(acts), 1):.1f}% of actions"),
        _kpi("Median response", "—" if pd.isna(med) else f"{med / 1000:.2f} s", "per action"),
    ]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _card(title, graph_id, md=6):
    return dbc.Col(dbc.Card([
        dbc.CardHeader(title, style={"fontWeight": 600, "background": "#F6F5FC"}),
        dbc.CardBody(dcc.Loading(dcc.Graph(id=graph_id, config={"displaylogo": False})),
                     style={"padding": "6px"}),
    ], className="shadow-sm h-100"), md=md, className="mb-4")


_TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "4px 8px",
                "textAlign": "left", "maxWidth": 260, "overflow": "hidden",
                "textOverflow": "ellipsis"},
    style_header={"fontWeight": 700, "background": "#F6F5FC"},
)

_login = dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
    html.H3("Admin · Usage dashboard"),
    html.P("Enter the admin password to view usage logs.", className="text-muted"),
    dbc.InputGroup([
        dbc.Input(id="adm-pw", type="password", placeholder="Password",
                  autoComplete="current-password", debounce=False),
        dbc.Button("Show", id="adm-pw-toggle", color="secondary", outline=True,
                   n_clicks=0, title="Show / hide password"),
    ], className="mb-2"),
    dbc.Button("Sign in", id="adm-login-btn", color="primary", n_clicks=0),
    html.Div(id="adm-login-msg", className="mt-2 text-danger"),
]), className="shadow-sm"), md=6, lg=4), justify="center", className="my-5")

_dash = html.Div([
    dbc.Row([
        dbc.Col([html.H3("Usage dashboard", className="mb-0"),
                 html.Div(id="adm-source", className="text-muted small")], md=4),
        dbc.Col(dbc.Select(id="adm-range", value="30", options=[
            {"label": "Last 24 hours", "value": "1"},
            {"label": "Last 7 days", "value": "7"},
            {"label": "Last 30 days", "value": "30"},
            {"label": "Last 90 days", "value": "90"},
            {"label": "Last 365 days", "value": "365"},
            {"label": "All", "value": "all"},
        ]), md=2),
        dbc.Col(dcc.Dropdown(id="adm-tools", multi=True, placeholder="All tools"), md=3),
        dbc.Col([
            dbc.Button("Refresh", id="adm-refresh", color="secondary", outline=True,
                       n_clicks=0, className="me-2"),
            dbc.DropdownMenu(label="Download", color="primary", className="d-inline-block me-2",
                             children=[
                dbc.DropdownMenuItem("Events (CSV)", id="adm-dl-events", n_clicks=0),
                dbc.DropdownMenuItem("Per-tool summary (CSV)", id="adm-dl-summary", n_clicks=0),
                dbc.DropdownMenuItem("Daily visitors (CSV)", id="adm-dl-daily", n_clicks=0),
            ]),
            dbc.Button("Sign out", id="adm-logout", color="link", n_clicks=0),
        ], md=3, className="text-md-end"),
    ], className="align-items-center g-2 mb-2"),
    dbc.Checklist(id="adm-flags", inline=True, switch=True, className="mb-3",
                  value=["no_internal"], options=[
        {"label": "Exclude team / internal traffic", "value": "no_internal"},
        {"label": "Exclude LBNL network", "value": "no_lbl"},
    ]),
    dbc.Row(id="adm-kpis", className="g-3"),
    dbc.Row([_card("Daily unique visitors by tool", "adm-fig-daily", 8),
              _card("Visitors per tool", "adm-fig-tools", 4)]),
    dbc.Row([_card("Engagement per tool (unique visitors)", "adm-fig-funnel", 6),
              _card("When people use the site", "adm-fig-heat", 6)]),
    dbc.Row([_card("Most used controls", "adm-fig-actions", 12)]),
    dbc.Row([_card("Where visitors come from (referrer)", "adm-fig-refs", 4),
             _card("Countries", "adm-fig-country", 4),
             _card("Devices / browsers", "adm-fig-ua", 4)]),
    dbc.Card([
        dbc.CardHeader("PV-Copilot analyses — one row per analysis run",
                       style={"fontWeight": 600, "background": "#F6F5FC"}),
        dbc.CardBody([
            dcc.Store(id="adm-cp-all"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="adm-cp-f-mode", multi=True, placeholder="Mode"), md=2),
                dbc.Col(dcc.Dropdown(id="adm-cp-f-data", multi=True, placeholder="Data"), md=3),
                dbc.Col(dcc.Dropdown(id="adm-cp-f-method", multi=True, placeholder="Method"), md=2),
                dbc.Col(dcc.Dropdown(id="adm-cp-f-status", multi=True, placeholder="Outcome",
                                     options=[{"label": "Completed ✓", "value": "done"},
                                              {"label": "Not completed", "value": "open"},
                                              {"label": "Had errors", "value": "error"}]), md=2),
                dbc.Col(dcc.Dropdown(id="adm-cp-f-visitor", multi=True, placeholder="Visitor"), md=2),
                dbc.Col(dbc.Button("Clear", id="adm-cp-f-clear", color="secondary", outline=True,
                                   n_clicks=0, className="w-100"), md=1),
            ], className="g-2 mb-2"),
            html.Div(id="adm-cp-count", className="text-muted small mb-2"),
            dash_table.DataTable(id="adm-cp-sessions", sort_action="native",
                                 filter_action="native", page_size=10,
                                 row_selectable="multi", selected_row_ids=[],
                                 **_TABLE_STYLE),
            html.Div(id="adm-cp-detail", className="mt-3"),
        ]),
    ], className="shadow-sm mb-4"),
    dbc.Card([dbc.CardHeader("Per-tool summary", style={"fontWeight": 600, "background": "#F6F5FC"}),
              dbc.CardBody(dash_table.DataTable(id="adm-summary", sort_action="native",
                                                **_TABLE_STYLE))],
             className="shadow-sm mb-4"),
    dbc.Card([dbc.CardHeader("Errors (5xx crashes + errors shown to users)", style={"fontWeight": 600, "background": "#F6F5FC"}),
              dbc.CardBody(dash_table.DataTable(id="adm-errors", sort_action="native",
                                                page_size=10, **_TABLE_STYLE))],
             className="shadow-sm mb-4"),
    dbc.Card([
        dbc.CardHeader("Team / internal visitors", style={"fontWeight": 600, "background": "#F6F5FC"}),
        dbc.CardBody([
            html.P(["Traffic from these browsers or networks is treated as testing by the team. ",
                    "Signing in here marks the current browser automatically. To mark another ",
                    "device (phone, second laptop, a colleague), sign in once on it, or open the ",
                    "site there and add it below."], className="text-muted small"),
            html.Div(id="adm-this-browser", className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Input(id="adm-mark-label", placeholder="Label, e.g. Baojie MacBook"), md=4),
                dbc.Col(dbc.Checkbox(id="adm-mark-ip", value=False,
                                     label="also mark my current network (IP hash) — not for shared LBNL Wi-Fi"),
                        md=5, className="pt-2"),
                dbc.Col(dbc.Button("Mark this browser as internal", id="adm-mark-btn",
                                   color="primary", outline=True, n_clicks=0), md=3),
            ], className="g-2 mb-3"),
            dash_table.DataTable(id="adm-internal", row_deletable=True,
                                 columns=[{"name": c, "id": c} for c in
                                          ["kind", "id", "label", "added"]],
                                 **_TABLE_STYLE),
            html.Div("Delete a row (✕) to stop treating it as internal.",
                     className="text-muted small mt-1"),
        ]),
    ], className="shadow-sm mb-4"),
    dbc.Card([dbc.CardHeader("Event log (latest first, up to 5,000 rows; download for all)",
                             style={"fontWeight": 600, "background": "#F6F5FC"}),
              dbc.CardBody(dash_table.DataTable(id="adm-events", sort_action="native",
                                                filter_action="native", page_size=20,
                                                **_TABLE_STYLE))],
             className="shadow-sm mb-4"),
], id="adm-dash", style={"display": "none"})

layout = dbc.Container([
    dcc.Store(id="adm-token", storage_type="session"),
    dcc.Store(id="adm-internal-ver", data=0),
    dcc.Download(id="adm-download"),
    html.Div(_login, id="adm-login"),
    _dash,
], fluid=False)   # Bootstrap .container widths = same edges as the site header


def get_layout():
    return layout


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("adm-pw", "type"), Output("adm-pw-toggle", "children"),
    Input("adm-pw-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def _toggle_pw_visibility(n):
    shown = bool(n) and n % 2 == 1
    return ("text", "Hide") if shown else ("password", "Show")


@app.callback(
    Output("adm-token", "data"), Output("adm-login-msg", "children"), Output("adm-pw", "value"),
    Input("adm-login-btn", "n_clicks"), Input("adm-pw", "n_submit"), Input("adm-logout", "n_clicks"),
    State("adm-pw", "value"),
    prevent_initial_call=True,
)
def _login_cb(_n, _s, _out, pw):
    if ctx.triggered_id == "adm-logout":
        return None, "", ""
    if not _pw_hash():
        return no_update, "Admin mode is not configured (ADMIN_PASSWORD_HASH is not set).", ""
    if _check_password(pw):
        _flag_internal_cookie()
        vid, _ip = usage_log.current_visitor()
        if vid and vid not in usage_report.load_internal().get("vids", {}):
            try:
                usage_report.mark_internal(
                    vid=vid, label=f"admin sign-in {time.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"admin: could not mark browser internal: {e}")
        return _token(), "", ""
    time.sleep(1.0)  # slow down guessing
    return no_update, "Wrong password.", ""


@app.callback(
    Output("adm-login", "style"), Output("adm-dash", "style"),
    Input("adm-token", "data"),
)
def _gate(tok):
    if _authed(tok):
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("adm-kpis", "children"),
    Output("adm-fig-daily", "figure"), Output("adm-fig-tools", "figure"),
    Output("adm-fig-funnel", "figure"), Output("adm-fig-heat", "figure"),
    Output("adm-fig-actions", "figure"), Output("adm-fig-refs", "figure"),
    Output("adm-fig-country", "figure"),
    Output("adm-cp-all", "data"),
    Output("adm-fig-ua", "figure"),
    Output("adm-summary", "data"), Output("adm-summary", "columns"),
    Output("adm-errors", "data"), Output("adm-errors", "columns"),
    Output("adm-events", "data"), Output("adm-events", "columns"),
    Output("adm-tools", "options"), Output("adm-source", "children"),
    Input("adm-token", "data"), Input("adm-range", "value"),
    Input("adm-tools", "value"), Input("adm-refresh", "n_clicks"),
    Input("adm-flags", "value"), Input("adm-internal-ver", "data"),
)
def _update(tok, days, tools, _r, flags, _v):
    if not _authed(tok):
        raise PreventUpdate
    full = _load(force=ctx.triggered_id == "adm-refresh")
    in_range = _filter(full, days, tools)
    df = _filter(full, days, tools, flags)
    n_hidden = len(in_range) - len(df)
    summary = usage_report.per_tool_summary(df)
    cp = usage_report.copilot_runs(df)
    cp = cp.astype(object).where(cp.notna(), None)

    err = df[df["is_error"]]
    if err.empty:
        err_t = pd.DataFrame(columns=["tool", "trigger", "status", "err", "count", "last_seen"])
    else:
        err_t = (err.groupby(["tool", "trigger", "status", "err"], dropna=False)
                    .agg(count=("ts", "size"), last_seen=("ts_local", "max")).reset_index()
                    .sort_values("count", ascending=False))
        err_t["last_seen"] = err_t["last_seen"].dt.strftime("%Y-%m-%d %H:%M")

    ev = df.sort_values("ts", ascending=False).head(5000).copy()
    ev["time (PT)"] = ev["ts_local"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ev["visitor"] = ev["who"].astype(str).str[:8]
    ev["team"] = ev["internal"].map({True: "yes", False: ""})
    ev_cols = ["time (PT)", "tool", "event", "trigger", "status", "err", "ms", "kb",
               "ua", "ref", "country", "net", "team", "visitor"]
    ev = ev[ev_cols]

    def cols(d):
        return [{"name": c, "id": c} for c in d.columns]

    opts = sorted(full["tool"].dropna().unique()) if not full.empty else []
    span = (f"{len(df):,} events" + (f" ({n_hidden:,} team/LBNL events hidden)" if n_hidden else "")
            + f" · source: {usage_log.storage_location()}")
    if _cache.get("err"):
        span += f" · load error: {_cache['err']}"

    return (
        _kpis(df), _fig_daily(df), _fig_tools(summary), _fig_funnel(df), _fig_heatmap(df),
        _fig_actions(df), _fig_refs(df), _fig_countries(df),
        cp.assign(id=cp["run"]).to_dict("records"),
        _fig_counts(df["ua"], "Events", PALETTE[3]),
        summary.round(1).to_dict("records"), cols(summary),
        err_t.to_dict("records"), cols(err_t),
        ev.astype(object).where(ev.notna(), None).to_dict("records"), cols(ev),
        opts, span,
    )


_CP_COLS = [("start (PT)", "start (PT)"), ("visitor", "visitor"), ("#", "#"),
            ("mode", "mode"), ("data", "data"), ("upload_kb", "upload KB"), ("rows", "rows"),
            ("span_yr", "span yr"), ("method", "method"), ("kept_%", "kept %"),
            ("reached", "reached"), ("wall_s", "time to result s"), ("run_s", "compute s"),
            ("ai_code", "AI coding"), ("ai_diag", "AI diagnostic"), ("ai_chat", "AI chat"),
            ("errors", "errors")]


def _opts(rows, key, split=False):
    vals = set()
    for r in rows:
        v = r.get(key)
        if v in (None, "", "—"):
            continue
        vals.update([x.strip() for x in str(v).split(",")] if split else [str(v)])
    return [{"label": v, "value": v} for v in sorted(vals)]


@app.callback(
    Output("adm-cp-f-mode", "value"), Output("adm-cp-f-data", "value"),
    Output("adm-cp-f-method", "value"), Output("adm-cp-f-status", "value"),
    Output("adm-cp-f-visitor", "value"),
    Input("adm-cp-f-clear", "n_clicks"),
    prevent_initial_call=True,
)
def _cp_clear(_n):
    return None, None, None, None, None


@app.callback(
    Output("adm-cp-sessions", "data"), Output("adm-cp-sessions", "columns"),
    Output("adm-cp-sessions", "selected_row_ids"), Output("adm-cp-count", "children"),
    Output("adm-cp-f-mode", "options"), Output("adm-cp-f-data", "options"),
    Output("adm-cp-f-method", "options"), Output("adm-cp-f-visitor", "options"),
    Input("adm-cp-all", "data"),
    Input("adm-cp-f-mode", "value"), Input("adm-cp-f-data", "value"),
    Input("adm-cp-f-method", "value"), Input("adm-cp-f-status", "value"),
    Input("adm-cp-f-visitor", "value"),
)
def _cp_filter(rows, modes, datas, methods, status, visitors):
    rows = rows or []
    out = []
    for r in rows:
        if modes and r.get("mode") not in modes:
            continue
        if datas and r.get("data") not in datas:
            continue
        if methods and not set(m.strip() for m in str(r.get("method")).split(",")) & set(methods):
            continue
        if visitors and r.get("visitor") not in visitors:
            continue
        if status:
            flags = {"done" if r.get("done") else "open"} | ({"error"} if r.get("errors") else set())
            if not flags & set(status):
                continue
        out.append(r)
    msg = (f"{len(out)} of {len(rows)} analyses · select one for its timeline, "
           "two or more to compare run times")
    return (out, [{"name": n, "id": k} for k, n in _CP_COLS], [], msg,
            _opts(rows, "mode"), _opts(rows, "data"), _opts(rows, "method", split=True),
            _opts(rows, "visitor"))


def _run_label(r):
    data = str(r.get("data") or "")
    data = data if len(data) <= 26 else data[:25] + "…"
    return (f"{r['start (PT)'][5:]} · #{r['#']} {r.get('mode')} · "
            f"{r.get('method')} · {data}")



_SEG_OF_LANE = {"Data": ("Data load / prescreen", STAGE_COLOR["data"]),
                "Filter": ("Filter", STAGE_COLOR["filter"]),
                "Degradation": ("Degradation", STAGE_COLOR["degradation"]),
                "AI coding": ("AI coding", STAGE_COLOR["ai_code"]),
                "AI diagnostic": ("AI diagnostic", STAGE_COLOR["ai_diag"]),
                "AI help chat": ("AI help chat", STAGE_COLOR["ai_chat"])}
_AI_LANES = {"AI coding", "AI diagnostic", "AI help chat"}
_PVPRO_SEG = ("PVPRO fit (background)", STAGE_COLOR["pvpro"])


def _run_steps(df, r):
    """Compute steps of one run in order: (legend name, colour, seconds, hover)."""
    t = usage_report.run_timeline(df, r["run"])
    out, seen = [], {}
    for _, ev in t.iterrows():
        if ev["x_stage"] == "pvpro_done" and pd.notna(ev["x_run_s"]):
            name, color = _PVPRO_SEG
            d = float(ev["x_run_s"])
        elif ev["lane"] in _AI_LANES:
            name, color = _SEG_OF_LANE[ev["lane"]]
            if str(ev["x_stage"]).startswith("ai_") and pd.notna(ev["x_run_s"]):
                d = float(ev["x_run_s"])
            elif ("ai-diagnostic-btn" in str(ev["trigger"]) and pd.isna(ev["x_stage"])
                  and pd.notna(ev["ms"]) and ev["ms"] >= 50):
                d = float(ev["ms"]) / 1000   # older logs: diagnostic ran in the click
            else:
                continue      # a click that only starts background AI work
        elif ev["lane"] in _SEG_OF_LANE and pd.notna(ev["ms"]) and ev["ms"] >= 50:
            name, color = _SEG_OF_LANE[ev["lane"]]
            d = float(ev["ms"]) / 1000
        else:
            continue
        seen[name] = seen.get(name, 0) + 1
        nth = f" (run {seen[name]})" if seen[name] > 1 else ""
        out.append((name, color, d, f"{ev['label']}{nth}<br>{d:.2f} s"))
    return out


def _fig_steps(df, sel):
    """Compute time per step, back to back (no waiting). Each execution of a
    step is its own segment, so a filter re-run shows up twice."""
    fig = go.Figure()
    traces, labels, totals = {}, [], []
    for r in sel:
        lab = _run_label(r)
        labels.append(lab)
        x0 = 0.0
        for name, color, d, h in _run_steps(df, r):
            tr = traces.setdefault(name, {"y": [], "x": [], "base": [], "h": [], "c": color})
            tr["y"].append(lab)
            tr["x"].append(d)
            tr["base"].append(x0)
            tr["h"].append(h)
            x0 += d
        totals.append((lab, x0))
    order = ["Data load / prescreen", "Filter", "Degradation", _PVPRO_SEG[0],
             "AI coding", "AI diagnostic", "AI help chat"]
    for name in order:
        if name not in traces:
            continue
        tr = traces[name]
        fig.add_trace(go.Bar(
            y=tr["y"], x=tr["x"], base=tr["base"], orientation="h", name=name,
            marker=dict(color=tr["c"], line=dict(width=1.5, color="white")),
            text=[f"{v:.1f}" for v in tr["x"]], textposition="inside",
            insidetextanchor="middle", textangle=0, textfont=dict(color="white"),
            customdata=tr["h"], hovertemplate="%{customdata}<extra>" + name + "</extra>"))
    if totals:
        fig.add_trace(go.Scatter(x=[v for _, v in totals], y=[l for l, _ in totals],
                                 mode="text", text=[f"  {v:.1f} s" for _, v in totals],
                                 textposition="middle right", showlegend=False,
                                 hoverinfo="skip", cliponaxis=False,
                                 textfont=dict(size=13, color="#1F2A5C")))
    fig.update_layout(barmode="overlay", legend_traceorder="normal",
                      uniformtext=dict(minsize=11, mode="hide"))
    fig.update_xaxes(title="Compute time (s) — steps back to back, waiting excluded")
    fig.update_yaxes(title=None, autorange="reversed", categoryorder="array",
                     categoryarray=labels)
    _style(fig, height=150 + 50 * len(labels))
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      margin=dict(r=80))
    return fig


def _realtime_block(ids):
    """Collapsed by default; the real-time chart is only drawn when opened
    (a plotly chart drawn while hidden comes out at the wrong width)."""
    return html.Div([
        dbc.Button("▸ Real-time view (includes waiting between steps)",
                   id="adm-cp-rt-btn", color="link", size="sm", n_clicks=0,
                   className="px-0"),
        dcc.Store(id="adm-cp-rt-ids", data=ids),
        html.Div(id="adm-cp-rt"),
    ], className="mt-1")


def _fig_gantt(df, sel):
    """One row per analysis on a shared clock (0 = that analysis started).
    Each bar is real computing, drawn where it happened; the gaps between
    bars are waiting / the user reading or clicking. Small grey dots are
    UI clicks."""
    segs = {}      # legend name -> dict(y, base, x, color, hover)
    ui = {"x": [], "y": [], "h": []}
    ends = {"x": [], "y": [], "t": []}
    labels = []
    for r in sel:
        t = usage_report.run_timeline(df, r["run"])
        if t.empty:
            continue
        lab = _run_label(r)
        labels.append(lab)
        dur = t["ms"].fillna(0) / 1000
        start = (t["ts"] - pd.to_timedelta(dur, unit="s"))
        # 0 = the analysis' data step starting (same clock as "result at");
        # an example pick / upload just before it shows at negative time
        main = t["x_stage"].isin(["load", "prescreen"])
        t0 = start[main].iloc[0] if main.any() else start.min()
        for (_, ev), st0, d in zip(t.iterrows(), start, dur):
            rel = (st0 - t0).total_seconds()
            if ((ev["x_stage"] == "pvpro_done" or str(ev["x_stage"]).startswith("ai_"))
                    and pd.notna(ev["x_run_s"])):
                name, color = (_PVPRO_SEG if ev["x_stage"] == "pvpro_done"
                               else _SEG_OF_LANE[ev["lane"]])
                rs = float(ev["x_run_s"])
                rel, d = (ev["ts"] - t0).total_seconds() - rs, rs
            elif ev["lane"] in _SEG_OF_LANE and d >= 0.05:
                name, color = _SEG_OF_LANE[ev["lane"]]
            else:
                ui["x"].append((ev["ts"] - t0).total_seconds())
                ui["y"].append(lab)
                ui["h"].append(ev["label"])
                continue
            sg = segs.setdefault(name, {"y": [], "base": [], "x": [], "color": color, "h": []})
            sg["y"].append(lab)
            sg["base"].append(rel)
            sg["x"].append(d)
            sg["h"].append(f"{ev['label']}<br>{rel:.1f} → {rel + d:.1f} s ({d:.1f} s)")
        tot = float(r.get("run_s") or 0)
        end = (t["ts"].max() - t0).total_seconds()
        ends["x"].append(end)
        ends["y"].append(lab)
        ends["t"].append(f"  compute {tot:.1f} s"
                         + (f" · result at {r['wall_s']:.0f} s" if r.get("wall_s") is not None else ""))
    fig = go.Figure()
    order = ["Data load / prescreen", "Filter", "Degradation", _PVPRO_SEG[0],
             "AI coding", "AI diagnostic", "AI help chat"]
    for name in order:
        if name not in segs:
            continue
        sg = segs[name]
        fig.add_trace(go.Bar(
            y=sg["y"], x=sg["x"], base=sg["base"], orientation="h", name=name,
            marker=dict(color=sg["color"], line=dict(width=1, color="white")),
            text=[f"{v:.1f}" for v in sg["x"]], textposition="inside",
            insidetextanchor="middle", textangle=0, textfont=dict(color="white"),
            customdata=sg["h"], hovertemplate="%{customdata}<extra>" + name + "</extra>"))
    if ui["x"]:
        fig.add_trace(go.Scatter(x=ui["x"], y=ui["y"], mode="markers", name="UI click",
                                 marker=dict(size=7, color=STAGE_COLOR["ui"]),
                                 customdata=ui["h"],
                                 hovertemplate="+%{x:.0f} s · %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ends["x"], y=ends["y"], mode="text", text=ends["t"],
                             textposition="middle right", showlegend=False, hoverinfo="skip",
                             textfont=dict(size=12, color="#1F2A5C"), cliponaxis=False))
    fig.update_layout(barmode="overlay", legend_traceorder="normal",
                      uniformtext=dict(minsize=11, mode="hide"))
    fig.update_xaxes(title="Seconds since the analysis started (0 = data step began)",
                     zeroline=True, zerolinecolor="#C9CDE0")
    fig.update_yaxes(title=None, autorange="reversed", categoryorder="array",
                     categoryarray=labels)
    _style(fig, height=150 + 50 * len(labels))
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      margin=dict(r=190))
    return fig


def _compare_table(sel):
    fields = [("Started", "start (PT)"), ("Visitor", "visitor"), ("Mode", "mode"),
              ("Method", "method"), ("Data", "data"), ("Upload KB", "upload_kb"),
              ("Rows", "rows"), ("Span (yr)", "span_yr"), ("Kept after filter %", "kept_%"),
              ("Reached", "reached"), ("Time to result (s)", "wall_s"),
              ("Compute total (s)", "run_s"), ("· data load (s)", "load_s"),
              ("· filter (s)", "filter_s"), ("· degradation (s)", "degradation_s"),
              ("· PVPRO fit (s)", "pvpro_s"), ("· AI coding (s)", "ai_code_s"),
              ("· AI diagnostic (s)", "ai_diag_s"), ("· AI help chat (s)", "ai_chat_s"),
              ("AI coding (times)", "ai_code"), ("AI diagnostic (times)", "ai_diag"),
              ("AI help chat (times)", "ai_chat"),
              ("Errors", "errors")]
    names = [f"#{r['#']} {r['start (PT)'][5:]}" for r in sel]
    data = [{"metric": lab, **{n: r.get(k) for n, r in zip(names, sel)}} for lab, k in fields]
    return dash_table.DataTable(
        data=data, columns=[{"name": "", "id": "metric"}] + [{"name": n, "id": n} for n in names],
        style_cell_conditional=[{"if": {"column_id": "metric"}, "fontWeight": 600,
                                 "width": "180px"}],
        style_data_conditional=[{"if": {"filter_query": '{metric} contains "Time to result"'},
                                 "backgroundColor": "#EEF0FB", "fontWeight": 700}],
        **_TABLE_STYLE)


@app.callback(
    Output("adm-cp-detail", "children"),
    Input("adm-cp-sessions", "selected_row_ids"),
    State("adm-cp-all", "data"),
    State("adm-token", "data"),
    prevent_initial_call=True,
)
def _session_detail(sel_ids, rows, tok):
    if not _authed(tok) or not sel_ids or not rows:
        return html.Div("Select an analysis above to see what happened in it; "
                        "select two or more to compare them.", className="text-muted small")
    by_id = {r["id"]: r for r in rows}
    sel = [by_id[i] for i in sel_ids if i in by_id]
    if len(sel) >= 2:
        sel = sorted(sel, key=lambda r: (r["start (PT)"], r["#"]))
        return html.Div([
            html.H6(f"Comparing {len(sel)} analyses — compute time per step", className="mt-2"),
            html.Div("Each segment is one step execution (a re-run filter or degradation "
                     "shows up again); waiting between steps is left out. Hover for details.",
                     className="text-muted small"),
            dcc.Graph(figure=_fig_steps(_load(), sel), config={"displayModeBar": False}),
            _realtime_block([r["id"] for r in sel]),
            html.H6("Side by side", className="mt-3"),
            _compare_table(sel),
        ])
    row = sel[0]
    t = usage_report.run_timeline(_load(), row["run"])
    if t.empty:
        return html.Div("No events for this analysis.", className="text-muted small")
    t0 = t["ts"].iloc[0]
    steps = pd.DataFrame({
        "time (PT)": t["ts_local"].dt.strftime("%H:%M:%S"),
        "+s": ((t["ts"] - t0).dt.total_seconds()).round(0).astype(int),
        "lane": t["lane"],
        "what": t["label"],
        "server s": (t["ms"] / 1000).round(2),
        "error": t["err"].where(t["err"].notna(),
                                t["status"].where(t["status"] >= 500)).astype(object),
    })
    steps = steps.astype(object).where(steps.notna(), None)
    up = f"{row['upload_kb']:,} KB" if row.get("upload_kb") else ""
    span = f"{row['span_yr']} yr" if row.get("span_yr") is not None else ""
    wall = f"{row['wall_s']} s" if row.get("wall_s") is not None else ""
    return html.Div([
        html.Div([
            _chip("Started", row["start (PT)"]),
            _chip("Analysis", f"#{row['#']} of this visit"),
            _chip("Mode", row["mode"]), _chip("Method", row["method"]),
            _chip("Data", row["data"]), _chip("Upload", up),
            _chip("Rows", f"{row['rows']:,}" if row.get("rows") else ""), _chip("Span", span),
            _chip("Kept after filter", f"{row['kept_%']}%" if row.get("kept_%") is not None else ""),
            _chip("Reached", row["reached"]),
            _chip("Time to result", wall), _chip("Compute", f"{row['run_s']} s"),
            _chip("AI coding", f"{row['ai_code']}× · {row['ai_code_s']} s" if row.get("ai_code") else "no"),
            _chip("AI diagnostic", f"{row['ai_diag']}× · {row['ai_diag_s']} s" if row.get("ai_diag") else "no"),
            _chip("AI help chat", f"{row['ai_chat']}× · {row['ai_chat_s']} s" if row.get("ai_chat") else "no"),
            _chip("Errors", str(row["errors"])),
        ], className="adm-chips"),
        dcc.Graph(figure=_fig_steps(_load(), [row]), config={"displayModeBar": False}),
        _realtime_block([row["id"]]),
        dash_table.DataTable(data=steps.to_dict("records"),
                             columns=[{"name": c, "id": c} for c in steps.columns],
                             page_size=15, **_TABLE_STYLE),
    ])


@app.callback(
    Output("adm-cp-rt", "children"), Output("adm-cp-rt-btn", "children"),
    Input("adm-cp-rt-btn", "n_clicks"),
    State("adm-cp-rt-ids", "data"), State("adm-cp-all", "data"), State("adm-token", "data"),
    prevent_initial_call=True,
)
def _cp_realtime(n, ids, rows, tok):
    closed = "▸ Real-time view (includes waiting between steps)"
    if not _authed(tok) or not n or n % 2 == 0 or not ids:
        return None, closed
    by_id = {r["id"]: r for r in rows or []}
    sel = sorted([by_id[i] for i in ids if i in by_id], key=lambda r: (r["start (PT)"], r["#"]))
    return html.Div([
        html.Div("Bars drawn when they happened (0 = data step began); gaps are waiting "
                 "or the user reading / clicking. Right: total compute and when the result "
                 "appeared.", className="text-muted small"),
        dcc.Graph(figure=_fig_gantt(_load(), sel), config={"displayModeBar": False}),
    ]), "▾ Real-time view (includes waiting between steps)"


@app.callback(
    Output("adm-download", "data"),
    Input("adm-dl-events", "n_clicks"), Input("adm-dl-summary", "n_clicks"),
    Input("adm-dl-daily", "n_clicks"),
    State("adm-token", "data"), State("adm-range", "value"), State("adm-tools", "value"),
    State("adm-flags", "value"),
    prevent_initial_call=True,
)
def _download(_e, _s, _d, tok, days, tools, flags):
    if not _authed(tok) or not ctx.triggered or not ctx.triggered[0]["value"]:
        raise PreventUpdate
    df = _filter(_load(), days, tools, flags)
    stamp = pd.Timestamp.now(tz=usage_report.TZ).strftime("%Y%m%d")
    if ctx.triggered_id == "adm-dl-summary":
        return dcc.send_data_frame(usage_report.per_tool_summary(df).to_csv,
                                   f"pvtools_usage_summary_{stamp}.csv", index=False)
    if ctx.triggered_id == "adm-dl-daily":
        d = (df.assign(day=df["ts_local"].dt.date).groupby(["day", "tool"])["who"].nunique()
               .unstack(fill_value=0)) if not df.empty else pd.DataFrame()
        return dcc.send_data_frame(d.to_csv, f"pvtools_daily_visitors_{stamp}.csv")
    out = df.drop(columns=["who", "session", "is_upload", "lbl_net", "ts"], errors="ignore")
    if not out.empty:
        out = out.rename(columns={"ts_local": "time_pt"})
        out["time_pt"] = out["time_pt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return dcc.send_data_frame(out.to_csv, f"pvtools_usage_events_{stamp}.csv", index=False)


# ---------------------------------------------------------------------------
# Team / internal visitors
# ---------------------------------------------------------------------------
def _internal_rows():
    reg = usage_report.load_internal()
    rows = [{"kind": "browser", "id": k, **v} for k, v in reg.get("vids", {}).items()]
    rows += [{"kind": "network", "id": k, **v} for k, v in reg.get("ips", {}).items()]
    return sorted(rows, key=lambda r: r.get("added", ""), reverse=True)


@app.callback(
    Output("adm-internal", "data"), Output("adm-this-browser", "children"),
    Output("adm-internal-ver", "data"),
    Input("adm-token", "data"), Input("adm-mark-btn", "n_clicks"),
    Input("adm-internal", "data_timestamp"),          # fires only on user edits (row delete)
    State("adm-internal", "data"), State("adm-mark-label", "value"),
    State("adm-mark-ip", "value"), State("adm-internal-ver", "data"),
)
def _internal(tok, _n, _ts, table, label, with_ip, ver):
    """One callback for view + mark + delete, so there is no dependency cycle
    (adm-internal-ver only feeds the main dashboard update)."""
    if not _authed(tok):
        raise PreventUpdate
    changed = False
    trig = ctx.triggered_id
    if trig == "adm-mark-btn":
        vid, ip = usage_log.current_visitor()
        usage_report.mark_internal(vid=vid, ip_hash=ip if with_ip else None,
                                   label=(label or "").strip() or "marked from admin page")
        _flag_internal_cookie()
        changed = True
    elif trig == "adm-internal" and table is not None:
        # a row was deleted in the table -> drop it from the registry
        keep = {(r["kind"], r["id"]) for r in table}
        reg = usage_report.load_internal(max_age_s=0)
        gone_v = [k for k in reg.get("vids", {}) if ("browser", k) not in keep]
        gone_i = [k for k in reg.get("ips", {}) if ("network", k) not in keep]
        for k in gone_v:
            reg["vids"].pop(k, None)
        for k in gone_i:
            reg["ips"].pop(k, None)
        if gone_v or gone_i:
            usage_report.save_internal(reg)
            changed = True

    vid, _ip = usage_log.current_visitor()
    reg = usage_report.load_internal()
    if not vid:
        me = html.Span("This browser has no visitor cookie yet (reload the page).",
                       className="text-warning")
    elif vid in reg.get("vids", {}):
        me = html.Span(["This browser (", html.Code(vid), ") is marked internal ✓"],
                       className="text-success")
    else:
        me = html.Span(["This browser (", html.Code(vid), ") is not marked internal."])
    return _internal_rows(), me, ((ver or 0) + 1) if changed else no_update
