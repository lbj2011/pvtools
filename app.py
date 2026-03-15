import os
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from flask import request, abort

# ----------------------------
# Create Dash app
# ----------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{
        "name": "google-site-verification",
        "content": "S1RjgJU6ZoVdko93JeLNEnn5viVxN1cXL2me3LB9J5I",
    }],
)

server = app.server
server.secret_key = os.environ.get("secret_key", "secret")

# Allow callbacks before layout
app.config["suppress_callback_exceptions"] = True
app.title = "PVTOOLS"

# Dash scripts local
app.scripts.config.serve_locally = True


# ----------------------------
# Crawler / bot protection
# ----------------------------
BLOCK_PATHS = [
    "wp-login",
    "wp-admin",
    "wp-content",
    "wp-includes",
    "xmlrpc.php",
    "wp_filemanager",
    "phpmyadmin"
]

BAD_AGENTS = [
    "sqlmap",
    "nikto",
    "scanner"
]

@server.before_request
def block_bots():
    path = request.path.lower()
    ua = request.headers.get("User-Agent", "").lower()

    if any(p in path for p in BLOCK_PATHS):
        abort(404)

    if any(b in ua for b in BAD_AGENTS):
        abort(403)


# ----------------------------
# robots.txt
# ----------------------------
@server.route("/robots.txt")
def robots():
    return "User-agent: *\nDisallow: /", 200, {"Content-Type": "text/plain"}


# ----------------------------
# Google Analytics
# ----------------------------
GA_ID = "G-WENJERWTTT"

ga_script = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_ID}');
</script>

<script>
  window.addEventListener('DOMContentLoaded', function() {{
    const oldPushState = history.pushState;
    history.pushState = function() {{
      oldPushState.apply(history, arguments);
      gtag('config', '{GA_ID}', {{
        page_path: window.location.pathname
      }});
    }};
    window.addEventListener('popstate', function() {{
      gtag('config', '{GA_ID}', {{
        page_path: window.location.pathname
      }});
    }});
  }});
</script>
"""


# ----------------------------
# Custom index HTML
# ----------------------------
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        {ga_script}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""


# ----------------------------
# Example layout (replace with yours)
# ----------------------------
app.layout = html.Div([
    html.H1("PVTOOLS"),
    html.P("PV tools dashboard"),
])


# ----------------------------
# Run local
# ----------------------------
if __name__ == "__main__":
    app.run_server(debug=True)