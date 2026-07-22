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
    # PV Copilot polls background jobs frequently.  Keep the browser tab title
    # stable instead of flashing Dash's default "Updating..." on every poll.
    update_title=None,
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
    return "User-agent: *\nDisallow:", 200, {"Content-Type": "text/plain"}


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
        <!-- KaTeX (replaces MathJax for reliable rendering inside nested <details>) -->
        <link rel="stylesheet"
              href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
              integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+"
              crossorigin="anonymous">
        <script defer
                src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
                integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg"
                crossorigin="anonymous"></script>
        <script defer
                src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
                integrity="sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk"
                crossorigin="anonymous"></script>
        <style>
            /* Prevent Bootstrap number input validation red color */
            input[type=number] {{ color: #000 !important; }}

            /* Kill purple color on dcc.RadioItems labels (Bootstrap link color bleeds in) */
            .form-check-label,
            .form-check-label:hover,
            .form-check-label:focus,
            .form-check-label b,
            .form-check-label * {{
                color: inherit !important;
                text-decoration: none !important;
            }}
            /* Remove any Bootstrap active/checked highlight on label text */
            .form-check-input:checked ~ .form-check-label,
            .form-check-input:checked ~ .form-check-label * {{
                color: inherit !important;
            }}
            /* Metric radio labels — force no color change on hover using element+class selector */
            label.metric-radio-label,
            label.metric-radio-label:hover,
            label.metric-radio-label:focus,
            label.metric-radio-label:active {{
                color: #212529 !important;
                text-decoration: none !important;
            }}
            label.metric-radio-label b,
            label.metric-radio-label:hover b,
            label.metric-radio-label:hover * {{
                color: inherit !important;
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
        <script>
        /* KaTeX auto-render: runs on initial load + whenever any <details>
           (outer or nested) opens. Uses MutationObserver to also catch
           equations that are inserted dynamically by Dash callbacks.       */
        (function() {{
            function renderMath(root) {{
                if (!window.renderMathInElement) return;
                try {{
                    window.renderMathInElement(root || document.body, {{
                        delimiters: [
                            {{left: "$$", right: "$$", display: true}},
                            {{left: "\\\\(", right: "\\\\)", display: false}},
                            {{left: "\\\\[", right: "\\\\]", display: true}},
                            {{left: "$",  right: "$",  display: false}}
                        ],
                        ignoredClasses: ["mathjax-ignore", "katex-ignore"],
                        throwOnError: false
                    }});
                }} catch (e) {{ console.warn("KaTeX render error:", e); }}
            }}

            function bootKaTeX() {{
                renderMath(document.body);
                /* Re-render when ANY <details> (including nested) opens */
                document.addEventListener("toggle", function(e) {{
                    if (e.target && e.target.tagName === "DETAILS" && e.target.open) {{
                        renderMath(e.target);
                    }}
                }}, true);
                /* Catch Dash callback re-renders that swap DOM nodes */
                var mo = new MutationObserver(function(muts) {{
                    for (var i = 0; i < muts.length; i++) {{
                        var m = muts[i];
                        for (var j = 0; j < m.addedNodes.length; j++) {{
                            var n = m.addedNodes[j];
                            if (n.nodeType === 1) renderMath(n);
                        }}
                    }}
                }});
                mo.observe(document.body, {{ childList: true, subtree: true }});
            }}

            if (window.renderMathInElement) {{
                bootKaTeX();
            }} else {{
                /* KaTeX script has `defer`, wait for it */
                window.addEventListener("DOMContentLoaded", function() {{
                    if (window.renderMathInElement) {{
                        bootKaTeX();
                    }} else {{
                        /* Fallback: poll briefly until auto-render.min.js attaches */
                        var tries = 0;
                        var iv = setInterval(function() {{
                            tries++;
                            if (window.renderMathInElement) {{
                                clearInterval(iv);
                                bootKaTeX();
                            }} else if (tries > 40) {{
                                clearInterval(iv);
                                console.warn("KaTeX auto-render failed to load");
                            }}
                        }}, 100);
                    }}
                }});
            }}
        }})();
        </script>
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
