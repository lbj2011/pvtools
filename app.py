import os

from dash import dash, html, dcc
import dash_bootstrap_components as dbc

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

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{
                    'name': "google-site-verification",
                    'content': "S1RjgJU6ZoVdko93JeLNEnn5viVxN1cXL2me3LB9J5I",
                }],
                )

server = app.server
server.secret_key = os.environ.get('secret_key', 'secret')

# Allow to set callbacks before setting layout:
app.config['suppress_callback_exceptions']=True
app.title = 'PVTOOLS'

# For google analytics to work:
app.scripts.config.serve_locally = True

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

# Add meta tag for google search



# app.scripts.append_script({
#     'external_url': 'https://cdn.jsdelivr.net/gh/lppier/lppier.github.io/async_src.js'
# })
# app.scripts.append_script({
#     'external_url': 'https://cdn.jsdelivr.net/gh/lppier/lppier.github.io/gtag.js'
# })

#
# app.config.suppress_callback_exceptions = True
# # app.config.suppress_callback_exceptions = False
# app.css.config.serve_locally = False
# app.scripts.config.serve_locally = False
#


if __name__ == '__main__':
    app.run_server(debug=True)