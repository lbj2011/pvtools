"""Privacy & data use page (/privacy). Keep in sync with utils/usage_log.py,
pages/pvcopilot.py (session cache TTL) and the LLM clients used by each tool."""
from dash import html
import dash_bootstrap_components as dbc


def _section(title, *children):
    return html.Div([html.H4(title, className="mt-4"), *children])


layout = dbc.Container([
    html.H2("Privacy & data use"),
    html.P("PVTOOLS does not require an account and does not ask for your name or "
           "email. This page explains what happens to the data you use with the tools "
           "and what we record about how the site is used."),

    _section("Data and images you upload",
        html.Ul([
            html.Li("Uploaded files and images are used only to run the analysis you "
                    "request. We do not keep them, publish them, or use them to train models."),
            html.Li("PV-Copilot keeps a temporary copy of your uploaded dataset on the "
                    "server so that a page refresh does not lose your session. It is "
                    "deleted automatically after at most 2 hours, and whenever the "
                    "server restarts."),
            html.Li("PV Image Analysis keeps the image in your browser only; it is not "
                    "written to the server."),
        ])),

    _section("AI features and third-party services",
        html.P("Some features send content to external AI providers to generate a response:"),
        html.Ul([
            html.Li("PV Image Analysis: the image you analyze is sent through OpenRouter to "
                    "the model you choose (OpenAI, Google or Anthropic)."),
            html.Li("PV-Copilot: the column names of your file (not the data values) are "
                    "sent to OpenAI to identify the variables. If you use the AI "
                    "assistant, your question and a summary of your analysis results "
                    "are sent to OpenAI."),
            html.Li("Field Degradation AI chat: your question is sent to OpenAI."),
        ]),
        html.P("These providers process the content under their own terms. Please do not "
               "upload data or images you are not allowed to share with them.")),

    _section("Usage statistics",
        html.P("To report how the tools are used to our funders (U.S. DOE and DuraMAT) "
               "and to find errors, we record anonymous usage events:"),
        html.Ul([
            html.Li("which tool page was opened and which buttons or controls were used "
                    "(for example \"Run the analysis\" or \"upload\");"),
            html.Li("time, response time and whether an error occurred;"),
            html.Li("the size of an upload (not its content or file name);"),
            html.Li("device type and browser family, and the website you came from "
                    "(domain only);"),
            html.Li("a random identifier stored in a cookie, and a one-way hash of your "
                    "IP address, so we can count returning visitors. We do not store "
                    "your IP address itself."),
        ]),
        html.P("We never record the data you upload, your images, file names, the "
               "questions you ask the AI or its answers. Usage records are stored "
               "privately at LBNL and are reported only as aggregate numbers.")),

    _section("Google Analytics",
        html.P("We also use Google Analytics to understand overall traffic (pages viewed, "
               "approximate location, device type). Google Analytics sets its own cookies "
               "and its data is processed by Google under Google's privacy policy. It does "
               "not receive your uploaded data, images or AI questions. You can opt out "
               "with Google's browser add-on or by blocking cookies for this site.")),

    _section("Cookies",
        html.P("PVTOOLS sets a random identifier cookie used for the usage statistics "
               "above and to protect the site from repeated automated reloads; it "
               "contains no personal information. Google Analytics sets its own cookies "
               "(names starting with _ga).")),

    _section("Questions",
        html.P(["Questions or requests about this page: open an issue on ",
                html.A("GitHub", href="https://github.com/lbj2011/pvtools/issues",
                       target="_blank"), "."])),
], style={"maxWidth": "860px"}, className="mb-5")


def get_layout():
    return layout
