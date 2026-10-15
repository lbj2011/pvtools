"""Smoke tests: the Dash app imports, registers its callbacks, and serves pages.

These catch the most common breakages from a bad merge (syntax errors,
missing imports, duplicate callback outputs) without needing a browser.
"""
import json

import pytest


@pytest.fixture(scope="module")
def client():
    import index  # registers every page's layout + callbacks
    return index.app.server.test_client()


def test_pvcopilot_page_imports():
    import app
    import pages.pvcopilot  # noqa: F401
    assert len(app.app.callback_map) > 50


@pytest.mark.parametrize("path", ["/", "/pv-copilot", "/_dash-layout", "/_dash-dependencies"])
def test_routes_return_200(client, path):
    assert client.get(path).status_code == 200


def test_pvcopilot_layout_renders(client):
    """Fire the URL-router callback for /pv-copilot, as the browser would."""
    payload = {
        "output": "page-content.children",
        "outputs": {"id": "page-content", "property": "children"},
        "inputs": [{"id": "url", "property": "pathname", "value": "/pv-copilot"}],
        "changedPropIds": ["url.pathname"],
    }
    r = client.post("/_dash-update-component", data=json.dumps(payload),
                    content_type="application/json")
    assert r.status_code == 200
    assert "page-content" in r.get_json()["response"]
