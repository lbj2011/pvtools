import os
import json
import networkx as nx

LEVEL_X = {
    "stressor": 100,
    "mechanism": 400,
    "failure": 700,
    "performance_loss": 1000
}

def split_into_pathways(nodes):
    id_map = {n["id"]: n for n in nodes}
    visited = set()
    pathways = []

    def dfs(node_id, group):
        if node_id in visited:
            return

        visited.add(node_id)
        group.add(node_id)

        node = id_map[node_id]

        # forward
        for child in node["child_ids"]:
            dfs(child, group)

        # reverse
        for n in nodes:
            if node_id in n["child_ids"]:
                dfs(n["id"], group)

    # ✅ THIS MUST BE OUTSIDE dfs
    for n in nodes:
        if n["id"] not in visited:
            group = set()
            dfs(n["id"], group)
            pathways.append(group)

    # ✅ filter small pathways
    pathways = [p for p in pathways if len(p) >= 2]

    return pathways

def build_elements(json_data, pathway_idx=0):
    """
    Build Cytoscape elements for a selected pathway
    """
    nodes = json_data["pathways_graph"]

    # split into pathways
    pathways = split_into_pathways(nodes)

    if not pathways:
        return [], 0

    # safety check
    if pathway_idx >= len(pathways):
        return [], len(pathways)

    selected_ids = pathways[pathway_idx]

    elements = []

    # horizontal layout by category
    category_x = {
        "stressor": 0,
        "mechanism": 1,
        "failure": 2,
        "performance_impact": 3
    }

    # track vertical spacing per category (better layout)
    y_counter = {
        "stressor": 0,
        "mechanism": 0,
        "failure": 0,
        "performance_impact": 0
    }

    for n in nodes:
        if n["id"] not in selected_ids:
            continue

        cat = n["category"]

        # tooltip text
        tooltip = (
            f"{n['brief_term']} ({cat})\n"
            f"{n.get('detailed_term', '')}\n"
            f"{n.get('reasoning', '')}"
        )

        # position
        x = category_x.get(cat, 0) * 200
        y = y_counter[cat] * 80
        y_counter[cat] += 1

        elements.append({
            "data": {
                "id": str(n["id"]),
                "label": n["brief_term"],
                "category": cat,
                "title": tooltip
            },
            "position": {"x": x, "y": y}
        })

        # edges
        for child in n["child_ids"]:
            if child in selected_ids:
                elements.append({
                    "data": {
                        "source": str(n["id"]),
                        "target": str(child)
                    }
                })

    return elements, len(pathways)

## Common pathway builder
def load_nodes(folder):
    nodes = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(root, f)
                data = json.load(open(path))
                if "id" in data:
                    nodes[data["id"]] = data
    return nodes

def build_graph(nodes):
    G = nx.DiGraph()

    for node_id, node in nodes.items():
        G.add_node(node_id, label=node.get("title", node_id), type=node["node_type"])

        # parent edges
        for p in node.get("parent_nodes", []):
            G.add_edge(p["id"], node_id)

        # child edges
        for c in node.get("child_nodes", []):
            G.add_edge(node_id, c["id"])

    return G

def simplify_label(node):
    if "display_name" in node:
        return node["display_name"]

    title = node.get("title", "")
    return title.split(" in ")[0][:40]  # simple fallback


def to_cytoscape_positioned(nodes, G):
    elements = []
    y_offsets = {}   # initialised lazily so unknown node types don't crash

    spacing_y = 80

    for node_id, node in nodes.items():
        node_type = node["node_type"]
        x = LEVEL_X.get(node_type, 1200)

        if node_type not in y_offsets:
            y_offsets[node_type] = 0
        y = y_offsets[node_type]
        y_offsets[node_type] += spacing_y

        elements.append({
            "data": {
                "id": node_id,
                "label": simplify_label(node),
                "type": node_type
            },
            "position": {"x": x, "y": y}
        })

    for source, target in G.edges():
        elements.append({
            "data": {"source": source, "target": target}
        })

    return elements