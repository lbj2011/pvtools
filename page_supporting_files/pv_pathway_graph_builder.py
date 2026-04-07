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