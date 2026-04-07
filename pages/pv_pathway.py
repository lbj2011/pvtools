from dash import Dash
from page_supporting_files.pv_pathway_data_loader import load_dataframe, build_index_map
from page_supporting_files.pv_pathway_layout import create_layout
from page_supporting_files.pv_pathway_callbacks import register_callbacks
import json

from app import app

FILE_LIST, DATA, DF = load_dataframe('data/pv_pathway_data_2604.parquet')
INDEX_MAP = build_index_map(FILE_LIST)
register_callbacks(app, DATA, INDEX_MAP, DF, FILE_LIST)

# =========================
# DEBUG: check bad graphs
# =========================
# from page_supporting_files.pv_pathway_graph_builder import build_elements

# bad_eids = []
# good_eids = []

# for idx, row in DF.iterrows():
#     eid = row["eid"]

#     try:
#         elements, num_pathways = build_elements(
#             {"pathways_graph": row["pathways_graph"]}
#         )

#         if not elements or num_pathways == 0:
#             bad_eids.append(eid)
#         else:
#             good_eids.append(eid)

#     except Exception as e:
#         bad_eids.append(eid)
#         print(f"❌ Failed: {eid} → {e}")

# print(f"Good rows: {len(good_eids)}")
# print(f"Bad rows: {len(bad_eids)}")

# with open("good_eids.json", "w") as f:
#     json.dump(good_eids, f, indent=2)

def get_layout():
    return create_layout(FILE_LIST)

# Run
if __name__ == '__main__':
    app.run(debug=True)