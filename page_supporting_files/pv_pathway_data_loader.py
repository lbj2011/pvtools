import os
import json
import pandas as pd

def load_dataframe(path):
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".parquet":
        df = pd.read_parquet(path)

    elif ext == ".pkl":
        df = pd.read_pickle(path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


    file_list = df["eid"].tolist()
    data = {}

    for _, row in df.iterrows():
        key = row["eid"]

        data[key] = {
            "pathways_graph": row["pathways_graph"]
        }

    return file_list, data, df

def build_index_map(file_list):
    return {i: name for i, name in enumerate(file_list)}