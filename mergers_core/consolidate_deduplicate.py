# Run by consolidate.sh

import pandas as pd


df = pd.read_csv("data/results/results.csv", dtype={"district": str})
columns_to_deduplicate = [
    "district",
    "dissimilarity_weight",
    "population_metric_weight",
]

duplicated_mask = df.duplicated(subset=columns_to_deduplicate, keep="first")
if duplicated_mask.any():
    print("dropped duplicates(s):")
    dropped_keys = df[duplicated_mask][columns_to_deduplicate].drop_duplicates()
    print(dropped_keys)
else:
    print("no duplicates")

df = df[~duplicated_mask]
df.to_csv("results.csv", index=False)
